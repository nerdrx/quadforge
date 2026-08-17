"""Mesh analysis: hard-edge detection, material boundaries, curvature/paint
density, and the curvature-aware post-passes that give QuadriFlow adaptivity it
does not natively support.

Everything here is numpy-over-foreach for speed and must import cleanly with no
UI / context access.
"""

from __future__ import annotations

import numpy as np

import bmesh

DENSITY_ATTR = "qf_density"

# ---------------------------------------------------------------------------
# low level array helpers
# ---------------------------------------------------------------------------


def verts_co(mesh) -> np.ndarray:
    n = len(mesh.vertices)
    a = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", a)
    return a.reshape(n, 3)


def edge_verts(mesh) -> np.ndarray:
    n = len(mesh.edges)
    a = np.empty(n * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", a)
    return a.reshape(n, 2)


def poly_normals(mesh) -> np.ndarray:
    n = len(mesh.polygons)
    a = np.empty(n * 3, dtype=np.float64)
    mesh.polygons.foreach_get("normal", a)
    return a.reshape(n, 3)


def vert_normals(mesh) -> np.ndarray:
    n = len(mesh.vertices)
    a = np.empty(n * 3, dtype=np.float64)
    mesh.vertex_normals.foreach_get("vector", a)
    return a.reshape(n, 3)


def loop_face_map(mesh):
    """Return (loop_edge_index, loop_face_index) int arrays."""
    nl = len(mesh.loops)
    npoly = len(mesh.polygons)
    loop_edge = np.empty(nl, dtype=np.int32)
    mesh.loops.foreach_get("edge_index", loop_edge)
    ltot = np.empty(npoly, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", ltot)
    loop_face = np.repeat(np.arange(npoly, dtype=np.int32), ltot)
    return loop_edge, loop_face


def edge_face_incidence(mesh):
    """Return (counts, f0, f1) per edge. f0/f1 are -1 when the edge has fewer
    than 1/2 incident faces."""
    ne = len(mesh.edges)
    loop_edge, loop_face = loop_face_map(mesh)
    counts = np.bincount(loop_edge, minlength=ne).astype(np.int32)
    order = np.argsort(loop_edge, kind="stable")
    f_sorted = loop_face[order]
    starts = np.zeros(ne, dtype=np.int64)
    if ne:
        starts[1:] = np.cumsum(counts)[:-1]
    f0 = np.full(ne, -1, dtype=np.int32)
    f1 = np.full(ne, -1, dtype=np.int32)
    m1 = counts >= 1
    m2 = counts >= 2
    if m1.any():
        f0[m1] = f_sorted[starts[m1]]
    if m2.any():
        f1[m2] = f_sorted[starts[m2] + 1]
    return counts, f0, f1


def dihedral_angles(mesh):
    """Per-edge dihedral angle in radians; 0.0 for boundary / non-manifold."""
    counts, f0, f1 = edge_face_incidence(mesh)
    ang = np.zeros(len(mesh.edges), dtype=np.float64)
    man = counts == 2
    if not man.any():
        return ang, counts
    pn = poly_normals(mesh)
    d = np.einsum("ij,ij->i", pn[f0[man]], pn[f1[man]])
    ang[man] = np.arccos(np.clip(d, -1.0, 1.0))
    return ang, counts


def world_area(mesh, matrix) -> float:
    """Surface area of `mesh` after applying `matrix` (4x4)."""
    try:
        mesh.calc_loop_triangles()
    except Exception:
        pass
    nt = len(mesh.loop_triangles)
    if nt == 0:
        return 0.0
    tv = np.empty(nt * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tv)
    tv = tv.reshape(nt, 3)
    co = verts_co(mesh)
    m = np.array(matrix, dtype=np.float64)  # row-major 4x4
    co = co @ m[:3, :3].T + m[:3, 3]
    a = co[tv[:, 0]]
    b = co[tv[:, 1]]
    c = co[tv[:, 2]]
    cr = np.cross(b - a, c - a)
    return float(0.5 * np.linalg.norm(cr, axis=1).sum())


def _read_float_attr(mesh, name, domain, length):
    attr = mesh.attributes.get(name)
    if attr is None or attr.domain != domain or attr.data_type != "FLOAT":
        return None
    if len(attr.data) != length:
        return None
    out = np.empty(length, dtype=np.float64)
    attr.data.foreach_get("value", out)
    return out


def _write_float_attr(mesh, name, domain, values):
    attr = mesh.attributes.get(name)
    if attr is not None and (attr.domain != domain or attr.data_type != "FLOAT"):
        mesh.attributes.remove(attr)
        attr = None
    if attr is None:
        attr = mesh.attributes.new(name, "FLOAT", domain)
    attr = mesh.attributes[name]
    attr.data.foreach_set("value", np.ascontiguousarray(values, dtype=np.float32))
    return attr


# ---------------------------------------------------------------------------
# hard edges
# ---------------------------------------------------------------------------


def mark_hard_edges(work_obj, s) -> int:
    """Mark hard edges as sharp on `work_obj`. Returns the number of sharp edges."""
    mesh = work_obj.data
    ne = len(mesh.edges)
    if ne == 0:
        return 0

    sharp = np.zeros(ne, dtype=bool)

    if getattr(s, "use_marked_sharp", False):
        existing = np.zeros(ne, dtype=bool)
        mesh.edges.foreach_get("use_edge_sharp", existing)
        sharp |= existing
        seam = np.zeros(ne, dtype=bool)
        mesh.edges.foreach_get("use_seam", seam)
        sharp |= seam
        crease = _read_float_attr(mesh, "crease_edge", "EDGE", ne)
        if crease is not None:
            sharp |= crease > 0.5

    counts = None
    if getattr(s, "detect_hard_edges", True):
        ang, counts = dihedral_angles(mesh)
        thr = float(getattr(s, "hard_edge_angle", 0.6981317))
        sharp |= ang >= thr

    if counts is None:
        counts, _f0, _f1 = edge_face_incidence(mesh)
    # non-manifold edges always count as hard - they are creases by nature
    sharp |= counts > 2

    mesh.edges.foreach_set("use_edge_sharp", sharp)
    mesh.update()
    return int(sharp.sum())


def material_boundaries_to_sharp(work_obj) -> int:
    """Edges between faces with different material_index become sharp."""
    mesh = work_obj.data
    ne = len(mesh.edges)
    npoly = len(mesh.polygons)
    if ne == 0 or npoly == 0:
        return 0
    mats = np.empty(npoly, dtype=np.int32)
    mesh.polygons.foreach_get("material_index", mats)
    if mats.max(initial=0) == mats.min(initial=0):
        return 0
    counts, f0, f1 = edge_face_incidence(mesh)
    man = counts == 2
    boundary = np.zeros(ne, dtype=bool)
    if man.any():
        boundary[man] = mats[f0[man]] != mats[f1[man]]
    if not boundary.any():
        return 0
    sharp = np.zeros(ne, dtype=bool)
    mesh.edges.foreach_get("use_edge_sharp", sharp)
    sharp |= boundary
    mesh.edges.foreach_set("use_edge_sharp", sharp)
    mesh.update()
    return int(boundary.sum())


def uv_island_boundaries_to_sharp(work_obj, eps: float = 1e-5) -> int:
    """UV island boundaries become sharp feature edges, so the solvers align
    edge flow along them and texture seams survive the remesh.

    An edge is an island boundary when its two adjacent faces disagree about
    either endpoint's UV coordinate (or when it is explicitly marked as a
    seam)."""
    mesh = work_obj.data
    uvl = mesh.uv_layers.active
    ne = len(mesh.edges)
    nl = len(mesh.loops)
    if uvl is None or ne == 0 or nl == 0:
        return 0

    uv = np.empty(nl * 2)
    uvl.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    lv = np.empty(nl, dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", lv)
    # loop -> owning face
    starts = np.empty(len(mesh.polygons), dtype=np.int64)
    totals = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", totals)
    lf = np.repeat(np.arange(len(mesh.polygons), dtype=np.int64), totals)

    # (face, vert) -> loop lookup via sorted composite keys
    nv = len(mesh.vertices)
    keys = lf * nv + lv
    order = np.argsort(keys)
    sorted_keys = keys[order]

    def uv_at(face_idx, vert_idx):
        k = face_idx * nv + vert_idx
        pos = np.searchsorted(sorted_keys, k)
        pos = np.clip(pos, 0, nl - 1)
        hit = sorted_keys[pos] == k
        li = order[pos]
        out = np.zeros((len(k), 2))
        out[hit] = uv[li[hit]]
        return out, hit

    counts, f0, f1 = edge_face_incidence(mesh)
    ev = np.empty(ne * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)

    man = counts == 2
    boundary = np.zeros(ne, dtype=bool)
    if man.any():
        idx = np.nonzero(man)[0]
        for corner in (0, 1):
            v = ev[idx, corner]
            a, ha = uv_at(f0[idx], v)
            b, hb = uv_at(f1[idx], v)
            ok = ha & hb
            diff = np.zeros(len(idx), dtype=bool)
            diff[ok] = np.abs(a[ok] - b[ok]).max(axis=1) > eps
            boundary[idx] |= diff

    seams = np.zeros(ne, dtype=bool)
    mesh.edges.foreach_get("use_seam", seams)
    boundary |= seams

    if not boundary.any():
        return 0
    sharp = np.zeros(ne, dtype=bool)
    mesh.edges.foreach_get("use_edge_sharp", sharp)
    sharp |= boundary
    mesh.edges.foreach_set("use_edge_sharp", sharp)
    mesh.update()
    return int(boundary.sum())


# ---------------------------------------------------------------------------
# density / curvature
# ---------------------------------------------------------------------------


def curvature_per_vertex(mesh) -> np.ndarray:
    """Normalised (0..1) curvature estimate per vertex from 1-ring normal
    variation, robust-normalised against the 95th percentile."""
    nv = len(mesh.vertices)
    if nv == 0:
        return np.zeros(0)
    ev = edge_verts(mesh)
    if len(ev) == 0:
        return np.zeros(nv)
    vn = vert_normals(mesh)
    a = ev[:, 0]
    b = ev[:, 1]
    dev = 1.0 - np.einsum("ij,ij->i", vn[a], vn[b])  # 0 .. 2
    np.clip(dev, 0.0, 2.0, out=dev)
    acc = np.bincount(a, weights=dev, minlength=nv) + np.bincount(b, weights=dev, minlength=nv)
    cnt = np.bincount(a, minlength=nv) + np.bincount(b, minlength=nv)
    cnt = np.maximum(cnt, 1)
    curv = acc / cnt
    hi = float(np.percentile(curv, 95.0)) if nv > 1 else float(curv.max(initial=0.0))
    if hi <= 1e-12:
        return np.zeros(nv)
    return np.clip(curv / hi, 0.0, 1.0)


def build_density_attr(work_obj, s) -> None:
    """Write the float point attribute 'qf_density'.

    density > 1 -> denser (smaller quads) wanted, < 1 -> sparser.
    Uniform (1.0 everywhere) when adaptive_size == 0 and no paint layer.
    """
    mesh = work_obj.data
    nv = len(mesh.vertices)
    if nv == 0:
        return

    paint = None
    if getattr(s, "use_paint_density", False):
        paint = _read_float_attr(mesh, DENSITY_ATTR, "POINT", nv)
        if paint is not None:
            paint = np.clip(paint, 0.0, 2.0)

    a = float(getattr(s, "adaptive_size", 0.0)) / 100.0
    if a > 0.0:
        curv = curvature_per_vertex(mesh)
        density = 1.0 + a * (2.0 * curv - 1.0)
    else:
        density = np.ones(nv, dtype=np.float64)

    if paint is not None:
        density = density * paint

    np.clip(density, 0.05, 4.0, out=density)
    _write_float_attr(mesh, DENSITY_ATTR, "POINT", density)
    mesh.update()


def read_density(mesh) -> np.ndarray | None:
    return _read_float_attr(mesh, DENSITY_ATTR, "POINT", len(mesh.vertices))


def density_target_scale(mesh) -> float:
    """Mean density -> multiplicative correction for the requested face count."""
    d = read_density(mesh)
    if d is None or d.size == 0:
        return 1.0
    m = float(np.mean(d))
    if not np.isfinite(m) or m <= 0.0:
        return 1.0
    return float(np.clip(m, 0.25, 4.0))


# ---------------------------------------------------------------------------
# adaptive post-pass: conservative all-quad edge-loop removal
# ---------------------------------------------------------------------------


def _sample_density_onto(new_mesh, src_mesh) -> np.ndarray | None:
    """Nearest-vertex transfer of qf_density from src_mesh onto new_mesh verts."""
    from mathutils import kdtree

    d = read_density(src_mesh)
    if d is None:
        return None
    src_co = verts_co(src_mesh)
    n = len(src_co)
    if n == 0:
        return None
    tree = kdtree.KDTree(n)
    for i, c in enumerate(src_co):
        tree.insert((float(c[0]), float(c[1]), float(c[2])), i)
    tree.balance()
    new_co = verts_co(new_mesh)
    out = np.ones(len(new_co), dtype=np.float64)
    for i, c in enumerate(new_co):
        _co, idx, _dist = tree.find((float(c[0]), float(c[1]), float(c[2])))
        if idx is not None:
            out[i] = d[idx]
    return out


def _walk_edge_loop(start_edge):
    """Walk a bmesh edge loop through valence-4 all-quad vertices.

    Returns (edges, closed). Returns (None, False) if the loop leaves the
    regular quad region (irregular vertex, boundary, non-quad face)."""
    edges = [start_edge]
    seen = {start_edge}
    for direction in (0, 1):
        e = start_edge
        v = start_edge.verts[direction]
        while True:
            if len(v.link_edges) != 4:
                return None, False
            if len(v.link_faces) != 4:
                return None, False
            if any(len(f.verts) != 4 for f in v.link_faces):
                return None, False
            e_faces = set(e.link_faces)
            nxt = None
            for ce in v.link_edges:
                if ce is e:
                    continue
                if not (set(ce.link_faces) & e_faces):
                    nxt = ce
                    break
            if nxt is None:
                return None, False
            if nxt is start_edge:
                return edges, True
            if nxt in seen:
                return None, False
            seen.add(nxt)
            edges.append(nxt)
            e = nxt
            v = e.other_vert(v)
    return None, False


def _locked_verts(mesh) -> np.ndarray:
    """Vertices that must not move: on sharp edges, seams or open boundaries."""
    nv = len(mesh.vertices)
    ne = len(mesh.edges)
    lock = np.zeros(nv, dtype=bool)
    if ne == 0:
        return lock
    ev = edge_verts(mesh)
    sharp = np.zeros(ne, dtype=bool)
    mesh.edges.foreach_get("use_edge_sharp", sharp)
    seam = np.zeros(ne, dtype=bool)
    mesh.edges.foreach_get("use_seam", seam)
    counts, _f0, _f1 = edge_face_incidence(mesh)
    bad = sharp | seam | (counts != 2)
    if bad.any():
        lock[ev[bad, 0]] = True
        lock[ev[bad, 1]] = True
    return lock


def density_relax(new_obj, src_mesh, s, iterations: int = 0, blend: float = 0.5) -> dict:
    """Density-driven tangential relaxation.

    Every edge is a spring whose rest length is inversely proportional to the
    local ``qf_density``; vertices are re-projected onto the original surface
    after each step. Quads end up smaller where density is high and larger where
    it is low. Purely positional - the all-quad topology and the face count stay
    exactly as the solver produced them.

    A quality guard rolls back to the last known-good state if a face is about
    to collapse, so the pass can only ever improve or no-op.
    """
    from mathutils.bvhtree import BVHTree

    out = {"iterations": 0, "moved": 0, "max_shift": 0.0, "rolled_back": False}
    mesh = new_obj.data
    nv = len(mesh.vertices)
    if nv == 0 or len(mesh.edges) == 0:
        return out

    dens = _sample_density_onto(mesh, src_mesh)
    if dens is None:
        return out
    spread = float(np.percentile(dens, 95) - np.percentile(dens, 5))
    if spread < 1e-3:
        return out
    dens = np.clip(dens, 0.05, 4.0)

    # BVH of the source surface for re-projection
    try:
        src_mesh.calc_loop_triangles()
        nt = len(src_mesh.loop_triangles)
        if nt == 0:
            return out
        tv = np.empty(nt * 3, dtype=np.int32)
        src_mesh.loop_triangles.foreach_get("vertices", tv)
        src_co = verts_co(src_mesh)
        bvh = BVHTree.FromPolygons(
            [tuple(map(float, c)) for c in src_co],
            [tuple(int(x) for x in t) for t in tv.reshape(nt, 3)],
            all_triangles=True,
        )
    except Exception:
        return out

    ev = edge_verts(mesh)
    a = ev[:, 0].astype(np.int64)
    b = ev[:, 1].astype(np.int64)
    lock = _locked_verts(mesh)
    free = ~lock
    if not free.any():
        return out

    co = verts_co(mesh)
    valence = np.maximum(
        np.bincount(a, minlength=nv) + np.bincount(b, minlength=nv), 1
    ).astype(np.float64)

    # rest length per edge: inversely proportional to density, globally
    # normalised so the total wanted length matches the length available
    rest = 1.0 / (0.5 * (dens[a] + dens[b]))
    cur = np.linalg.norm(co[a] - co[b], axis=1)
    if rest.sum() <= 1e-12 or cur.sum() <= 1e-12:
        return out
    rest *= cur.sum() / rest.sum()
    # rollback floor: only near-degenerate edges are considered a failure
    min_len = 0.25 * min(float(cur.min()), float(rest.min()))
    lap_w = 0.18

    if iterations <= 0:
        iterations = 60 if nv <= 20000 else (25 if nv <= 60000 else 10)
    iterations = int(max(1, min(iterations, 400)))

    good_co = co.copy()
    total_shift = 0.0
    moved = 0
    for it in range(iterations):
        e = co[b] - co[a]
        length = np.linalg.norm(e, axis=1)
        ok = length > 1e-12
        direction = np.zeros_like(e)
        direction[ok] = e[ok] / length[ok, None]
        stretch = (length - rest)[:, None] * direction

        force = np.zeros((nv, 3), dtype=np.float64)
        np.add.at(force, a, stretch)
        np.add.at(force, b, -stretch)
        force /= valence[:, None]

        # light Laplacian keeps the quads from skewing
        lap_num = np.zeros((nv, 3), dtype=np.float64)
        np.add.at(lap_num, a, co[b])
        np.add.at(lap_num, b, co[a])
        lap = lap_num / valence[:, None] - co

        delta = blend * ((1.0 - lap_w) * force + lap_w * lap)
        # clamp against the *shortest incident* edge: a mean- or global-based
        # clamp lets one step collapse the short edges of an uneven mesh
        h_min = np.full(nv, np.inf)
        np.minimum.at(h_min, a, length)
        np.minimum.at(h_min, b, length)
        h_min[~np.isfinite(h_min)] = 0.0
        max_step = 0.20 * h_min
        dl = np.linalg.norm(delta, axis=1)
        scale = np.ones(nv)
        big = dl > max_step
        scale[big] = max_step[big] / dl[big]
        delta *= scale[:, None]
        delta[~free] = 0.0

        new_co = co + delta
        idx = np.nonzero(np.linalg.norm(delta, axis=1) > 1e-12)[0]
        moved = int(len(idx))
        if moved == 0:
            break
        for i in idx:
            p = (float(new_co[i, 0]), float(new_co[i, 1]), float(new_co[i, 2]))
            hit = bvh.find_nearest(p)
            if hit is None or hit[0] is None:
                continue
            snapped = np.array(hit[0], dtype=np.float64)
            # on thin geometry the nearest point can sit on a different sheet;
            # a jump that large would fold the mesh, so keep the free position
            if np.linalg.norm(snapped - new_co[i]) > 0.6 * h_min[i]:
                continue
            new_co[i] = snapped

        # quality guard: never let an edge collapse
        nl = np.linalg.norm(new_co[a] - new_co[b], axis=1)
        if float(nl.min()) < min_len:
            out["rolled_back"] = True
            co = good_co
            break

        total_shift = max(total_shift, float(np.linalg.norm(new_co - co, axis=1).max()))
        co = new_co
        out["iterations"] += 1
        if it % 5 == 0:
            good_co = co.copy()

    if out["iterations"] == 0:
        return out

    mesh.vertices.foreach_set("co", np.ascontiguousarray(co.ravel(), dtype=np.float32))
    mesh.update()
    out["moved"] = moved
    out["max_shift"] = round(total_shift, 6)
    out["density_spread"] = round(spread, 4)
    return out


def adaptive_decimate(new_obj, src_mesh, s, face_target: int) -> dict:
    """Remove whole edge loops that live entirely in low-density regions.

    Only closed loops through fully regular (valence-4, all-quad) neighbourhoods
    are considered, so the result stays 100% quads. Returns a stats dict.
    """
    stats = {"loops_removed": 0, "faces_before": len(new_obj.data.polygons), "faces_after": 0}
    mesh = new_obj.data
    if len(mesh.polygons) <= face_target:
        stats["faces_after"] = len(mesh.polygons)
        return stats

    vdens = _sample_density_onto(mesh, src_mesh)
    if vdens is None:
        stats["faces_after"] = len(mesh.polygons)
        return stats

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # per face mean density
    fdens = {}
    for f in bm.faces:
        fdens[f.index] = float(np.mean([vdens[v.index] for v in f.verts]))

    # threshold: only touch the sparser half of the mesh
    all_d = np.array(list(fdens.values())) if fdens else np.zeros(0)
    if all_d.size == 0:
        bm.free()
        stats["faces_after"] = len(mesh.polygons)
        return stats
    thresh = float(np.percentile(all_d, 55.0))

    candidates = []
    visited = set()
    for e in bm.edges:
        if e.index in visited or e.seam or not e.smooth:
            continue
        if len(e.link_faces) != 2:
            continue
        loop, closed = _walk_edge_loop(e)
        if not closed or loop is None:
            visited.add(e.index)
            continue
        for le in loop:
            visited.add(le.index)
        # reject loops touching sharp / seam edges or dense faces
        bad = False
        faces = set()
        dsum = 0.0
        for le in loop:
            if (not le.smooth) or le.seam or len(le.link_faces) != 2:
                bad = True
                break
            for f in le.link_faces:
                faces.add(f.index)
        if bad or not faces:
            continue
        for fi in faces:
            dsum += fdens.get(fi, 1.0)
        mean_d = dsum / len(faces)
        if mean_d > thresh:
            continue
        if max(fdens.get(fi, 1.0) for fi in faces) > thresh * 1.15 + 1e-9:
            continue
        candidates.append((mean_d, len(loop), [le.index for le in loop], faces))

    if not candidates:
        bm.free()
        stats["faces_after"] = len(mesh.polygons)
        return stats

    candidates.sort(key=lambda c: c[0])

    budget = len(bm.faces) - face_target
    used_faces = set()
    chosen = []
    for mean_d, nedges, eidx, faces in candidates:
        if budget <= 0:
            break
        if faces & used_faces:
            continue
        if nedges > budget:
            continue
        used_faces |= faces
        chosen.extend(eidx)
        budget -= nedges
        stats["loops_removed"] += 1

    if not chosen:
        bm.free()
        stats["faces_after"] = len(mesh.polygons)
        return stats

    bm.edges.ensure_lookup_table()
    edges = [bm.edges[i] for i in chosen]
    try:
        bmesh.ops.dissolve_edges(bm, edges=edges, use_verts=True, use_face_split=False)
    except Exception:
        bm.free()
        stats["faces_after"] = len(mesh.polygons)
        stats["loops_removed"] = 0
        return stats

    # safety: if the dissolve produced non quads, throw the result away
    nonquad = sum(1 for f in bm.faces if len(f.verts) != 4)
    if nonquad:
        bm.free()
        stats["faces_after"] = len(mesh.polygons)
        stats["loops_removed"] = 0
        stats["rejected_nonquad"] = nonquad
        return stats

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    stats["faces_after"] = len(mesh.polygons)
    return stats
