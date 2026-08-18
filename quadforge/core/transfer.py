"""QuadForge — data preservation (core/transfer.py).

Captures everything that lives on a mesh *before* remeshing and re-projects it
onto the new topology afterwards using a surface-nearest / barycentric mapping.

Public API (see CONTRACTS.md):

    capture(obj) -> Snapshot
    apply(snapshot, new_obj, s) -> dict

The snapshot is a plain Python object holding numpy arrays only (plus material
datablock references, which survive the mesh swap and are re-resolved by name if
they do not).  No mesh / object / key datablock is kept alive by a Snapshot.

Everything is vectorised with numpy + foreach_get / foreach_set; the Python
level loops are the BVH nearest-surface queries (~2.5 M queries/s), the
vertex-group write-back (~1.6 M weights/s), the per-arc crease/bevel chain
router and the per-corner UV clearance probes — the last two only ever walk the
handful of edges and corners that are actually contested, and the first two are
fast enough for multi-hundred-thousand vertex meshes.
"""

from __future__ import annotations

import time

import numpy as np

import bpy
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

__all__ = ["Snapshot", "capture", "apply"]


# Generic attribute names Blender 4.1+ / 5.x uses for creases & bevel weights.
CREASE_EDGE = "crease_edge"
CREASE_VERT = "crease_vert"
BWEIGHT_EDGE = "bevel_weight_edge"
BWEIGHT_VERT = "bevel_weight_vert"

_EPS = 1e-12
_WEIGHT_EPS = 1e-6
# How far a loop's sample point is pulled towards its face centre when looking
# up UVs.  Large enough to land on the correct side of a UV seam, small enough
# not to skip across a whole source face.
_LOOP_INSET = 0.08
# Two source polygons belong to the same UV region when their shared edge has
# matching UVs on both sides (in *every* layer) to within this tolerance.
_UV_SEAM_EPS = 1e-6
# Above this many marked source edges the chain router gives up its per-arc
# work and the legacy nearest-edge match takes over (a mesh where *everything*
# is creased has no chains worth following anyway).
_MAX_CHAIN_EDGES = 40000
# Corridor widths (in units of the local edge scale) tried in turn when routing
# a source chain through the output edge graph.
_CORRIDOR_STEPS = (1.0, 1.75, 3.0)
# Material vote: sample points per corner triangle of an output face (1..3).
_MATERIAL_SAMPLES = 3
# Bounded UV extrapolation: how far (UV units) island clearances are measured
# at all.  ~20 texels on a 2K map; islands farther apart than this are simply
# "far enough", and half of it is the most any corner may ever extrapolate.
_UV_CLEARANCE_CAP = 0.01
_UV_EXTRAPOLATE = True


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _fget(coll, attr, size, dtype):
    a = np.empty(size, dtype)
    coll.foreach_get(attr, a)
    return a


def _attr_array(mesh, name, domain, count):
    """Read a FLOAT attribute as (count,) f4, or None when absent/empty."""
    at = mesh.attributes.get(name)
    if at is None or at.domain != domain or at.data_type != 'FLOAT':
        return None
    if len(at.data) != count or count == 0:
        return None
    a = np.empty(count, 'f4')
    at.data.foreach_get("value", a)
    if not np.any(a):
        return None
    return a


def _ensure_float_attr(mesh, name, domain, count):
    at = mesh.attributes.get(name)
    if at is not None and (at.domain != domain or at.data_type != 'FLOAT'):
        return None
    if at is None:
        at = mesh.attributes.new(name, 'FLOAT', domain)
    if len(at.data) != count:
        return None
    return at


def _poly_arrays(mesh):
    npoly = len(mesh.polygons)
    nloop = len(mesh.loops)
    starts = _fget(mesh.polygons, "loop_start", npoly, 'i4')
    totals = _fget(mesh.polygons, "loop_total", npoly, 'i4')
    lverts = _fget(mesh.loops, "vertex_index", nloop, 'i4')
    return starts, totals, lverts


def _triangulate(starts, totals, loop_verts):
    """Fan-triangulate polygons.  Fully vectorised.

    Returns (tri_loops (nt,3) i4, tris (nt,3) i4, tri_poly (nt,) i4).
    """
    if starts.size == 0:
        z3 = np.zeros((0, 3), 'i4')
        return z3, z3, np.zeros(0, 'i4')
    per = np.maximum(totals - 2, 0).astype('i8')
    total = int(per.sum())
    if total == 0:
        z3 = np.zeros((0, 3), 'i4')
        return z3, z3, np.zeros(0, 'i4')
    poly_idx = np.repeat(np.arange(starts.size, dtype='i4'), per)
    ends = np.cumsum(per)
    k = np.arange(total, dtype='i8') - np.repeat(ends - per, per) + 1
    s = starts[poly_idx].astype('i8')
    tri_loops = np.stack([s, s + k, s + k + 1], axis=1).astype('i4')
    tris = loop_verts[tri_loops]
    return tri_loops, tris, poly_idx


def _poly_centers(mesh):
    npoly = len(mesh.polygons)
    if npoly == 0:
        return np.zeros((0, 3), 'f8')
    try:
        c = _fget(mesh.polygons, "center", npoly * 3, 'f8')
        return c.reshape(-1, 3)
    except Exception:
        starts, totals, lverts = _poly_arrays(mesh)
        V = _fget(mesh.vertices, "co", len(mesh.vertices) * 3, 'f8').reshape(-1, 3)
        out = np.zeros((npoly, 3), 'f8')
        acc = np.zeros((npoly, 3), 'f8')
        pidx = np.repeat(np.arange(npoly), totals)
        np.add.at(acc, pidx, V[lverts])
        out[:] = acc / np.maximum(totals, 1)[:, None]
        return out


def _material_ok(mat):
    if mat is None:
        return False
    try:
        mat.name
        return True
    except ReferenceError:
        return False


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class Snapshot:
    """Standalone (numpy-only) copy of everything worth preserving."""

    def __init__(self):
        self.object_name = ""
        self.mesh_name = ""
        self.warnings = []

        # geometry -------------------------------------------------------
        self.verts = np.zeros((0, 3), 'f8')        # surface positions (BVH)
        self.base_verts = np.zeros((0, 3), 'f8')   # base (shape-key space)
        self.loop_starts = np.zeros(0, 'i4')
        self.loop_totals = np.zeros(0, 'i4')
        self.loop_verts = np.zeros(0, 'i4')
        self.edges = np.zeros((0, 2), 'i4')
        self.tri_loops = np.zeros((0, 3), 'i4')
        self.tris = np.zeros((0, 3), 'i4')
        self.tri_poly = np.zeros(0, 'i4')

        # materials ------------------------------------------------------
        self.materials = []            # list of (Material|None, name|"")
        self.poly_material = np.zeros(0, 'i4')

        # uvs ------------------------------------------------------------
        self.uv_layers = []            # [{'name','uv'(nl,2)f4,'active','active_render'}]

        # vertex groups --------------------------------------------------
        self.vgroups = []              # [{'name','weights'(nv,)f4,'lock'}]

        # shape keys -----------------------------------------------------
        self.shape_keys = []           # [{...,'delta'(nv,3)f4}]
        self.key_name = ""
        self.key_use_relative = True
        self.key_reference_name = ""

        # edge / vertex scalar data -------------------------------------
        self.crease_edge = None
        self.crease_vert = None
        self.bweight_edge = None
        self.bweight_vert = None
        self.sharp_edge = None

        self.has_custom_normals = False

        self._bvh = None
        self._uv_regions = None        # per-polygon UV-island id (lazy)
        self._uv_region_index = None   # (order, start, end) into snap.tris
        self._uv_region_bvh = {}       # island id -> (BVHTree, global tri ids)
        self._uv_clearance = {}        # UV-clearance cache (layer BVHs, gaps)

    # -- derived ---------------------------------------------------------
    @property
    def nverts(self):
        return self.verts.shape[0]

    @property
    def nloops(self):
        return self.loop_verts.shape[0]

    @property
    def npolys(self):
        return self.loop_starts.shape[0]

    def bvh(self):
        """Lazily built BVH over the fan-triangulated snapshot surface."""
        if self._bvh is None:
            if self.tris.shape[0] == 0:
                return None
            self._bvh = BVHTree.FromPolygons(
                self.verts.tolist(),
                [tuple(t) for t in self.tris.tolist()],
                all_triangles=True,
            )
        return self._bvh

    def free(self):
        self._bvh = None
        self._uv_region_bvh = {}
        self._uv_clearance = {}

    def __repr__(self):
        return (
            "<qf.Snapshot %r verts=%d polys=%d uv=%d vg=%d keys=%d mats=%d>"
            % (self.object_name, self.nverts, self.npolys, len(self.uv_layers),
               len(self.vgroups), len(self.shape_keys), len(self.materials))
        )


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

def capture(obj, use_evaluated=True):
    """Snapshot everything preservable on ``obj`` before it gets remeshed.

    Geometry, UVs, vertex groups and shape keys are read from the *base* mesh so
    that every array shares one vertex indexing.  When the object additionally
    has purely deforming modifiers (evaluated mesh has identical vertex / loop
    counts) the evaluated positions are used for the BVH surface, so the mapping
    matches the surface the remesher actually saw.
    """
    snap = Snapshot()
    if obj is None or obj.type != 'MESH' or obj.data is None:
        snap.warnings.append("capture: not a mesh object")
        return snap

    me = obj.data
    snap.object_name = obj.name
    snap.mesh_name = me.name

    nv = len(me.vertices)
    nl = len(me.loops)
    ne = len(me.edges)
    npo = len(me.polygons)

    # --- geometry -------------------------------------------------------
    base = _fget(me.vertices, "co", nv * 3, 'f8').reshape(-1, 3)
    snap.base_verts = base
    snap.verts = base
    snap.loop_starts, snap.loop_totals, snap.loop_verts = _poly_arrays(me)
    if ne:
        snap.edges = _fget(me.edges, "vertices", ne * 2, 'i4').reshape(-1, 2)
    snap.tri_loops, snap.tris, snap.tri_poly = _triangulate(
        snap.loop_starts, snap.loop_totals, snap.loop_verts)

    # --- evaluated surface (deform-only modifiers) ----------------------
    if use_evaluated and obj.modifiers:
        try:
            dg = bpy.context.evaluated_depsgraph_get()
            ev = obj.evaluated_get(dg)
            em = ev.to_mesh()
            if em is not None and len(em.vertices) == nv and len(em.loops) == nl:
                snap.verts = _fget(em.vertices, "co", nv * 3, 'f8').reshape(-1, 3)
            elif em is not None:
                snap.warnings.append(
                    "modifiers change topology (%d->%d verts); mapping uses the "
                    "base mesh" % (nv, len(em.vertices)))
            ev.to_mesh_clear()
        except Exception as ex:                                  # pragma: no cover
            snap.warnings.append("evaluated mesh unavailable: %s" % ex)

    # --- materials ------------------------------------------------------
    for slot in obj.material_slots:
        mat = slot.material
        snap.materials.append((mat, mat.name if _material_ok(mat) else ""))
    if not snap.materials:
        for mat in me.materials:
            snap.materials.append((mat, mat.name if _material_ok(mat) else ""))
    if npo:
        snap.poly_material = _fget(me.polygons, "material_index", npo, 'i4')

    # --- uv layers ------------------------------------------------------
    active_name = me.uv_layers.active.name if me.uv_layers.active else ""
    render_name = ""
    for uvl in me.uv_layers:
        if getattr(uvl, "active_render", False):
            render_name = uvl.name
    for uvl in me.uv_layers:
        uv = np.empty(nl * 2, 'f4')
        uvl.data.foreach_get("uv", uv)
        snap.uv_layers.append({
            'name': uvl.name,
            'uv': uv.reshape(-1, 2),
            'active': uvl.name == active_name,
            'active_render': uvl.name == render_name,
        })

    # --- vertex groups --------------------------------------------------
    ngroups = len(obj.vertex_groups)
    if ngroups and nv:
        W = np.zeros((ngroups, nv), 'f4')
        for i, v in enumerate(me.vertices):
            for g in v.groups:
                gi = g.group
                if 0 <= gi < ngroups:
                    W[gi, i] = g.weight
        for gi, vg in enumerate(obj.vertex_groups):
            snap.vgroups.append({
                'name': vg.name,
                'weights': W[gi],
                'lock': bool(vg.lock_weight),
            })

    # --- shape keys -----------------------------------------------------
    key = me.shape_keys
    if key is not None and key.key_blocks and nv:
        snap.key_name = key.name
        snap.key_use_relative = bool(key.use_relative)
        ref = key.reference_key
        snap.key_reference_name = ref.name if ref else key.key_blocks[0].name
        basis = np.empty(nv * 3, 'f4')
        (ref or key.key_blocks[0]).data.foreach_get("co", basis)
        basis = basis.reshape(-1, 3)
        for kb in key.key_blocks:
            co = np.empty(nv * 3, 'f4')
            kb.data.foreach_get("co", co)
            co = co.reshape(-1, 3)
            snap.shape_keys.append({
                'name': kb.name,
                'is_reference': (ref is not None and kb == ref),
                'delta': (co - basis),
                'value': float(kb.value),
                'slider_min': float(kb.slider_min),
                'slider_max': float(kb.slider_max),
                'mute': bool(kb.mute),
                'interpolation': kb.interpolation,
                'relative_key': kb.relative_key.name if kb.relative_key else "",
                'vertex_group': kb.vertex_group or "",
            })

    # --- creases / bevel weights / sharp --------------------------------
    snap.crease_edge = _attr_array(me, CREASE_EDGE, 'EDGE', ne)
    snap.crease_vert = _attr_array(me, CREASE_VERT, 'POINT', nv)
    snap.bweight_edge = _attr_array(me, BWEIGHT_EDGE, 'EDGE', ne)
    snap.bweight_vert = _attr_array(me, BWEIGHT_VERT, 'POINT', nv)

    at = me.attributes.get("sharp_edge")
    if at is not None and at.domain == 'EDGE' and at.data_type == 'BOOLEAN' and ne:
        b = np.empty(ne, '?')
        at.data.foreach_get("value", b)
        snap.sharp_edge = b if b.any() else None

    try:
        snap.has_custom_normals = bool(me.has_custom_normals)
    except AttributeError:
        snap.has_custom_normals = me.attributes.get("custom_normal") is not None

    return snap


# ---------------------------------------------------------------------------
# surface mapping
# ---------------------------------------------------------------------------

def _nearest_tris(snap, points, normals=None):
    """Nearest-surface query.  points (n,3) -> (tri_idx (n,) i4, loc (n,3) f8).

    tri_idx is -1 where nothing was hit.

    When ``normals`` (n,3) is given the query is side-aware: on thin shells
    (limbs, ears, cards) the plain nearest hit is frequently the *opposite*
    wall, which silently transfers the wrong weights/deltas. A hit whose face
    normal opposes the query normal is re-tried from a point nudged outward
    along the query normal, keeping the better-facing hit.
    """
    n = points.shape[0]
    tri_idx = np.full(n, -1, 'i4')
    loc = np.array(points, dtype='f8', copy=True)
    bvh = snap.bvh()
    if bvh is None or n == 0:
        return tri_idx, loc
    fn = bvh.find_nearest
    ti = tri_idx
    lo = loc
    for i, p in enumerate(points.tolist()):
        r = fn(p)
        if r is None or r[2] is None:
            continue
        ti[i] = r[2]
        lo[i] = r[0]
        if normals is None:
            continue
        nq = normals[i]
        if r[1].dot((float(nq[0]), float(nq[1]), float(nq[2]))) >= 0.0:
            continue
        d0 = float(r[3]) if r[3] is not None else 0.0
        base = np.asarray(p)
        for off in (2.5 * d0 + 1e-6, 6.0 * d0 + 1e-6):
            probe = base + nq * off
            r2 = fn((float(probe[0]), float(probe[1]), float(probe[2])))
            if r2 is None or r2[2] is None:
                continue
            hit = np.asarray(r2[0])
            if (r2[1].dot((float(nq[0]), float(nq[1]), float(nq[2]))) >= 0.0
                    and np.linalg.norm(hit - base) <= 4.0 * off):
                ti[i] = r2[2]
                lo[i] = hit
                break
    return tri_idx, loc


def _bary(snap, tri_idx, points):
    """Barycentric coords of the point on each triangle closest to ``points``.

    This is the exact closest-point-on-triangle solve (Ericson, *Real-Time
    Collision Detection*, §5.1.5) vectorised over all points.  Using it instead
    of "project onto the plane and clip the weights" matters a lot: clipping and
    renormalising slides the sample along the wrong direction whenever a point
    falls outside its triangle, which happens constantly where the source mesh
    is dense and curved (sphere poles, creased corners).

    Returns (n,3) f8 weights that are always >= 0 and sum to 1.
    """
    n = points.shape[0]
    w = np.zeros((n, 3), 'f8')
    w[:, 0] = 1.0
    if n == 0:
        return w
    ok = tri_idx >= 0
    if not ok.any():
        return w

    t = snap.tris[tri_idx[ok]]
    V = snap.verts
    a = V[t[:, 0]]
    b = V[t[:, 1]]
    c = V[t[:, 2]]
    p = points[ok]

    ab = b - a
    ac = c - a
    dot = lambda x, y: np.einsum('ij,ij->i', x, y)      # noqa: E731

    ap = p - a
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def _div(num, den):
        return num / np.where(np.abs(den) < _EPS, 1.0, den)

    den = va + vb + vc
    iv = _div(vb, den)
    iw = _div(vc, den)
    sub = np.stack([1.0 - iv - iw, iv, iw], axis=1)     # interior (default)

    z = np.zeros(d1.shape, 'f8')
    one = np.ones(d1.shape, 'f8')
    e_ab = np.clip(_div(d1, d1 - d3), 0.0, 1.0)
    e_ac = np.clip(_div(d2, d2 - d6), 0.0, 1.0)
    e_bc = np.clip(_div(d4 - d3, (d4 - d3) + (d5 - d6)), 0.0, 1.0)

    regions = (
        ((d1 <= 0) & (d2 <= 0),                          np.stack([one, z, z], 1)),
        ((d3 >= 0) & (d4 <= d3),                         np.stack([z, one, z], 1)),
        ((d6 >= 0) & (d5 <= d6),                         np.stack([z, z, one], 1)),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0),              np.stack([1.0 - e_ab, e_ab, z], 1)),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0),              np.stack([1.0 - e_ac, z, e_ac], 1)),
        ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), np.stack([z, 1.0 - e_bc, e_bc], 1)),
    )
    # Later regions are overwritten by earlier ones -> apply in reverse order.
    for cond, val in reversed(regions):
        sub = np.where(cond[:, None], val, sub)

    np.clip(sub, 0.0, None, out=sub)
    ssum = sub.sum(axis=1)
    degen = ~np.isfinite(ssum) | (ssum < _EPS)
    sub[degen] = (1.0, 0.0, 0.0)
    ssum = np.where(degen, 1.0, ssum)
    sub /= ssum[:, None]

    w[ok] = sub
    return w


def _interp_vert(snap, tri_idx, bary, arr):
    """Barycentric interpolation of a per-vertex array (nv,) or (nv,k)."""
    safe = np.where(tri_idx >= 0, tri_idx, 0)
    t = snap.tris[safe]
    if arr.ndim == 1:
        return np.einsum('ij,ij->i', arr[t].astype('f8'), bary)
    return np.einsum('ijk,ij->ik', arr[t].astype('f8'), bary)


def _interp_loop(snap, tri_idx, bary, arr):
    """Barycentric interpolation of a per-loop array (nl,k) using the hit
    triangle's *own* corners — this is what keeps UV seams crisp."""
    safe = np.where(tri_idx >= 0, tri_idx, 0)
    t = snap.tri_loops[safe]
    return np.einsum('ijk,ij->ik', arr[t].astype('f8'), bary)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def apply(snapshot, new_obj, s=None):
    """Re-project ``snapshot`` onto ``new_obj`` (in place).

    ``s`` is an ``obj.quadforge`` settings group (or anything exposing the same
    ``preserve_*`` booleans); ``None`` means "preserve everything".
    """
    t0 = time.perf_counter()
    rep = {
        'ok': False,
        'uvs': False,
        'uv_layers': 0,
        'uv_seam_faces': 0,
        'uv_seam_faces_fixed': 0,
        'uv_seam_corners_fixed': 0,
        'weights': 0,
        'shape_keys': 0,
        'materials': 0,
        'creases': 0,
        'creases_skipped': 0,
        'bevel_weights': 0,
        'bevel_weights_skipped': 0,
        'unmapped': 0,
        'has_custom_normals': False,
        'time_s': 0.0,
        'warnings': list(getattr(snapshot, 'warnings', ())),
    }

    def _flag(name):
        return True if s is None else bool(getattr(s, name, True))

    if snapshot is None or new_obj is None or new_obj.type != 'MESH':
        rep['warnings'].append("apply: invalid target object")
        rep['time_s'] = time.perf_counter() - t0
        return rep

    me = new_obj.data
    nv = len(me.vertices)
    nl = len(me.loops)
    ne = len(me.edges)
    npo = len(me.polygons)
    rep['has_custom_normals'] = bool(snapshot.has_custom_normals)

    if snapshot.nverts == 0 or snapshot.tris.shape[0] == 0:
        rep['warnings'].append("apply: empty snapshot, nothing transferred")
        rep['time_s'] = time.perf_counter() - t0
        return rep
    if nv == 0:
        rep['warnings'].append("apply: target mesh is empty")
        rep['time_s'] = time.perf_counter() - t0
        return rep

    NV = _fget(me.vertices, "co", nv * 3, 'f8').reshape(-1, 3)
    try:
        VN = _fget(me.vertex_normals, "vector", nv * 3, 'f8').reshape(-1, 3)
    except Exception:
        VN = None

    # ---- per-vertex mapping (side-aware on thin shells) ----------------
    v_tri, v_loc = _nearest_tris(snapshot, NV, normals=VN)
    rep['unmapped'] = int((v_tri < 0).sum())
    # Barycentric coords of the *vertex itself* projected onto its triangle
    # (not of the projected point) so interpolation follows the vertex, and is
    # exact for vertices that lie on the source surface.
    v_bary = _bary(snapshot, v_tri, NV)

    # Vertex-identity override: geometry kept verbatim (shells the solver
    # refused, grafted regions) must take its own source vertex's data.
    # Nearest-surface is ambiguous inside stacked/interpenetrating card shells
    # (fur, feathers) — a coincident stack-mate's triangle wins arbitrarily and
    # its different weights make the card fly off when posed. An exact position
    # match is rewritten as a one-hot barycentric on the matched source vertex,
    # which every downstream interpolation then honours automatically.
    try:
        from mathutils.kdtree import KDTree
        S = snapshot.verts
        bb = S.max(0) - S.min(0)
        eps = 1e-5 * float(np.linalg.norm(bb))
        kt = KDTree(len(S))
        ins = kt.insert
        for j, c in enumerate(S.tolist()):
            ins(c, j)
        kt.balance()
        T = snapshot.tris
        v2tri = np.full(len(S), -1, 'i8')
        v2corner = np.zeros(len(S), 'i8')
        tidx = np.arange(len(T))
        for corner in range(3):
            v2tri[T[:, corner]] = tidx
            v2corner[T[:, corner]] = corner
        exact = 0
        find = kt.find
        for i, p in enumerate(NV.tolist()):
            _c, j, d = find(p)
            if j is None or d is None or d > eps:
                continue
            t = int(v2tri[j])
            if t < 0:
                continue
            v_tri[i] = t
            v_bary[i] = 0.0
            v_bary[i, int(v2corner[j])] = 1.0
            exact += 1
        rep['exact_vertex_matches'] = exact
    except Exception as exc:
        rep['warnings'].append(f"vertex-identity pass failed: {exc}")

    # ---- vertex groups -------------------------------------------------
    if _flag('preserve_weights') and snapshot.vgroups:
        # The work mesh can arrive with stale groups (carried through the
        # solver / duplicated un-swapped by the exact-symmetry mirror, e.g.
        # right-hand weights on the mirrored left hand). Transferred weights
        # are authoritative — start from a clean slate.
        for g in list(new_obj.vertex_groups):
            new_obj.vertex_groups.remove(g)
        for g in snapshot.vgroups:
            w = _interp_vert(snapshot, v_tri, v_bary, g['weights'])
            np.clip(w, 0.0, 1.0, out=w)
            vg = new_obj.vertex_groups.get(g['name'])
            if vg is None:
                vg = new_obj.vertex_groups.new(name=g['name'])
            idx = np.nonzero(w > _WEIGHT_EPS)[0]
            add = vg.add
            for i in idx.tolist():
                add((i,), float(w[i]), 'REPLACE')
            try:
                vg.lock_weight = g['lock']
            except Exception:
                pass
            rep['weights'] += 1

    # ---- shape keys ----------------------------------------------------
    if _flag('preserve_shape_keys') and snapshot.shape_keys:
        rep['shape_keys'] = _apply_shape_keys(snapshot, new_obj, me, nv,
                                              NV, v_tri, v_bary, rep)

    # ---- UVs (per loop, seam aware) ------------------------------------
    if _flag('preserve_uvs') and snapshot.uv_layers and nl:
        _apply_uvs(snapshot, me, nl, npo, NV, rep)

    # ---- materials -----------------------------------------------------
    if _flag('preserve_materials') and snapshot.materials:
        rep['materials'] = _apply_materials(snapshot, new_obj, me, npo, rep)

    # ---- creases / bevel weights ---------------------------------------
    do_cr = _flag('preserve_creases')
    do_bw = _flag('preserve_bevel_weights')
    if (do_cr or do_bw) and ne:
        EV = _fget(me.edges, "vertices", ne * 2, 'i4').reshape(-1, 2)
        cache = {}
        if do_cr and snapshot.crease_edge is not None:
            n_ok, n_skip = _transfer_edge_scalar(
                snapshot, me, EV, NV, snapshot.crease_edge, CREASE_EDGE, cache)
            rep['creases'] += n_ok
            rep['creases_skipped'] += n_skip
        if do_bw and snapshot.bweight_edge is not None:
            n_ok, n_skip = _transfer_edge_scalar(
                snapshot, me, EV, NV, snapshot.bweight_edge, BWEIGHT_EDGE, cache)
            rep['bevel_weights'] += n_ok
            rep['bevel_weights_skipped'] += n_skip
    if do_cr and snapshot.crease_vert is not None:
        rep['creases'] += _transfer_vert_scalar(
            snapshot, me, NV, snapshot.crease_vert, CREASE_VERT)
    if do_bw and snapshot.bweight_vert is not None:
        rep['bevel_weights'] += _transfer_vert_scalar(
            snapshot, me, NV, snapshot.bweight_vert, BWEIGHT_VERT)

    me.update()
    rep['ok'] = True
    rep['time_s'] = time.perf_counter() - t0
    return rep


# ---------------------------------------------------------------------------

def _apply_shape_keys(snap, new_obj, me, nv, NV, v_tri, v_bary, rep):
    """Rebuild the whole shape-key stack from basis-relative deltas."""
    try:
        new_obj.shape_key_clear()
    except Exception:
        pass

    ref_name = snap.key_reference_name or "Basis"
    basis_src = None
    for k in snap.shape_keys:
        if k['is_reference'] or k['name'] == ref_name:
            basis_src = k
            break

    basis_kb = new_obj.shape_key_add(name=ref_name, from_mix=False)
    # Basis == current mesh positions; shape_key_add already does that, but be
    # explicit so we control the array we add deltas to.
    flat = NV.astype('f4').ravel()
    basis_kb.data.foreach_set("co", flat)
    if basis_src is not None:
        basis_kb.slider_min = basis_src['slider_min']
        basis_kb.slider_max = basis_src['slider_max']
        basis_kb.interpolation = basis_src['interpolation']

    made = {}
    order = []
    for k in snap.shape_keys:
        if k is basis_src:
            made[k['name']] = basis_kb
            order.append((k, basis_kb))
            continue
        delta = _interp_vert(snap, v_tri, v_bary, k['delta'])
        kb = new_obj.shape_key_add(name=k['name'], from_mix=False)
        kb.data.foreach_set("co", (NV + delta).astype('f4').ravel())
        made[k['name']] = kb
        order.append((k, kb))

    key = me.shape_keys
    if key is not None:
        if snap.key_name:
            try:
                key.name = snap.key_name
            except Exception:
                pass
        key.use_relative = snap.key_use_relative

    # second pass: relative_key / value / range / mute / vertex group
    for k, kb in order:
        rel = made.get(k['relative_key'])
        if rel is not None and rel != kb:
            try:
                kb.relative_key = rel
            except Exception:
                pass
        smin, smax = k['slider_min'], k['slider_max']
        for setter in (
            lambda: setattr(kb, 'slider_max', max(smax, kb.slider_min)),
            lambda: setattr(kb, 'slider_min', smin),
            lambda: setattr(kb, 'slider_max', smax),
        ):
            try:
                setter()
            except Exception:
                pass
        try:
            kb.value = k['value']
        except Exception:
            pass
        kb.mute = k['mute']
        try:
            kb.interpolation = k['interpolation']
        except Exception:
            pass
        vgn = k['vertex_group']
        if vgn and new_obj.vertex_groups.get(vgn) is not None:
            try:
                kb.vertex_group = vgn
            except Exception:
                pass

    n = len(order)
    if basis_src is None and n:
        rep['warnings'].append("snapshot had no reference key; synthesised Basis")
        n += 1
    return n


def _uv_regions(snap):
    """Per-source-polygon UV island id (lazily computed, cached on the snapshot).

    Two polygons share an id only when their common edge carries the same UVs on
    both sides in *every* UV layer, i.e. the texture runs continuously across it.
    A single partition therefore satisfies all layers at once.
    """
    if snap._uv_regions is not None:
        return snap._uv_regions
    npo = snap.npolys
    if npo == 0 or not snap.uv_layers:
        snap._uv_regions = np.zeros(npo, 'i4')
        return snap._uv_regions

    starts, totals, lv = snap.loop_starts, snap.loop_totals, snap.loop_verts
    nl = lv.size
    lp = np.repeat(np.arange(npo, dtype='i4'), totals)
    idx = np.arange(nl, dtype='i8')
    local = idx - np.repeat(starts.astype('i8'), totals)
    nxt = starts[lp].astype('i8') + (local + 1) % np.maximum(totals[lp], 1)

    # pair up the two loops that share each mesh edge
    ekey = np.sort(np.stack([lv, lv[nxt]], 1), axis=1)
    order = np.lexsort((ekey[:, 1], ekey[:, 0]))
    e = ekey[order]
    if e.shape[0] < 2:
        snap._uv_regions = np.arange(npo, dtype='i4')
        return snap._uv_regions
    same = np.all(e[1:] == e[:-1], axis=1)
    la = order[:-1][same]
    lb = order[1:][same]

    join = np.ones(la.size, bool)
    for layer in snap.uv_layers:
        uv = layer['uv']
        if uv.shape[0] != nl:
            continue
        a0, a1 = uv[la], uv[nxt[la]]
        b0, b1 = uv[lb], uv[nxt[lb]]
        m1 = ((np.abs(a0 - b1).max(1) < _UV_SEAM_EPS)
              & (np.abs(a1 - b0).max(1) < _UV_SEAM_EPS))
        m2 = ((np.abs(a0 - b0).max(1) < _UV_SEAM_EPS)
              & (np.abs(a1 - b1).max(1) < _UV_SEAM_EPS))
        join &= (m1 | m2)

    parent = list(range(npo))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pa = lp[la[join]]
    pb = lp[lb[join]]
    for x, y in zip(pa.tolist(), pb.tolist()):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
    roots = np.fromiter((find(i) for i in range(npo)), 'i4', npo)
    _u, inv = np.unique(roots, return_inverse=True)
    snap._uv_regions = inv.astype('i4')
    return snap._uv_regions


def _region_bvh(snap, region):
    """BVH over just the triangles of one UV island (built once, cached)."""
    if region in snap._uv_region_bvh:
        return snap._uv_region_bvh[region]
    if snap._uv_region_index is None:
        tri_region = _uv_regions(snap)[snap.tri_poly]
        order = np.argsort(tri_region, kind='stable')
        sr = tri_region[order]
        nreg = int(tri_region.max()) + 1 if tri_region.size else 0
        rr = np.arange(nreg + 1)
        snap._uv_region_index = (order, np.searchsorted(sr, rr[:-1]),
                                 np.searchsorted(sr, rr[:-1], side='right'))
    order, lo, hi = snap._uv_region_index
    if region < 0 or region >= lo.size:
        return None
    sel = order[lo[region]:hi[region]]
    if sel.size == 0:
        snap._uv_region_bvh[region] = None
        return None
    tris = snap.tris[sel]
    used, inv = np.unique(tris, return_inverse=True)
    try:
        bvh = BVHTree.FromPolygons(
            snap.verts[used].tolist(),
            [tuple(t) for t in inv.reshape(-1, 3).tolist()],
            all_triangles=True,
        )
    except Exception:
        bvh = None
    out = (bvh, sel) if bvh is not None else None
    snap._uv_region_bvh[region] = out
    return out


def _constrain_to_island(snap, l_tri, sample, dist, lp, starts, totals,
                         centers, lco, rep):
    """Pull every corner of an output face onto one UV island.

    Sampling each corner independently is what produces seam bleed: a corner
    just across a UV seam grabs the neighbouring island and the face ends up
    textured out of two unrelated parts of the layout — a smear across the whole
    UV space, not a subtle error.

    A face that landed in more than one island re-samples *all* of its corners
    inside each candidate island and keeps the island that moves the corners
    least in total, so the face is textured from one continuous piece and the
    UV shift it costs is the smallest available.

    The island has to be genuinely within reach (roughly the face's own size) for
    every corner; a face that really does bridge a gap between two shells is left
    alone rather than dragged across the gap.
    """
    npo = starts.size
    none = np.zeros(lp.size, bool)
    if npo == 0 or lp.size == 0:
        return l_tri, none
    regions = _uv_regions(snap)
    if regions.size == 0 or int(regions.max()) == 0:
        return l_tri, none

    ok = l_tri >= 0
    reg = np.full(lp.size, -1, 'i4')
    if ok.any():
        reg[ok] = regions[snap.tri_poly[l_tri[ok]]]

    # faces whose corners did not agree on an island
    v = reg >= 0
    rmin = np.full(npo, 1 << 30, 'i8')
    rmax = np.full(npo, -1, 'i8')
    np.minimum.at(rmin, lp[v], reg[v])
    np.maximum.at(rmax, lp[v], reg[v])
    faces = np.nonzero((rmax >= 0) & (rmin != rmax))[0]
    rep['uv_seam_faces'] = int(faces.size)
    if faces.size == 0:
        return l_tri, none

    # how far a corner may travel to reach its face's island
    extent = np.zeros(npo, 'f8')
    np.maximum.at(extent, lp, np.linalg.norm(lco - centers[lp], axis=1))
    guard = 2.0 * extent

    # gather the work: which island has to be probed for which corners
    corners = {}
    per_region = {}
    for f in faces.tolist():
        ks = list(range(int(starts[f]), int(starts[f]) + int(totals[f])))
        cands = sorted({int(reg[k]) for k in ks if reg[k] >= 0})
        if len(cands) < 2:
            continue
        corners[f] = (ks, cands)
        for r in cands:
            per_region.setdefault(r, []).extend(ks)

    probe = {}
    for r, ks in per_region.items():
        got = _region_bvh(snap, r)
        if got is None:
            continue
        bvh, sel = got
        fn = bvh.find_nearest
        for i in ks:
            p = sample[i]
            hit = fn((float(p[0]), float(p[1]), float(p[2])))
            if hit is None or hit[2] is None or hit[3] is None:
                continue
            probe[(i, r)] = (float(hit[3]), int(sel[int(hit[2])]))

    moved = np.zeros(lp.size, bool)
    for f, (ks, cands) in corners.items():
        lim = guard[f]
        best = None
        for r in cands:
            total = 0.0
            for i in ks:
                got = probe.get((i, r))
                if got is None or got[0] > dist[i] + lim:
                    total = None
                    break
                total += got[0]
            if total is not None and (best is None or total < best[0]):
                best = (total, r)
        if best is None:
            continue
        for i in ks:
            tri = probe[(i, best[1])][1]
            if tri != l_tri[i]:
                l_tri[i] = tri
                moved[i] = True
    rep['uv_seam_corners_fixed'] = int(moved.sum())
    rep['uv_seam_faces_fixed'] = int(np.unique(lp[moved]).size)
    return l_tri, moved


def _uv_layer_bvh(snap, li):
    """BVH over one UV layer's triangles, laid flat in the z=0 plane."""
    got = snap._uv_clearance.get(('bvh', li), False)
    if got is not False:
        return got
    bvh = None
    uv = snap.uv_layers[li]['uv']
    if snap.tris.shape[0] and uv.shape[0] == snap.nloops:
        U = uv[snap.tri_loops].reshape(-1, 2).astype('f8')
        V3 = np.zeros((U.shape[0], 3), 'f8')
        V3[:, :2] = U
        tri = np.arange(U.shape[0], dtype='i4').reshape(-1, 3)
        try:
            bvh = BVHTree.FromPolygons(V3.tolist(),
                                       [tuple(t) for t in tri.tolist()],
                                       all_triangles=True)
        except Exception:
            bvh = None
    snap._uv_clearance[('bvh', li)] = bvh
    return bvh


def _uv_island_gap(snap, li):
    """Per-island lower bound on the UV distance to any other island.

    Island bounding boxes only: boxes ``d`` apart mean the islands themselves
    are at least ``d`` apart, so an island whose box is already farther than the
    cap needs no triangle-level query at all.  Skipped (all zeros, i.e. "ask the
    triangles") on layouts with more islands than the pairwise table is worth.
    """
    key = ('gap', li)
    if key in snap._uv_clearance:
        return snap._uv_clearance[key]
    regions = _uv_regions(snap)
    nreg = int(regions.max()) + 1 if regions.size else 0
    gap = np.zeros(max(nreg, 1), 'f8')
    uv = snap.uv_layers[li]['uv']
    if 0 < nreg <= 512 and snap.tris.shape[0] and uv.shape[0] == snap.nloops:
        tri_region = regions[snap.tri_poly].astype('i8')
        U = uv[snap.tri_loops].astype('f8')                # (nt,3,2)
        mn = np.full((nreg, 2), np.inf)
        mx = np.full((nreg, 2), -np.inf)
        np.minimum.at(mn, tri_region, U.min(axis=1))
        np.maximum.at(mx, tri_region, U.max(axis=1))
        live = np.isfinite(mn).all(axis=1)
        d = np.zeros((nreg, nreg), 'f8')
        for ax in (0, 1):
            sep = np.maximum(mn[:, None, ax] - mx[None, :, ax],
                             mn[None, :, ax] - mx[:, None, ax])
            d += np.maximum(sep, 0.0) ** 2
        d = np.sqrt(d)
        d[~live, :] = np.inf
        d[:, ~live] = np.inf
        np.fill_diagonal(d, np.inf)
        g = d.min(axis=1)
        gap = np.where(np.isfinite(g), g, _UV_CLEARANCE_CAP)
    snap._uv_clearance[key] = gap
    return gap


def _uv_clearance(snap, li, points, own):
    """UV-space distance from each point to the nearest *other* island.

    Measured, not assumed: the whole layer's triangles are in one BVH, every
    hit within ``_UV_CLEARANCE_CAP`` is checked against the point's own island,
    and the nearest foreign one wins.  Points with no foreign island in range
    keep the cap, which is a lower bound on their true clearance — so half the
    returned value is always a displacement that provably cannot reach another
    island's texels.
    """
    cap = _UV_CLEARANCE_CAP
    n = points.shape[0]
    out = np.full(n, cap, 'f8')
    if n == 0:
        return out
    # islands whose bounding box is already farther than the cap are done
    ask = np.nonzero(_uv_island_gap(snap, li)[own] < cap)[0]
    if ask.size == 0:
        return out
    bvh = _uv_layer_bvh(snap, li)
    if bvh is None:
        return out * 0.0
    regions = _uv_regions(snap)
    tri_region = regions[snap.tri_poly]
    rng = bvh.find_nearest_range
    for i in ask.tolist():
        p = points[i]
        r0 = int(own[i])
        best = cap
        for hit in rng((p[0], p[1], 0.0), cap):
            if hit is None or hit[2] is None or hit[3] is None:
                continue
            if int(tri_region[int(hit[2])]) == r0:
                continue
            d = float(hit[3])
            if d < best:
                best = d
        out[i] = best
    return out


def _bary_free(snap, tri_idx, points):
    """Unclamped barycentric of ``points`` projected onto their triangle plane.

    Negative weights mean the point lies outside the triangle; interpolating a
    UV with them continues the source parameterisation past the island's edge
    instead of pinning the corner to its outline.
    """
    n = points.shape[0]
    w = np.zeros((n, 3), 'f8')
    w[:, 0] = 1.0
    ok = tri_idx >= 0
    if n == 0 or not ok.any():
        return w, np.zeros(n, bool)
    t = snap.tris[tri_idx[ok]]
    V = snap.verts
    a, b, c = V[t[:, 0]], V[t[:, 1]], V[t[:, 2]]
    v0, v1, v2 = b - a, c - a, points[ok] - a
    dot = lambda x, y: np.einsum('ij,ij->i', x, y)      # noqa: E731
    d00, d01, d11 = dot(v0, v0), dot(v0, v1), dot(v1, v1)
    d20, d21 = dot(v2, v0), dot(v2, v1)
    den = d00 * d11 - d01 * d01
    good = np.abs(den) > _EPS
    den = np.where(good, den, 1.0)
    vv = (d11 * d20 - d01 * d21) / den
    ww = (d00 * d21 - d01 * d20) / den
    sub = np.stack([1.0 - vv - ww, vv, ww], axis=1)
    sub[~good] = (1.0, 0.0, 0.0)
    w[ok] = sub
    valid = np.zeros(n, bool)
    valid[np.nonzero(ok)[0][good]] = True
    return w, valid


def _apply_uvs(snap, me, nl, npo, NV, rep):
    """Per-loop UV transfer.

    The lookup point for a loop is its vertex nudged towards the face centre so
    that loops on opposite sides of a UV seam pick different source triangles;
    the barycentric weights are then evaluated at the *vertex* position on that
    triangle, so the transferred UV is not shifted inwards.

    Corners are then pulled onto a single UV island per face (see
    ``_constrain_to_island``) so a face is never textured from two islands.
    """
    starts, totals, lverts = _poly_arrays(me)
    centers = _poly_centers(me)
    lp = np.repeat(np.arange(npo, dtype='i4'), totals) if npo else np.zeros(0, 'i4')
    lco = NV[lverts]
    sample = lco + (centers[lp] - lco) * _LOOP_INSET

    l_tri, l_loc = _nearest_tris(snap, sample)
    dist = np.linalg.norm(sample - l_loc, axis=1)
    l_tri, moved = _constrain_to_island(snap, l_tri, sample, dist, lp, starts,
                                        totals, centers, lco, rep)
    l_bary = _bary(snap, l_tri, lco)

    # Corners pulled onto an island can sit past its edge; their barycentric is
    # then clamped to the island's outline and the UV pins to the island hull.
    # Those are the only corners allowed to extrapolate, and only as far as the
    # measured clearance to the neighbouring island permits (see _uv_clearance).
    ext = np.zeros(nl, bool)
    if _UV_EXTRAPOLATE and moved.any():
        free, valid = _bary_free(snap, l_tri, lco)
        ext = moved & valid & (free.min(axis=1) < -1e-9)
        rep['uv_corners_extrapolated'] = int(ext.sum())
    if not ext.any():
        free = None
    else:
        regions = _uv_regions(snap)
        e_reg = regions[snap.tri_poly[np.where(l_tri >= 0, l_tri, 0)]]

    for li, layer in enumerate(snap.uv_layers):
        uvl = me.uv_layers.get(layer['name'])
        if uvl is None:
            try:
                uvl = me.uv_layers.new(name=layer['name'], do_init=False)
            except TypeError:
                uvl = me.uv_layers.new(name=layer['name'])
        if uvl is None:
            rep['warnings'].append("could not create UV layer %r" % layer['name'])
            continue
        uv = _interp_loop(snap, l_tri, l_bary, layer['uv'])
        if free is not None:
            k = np.nonzero(ext)[0]
            step = _interp_loop(snap, l_tri[k], free[k], layer['uv']) - uv[k]
            need = np.linalg.norm(step, axis=1)
            # the clamped UV sits on the island's own outline; measure from
            # there how much room there is before another island's texels
            room = 0.5 * _uv_clearance(snap, li, uv[k], e_reg[k])
            # never extrapolate further than the source triangle's own UV size:
            # a sliver at the island edge must not fling the corner away
            tuv = layer['uv'][snap.tri_loops[l_tri[k]]].astype('f8')
            span = np.linalg.norm(tuv - tuv.mean(axis=1)[:, None, :],
                                  axis=2).max(axis=1)
            allow = np.minimum(np.minimum(room, span), need)
            uv[k] += step * np.where(need > _EPS,
                                     allow / np.maximum(need, _EPS), 0.0)[:, None]
            if k.size:
                rep['uv_extrapolation_max'] = max(
                    rep.get('uv_extrapolation_max', 0.0), float(allow.max()))
                rep['uv_clearance_min'] = min(
                    rep.get('uv_clearance_min', 1.0), float((2.0 * room).min()))
        uvl.data.foreach_set("uv", uv.astype('f4').ravel())
        rep['uv_layers'] += 1
        if layer['active']:
            try:
                me.uv_layers.active = uvl
            except Exception:
                pass
        if layer['active_render']:
            try:
                uvl.active_render = True
            except Exception:
                pass
    rep['uvs'] = rep['uv_layers'] > 0


def _corner_triangles(me, npo, centers):
    """Split every polygon into (centre, corner, next corner) triangles.

    Returns (a, b, c, area, poly) per triangle; the triangles of one polygon
    tile it exactly, so their areas are the weights of an area-weighted vote.
    """
    starts, totals, lverts = _poly_arrays(me)
    nl = lverts.size
    z = np.zeros((0, 3), 'f8')
    if npo == 0 or nl == 0:
        return z, z, z, np.zeros(0, 'f8'), np.zeros(0, 'i4')
    V = _fget(me.vertices, "co", len(me.vertices) * 3, 'f8').reshape(-1, 3)
    lp = np.repeat(np.arange(npo, dtype='i4'), totals)
    idx = np.arange(nl, dtype='i8')
    local = idx - np.repeat(starts.astype('i8'), totals)
    nxt = starts[lp].astype('i8') + (local + 1) % np.maximum(totals[lp], 1)
    a = centers[lp]
    b = V[lverts]
    c = V[lverts[nxt]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return a, b, c, area, lp


def _bary_grid(k):
    """k interior barycentric coordinates of a triangle (k = 1 or 3)."""
    if k <= 1:
        return np.full((1, 3), 1.0 / 3.0)
    e = np.full((3, 3), 1.0 / 6.0)
    np.fill_diagonal(e, 2.0 / 3.0)
    return e


def _sample_materials(snap, tri, sel, k, nslots):
    """Vote weights of one sampling pass over the selected corner triangles."""
    a, b, c, area, lp = tri[0], tri[1], tri[2], tri[3], tri[4]
    a, b, c, area, lp = a[sel], b[sel], c[sel], area[sel], lp[sel]
    bary = _bary_grid(k)
    P = np.einsum('kb,nbj->nkj', bary,
                  np.stack([a, b, c], axis=1)).reshape(-1, 3)
    s_tri, _ = _nearest_tris(snap, P)
    safe = np.where(s_tri >= 0, s_tri, 0)
    mat = np.clip(snap.poly_material[snap.tri_poly[safe]], 0, nslots - 1)
    mat = np.where(s_tri >= 0, mat, 0).astype('i8')
    w = np.repeat(area / bary.shape[0], bary.shape[0])
    poly = np.repeat(lp, bary.shape[0]).astype('i8')
    return mat, w, poly, P.shape[0]


def _apply_materials(snap, new_obj, me, npo, rep):
    """Assign material slots by an area-weighted vote over each face.

    Sampling only the face centre decides a whole quad from a single point, so
    every output face straddling a material boundary is a coin flip: 0.63% of
    the reference avatar's surface came out under the wrong material.  Each face
    is instead split into corner triangles and voted on by area; faces whose
    samples disagree (the ones actually on a boundary) are re-sampled three
    times finer, which is where the accuracy is needed and nowhere else.
    """
    me.materials.clear()
    remap = []
    for mat, name in snap.materials:
        m = mat if _material_ok(mat) else (bpy.data.materials.get(name) if name else None)
        me.materials.append(m)
        remap.append(m)
    nslots = len(remap)
    if not (nslots and npo):
        return nslots
    if snap.poly_material.size == 0 or nslots == 1:
        me.polygons.foreach_set("material_index", np.zeros(npo, 'i4'))
        return nslots

    centers = _poly_centers(me)
    if _MATERIAL_SAMPLES < 1:                  # face-centre only (pre-0.6.1)
        f_tri, _ = _nearest_tris(snap, centers)
        safe = np.where(f_tri >= 0, f_tri, 0)
        mi = np.clip(snap.poly_material[snap.tri_poly[safe]], 0, nslots - 1)
        me.polygons.foreach_set("material_index", mi.astype('i4'))
        rep['material_samples'] = int(npo)
        return nslots

    tri = _corner_triangles(me, npo, centers)
    ntri = tri[3].size
    if ntri == 0:
        return nslots

    all_sel = np.arange(ntri)
    mat, w, poly, ns = _sample_materials(snap, tri, all_sel, 1, nslots)
    n_samples = ns

    if _MATERIAL_SAMPLES > 1:
        # a face whose coarse samples disagree straddles a boundary: only those
        # are worth the finer pass
        lo = np.full(npo, 1 << 30, 'i8')
        hi = np.full(npo, -1, 'i8')
        np.minimum.at(lo, poly, mat)
        np.maximum.at(hi, poly, mat)
        mixed = np.nonzero((hi >= 0) & (lo != hi))[0]
        if mixed.size:
            fine = np.zeros(npo, bool)
            fine[mixed] = True
            sel = np.nonzero(fine[tri[4]])[0]
            keep = ~fine[poly]
            mat2, w2, poly2, ns2 = _sample_materials(
                snap, tri, sel, _MATERIAL_SAMPLES, nslots)
            mat = np.concatenate([mat[keep], mat2])
            w = np.concatenate([w[keep], w2])
            poly = np.concatenate([poly[keep], poly2])
            n_samples += ns2
            rep['material_faces_refined'] = int(mixed.size)

    acc = np.zeros((npo, nslots), 'f8')
    np.add.at(acc, (poly, mat), w)
    mi = np.argmax(acc, axis=1).astype('i4')
    # degenerate faces (no area anywhere) have nothing to weigh: fall back to
    # the material of one of their samples, i.e. the old face-centre behaviour
    dead = acc.max(axis=1) <= 0.0
    if dead.any():
        first = np.zeros(npo, 'i8')
        first[poly[::-1]] = np.arange(poly.size)[::-1]
        mi[dead] = mat[first[dead]].astype('i4')
    me.polygons.foreach_set("material_index", mi)
    rep['material_samples'] = int(n_samples)
    return nslots


def _csr_adj(EV, nv):
    """CSR vertex->(neighbour, edge id) adjacency for an edge array."""
    n = EV.shape[0]
    a = np.concatenate([EV[:, 0], EV[:, 1]]).astype('i8')
    b = np.concatenate([EV[:, 1], EV[:, 0]]).astype('i8')
    e = np.concatenate([np.arange(n), np.arange(n)]).astype('i8')
    o = np.argsort(a, kind='stable')
    a, b, e = a[o], b[o], e[o]
    off = np.searchsorted(a, np.arange(nv + 1))
    return off, b, e


def _source_arcs(edges):
    """Split a marked-edge subgraph into simple arcs.

    Returns a list of ``(vertex_path, edge_path)``.  Arcs run between vertices
    where the subgraph branches or ends (degree != 2); anything left over is a
    pure cycle, which is cut in half so that it, too, has two endpoints to route
    between.  This is what makes the transfer chain-aware: the unit of matching
    is a *path*, not a single edge.
    """
    adj = {}
    for i, (a, b) in enumerate(edges.tolist()):
        if a == b:
            continue
        adj.setdefault(a, []).append(i)
        adj.setdefault(b, []).append(i)
    used = [False] * edges.shape[0]

    def walk(start, e0):
        vpath = [start]
        epath = []
        cur, eid = start, e0
        while True:
            used[eid] = True
            epath.append(eid)
            a, b = int(edges[eid][0]), int(edges[eid][1])
            cur = b if a == cur else a
            vpath.append(cur)
            if cur == start:
                break
            nb = adj.get(cur, ())
            if len(nb) != 2:
                break
            nxt = [e for e in nb if e != eid]
            if not nxt or used[nxt[0]]:
                break
            eid = nxt[0]
        return vpath, epath

    arcs = []
    for v in sorted(k for k, l in adj.items() if len(l) != 2):
        for e in adj[v]:
            if not used[e]:
                arcs.append(walk(v, e))
    for e in range(edges.shape[0]):
        if used[e]:
            continue
        vp, ep = walk(int(edges[e][0]), e)
        if len(ep) < 2:
            arcs.append((vp, ep))
            continue
        m = len(ep) // 2                      # cut the cycle into two arcs
        arcs.append((vp[:m + 1], ep[:m]))
        arcs.append((vp[m:], ep[m:]))
    return arcs


def _seg_dist(P, A, B):
    """Distance from each point in P (n,3) to each segment A[i]-B[i] (m,3)."""
    d = B - A
    ll = np.einsum('ij,ij->i', d, d)
    out = np.empty((P.shape[0], A.shape[0]), 'f8')
    block = max(1, int(4_000_000 // max(A.shape[0], 1)))
    for i in range(0, P.shape[0], block):
        p = P[i:i + block]
        ap = p[:, None, :] - A[None, :, :]
        t = np.einsum('ijk,jk->ij', ap, d) / np.maximum(ll, _EPS)[None, :]
        np.clip(t, 0.0, 1.0, out=t)
        out[i:i + block] = np.linalg.norm(ap - t[:, :, None] * d[None, :, :], axis=2)
    return out


def _out_graph(EV, NV, cache):
    """Adjacency / lengths / KD-trees of the output edge graph (built once)."""
    if cache is not None and 'graph' in cache:
        return cache['graph']
    nv, ne = NV.shape[0], EV.shape[0]
    off, nbr, nbre = _csr_adj(EV, nv)
    elen = np.linalg.norm(NV[EV[:, 1]] - NV[EV[:, 0]], axis=1)
    emid = (NV[EV[:, 0]] + NV[EV[:, 1]]) * 0.5
    kd_mid = KDTree(ne)
    for i, p in enumerate(emid.tolist()):
        kd_mid.insert(p, i)
    kd_mid.balance()
    kd_vert = KDTree(nv)
    for i, p in enumerate(NV.tolist()):
        kd_vert.insert(p, i)
    kd_vert.balance()
    g = (off, nbr, nbre, elen, emid, kd_mid, kd_vert)
    if cache is not None:
        cache['graph'] = g
    return g


def _route_arc(cand, cdist, ne, off, nbr, nbre, elen, v_start, v_end, blocked):
    """Cheapest corridor path between two output vertices (Dijkstra).

    ``cand`` are the corridor's output edge ids, ``cdist`` their distance to the
    source arc.  Cost is edge length inflated by how far the edge strays from
    the arc, so the path hugs the source chain instead of cutting across it.
    """
    import heapq

    keep = np.zeros(ne, bool)
    keep[cand] = True
    pen = np.zeros(ne, 'f8')
    scale = max(cdist.max(), _EPS) if cdist.size else 1.0
    pen[cand] = 1.0 + 3.0 * (cdist / scale)
    cost = elen * pen + 1e-9

    dist = {v_start: 0.0}
    prev = {}
    heap = [(0.0, int(v_start))]
    seen = set()
    while heap:
        d, v = heapq.heappop(heap)
        if v in seen:
            continue
        seen.add(v)
        if v == v_end:
            break
        for k in range(int(off[v]), int(off[v + 1])):
            e = int(nbre[k])
            if not keep[e]:
                continue
            u = int(nbr[k])
            if u in blocked and u != v_end:
                continue
            nd = d + float(cost[e])
            if nd < dist.get(u, np.inf):
                dist[u] = nd
                prev[u] = (v, e)
                heapq.heappush(heap, (nd, u))
    if v_end not in prev and v_end != v_start:
        return None
    path = []
    v = int(v_end)
    guard = 0
    while v != v_start:
        pv, pe = prev[v]
        path.append(pe)
        v = pv
        guard += 1
        if guard > 100000:
            return None
    return path[::-1]


def _route_edge_scalar(snap, EV, NV, src_idx, src_vals, cache=None):
    """Chain-aware transfer: every source path becomes one output path.

    The nearest-edge match this replaces assigns per *edge*: one source crease
    edge is picked up by every output edge that happens to be nearest to it, so
    a 6-edge crease came out as an 8-edge thicket with degree-4 junctions and
    2.4x the length.  Here the source's marked subgraph is split into arcs and
    each arc is routed through the output edge graph as a single path, so the
    result is a chain of comparable length with the source's own topology.

    Returns a per-output-edge value array, or None to fall back.
    """
    SV = snap.verts
    se = snap.edges[src_idx]
    ne = EV.shape[0]
    off, nbr, nbre, elen, emid, kd_mid, kd_vert = _out_graph(EV, NV, cache)

    def local_len(v):
        lo, hi = int(off[v]), int(off[v + 1])
        return float(elen[nbre[lo:hi]].mean()) if hi > lo else 0.0

    out = np.zeros(ne, 'f4')
    blocked = set()
    n_arcs = n_failed = 0
    for vpath, epath in _source_arcs(se):
        if not epath:
            continue
        n_arcs += 1
        A = SV[np.asarray(vpath, 'i8')]
        seg_a, seg_b = A[:-1], A[1:]
        slen = np.linalg.norm(seg_b - seg_a, axis=1)
        arc_len = float(slen.sum())
        vals = src_vals[src_idx[np.asarray(epath, 'i8')]]

        _c, vs, _d = kd_vert.find(A[0].tolist())
        _c, ve, _d = kd_vert.find(A[-1].tolist())
        if vs is None or ve is None:
            n_failed += 1
            continue
        lloc = max(local_len(vs), local_len(ve), _EPS)
        R = 0.75 * (lloc + float(slen.mean()))

        # corridor: output edges whose midpoint stays within R of the arc.
        # Widened in two steps when the narrow corridor has no connected route
        # (a chain that runs diagonally across the new quads needs elbow room);
        # the length guard below is what actually keeps a path honest.
        seen = set()
        probe = np.concatenate([A, (seg_a + seg_b) * 0.5], axis=0)
        rng = kd_mid.find_range
        for p in probe.tolist():
            for _co, i, _d in rng(p, _CORRIDOR_STEPS[-1] * R + lloc):
                seen.add(int(i))
        if not seen:
            n_failed += 1
            continue
        all_cand = np.fromiter(sorted(seen), 'i8', len(seen))
        all_dist = _seg_dist(emid[all_cand], seg_a, seg_b).min(axis=1)
        smid = (seg_a + seg_b) * 0.5

        def _single():
            """One output edge for a chain the new mesh cannot resolve."""
            near = all_dist <= R
            if not near.any():
                return False
            c, d = all_cand[near], all_dist[near]
            dirn = NV[EV[c, 1]] - NV[EV[c, 0]]
            dirn /= np.maximum(np.linalg.norm(dirn, axis=1), _EPS)[:, None]
            av = A[-1] - A[0]
            av = av / max(float(np.linalg.norm(av)), _EPS)
            align = np.abs(dirn @ av)
            ok = align >= 0.5
            if not ok.any():
                return False
            i = int(c[ok][np.argmin(d[ok])])
            j = int(np.argmin(np.linalg.norm(smid - emid[i], axis=1)))
            out[i] = max(out[i], vals[j])
            return True

        if vs == ve:
            # the arc is shorter than one output edge: keep the single best
            # candidate rather than every edge that brushes past it
            if not _single():
                n_failed += 1
            continue

        limit = max(2.5 * arc_len, arc_len + 2.0 * lloc)
        path = None
        for mult in _CORRIDOR_STEPS:
            near = all_dist <= mult * R
            if not near.any():
                continue
            got = _route_arc(all_cand[near], all_dist[near], ne, off, nbr,
                             nbre, elen, vs, ve, blocked)
            if got is not None and float(elen[got].sum()) <= limit:
                path = got
                break
        if path is None:
            # A source chain finer than the new mesh's own edges has no honest
            # path to become; one edge keeps the feature without inflating it.
            if not (arc_len <= 1.5 * lloc and _single()):
                n_failed += 1
            continue
        for e in path:
            j = int(np.argmin(np.linalg.norm(smid - emid[e], axis=1)))
            out[e] = max(out[e], vals[j])
        for v in (int(EV[e][0]) for e in path):
            blocked.add(v)
        for v in (int(EV[e][1]) for e in path):
            blocked.add(v)
        blocked.discard(int(vs))
        blocked.discard(int(ve))

    if n_arcs == 0 or n_failed > 0.5 * n_arcs:
        return None                    # routing is not working here; fall back
    return out


def _unrepresented(snap, src_idx, EV, NV, out):
    """Source marked edges with no transferred edge anywhere near them."""
    hit = np.nonzero(out > 0.0)[0]
    if hit.size == 0:
        return int(src_idx.size)
    mid = (NV[EV[hit, 0]] + NV[EV[hit, 1]]) * 0.5
    kd = KDTree(hit.size)
    for i, p in enumerate(mid.tolist()):
        kd.insert(p, i)
    kd.balance()
    se = snap.edges[src_idx]
    sa, sb = snap.verts[se[:, 0]], snap.verts[se[:, 1]]
    smid = (sa + sb) * 0.5
    slen = np.linalg.norm(sb - sa, axis=1)
    nlen = np.linalg.norm(NV[EV[hit, 1]] - NV[EV[hit, 0]], axis=1)
    n_bad = 0
    find = kd.find
    for i, p in enumerate(smid.tolist()):
        _co, j, d = find(p)
        if j is None or d > 0.75 * (slen[i] + nlen[j]):
            n_bad += 1
    return n_bad


def _transfer_edge_scalar(snap, me, EV, NV, src_vals, attr_name, cache=None):
    """Transfer a per-edge float (crease / bevel weight) onto the new mesh.

    Chain-aware when the marked subgraph is small enough to route arc by arc
    (``_route_edge_scalar``); otherwise the legacy nearest-edge match below.
    """
    src_idx = np.nonzero(src_vals > 0.0)[0]
    if src_idx.size == 0 or EV.shape[0] == 0:
        return 0, 0
    out = None
    if src_idx.size <= _MAX_CHAIN_EDGES:
        try:
            out = _route_edge_scalar(snap, EV, NV, src_idx, src_vals, cache)
        except Exception:
            out = None
    if out is not None:
        n_ok = int((out > 0.0).sum())
        if n_ok:
            at = _ensure_float_attr(me, attr_name, 'EDGE', EV.shape[0])
            if at is None:
                return 0, int(src_idx.size)
            at.data.foreach_set("value", out)
        return n_ok, _unrepresented(snap, src_idx, EV, NV, out)
    return _nearest_edge_scalar(snap, me, EV, NV, src_vals, attr_name, src_idx)


def _nearest_edge_scalar(snap, me, EV, NV, src_vals, attr_name, src_idx):
    """Conservative nearest-edge transfer of a per-edge float.

    Only source edges with a non-zero value are candidates; a new edge picks one
    up when its midpoint is close to the source edge midpoint (relative to the
    two edge lengths) and the two edges point roughly the same way.
    """
    SV = snap.verts
    se = snap.edges[src_idx]
    sa, sb = SV[se[:, 0]], SV[se[:, 1]]
    smid = (sa + sb) * 0.5
    sdir = sb - sa
    slen = np.linalg.norm(sdir, axis=1)
    sdir = sdir / np.maximum(slen, _EPS)[:, None]

    kd = KDTree(src_idx.size)
    for i, p in enumerate(smid.tolist()):
        kd.insert(p, i)
    kd.balance()

    na, nb = NV[EV[:, 0]], NV[EV[:, 1]]
    nmid = (na + nb) * 0.5
    ndir = nb - na
    nlen = np.linalg.norm(ndir, axis=1)
    ndir = ndir / np.maximum(nlen, _EPS)[:, None]

    ne = EV.shape[0]
    hit = np.full(ne, -1, 'i4')
    dist = np.zeros(ne, 'f8')
    find = kd.find
    for i, p in enumerate(nmid.tolist()):
        _co, idx, d = find(p)
        if idx is not None:
            hit[i] = idx
            dist[i] = d

    ok = hit >= 0
    h = np.where(ok, hit, 0)
    tol = 0.25 * (slen[h] + nlen)
    align = np.abs(np.einsum('ij,ij->i', ndir, sdir[h]))
    good = ok & (dist <= tol) & (align >= 0.5)   # within ~60 degrees

    out = np.zeros(ne, 'f4')
    out[good] = src_vals[src_idx][h[good]]
    n_ok = int(good.sum())
    n_skip = int(src_idx.size - np.unique(h[good]).size) if n_ok else int(src_idx.size)
    if n_ok:
        at = _ensure_float_attr(me, attr_name, 'EDGE', ne)
        if at is None:
            return 0, int(src_idx.size)
        at.data.foreach_set("value", out)
    return n_ok, max(n_skip, 0)


def _transfer_vert_scalar(snap, me, NV, src_vals, attr_name):
    src_idx = np.nonzero(src_vals > 0.0)[0]
    nv = NV.shape[0]
    if src_idx.size == 0 or nv == 0:
        return 0
    SV = snap.verts[src_idx]
    kd = KDTree(src_idx.size)
    for i, p in enumerate(SV.tolist()):
        kd.insert(p, i)
    kd.balance()

    # tolerance: half the mean source edge length (conservative)
    if snap.edges.shape[0]:
        el = np.linalg.norm(snap.verts[snap.edges[:, 1]] - snap.verts[snap.edges[:, 0]], axis=1)
        tol = 0.5 * float(el.mean())
    else:
        tol = float(np.linalg.norm(snap.verts.max(0) - snap.verts.min(0))) * 0.01

    out = np.zeros(nv, 'f4')
    find = kd.find
    n_ok = 0
    for i, p in enumerate(NV.tolist()):
        _co, idx, d = find(p)
        if idx is not None and d <= tol:
            out[i] = src_vals[src_idx[idx]]
            n_ok += 1
    if n_ok:
        at = _ensure_float_attr(me, attr_name, 'POINT', nv)
        if at is None:
            return 0
        at.data.foreach_set("value", out)
    return n_ok
