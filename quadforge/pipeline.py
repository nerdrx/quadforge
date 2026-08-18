"""QuadForge orchestration.

``run_remesh(context, obj, s)`` is the single entry point used by the operators.
It never raises: every failure comes back as ``{'ok': False, 'error': ...}``.

Everything works headless (``blender --background``); the only operator used is
``object.quadriflow_remesh``, always through ``context.temp_override``.
"""

from __future__ import annotations

import json
import time

import numpy as np

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from .core import analysis, guides

ORIGINALS_COLLECTION = "QuadForge Originals"
WORK_SUFFIX = "_qf_work"
RESULT_SUFFIX = "_quad"

_AXIS_NAMES = ("X", "Y", "Z")

# An input shell this fraction of whose faces ended up far from the solved
# surface is treated as dropped and grafted back verbatim (restore_lost_regions).
SHELL_LOST_FRACTION = 0.75


# ---------------------------------------------------------------------------
# optional sibling modules (owned by other agents - never hard-require them)
# ---------------------------------------------------------------------------


def _try_import(name):
    try:
        mod = __import__(f"{__package__}.core.{name}", fromlist=[name])
        return mod
    except Exception:
        return None


def _get_backend(s):
    backend = getattr(s, "backend", "QUADRIFLOW")
    if backend == "NATIVE":
        try:
            from .backends import native
            if hasattr(native, "remesh"):
                return native, "NATIVE"
        except Exception:
            pass
    from .backends import quadriflow
    return quadriflow, "QUADRIFLOW"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def symmetry_axes(s):
    return [i for i, f in enumerate((
        bool(getattr(s, "symmetry_x", False)),
        bool(getattr(s, "symmetry_y", False)),
        bool(getattr(s, "symmetry_z", False)),
    )) if f]


def face_target_from_settings(obj, mesh, s) -> int:
    """Resolve FACES / RATIO / EDGE into an absolute quad count."""
    mode = getattr(s, "mode", "FACES")
    if mode == "RATIO":
        base = len(mesh.polygons)
        target = int(round(base * float(getattr(s, "target_ratio", 1.0))))
    elif mode == "EDGE":
        elen = float(getattr(s, "target_edge_length", 0.1))
        elen = max(elen, 1e-6)
        area = analysis.world_area(mesh, obj.matrix_world)
        target = int(round(area / (elen * elen)))
    else:
        target = int(getattr(s, "target_count", 5000))
    return int(max(12, min(target, 8_000_000)))


def mesh_quick_stats(obj) -> dict:
    mesh = obj.data
    npoly = len(mesh.polygons)
    if npoly == 0:
        return {"faces": 0, "quads": 0, "tris": 0, "ngons": 0, "quad_pct": 0.0,
                "verts": len(mesh.vertices)}
    sizes = np.empty(npoly, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", sizes)
    quads = int((sizes == 4).sum())
    tris = int((sizes == 3).sum())
    ngons = int((sizes > 4).sum())
    return {
        "faces": npoly,
        "quads": quads,
        "tris": tris,
        "ngons": ngons,
        "quad_pct": round(100.0 * quads / npoly, 3),
        "verts": len(mesh.vertices),
    }


def symmetry_error(obj, axis: int) -> float:
    """Max nearest-neighbour mismatch between the mesh and its mirror."""
    mesh = obj.data
    co = analysis.verts_co(mesh)
    n = len(co)
    if n == 0:
        return 0.0
    mir = co.copy()
    mir[:, axis] *= -1.0
    from mathutils import kdtree
    tree = kdtree.KDTree(n)
    for i, c in enumerate(co):
        tree.insert((float(c[0]), float(c[1]), float(c[2])), i)
    tree.balance()
    worst = 0.0
    for c in mir:
        _co, _idx, d = tree.find((float(c[0]), float(c[1]), float(c[2])))
        if d is not None and d > worst:
            worst = d
    return float(worst)


def _mean_edge_length(mesh) -> float:
    ev = analysis.edge_verts(mesh)
    if len(ev) == 0:
        return 0.0
    co = analysis.verts_co(mesh)
    return float(np.linalg.norm(co[ev[:, 0]] - co[ev[:, 1]], axis=1).mean())


# ---------------------------------------------------------------------------
# working object
# ---------------------------------------------------------------------------


def make_work_object(context, obj):
    """Evaluated (modifiers applied) copy of ``obj`` linked into the scene."""
    depsgraph = context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(
        eval_obj, preserve_all_data_layers=True, depsgraph=depsgraph
    )
    mesh.name = obj.data.name + WORK_SUFFIX
    work = bpy.data.objects.new(obj.name + WORK_SUFFIX, mesh)
    work.matrix_world = obj.matrix_world.copy()
    # materials come along with the mesh datablock
    context.scene.collection.objects.link(work)
    return work


def discard_object(obj):
    if obj is None:
        return
    mesh = obj.data if obj.type == "MESH" else None
    try:
        bpy.data.objects.remove(obj, do_unlink=True)
    except Exception:
        pass
    if mesh is not None and mesh.users == 0:
        try:
            bpy.data.meshes.remove(mesh)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# exact symmetry
# ---------------------------------------------------------------------------


def split_small_shells_aside(work_obj, limit: int):
    """Detach connected components smaller than ``limit`` faces into a side
    mesh and return it (or None). Used by the exact-symmetry path: bisecting
    thin centerline shells (hair plates, teeth, ruff leaves) shreds them into
    seam pinholes — small shells are kept whole and rejoined after the mirror,
    preserving their authored (already symmetric) topology."""
    mesh = work_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    seen = set()
    comps = []
    for f0 in bm.faces:
        if f0.index in seen:
            continue
        comp = []
        stack = [f0]
        seen.add(f0.index)
        while stack:
            f = stack.pop()
            comp.append(f)
            for e in f.edges:
                for nf in e.link_faces:
                    if nf.index not in seen:
                        seen.add(nf.index)
                        stack.append(nf)
        comps.append(comp)
    if len(comps) <= 1:
        bm.free()
        return None
    biggest = max(range(len(comps)), key=lambda i: len(comps[i]))
    small_faces = [f for i, comp in enumerate(comps)
                   if i != biggest and len(comp) < limit
                   for f in comp]
    if not small_faces:
        bm.free()
        return None
    side_mesh = bpy.data.meshes.new(mesh.name + "_side")
    kept = set(f.index for f in small_faces)
    tmp = bm.copy()
    tmp.faces.ensure_lookup_table()
    doomed = [f for f in tmp.faces if f.index not in kept]
    bmesh.ops.delete(tmp, geom=doomed, context='FACES')
    # These shells are rejoined verbatim after the mirror, so they are the one
    # part of the mesh that never reaches the backend's preclean. Give them the
    # same minimal degeneracy pass here, otherwise authored debris (exactly
    # coincident vertices, and the zero-length edges and slivers between them)
    # is copied straight into the result and reads as seam damage.
    weld = max(1e-6, 1e-4 * (_mean_edge_length(mesh) or 1e-2))
    try:
        bmesh.ops.remove_doubles(tmp, verts=tmp.verts[:], dist=weld)
    except Exception:
        pass
    slivers = [f for f in tmp.faces if f.calc_area() <= 1e-14]
    if slivers:
        bmesh.ops.delete(tmp, geom=slivers, context='FACES')
    stray_e = [e for e in tmp.edges if not e.link_faces]
    if stray_e:
        bmesh.ops.delete(tmp, geom=stray_e, context='EDGES')
    stray_v = [v for v in tmp.verts if not v.link_faces]
    if stray_v:
        bmesh.ops.delete(tmp, geom=stray_v, context='VERTS')
    tmp.to_mesh(side_mesh)
    tmp.free()
    bmesh.ops.delete(bm, geom=small_faces, context='FACES')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    if len(side_mesh.polygons) == 0 or len(mesh.polygons) == 0:
        try:
            bpy.data.meshes.remove(side_mesh)
        except Exception:
            pass
        return None
    return side_mesh


def rejoin_side_mesh(work_obj, side_mesh) -> None:
    bm = bmesh.new()
    bm.from_mesh(work_obj.data)
    bm.from_mesh(side_mesh)
    bm.to_mesh(work_obj.data)
    bm.free()
    work_obj.data.update()
    try:
        bpy.data.meshes.remove(side_mesh)
    except Exception:
        pass


def bisect_to_half(work_obj, axes, eps: float, pad: float = 0.0) -> bool:
    """Cut the mesh at every symmetry plane, keep the negative side, and snap
    the cut vertices exactly onto the planes. False if nothing survived.

    With ``pad`` > 0 the cut happens at +pad instead of the plane itself, so
    the solver keeps full local context around the plane — features that sit
    within a couple of edge lengths of the centerline (inner toes, nose tips)
    would otherwise be flattened by the pinned cut boundary. The surplus band
    is trimmed back to the exact plane by a second, unpadded call after
    solving."""
    mesh = work_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for ax in axes:
        no = Vector((0.0, 0.0, 0.0))
        no[ax] = 1.0
        co = Vector((0.0, 0.0, 0.0))
        co[ax] = pad
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        try:
            bmesh.ops.bisect_plane(
                bm, geom=geom, dist=eps,
                plane_co=tuple(co), plane_no=no,
                use_snap_center=False, clear_outer=True, clear_inner=False,
            )
        except Exception:
            bm.free()
            return False
        if not bm.faces:
            bm.free()
            return False
        for v in bm.verts:
            if pad == 0.0 and abs(v.co[ax]) <= eps:
                v.co[ax] = 0.0
        # The cut leaves sliver edges/faces where the plane grazed the input;
        # QuadriFlow can spin (near-)forever on those. Collapse cut edges much
        # shorter than their neighbours (midpoints stay on the cut plane).
        cut_edges = [e for e in bm.edges
                     if abs(e.verts[0].co[ax] - pad) <= eps
                     and abs(e.verts[1].co[ax] - pad) <= eps]
        if len(cut_edges) >= 4:
            lens = sorted(e.calc_length() for e in cut_edges)
            median = lens[len(lens) // 2]
            short = [e for e in cut_edges if e.calc_length() < 0.2 * median]
            if short:
                bmesh.ops.collapse(bm, edges=short, uvs=False)
                # Only re-snap what the collapse actually touched. Snapping a
                # whole 0.2-edge-length band (as this used to) drags the second
                # vertex row onto the plane as well, which buries interior
                # edges and whole faces *inside* the symmetry plane; mirroring
                # then folds those onto their own copies (4-face edges,
                # zero-thickness double flaps) and tears the seam back open.
                if pad == 0.0:
                    for v in bm.verts:
                        if abs(v.co[ax]) <= eps:
                            v.co[ax] = 0.0
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return len(mesh.polygons) > 0


def _clear_plane_interior(bm, ax: int, eps: float, nudge: float) -> int:
    """Make the symmetry plane touch nothing but the open cut ring.

    A half that is about to be mirrored may carry *interior* geometry lying
    exactly in the plane — faces flattened into it, edges whose two faces both
    sit inside the half. ``bmesh.ops.mirror`` folds that geometry onto its own
    reflection: in-plane edges end up with four faces and in-plane faces become
    zero-thickness double flaps. The weld then cannot resolve them, which is
    where the seam holes and the coincident-vertex debris came from.

    Two local repairs, applied until the plane is clean:

    * a face lying entirely in the plane is a zero-thickness membrane; deleting
      it hands its rim to the mirror, which is the topology that was meant. A
      face is only removed when it has a neighbour outside the plane, so a whole
      shell that legitimately lives in the plane is never touched.
    * any other vertex sitting in the plane without belonging to the cut ring is
      pushed one hair inside the half, far below any visible scale but well
      above the weld epsilon.

    Returns the number of elements repaired.
    """
    def on_plane(v):
        return abs(v.co[ax]) <= eps

    repaired = 0
    for _ in range(4):
        flat = set()
        for f in bm.faces:
            if all(on_plane(v) for v in f.verts):
                flat.add(f)
        # iterate bm.faces (index order), not the set: BMFace sets hash by
        # id() and their order varies per process, which fed a varying
        # deletion order to bmesh and made exact-symmetry nondeterministic
        doomed = [f for f in bm.faces
                  if f in flat
                  and any(nf is not f and nf not in flat
                          for e in f.edges for nf in e.link_faces)]
        if doomed:
            bmesh.ops.delete(bm, geom=doomed, context='FACES')
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            repaired += len(doomed)
        # vertices entitled to stay on the plane: the cut ring, i.e. those
        # carrying a boundary edge that lies in the plane
        ring = set()
        for e in bm.edges:
            if len(e.link_faces) == 1 and on_plane(e.verts[0]) and on_plane(e.verts[1]):
                ring.add(e.verts[0])
                ring.add(e.verts[1])
        stray = [v for v in bm.verts
                 if v.link_faces and on_plane(v) and v not in ring]
        for v in stray:
            v.co[ax] = -nudge
        repaired += len(stray)
        if not doomed and not stray:
            break
    return repaired


def _fuse_seam_tris(work_obj, axes) -> int:
    """join_triangles on the tri row a post-solve plane cut leaves behind."""
    mesh = work_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    band = 2.0 * (_mean_edge_length(mesh) or 1e-3)
    tris = [f for f in bm.faces
            if len(f.verts) == 3
            and any(any(abs(v.co[ax]) < band for v in f.verts) for ax in axes)]
    fused = 0
    if tris:
        before = len(bm.faces)
        bmesh.ops.join_triangles(bm, faces=tris,
                                 angle_face_threshold=3.15,
                                 angle_shape_threshold=3.15)
        fused = before - len(bm.faces)
        bm.to_mesh(mesh)
        mesh.update()
    bm.free()
    return fused


def _boundary_loops(bm):
    """Connected components of boundary edges, as lists of edges."""
    edges = [e for e in bm.edges if len(e.link_faces) == 1]
    seen = set()
    loops = []
    for start in edges:
        if start in seen:
            continue
        comp = []
        stack = [start]
        seen.add(start)
        while stack:
            e = stack.pop()
            comp.append(e)
            for v in e.verts:
                for ne in v.link_edges:
                    if ne not in seen and len(ne.link_faces) == 1:
                        seen.add(ne)
                        stack.append(ne)
        loops.append(comp)
    return loops


def mirror_weld(work_obj, axes, snap_tol: float, weld_eps: float = 1e-7) -> int:
    """Snap the cut boundary exactly onto the planes, then mirror + weld so the
    result is bit-exact symmetric. Returns the number of boundary edges that
    remained near a symmetry plane (0 = watertight seam)."""
    mesh = work_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    for ax in axes:
        # Snap whole cut loops, not just verts inside a fixed distance: the
        # solver may drift cut verts off the plane by a fraction of the LOCAL
        # edge length, which on coarse regions is far beyond any global
        # tolerance (this is what left holes along the seam).
        for comp in _boundary_loops(bm):
            verts = {v for e in comp for v in e.verts}
            span = max(abs(v.co[ax]) for v in verts)
            lens = sorted(e.calc_length() for e in comp)
            median = lens[len(lens) // 2] if lens else 0.0
            if span <= max(snap_tol, 0.5 * median):
                for v in verts:
                    v.co[ax] = 0.0
        for v in bm.verts:
            if v.is_boundary and abs(v.co[ax]) <= snap_tol:
                v.co[ax] = 0.0

    for ax in axes:
        # the mirror only produces a watertight seam when the plane carries
        # nothing but the open cut ring
        _clear_plane_interior(bm, ax, weld_eps, max(weld_eps * 10.0, snap_tol * 1e-3))
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.mirror(
            bm, geom=geom, matrix=Matrix.Identity(4),
            merge_dist=0.0, axis=_AXIS_NAMES[ax],
        )
        bm.verts.ensure_lookup_table()
        plane_verts = [v for v in bm.verts if abs(v.co[ax]) <= weld_eps]
        if plane_verts:
            bmesh.ops.remove_doubles(bm, verts=plane_verts, dist=weld_eps)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        # Close residual seam cracks: cut verts that drifted past every earlier
        # snap were mirrored into near-coincident pairs straddling the plane.
        residual = [v for v in bm.verts if v.is_boundary and abs(v.co[ax]) <= snap_tol]
        if residual:
            for v in residual:
                v.co[ax] = 0.0
            bmesh.ops.remove_doubles(bm, verts=residual, dist=max(weld_eps, snap_tol * 0.1))
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

    # Any small boundary loop at (or straddling) a symmetry plane is a pinhole
    # the cut/weld left behind; fill it. Larger near-plane loops are treated as
    # legitimate original openings and left alone. The vert set of a straddling
    # pinhole is its own mirror image, so exact symmetry is preserved.
    #
    # Both this pass and the leftover count below classify whole boundary
    # components (from _boundary_loops, which walks bm.edges in index order): a
    # component counts as a seam defect only when *every* one of its edges hugs
    # a symmetry plane. The old code grew each component out of a set of
    # near-plane edges and marked it "open" only when it happened to bump into
    # an edge outside that set — a test the shared `seen` guard could skip
    # entirely, so the verdict depended on traversal order, and for the count
    # below on Python's set iteration order, which varies between runs.
    for ax in axes:
        tol = 2.0 * max(snap_tol, weld_eps)

        def confined(comp, _ax=ax, _tol=tol):
            return all(abs(e.verts[0].co[_ax]) <= _tol
                       and abs(e.verts[1].co[_ax]) <= _tol for e in comp)

        pinholes = []
        for comp in _boundary_loops(bm):
            if 3 <= len(comp) <= 6 and confined(comp):
                pinholes.extend(comp)
        if pinholes:
            try:
                bmesh.ops.holes_fill(bm, edges=pinholes, sides=6)
            except Exception:
                pass
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    # Count remaining seam defects: boundary loops confined to a symmetry
    # plane. Boundary chains merely crossing the plane are original openings.
    leftover = 0
    for ax in axes:
        tol = 2.0 * max(snap_tol, weld_eps)
        for comp in _boundary_loops(bm):
            if all(abs(e.verts[0].co[ax]) <= tol and abs(e.verts[1].co[ax]) <= tol
                   for e in comp):
                leftover += len(comp)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return leftover


def restore_lost_regions(work_obj, pre_mesh, report) -> int:
    """Graft back input regions the solver silently dropped.

    Blender's QuadriFlow discards interior cavities (mouth bags, nested shells)
    when the mesh has open boundaries — e.g. on the bisected half used for
    exact symmetry. Any connected input region whose faces are all far from the
    solved surface is copied back verbatim (original topology) and reported.

    A region is restored when it is either substantial (>= 16 faces well clear
    of the result) or an ENTIRE input shell the solver left essentially
    uncovered. The size gate alone used to miss the small-shell case completely:
    with Keep Small Shells off, hair cards and teeth reach the solver, are too
    thin for the lattice to express, and vanish — 8 of 9 shells on the plate
    fixture, 100 of 172 on Dinasty. A whole shell going missing is never noise,
    whatever its face count, so shells are classified in their own right.
    Returns the number of restored faces."""
    if not len(pre_mesh.polygons) or not len(work_obj.data.polygons):
        return 0

    solved = work_obj.data
    bm = bmesh.new()
    bm.from_mesh(solved)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    verts = [tuple(v.co) for v in bm.verts]
    tris = [[v.index for v in f.verts] for f in bm.faces]
    bm.free()
    if not tris:
        return 0
    try:
        bvh = BVHTree.FromPolygons(verts, tris, all_triangles=True)
    except Exception:
        return 0

    n = len(pre_mesh.polygons)
    centers = np.empty(n * 3)
    pre_mesh.polygons.foreach_get("center", centers)
    centers = centers.reshape(-1, 3)
    co = np.empty(len(pre_mesh.vertices) * 3)
    pre_mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    diag = float(np.linalg.norm(co.max(0) - co.min(0))) or 1.0

    dist = np.empty(n)
    for i in range(n):
        hit = bvh.find_nearest(Vector(centers[i]))
        dist[i] = hit[3] if hit[0] is not None else 0.0
    far = dist > 0.02 * diag
    if not far.any():
        return 0

    pre_bm = bmesh.new()
    pre_bm.from_mesh(pre_mesh)
    pre_bm.faces.ensure_lookup_table()
    lost = set()

    # 1. whole input shells the result does not cover. Judged on the shell, not
    #    on a component of far faces: a dropped hair card typically still has
    #    one face grazing the tolerance, and requiring *every* face to be far
    #    let all of them through.
    seen = set()
    for f0 in pre_bm.faces:
        if f0.index in seen:
            continue
        shell = []
        stack = [f0]
        seen.add(f0.index)
        while stack:
            f = stack.pop()
            shell.append(f.index)
            for e in f.edges:
                for nf in e.link_faces:
                    if nf.index not in seen:
                        seen.add(nf.index)
                        stack.append(nf)
        if len(shell) >= 3 and float(far[shell].mean()) >= SHELL_LOST_FRACTION:
            lost.update(shell)

    # 2. partial drops (interior cavities, nested geometry) inside a shell the
    #    solver did otherwise cover
    seen = set(lost)
    for fi in np.nonzero(far)[0]:
        if int(fi) in seen:
            continue
        comp = []
        stack = [pre_bm.faces[int(fi)]]
        seen.add(int(fi))
        while stack:
            f = stack.pop()
            comp.append(f.index)
            for e in f.edges:
                for nf in e.link_faces:
                    if nf.index not in seen and far[nf.index]:
                        seen.add(nf.index)
                        stack.append(nf)
        if len(comp) >= 16 and float(dist[comp].mean()) > 0.04 * diag:
            lost.update(comp)
    if not lost:
        pre_bm.free()
        return 0

    doomed = [f for f in pre_bm.faces if f.index not in lost]
    bmesh.ops.delete(pre_bm, geom=doomed, context='FACES')
    graft = bpy.data.meshes.new(pre_mesh.name + "_graft")
    pre_bm.to_mesh(graft)
    pre_bm.free()

    out = bmesh.new()
    out.from_mesh(solved)
    out.from_mesh(graft)
    out.to_mesh(solved)
    out.free()
    solved.update()
    try:
        bpy.data.meshes.remove(graft)
    except Exception:
        pass
    report.setdefault("warnings", []).append(
        "solver dropped %d faces of interior/nested geometry; original topology "
        "was restored for those regions" % len(lost)
    )
    report["restored_faces"] = len(lost)
    return len(lost)


def seal_solver_holes(work_obj, ref_mesh, max_loop: int = 12) -> int:
    """Close small boundary loops the solver tore into a watertight input.

    QuadriFlow reports success and still hands back a torn quad mesh for a shell
    far below its useful resolution. With Keep Small Shells off every hair card,
    tooth and button goes through the solver, so those tears end up in the
    result and read as open seams — and the exact-symmetry mirror cannot close
    them, because they are nowhere near the symmetry plane.

    Watertight in, watertight out: the pass runs only when the *input* had no
    boundary at all, so a mesh with genuine openings (a plane, a cloth panel, a
    half-open mouth bag) is never touched. Only small loops are filled — a large
    one means a whole region went missing, which is restore_lost_regions' job,
    not something to cap with an n-gon. Filling adds no vertices, so exact
    symmetry survives it untouched.

    Returns the number of boundary edges sealed.
    """
    if ref_mesh is None or not len(ref_mesh.polygons) or not len(work_obj.data.polygons):
        return 0
    ref_bm = bmesh.new()
    ref_bm.from_mesh(ref_mesh)
    ref_open = any(len(e.link_faces) == 1 for e in ref_bm.edges)
    ref_bm.free()
    if ref_open:
        return 0

    bm = bmesh.new()
    bm.from_mesh(work_obj.data)
    fill = [e for comp in _boundary_loops(bm) if len(comp) <= max_loop
            for e in comp]
    if not fill:
        bm.free()
        return 0
    before = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    try:
        res = bmesh.ops.holes_fill(bm, edges=fill, sides=max_loop)
    except Exception:
        bm.free()
        return 0
    # only the new caps get their winding computed; a whole-mesh recalc would
    # re-run the outward heuristic that fix_orientation exists to replace
    new_faces = [f for f in (res or {}).get("faces", ()) if f.is_valid]
    if new_faces:
        bmesh.ops.recalc_face_normals(bm, faces=new_faces)
    sealed = before - sum(1 for e in bm.edges if len(e.link_faces) == 1)
    if sealed <= 0:
        bm.free()
        return 0
    bm.to_mesh(work_obj.data)
    bm.free()
    work_obj.data.update()
    return sealed


def fix_orientation(work_obj, ref_mesh) -> int:
    """Flip whole shells whose normals disagree with the nearest source
    surface. recalc_face_normals' outward heuristic is unreliable on open or
    multi-shell meshes, so orientation is voted per shell against the input.
    Returns the number of faces flipped."""
    me = work_obj.data
    if not len(me.polygons) or not len(ref_mesh.polygons):
        return 0

    ref_bm = bmesh.new()
    ref_bm.from_mesh(ref_mesh)
    bmesh.ops.triangulate(ref_bm, faces=ref_bm.faces[:])
    ref_verts = [v.co[:] for v in ref_bm.verts]
    ref_faces = [[v.index for v in f.verts] for f in ref_bm.faces]
    ref_normals = [f.normal.copy() for f in ref_bm.faces]
    ref_bm.free()
    try:
        bvh = BVHTree.FromPolygons(ref_verts, ref_faces, all_triangles=True)
    except Exception:
        return 0

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    visited = set()
    flipped_faces = 0
    changed = False
    for f0 in bm.faces:
        if f0.index in visited:
            continue
        comp = []
        stack = [f0]
        visited.add(f0.index)
        while stack:
            f = stack.pop()
            comp.append(f)
            for e in f.edges:
                for nf in e.link_faces:
                    if nf.index not in visited:
                        visited.add(nf.index)
                        stack.append(nf)
        step = max(1, len(comp) // 200)
        score = 0.0
        for f in comp[::step]:
            hit = bvh.find_nearest(f.calc_center_median())
            if hit[0] is not None and hit[2] is not None:
                score += f.normal.dot(ref_normals[hit[2]])
        if score < 0.0:
            bmesh.ops.reverse_faces(bm, faces=comp)
            flipped_faces += len(comp)
            changed = True
    if changed:
        bm.to_mesh(me)
        me.update()
    bm.free()
    return flipped_faces


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


def preprocess(context, work_obj, s, report: dict) -> None:
    mesh = work_obj.data
    report["input_faces"] = len(mesh.polygons)
    report["input_verts"] = len(mesh.vertices)

    if not getattr(s, "use_marked_sharp", False):
        # start from a clean slate so stale sharp flags don't steer the solver
        zeros = np.zeros(len(mesh.edges), dtype=bool)
        mesh.edges.foreach_set("use_edge_sharp", zeros)

    try:
        report["hard_edges"] = analysis.mark_hard_edges(work_obj, s)
    except Exception as exc:
        report.setdefault("warnings", []).append(f"hard edge detection failed: {exc}")
        report["hard_edges"] = 0

    # Flag-based flow features are a Native-backend capability: measured
    # (identical output hashes) Blender's QuadriFlow derives features from
    # dihedral angle only and ignores sharp FLAGS on flat geometry.
    # Guides are not listed here: run_remesh reroutes guided QuadriFlow
    # solves to the native backend right after this preprocess.
    if report.get("backend", "QUADRIFLOW") == "QUADRIFLOW":
        flag_features = [name for flag, name in (
            (getattr(s, "use_materials", False), "material boundaries"),
            (getattr(s, "use_uv_seams", False), "UV island boundaries"),
        ) if flag]
        if flag_features:
            report.setdefault("warnings", []).append(
                "QuadriFlow ignores flag-marked features on flat geometry - "
                + ", ".join(flag_features)
                + " will only influence flow where real dihedral angles "
                  "coincide; use the Native backend for these")

    if getattr(s, "use_uv_seams", False):
        try:
            report["uv_seam_edges"] = analysis.uv_island_boundaries_to_sharp(work_obj)
        except Exception as exc:
            report.setdefault("warnings", []).append(f"UV seam detection failed: {exc}")

    if getattr(s, "use_materials", False):
        try:
            report["material_boundary_edges"] = analysis.material_boundaries_to_sharp(work_obj)
        except Exception as exc:
            report.setdefault("warnings", []).append(f"material boundaries failed: {exc}")

    try:
        analysis.build_density_attr(work_obj, s)
    except Exception as exc:
        report.setdefault("warnings", []).append(f"density attribute failed: {exc}")

    if getattr(s, "use_guides", False):
        coll = getattr(s, "guide_collection", None)
        objs = list(coll.all_objects) if coll is not None else []
        if not objs:
            report.setdefault("warnings", []).append(
                "Use Guides is on but the guide collection is empty"
            )
            report["guide_edges"] = 0
        else:
            try:
                report["guide_edges"] = guides.project_guides(work_obj, objs, s)
                if report["guide_edges"] == 0:
                    report.setdefault("warnings", []).append(
                        "guides produced no surface paths (too far from the mesh?)"
                    )
            except Exception as exc:
                report.setdefault("warnings", []).append(f"guide projection failed: {exc}")
                report["guide_edges"] = 0


# ---------------------------------------------------------------------------
# remesh drivers
# ---------------------------------------------------------------------------


def _adaptive_boost(s) -> float:
    """Extra faces requested up-front so the adaptive post-pass has slack to
    remove low-curvature edge loops.

    Kept small on purpose: adaptivity is delivered by the density relaxation,
    which is count-preserving, and safe edge-loop removal is opportunistic. A
    large boost would just overshoot the user's target.
    """
    a = float(getattr(s, "adaptive_size", 0.0)) / 100.0
    if a <= 0.0 or not bool(getattr(s, "adapt_quad_count", True)):
        return 1.0
    return 1.0 + 0.10 * a


def _call_backend(context, backend, work_obj, s, target, *, force_boundary=False,
                  symmetry=None, report=None):
    """Call ``backend.remesh``, passing the extra QuadForge hints only if the
    backend accepts them (the contract signature is the 4-argument one).
    When ``report`` is given, input regions the solver silently dropped are
    grafted back afterwards (see restore_lost_regions)."""
    kwargs = {}
    try:
        import inspect
        params = inspect.signature(backend.remesh).parameters
        if "force_preserve_boundary" in params:
            kwargs["force_preserve_boundary"] = force_boundary
        if "symmetry" in params:
            kwargs["symmetry"] = symmetry
    except (TypeError, ValueError):
        kwargs = {}
    pre = work_obj.data.copy() if report is not None else None
    try:
        stats = backend.remesh(context, work_obj, s, int(target), **kwargs)
        if pre is not None:
            try:
                restore_lost_regions(work_obj, pre, report)
            except Exception as exc:
                report.setdefault("warnings", []).append(
                    f"lost-region check failed: {exc}"
                )
        return stats
    finally:
        if pre is not None:
            try:
                bpy.data.meshes.remove(pre)
            except Exception:
                pass


def run_backend(context, backend, work_obj, s, face_target: int, report: dict,
                post_pass=None) -> None:
    """Backend call including the exact-symmetry bisect / mirror-weld path.

    ``post_pass(work_obj, target)`` runs on the solver output *before* mirroring,
    so it can never break exact symmetry. ``target`` is the face count wanted for
    the mesh as it stands (halved per symmetry axis in the exact path).
    """
    axes = symmetry_axes(s)
    exact = bool(getattr(s, "exact_symmetry", False)) and bool(axes)
    boost = _adaptive_boost(s)
    try:
        report["mean_density"] = round(analysis.density_target_scale(work_obj.data), 4)
    except Exception:
        pass

    requested = int(max(12, round(face_target * boost)))
    report["requested_faces"] = requested
    report["adaptive_boost"] = round(boost, 4)

    if not exact:
        report["symmetry_mode"] = "solver" if axes else "none"
        # shell preservation for every backend, not just the exact-sym path:
        # small authored shells (hair, teeth, piercings) stay verbatim
        side_mesh = None
        if bool(getattr(s, "preserve_small_shells", True)):
            limit = int(getattr(s, "small_shell_limit", 0) or 0)
            if limit <= 0:
                limit = max(64, int(0.02 * len(work_obj.data.polygons)))
            try:
                side_mesh = split_small_shells_aside(work_obj, limit)
                if side_mesh is not None:
                    report["side_shell_faces"] = len(side_mesh.polygons)
                    # the target should approximate the TOTAL output:
                    # deduct preserved faces from the solve budget — but only
                    # while that leaves a healthy budget; a preserved set
                    # bigger than the target must not starve the solved
                    # surface, it just overshoots (with a warning)
                    side_n = len(side_mesh.polygons)
                    if side_n <= 0.6 * requested:
                        requested = requested - side_n
                        report["solve_budget"] = requested
                    else:
                        report.setdefault("warnings", []).append(
                            "preserved shells (%d faces) approach or exceed "
                            "the target; total output will overshoot" % side_n)
            except Exception as exc:
                report.setdefault("warnings", []).append(
                    f"small-shell split failed: {exc}")
                side_mesh = None
        try:
            stats = _call_backend(context, backend, work_obj, s, requested, report=report)
            report["backend_stats"] = stats if isinstance(stats, dict) else {}
            if post_pass is not None:
                post_pass(work_obj, face_target)
        finally:
            if side_mesh is not None:
                rejoin_side_mesh(work_obj, side_mesh)
        return

    report["symmetry_mode"] = "exact"
    report["symmetry_axes"] = [_AXIS_NAMES[a] for a in axes]
    eps = max(_mean_edge_length(work_obj.data) * 1e-3, 1e-7)

    # keep small shells out of the bisect: cutting thin centerline shells
    # (hair, teeth, ruff) shreds them into seam pinholes
    side_mesh = None
    if bool(getattr(s, "preserve_small_shells", True)):
        limit = int(getattr(s, "small_shell_limit", 0) or 0)
        if limit <= 0:
            limit = max(64, int(0.02 * len(work_obj.data.polygons)))
        try:
            side_mesh = split_small_shells_aside(work_obj, limit)
            if side_mesh is not None:
                report["side_shell_faces"] = len(side_mesh.polygons)
                side_n = len(side_mesh.polygons)
                if side_n <= 0.6 * requested:
                    requested = requested - side_n
                    report["solve_budget"] = requested
                else:
                    report.setdefault("warnings", []).append(
                        "preserved shells (%d faces) approach or exceed "
                        "the target; total output will overshoot" % side_n)
        except Exception as exc:
            report.setdefault("warnings", []).append(
                f"small-shell split failed: {exc}")
            side_mesh = None

    # pad the cut by ~3 target edge lengths: the pinned cut boundary flattens
    # features within a couple of edges of the plane (inner toes, nose tips);
    # the surplus band is trimmed back after solving
    try:
        area = analysis.world_area(work_obj.data, work_obj.matrix_world)
        pad = 3.0 * float(np.sqrt(max(area, 1e-12) / max(requested, 12)))
    except Exception:
        pad = 0.0
    report["symmetry_pad"] = round(pad, 5)

    # keep a copy so we can fall back if bisecting destroys the mesh
    backup = bmesh.new()
    backup.from_mesh(work_obj.data)
    ok = bisect_to_half(work_obj, axes, eps, pad=pad)
    if not ok:
        backup.to_mesh(work_obj.data)
        backup.free()
        work_obj.data.update()
        if side_mesh is not None:
            rejoin_side_mesh(work_obj, side_mesh)
            side_mesh = None
        report.setdefault("warnings", []).append(
            "exact symmetry: bisecting left no geometry, falling back to solver symmetry"
        )
        report["symmetry_mode"] = "solver"
        stats = _call_backend(context, backend, work_obj, s, requested, report=report)
        report["backend_stats"] = stats if isinstance(stats, dict) else {}
        if post_pass is not None:
            post_pass(work_obj, face_target)
        return
    backup.free()

    half_target = int(max(12, round(requested / float(2 ** len(axes)))))
    report["half_target"] = half_target
    stats = _call_backend(
        context, backend, work_obj, s, half_target,
        force_boundary=True, symmetry=(False, False, False), report=report,
    )
    report["backend_stats"] = stats if isinstance(stats, dict) else {}

    if post_pass is not None:
        post_pass(work_obj, int(max(12, round(face_target / float(2 ** len(axes))))))

    if pad > 0.0:
        # cut the solved padded half back to the exact plane with the proven
        # bisect (clean plane boundary, sliver collapse), then fuse the tri
        # row the cut leaves into quads where possible
        if not bisect_to_half(work_obj, axes, eps):
            report.setdefault("warnings", []).append(
                "exact symmetry: post-solve trim failed; seam may be off-plane")
        _fuse_seam_tris(work_obj, axes)
        # the cut grazes solved verts, leaving coincident-vertex debris
        # (zero-length boundary edges) on the plane — weld it away
        bmc = bmesh.new()
        bmc.from_mesh(work_obj.data)
        med = _mean_edge_length(work_obj.data) or 1e-3
        # candidates are verts *on* the plane only: widening this band lets the
        # weld pull second-row vertices onto the plane, which is exactly the
        # in-plane interior geometry the mirror cannot handle
        near = [v for v in bmc.verts
                if any(abs(v.co[ax]) < 1e-3 * med for ax in axes)]
        if near:
            bmesh.ops.remove_doubles(bmc, verts=near, dist=1e-4 * med)
            deg = [f for f in bmc.faces if f.calc_area() <= 1e-16]
            if deg:
                bmesh.ops.delete(bmc, geom=deg, context='FACES')
            bmc.to_mesh(work_obj.data)
            work_obj.data.update()
        bmc.free()

    snap_tol = max(_mean_edge_length(work_obj.data) * 0.25, 1e-6)
    leftover = mirror_weld(work_obj, axes, snap_tol)
    if side_mesh is not None:
        rejoin_side_mesh(work_obj, side_mesh)
        side_mesh = None
    report["seam_open_edges"] = leftover
    if leftover:
        report.setdefault("warnings", []).append(
            f"exact symmetry: {leftover} boundary edges remain near the seam"
        )


# ---------------------------------------------------------------------------
# scene integration
# ---------------------------------------------------------------------------


def originals_collection(context):
    coll = bpy.data.collections.get(ORIGINALS_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(ORIGINALS_COLLECTION)
    scene_coll = context.scene.collection
    if coll.name not in scene_coll.children:
        try:
            scene_coll.children.link(coll)
        except Exception:
            pass
    coll.hide_viewport = True
    coll.hide_render = True
    coll.hide_select = True
    return coll


def stow_original(context, obj):
    coll = originals_collection(context)
    for c in list(obj.users_collection):
        if c is coll:
            continue
        try:
            c.objects.unlink(obj)
        except Exception:
            pass
    if obj.name not in coll.objects:
        try:
            coll.objects.link(obj)
        except Exception:
            pass
    obj.hide_viewport = True
    obj.hide_render = True
    try:
        obj.hide_set(True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_remesh(context, obj, s) -> dict:
    t0 = time.perf_counter()
    report = {"warnings": [], "limitations": []}
    work = None
    result = None
    prev_active = None

    try:
        if obj is None or obj.type != "MESH":
            return _fail("QuadForge needs a mesh object", report, t0, s=s)
        if obj.data is None or len(obj.data.polygons) == 0:
            return _fail(f"'{obj.name}' has no faces to remesh", report, t0, s=s)
        if s is None:
            return _fail("missing QuadForge settings", report, t0)

        view_layer = getattr(context, "view_layer", None)
        if view_layer is not None:
            prev_active = view_layer.objects.active
        if getattr(obj, "mode", "OBJECT") != "OBJECT":
            try:
                with context.temp_override(active_object=obj, object=obj,
                                           selected_objects=[obj]):
                    bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                return _fail("could not leave edit mode", report, t0, s=s)

        transfer = _try_import("transfer")
        reporting = _try_import("report")

        snapshot = None
        if transfer is not None and hasattr(transfer, "capture"):
            try:
                snapshot = transfer.capture(obj)
            except Exception as exc:
                report["warnings"].append(f"data snapshot failed: {exc}")

        backend, backend_name = _get_backend(s)
        report["backend"] = backend_name
        if getattr(s, "backend", "QUADRIFLOW") == "NATIVE" and backend_name != "NATIVE":
            report["warnings"].append("native backend unavailable, used QuadriFlow")

        # Remesh the REST shape: non-zero shape-key sliders would bake the
        # deformed surface as the new Basis and then re-apply the key deltas on
        # top (double deformation). Snapshot already stored the slider values;
        # zero them for the evaluated copy and restore right after.
        _key_restore = []
        try:
            for kb in (obj.data.shape_keys.key_blocks
                       if obj.data.shape_keys else ()):
                if kb.value != 0.0:
                    _key_restore.append((kb, kb.value))
                    kb.value = 0.0
        except Exception:
            _key_restore = []

        try:
            work = make_work_object(context, obj)
        finally:
            for kb, val in _key_restore:
                try:
                    kb.value = val
                except Exception:
                    pass
        face_target = face_target_from_settings(obj, work.data, s)
        report["target_faces"] = face_target
        report["mode"] = getattr(s, "mode", "FACES")

        preprocess(context, work, s, report)

        # Guided solves can't run on QuadriFlow: the operator hands the solver
        # bare vertex/triangle arrays, so the projected sharp paths never reach
        # it (measured: bit-identical output with/without marks). Reroute to
        # the native field solver, which consumes both the sharp paths and the
        # qf_guide attribute. Only when guides actually landed on the surface -
        # a junk or empty guide collection shouldn't change the solver.
        if (backend_name == "QUADRIFLOW" and getattr(s, "use_guides", False)
                and int(report.get("guide_edges") or 0) > 0):
            native_mod = None
            try:
                from .backends import native as native_mod
            except Exception:
                native_mod = None
            if native_mod is not None and hasattr(native_mod, "remesh"):
                backend, backend_name = native_mod, "NATIVE"
                report["backend"] = "NATIVE"
                report["guides_rerouted"] = True
                report["warnings"].append(
                    "guides cannot steer QuadriFlow - this solve was routed "
                    "to the Native backend")
            else:
                report["warnings"].append(
                    "guides have no effect on QuadriFlow and the Native "
                    "backend is unavailable - remeshing without guide "
                    "influence")

        # reference for post-remesh orientation repair (pre-bisect, full mesh)
        ref_mesh = work.data.copy()

        # keep the pre-remesh mesh around for the adaptive post-pass
        src_mesh = None
        adaptive = float(getattr(s, "adaptive_size", 0.0)) > 0.0
        use_paint = bool(getattr(s, "use_paint_density", False))
        post_pass = None
        if (adaptive or use_paint) and backend_name == "QUADRIFLOW":
            src_mesh = work.data.copy()
            boost = _adaptive_boost(s)

            def post_pass(_obj, _target, _src=src_mesh):
                # 1. density-weighted relaxation: real quad size variation,
                #    topology and face count untouched.
                try:
                    report["adaptive_relax"] = analysis.density_relax(_obj, _src, s)
                except Exception as exc:
                    report["warnings"].append(f"density relaxation failed: {exc}")
                # 2. optional conservative all-quad edge-loop removal in the
                #    sparse regions, only when count drift is allowed.
                if boost > 1.0:
                    try:
                        report["adaptive_decimate"] = analysis.adaptive_decimate(
                            _obj, _src, s, int(_target)
                        )
                    except Exception as exc:
                        report["warnings"].append(f"adaptive decimate failed: {exc}")
        elif adaptive:
            report["limitations"].append(
                "adaptive post-pass only runs on the QuadriFlow backend"
            )

        try:
            run_backend(context, backend, work, s, face_target, report, post_pass=post_pass)

            if len(work.data.polygons) == 0:
                raise RuntimeError("the solver returned an empty mesh")

            bstats = report.get("backend_stats") or {}
            for w in bstats.get("warnings") or []:
                report["warnings"].append(w)
            if bstats.get("unsolvable_parts"):
                report["warnings"].append(
                    "%d degenerate shell(s) (%d faces) refused by the solver; "
                    "their original topology was kept"
                    % (bstats["unsolvable_parts"], bstats.get("unsolvable_part_faces", 0))
                )

            try:
                sealed = seal_solver_holes(work, ref_mesh)
                if sealed:
                    report["sealed_solver_holes"] = sealed
                    report["warnings"].append(
                        "%d boundary edge(s) the solver tore into a watertight "
                        "input were sealed" % sealed
                    )
            except Exception as exc:
                report["warnings"].append(f"hole sealing failed: {exc}")

            try:
                report["orientation_flipped_faces"] = fix_orientation(work, ref_mesh)
            except Exception as exc:
                report["warnings"].append(f"orientation repair failed: {exc}")

            # purge degenerate debris (zero-length edges, zero-area faces)
            # before data transfer sees the mesh
            try:
                if work.data.validate(verbose=False):
                    report["validated"] = True
                work.data.update()
            except Exception:
                pass
        finally:
            try:
                bpy.data.meshes.remove(ref_mesh)
            except Exception:
                pass

        if adaptive:
            relaxed = (report.get("adaptive_relax") or {}).get("iterations", 0)
            removed = (report.get("adaptive_decimate") or {}).get("loops_removed", 0)
            if not relaxed:
                report["limitations"].append(
                    "adaptivity had no effect: the curvature/paint density field was "
                    "flat over this mesh"
                )
            elif not removed:
                report["limitations"].append(
                    "QuadriFlow takes no density input, so adaptivity is delivered by "
                    "density-weighted relaxation (quad size varies, quad count does not). "
                    "No edge loop could be removed without breaking the all-quad output."
                )

        if src_mesh is not None:
            try:
                bpy.data.meshes.remove(src_mesh)
            except Exception:
                pass
            src_mesh = None

        # shading: follow the majority of the source
        try:
            smooth = np.zeros(len(obj.data.polygons), dtype=bool)
            obj.data.polygons.foreach_get("use_smooth", smooth)
            if smooth.size and smooth.mean() > 0.5:
                work.data.polygons.foreach_set(
                    "use_smooth", np.ones(len(work.data.polygons), dtype=bool)
                )
                work.data.update()
        except Exception:
            pass

        # ---- data transfer -------------------------------------------------
        if snapshot is not None and hasattr(transfer, "apply"):
            try:
                report["transfer"] = transfer.apply(snapshot, work, s)
            except Exception as exc:
                report["warnings"].append(f"data transfer failed: {exc}")
        elif transfer is None:
            report["limitations"].append(
                "core.transfer not available - UVs / weights / shape keys were not "
                "transferred (QuadriFlow attribute preservation only)"
            )

        # ---- promote the working object to the result ----------------------
        source_collections = [c for c in obj.users_collection]
        result = work
        work = None
        result.name = obj.name + RESULT_SUFFIX
        result.data.name = obj.data.name + RESULT_SUFFIX
        result.matrix_world = obj.matrix_world.copy()
        try:
            result.parent = obj.parent
            result.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
        except Exception:
            pass

        scene_coll = context.scene.collection
        for c in list(result.users_collection):
            if c is not scene_coll:
                continue
            if source_collections and scene_coll not in source_collections:
                try:
                    scene_coll.objects.unlink(result)
                except Exception:
                    pass
        for c in source_collections:
            if result.name not in c.objects:
                try:
                    c.objects.link(result)
                except Exception:
                    pass
        if not result.users_collection:
            scene_coll.objects.link(result)

        keep_original = bool(getattr(s, "keep_original", True))
        if keep_original:
            stow_original(context, obj)

        # ---- report --------------------------------------------------------
        stats = mesh_quick_stats(result)
        if reporting is not None and hasattr(reporting, "mesh_report"):
            try:
                stats.update(reporting.mesh_report(result))
            except Exception as exc:
                report["warnings"].append(f"quality report failed: {exc}")
        axes = symmetry_axes(s)
        for ax in axes:
            try:
                stats[f"symmetry_error_{_AXIS_NAMES[ax].lower()}"] = symmetry_error(result, ax)
            except Exception:
                pass
        stats["time_s"] = round(time.perf_counter() - t0, 4)
        stats["target"] = face_target
        stats["backend"] = backend_name
        # surface the preprocessing counts so callers don't have to parse
        # last_report just to show what QuadForge did
        for key in ("input_faces", "input_verts", "hard_edges", "guide_edges",
                    "material_boundary_edges", "uv_seam_edges", "seam_open_edges",
                    "orientation_flipped_faces", "restored_faces",
                    "symmetry_mode", "requested_faces"):
            if key in report:
                stats.setdefault(key, report[key])
        report["stats"] = stats

        if view_layer is not None:
            try:
                view_layer.objects.active = result
                result.select_set(True)
            except Exception:
                pass

        copy_settings(s, result.quadforge)
        _write_last_report(result, s, report, ok=True)
        out = {"ok": True, "error": None, "object": result, "stats": stats, "report": report}

        # LAST: removing the original invalidates `s` (it lives on that object),
        # so nothing may touch the settings after this point.
        if not keep_original:
            discard_object(obj)
            obj = None
            s = None
        return out

    except Exception as exc:  # noqa: BLE001 - the pipeline must never raise
        import traceback
        report["traceback"] = traceback.format_exc(limit=8)
        return _fail(str(exc) or exc.__class__.__name__, report, t0, s=s)
    finally:
        if work is not None:
            discard_object(work)
        if prev_active is not None and result is None:
            try:
                context.view_layer.objects.active = prev_active
            except Exception:
                pass


_SKIP_COPY = {"rna_type", "last_report"}


def copy_settings(src, dst) -> None:
    """Mirror the QF_Settings values from one object onto another.

    The ``preset`` enum's update callback is suppressed: every covered value is
    copied explicitly here, so re-applying the preset would only risk clobbering
    edits the user made after picking it.
    """
    if src is None or dst is None:
        return
    try:
        props = src.bl_rna.properties
    except Exception:
        return
    try:
        from .properties import suppress_preset_update
    except Exception:  # pragma: no cover - defensive
        import contextlib
        suppress_preset_update = contextlib.nullcontext
    with suppress_preset_update():
        for p in props:
            if p.identifier in _SKIP_COPY or p.is_readonly:
                continue
            try:
                setattr(dst, p.identifier, getattr(src, p.identifier))
            except Exception:
                continue


def _fail(msg, report, t0, s=None):
    report["error"] = msg
    report["stats"] = {"time_s": round(time.perf_counter() - t0, 4)}
    if s is not None:
        try:
            s.last_report = json.dumps({"ok": False, "error": msg, **report})
        except Exception:
            pass
    return {"ok": False, "error": msg, "object": None, "stats": report["stats"],
            "report": report}


def _write_last_report(result_obj, s, report, ok: bool) -> None:
    payload = {"ok": ok}
    payload.update(report)
    try:
        text = json.dumps(payload, default=str)
    except Exception:
        text = json.dumps({"ok": ok, "error": "report not serialisable"})
    try:
        s.last_report = text
    except Exception:
        pass
    try:
        if result_obj is not None:
            result_obj.quadforge.last_report = text
    except Exception:
        pass
