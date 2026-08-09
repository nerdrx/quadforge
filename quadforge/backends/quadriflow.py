"""QuadriFlow backend.

Thin, robust wrapper around ``bpy.ops.object.quadriflow_remesh`` that

* pre-cleans the input into something QuadriFlow accepts (it needs a manifold
  triangle-able mesh),
* drives symmetry through ``mesh.use_mirror_x/y/z`` (verified: that, together
  with ``use_mesh_symmetry=True``, is what the operator reads),
* optionally iterates to land within 10% of the requested face count.
"""

from __future__ import annotations

import bmesh
import bpy
import numpy as np

MERGE_EPS = 1e-6
STRICT_TOL = 0.10
STRICT_MAX_RETRIES = 3
# QuadriFlow refuses meshes whose edges fall under an internal absolute epsilon
# (observed on real assets: refused at 1st-percentile edge length 1.8e-3, fine
# at 1.8e-2). Working meshes are scaled so the 1st percentile clears this.
QF_MIN_P1_EDGE = 0.02


class BackendError(RuntimeError):
    """Raised with a user readable explanation of why the mesh can't be remeshed."""


# ---------------------------------------------------------------------------
# pre-clean / validation
# ---------------------------------------------------------------------------


def _delete_loose(bm):
    """Drop wire edges and free-floating vertices. Returns (edges, verts)."""
    loose_edges = [e for e in bm.edges if not e.link_faces]
    if loose_edges:
        bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
    loose_verts = [v for v in bm.verts if not v.link_faces]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
    if loose_edges or loose_verts:
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
    return len(loose_edges), len(loose_verts)


def _split_pinched_verts(bm) -> int:
    """Repair bow-tie / pinched vertices.

    A pinched vertex is one where two or more face fans meet at a single point
    (Blender's own Suzanne has two, and QuadriFlow output often has a few).
    Each extra fan gets its own coincident copy of the vertex, which makes the
    mesh manifold without changing a single coordinate.
    """
    fixed = 0
    bad = [v for v in bm.verts if not v.is_manifold and not v.is_boundary]
    if not bad:
        return 0

    dead_faces = []
    for v in bad:
        faces = list(v.link_faces)
        if len(faces) < 2:
            continue
        # group the incident faces into fans connected through edges at v
        edges = set(v.link_edges)
        parent = {f: f for f in faces}

        def find(x):
            while parent[x] is not x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in edges:
            linked = [f for f in e.link_faces if f in parent]
            for i in range(1, len(linked)):
                ra, rb = find(linked[0]), find(linked[i])
                if ra is not rb:
                    parent[rb] = ra

        fans = {}
        for f in faces:
            fans.setdefault(find(f), []).append(f)
        if len(fans) < 2:
            continue

        for fan in list(fans.values())[1:]:
            nv = bm.verts.new(v.co)
            nv.normal = v.normal
            for f in fan:
                new_verts = [nv if x is v else x for x in f.verts]
                try:
                    nf = bm.faces.new(new_verts, f)
                except ValueError:
                    continue
                nf.material_index = f.material_index
                nf.smooth = f.smooth
                dead_faces.append(f)
        fixed += 1

    if dead_faces:
        bmesh.ops.delete(bm, geom=dead_faces, context="FACES_ONLY")
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
    return fixed


def preclean(work_obj, merge_eps: float = MERGE_EPS) -> dict:
    """Make the mesh QuadriFlow-safe. Raises BackendError when it can't be fixed.

    Returns a stats dict describing what was changed.
    """
    mesh = work_obj.data
    stats = {
        "merged_verts": 0,
        "loose_verts": 0,
        "loose_edges": 0,
        "ngons_triangulated": 0,
        "boundary_edges": 0,
    }
    if mesh is None or len(mesh.polygons) == 0:
        raise BackendError("input mesh has no faces")

    bm = bmesh.new()
    bm.from_mesh(mesh)

    v_before = len(bm.verts)
    try:
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=merge_eps)
    except Exception:
        pass
    stats["merged_verts"] = v_before - len(bm.verts)

    ne, nv = _delete_loose(bm)
    stats["loose_edges"] = ne
    stats["loose_verts"] = nv

    if not bm.faces:
        bm.free()
        raise BackendError("input mesh has no faces after removing loose geometry")

    ngons = [f for f in bm.faces if len(f.verts) > 4]
    if ngons:
        stats["ngons_triangulated"] = len(ngons)
        bmesh.ops.triangulate(bm, faces=ngons, quad_method="BEAUTY", ngon_method="BEAUTY")

    # zero-area faces confuse the solver
    degenerate = [f for f in bm.faces if f.calc_area() <= 1e-14]
    if degenerate:
        bmesh.ops.delete(bm, geom=degenerate, context="FACES_ONLY")
        _delete_loose(bm)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    # manifold validation - pinched vertices are repairable, T-junction edges
    # (three or more faces on one edge) are not. Splitting orphans the old
    # edges, so loose geometry has to go once more afterwards.
    split = _split_pinched_verts(bm)
    if split:
        _delete_loose(bm)
        split += _split_pinched_verts(bm)
        if split:
            _delete_loose(bm)
    stats["pinched_verts_split"] = split

    bad_edges = [e for e in bm.edges if len(e.link_faces) > 2]
    boundary = [e for e in bm.edges if len(e.link_faces) == 1]
    stats["boundary_edges"] = len(boundary)
    bad_verts = [v for v in bm.verts if not v.is_manifold and not v.is_boundary]

    if bad_edges or bad_verts:
        bm.free()
        parts = []
        if bad_edges:
            parts.append(f"{len(bad_edges)} edge(s) shared by more than two faces")
        if bad_verts:
            parts.append(f"{len(bad_verts)} non-manifold vertex/vertices (bow-ties or pinched fans)")
        raise BackendError(
            "QuadriFlow needs a manifold mesh - found " + " and ".join(parts) +
            ". Fix with Mesh > Clean Up > Merge by Distance / Select > All by Trait > "
            "Non Manifold, or use the Native backend."
        )

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return stats


# ---------------------------------------------------------------------------
# operator invocation
# ---------------------------------------------------------------------------


def _autoscale_factor(mesh) -> float:
    """Factor to scale the mesh up by so QuadriFlow's absolute edge-length
    epsilon can't reject it. 1.0 when no scaling is needed."""
    if not len(mesh.edges) or not len(mesh.vertices):
        return 1.0
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    ev = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    lengths = np.linalg.norm(co[ev[:, 0]] - co[ev[:, 1]], axis=1)
    lengths = lengths[lengths > 0.0]
    if lengths.size == 0:
        return 1.0
    p1 = float(np.percentile(lengths, 1))
    if p1 >= QF_MIN_P1_EDGE:
        return 1.0
    return float(min(QF_MIN_P1_EDGE / max(p1, 1e-12), 1e6))


def _scale_mesh(mesh, factor: float) -> None:
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    mesh.vertices.foreach_set("co", co * factor)
    mesh.update()


def _apply_symmetry_flags(mesh, symmetry) -> bool:
    sx, sy, sz = symmetry
    mesh.use_mirror_x = bool(sx)
    mesh.use_mirror_y = bool(sy)
    mesh.use_mirror_z = bool(sz)
    return bool(sx or sy or sz)


def _override(context, work_obj):
    ov = {
        "object": work_obj,
        "active_object": work_obj,
        "selected_objects": [work_obj],
        "selected_editable_objects": [work_obj],
    }
    scene = getattr(context, "scene", None)
    if scene is not None:
        ov["scene"] = scene
    vl = getattr(context, "view_layer", None)
    if vl is not None:
        ov["view_layer"] = vl
    return ov


def _run_op(context, work_obj, *, symmetry_on, preserve_sharp, preserve_boundary,
            preserve_attributes, smooth_normals, target_faces, seed, mesh_area=-1.0):
    view_layer = getattr(context, "view_layer", None)
    if view_layer is not None:
        try:
            view_layer.objects.active = work_obj
        except Exception:
            pass
    try:
        work_obj.select_set(True)
    except Exception:
        pass

    kwargs = dict(
        use_mesh_symmetry=bool(symmetry_on),
        use_preserve_sharp=bool(preserve_sharp),
        use_preserve_boundary=bool(preserve_boundary),
        preserve_attributes=bool(preserve_attributes),
        smooth_normals=bool(smooth_normals),
        mode="FACES",
        target_faces=int(max(4, target_faces)),
        mesh_area=float(mesh_area),
        seed=int(seed),
    )
    with context.temp_override(**_override(context, work_obj)):
        res = bpy.ops.object.quadriflow_remesh(**kwargs)
    if "CANCELLED" in res:
        raise BackendError(
            "QuadriFlow cancelled - the input is most likely non-manifold or "
            "too degenerate for the solver"
        )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def remesh(context, work_obj, s, face_target: int,
           force_preserve_boundary: bool = False,
           symmetry=None,
           skip_preclean: bool = False) -> dict:
    """Remesh ``work_obj`` in place. Returns a stats dict.

    ``symmetry`` overrides ``s.symmetry_x/y/z`` (used by the exact-symmetry path
    which has already bisected the mesh and must not mirror again).
    """
    stats = {"attempts": 0, "requested": int(face_target), "preclean": {}}

    if not skip_preclean:
        stats["preclean"] = preclean(work_obj)

    mesh = work_obj.data
    if symmetry is None:
        symmetry = (
            bool(getattr(s, "symmetry_x", False)),
            bool(getattr(s, "symmetry_y", False)),
            bool(getattr(s, "symmetry_z", False)),
        )
    symmetry_on = _apply_symmetry_flags(mesh, symmetry)

    preserve_sharp = bool(
        getattr(s, "detect_hard_edges", True)
        or getattr(s, "use_marked_sharp", False)
        or getattr(s, "use_materials", False)
        or getattr(s, "use_guides", False)
    )
    preserve_boundary = bool(force_preserve_boundary or getattr(s, "preserve_boundaries", True))
    preserve_attributes = bool(
        getattr(s, "preserve_uvs", True)
        or getattr(s, "preserve_weights", True)
        or getattr(s, "preserve_materials", True)
    )
    seed = int(getattr(s, "seed", 0))
    strict = bool(getattr(s, "strict_count", False))

    autoscale = _autoscale_factor(mesh)
    if autoscale != 1.0:
        _scale_mesh(mesh, autoscale)
    stats["autoscale"] = autoscale

    # snapshot the input so strict-count retries always start from the source
    backup = None
    if strict:
        backup = bmesh.new()
        backup.from_mesh(mesh)

    target = int(max(4, face_target))
    actual = 0
    try:
        for attempt in range(STRICT_MAX_RETRIES + 1):
            if attempt > 0 and backup is not None:
                backup.to_mesh(mesh)
                mesh.update()
                _apply_symmetry_flags(mesh, symmetry)
            stats["attempts"] = attempt + 1
            _run_op(
                context, work_obj,
                symmetry_on=symmetry_on,
                preserve_sharp=preserve_sharp,
                preserve_boundary=preserve_boundary,
                preserve_attributes=preserve_attributes,
                smooth_normals=False,
                target_faces=target,
                seed=seed,
            )
            actual = len(work_obj.data.polygons)
            if not strict or actual == 0 or face_target <= 0:
                break
            err = abs(actual - face_target) / float(face_target)
            if err <= STRICT_TOL:
                break
            if attempt == STRICT_MAX_RETRIES:
                break
            scale = float(face_target) / float(actual)
            scale = min(max(scale, 0.25), 4.0)
            new_target = int(round(target * scale))
            if new_target == target:
                new_target = target + (1 if scale > 1.0 else -1) * max(1, target // 20)
            target = max(4, new_target)
    finally:
        if backup is not None:
            backup.free()
        if autoscale != 1.0:
            _scale_mesh(work_obj.data, 1.0 / autoscale)

    stats["final_target"] = target
    stats["faces"] = actual or len(work_obj.data.polygons)
    return stats
