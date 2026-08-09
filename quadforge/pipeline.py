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

from .core import analysis, guides

ORIGINALS_COLLECTION = "QuadForge Originals"
WORK_SUFFIX = "_qf_work"
RESULT_SUFFIX = "_quad"

_AXIS_NAMES = ("X", "Y", "Z")


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


def bisect_to_half(work_obj, axes, eps: float) -> bool:
    """Cut the mesh at every symmetry plane, keep the negative side, and snap
    the cut vertices exactly onto the planes. False if nothing survived."""
    mesh = work_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for ax in axes:
        no = Vector((0.0, 0.0, 0.0))
        no[ax] = 1.0
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        try:
            bmesh.ops.bisect_plane(
                bm, geom=geom, dist=eps,
                plane_co=(0.0, 0.0, 0.0), plane_no=no,
                use_snap_center=False, clear_outer=True, clear_inner=False,
            )
        except Exception:
            bm.free()
            return False
        if not bm.faces:
            bm.free()
            return False
        for v in bm.verts:
            if abs(v.co[ax]) <= eps:
                v.co[ax] = 0.0
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return len(mesh.polygons) > 0


def mirror_weld(work_obj, axes, snap_tol: float, weld_eps: float = 1e-7) -> None:
    """Snap the cut boundary exactly onto the planes, then mirror + weld so the
    result is bit-exact symmetric."""
    mesh = work_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    for ax in axes:
        for v in bm.verts:
            if v.is_boundary and abs(v.co[ax]) <= snap_tol:
                v.co[ax] = 0.0

    for ax in axes:
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

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


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
                  symmetry=None):
    """Call ``backend.remesh``, passing the extra QuadForge hints only if the
    backend accepts them (the contract signature is the 4-argument one)."""
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
    return backend.remesh(context, work_obj, s, int(target), **kwargs)


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
        stats = _call_backend(context, backend, work_obj, s, requested)
        report["backend_stats"] = stats if isinstance(stats, dict) else {}
        if post_pass is not None:
            post_pass(work_obj, face_target)
        return

    report["symmetry_mode"] = "exact"
    report["symmetry_axes"] = [_AXIS_NAMES[a] for a in axes]
    eps = max(_mean_edge_length(work_obj.data) * 1e-3, 1e-7)

    # keep a copy so we can fall back if bisecting destroys the mesh
    backup = bmesh.new()
    backup.from_mesh(work_obj.data)
    ok = bisect_to_half(work_obj, axes, eps)
    if not ok:
        backup.to_mesh(work_obj.data)
        backup.free()
        work_obj.data.update()
        report.setdefault("warnings", []).append(
            "exact symmetry: bisecting left no geometry, falling back to solver symmetry"
        )
        report["symmetry_mode"] = "solver"
        stats = _call_backend(context, backend, work_obj, s, requested)
        report["backend_stats"] = stats if isinstance(stats, dict) else {}
        if post_pass is not None:
            post_pass(work_obj, face_target)
        return
    backup.free()

    half_target = int(max(12, round(requested / float(2 ** len(axes)))))
    report["half_target"] = half_target
    stats = _call_backend(
        context, backend, work_obj, s, half_target,
        force_boundary=True, symmetry=(False, False, False),
    )
    report["backend_stats"] = stats if isinstance(stats, dict) else {}

    if post_pass is not None:
        post_pass(work_obj, int(max(12, round(face_target / float(2 ** len(axes))))))

    snap_tol = max(_mean_edge_length(work_obj.data) * 0.25, 1e-6)
    mirror_weld(work_obj, axes, snap_tol)


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

        work = make_work_object(context, obj)
        face_target = face_target_from_settings(obj, work.data, s)
        report["target_faces"] = face_target
        report["mode"] = getattr(s, "mode", "FACES")

        preprocess(context, work, s, report)

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

        run_backend(context, backend, work, s, face_target, report, post_pass=post_pass)

        if len(work.data.polygons) == 0:
            raise RuntimeError("the solver returned an empty mesh")

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
                    "material_boundary_edges", "symmetry_mode", "requested_faces"):
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
    """Mirror the QF_Settings values from one object onto another."""
    if src is None or dst is None:
        return
    try:
        props = src.bl_rna.properties
    except Exception:
        return
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
