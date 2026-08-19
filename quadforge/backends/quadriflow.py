"""QuadriFlow backend.

Thin, robust wrapper around ``bpy.ops.object.quadriflow_remesh`` that

* pre-cleans the input into something QuadriFlow accepts (it needs a manifold
  triangle-able mesh),
* drives symmetry through ``mesh.use_mirror_x/y/z`` (verified: that, together
  with ``use_mesh_symmetry=True``, is what the operator reads),
* optionally iterates to land within 10% of the requested face count.
"""

from __future__ import annotations

import json

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

    # T-junction edges (3+ faces on one edge, common where game-mesh card fans
    # share a spine) are repairable too: splitting detaches the fans into
    # separately-welded sheets without moving any geometry.
    bad_edges = [e for e in bm.edges if len(e.link_faces) > 2]
    if bad_edges:
        stats["overshared_edges_split"] = len(bad_edges)
        bmesh.ops.split_edges(bm, edges=bad_edges)
        _delete_loose(bm)
        split2 = _split_pinched_verts(bm)
        if split2:
            _delete_loose(bm)
            stats["pinched_verts_split"] = stats.get("pinched_verts_split", 0) + split2

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


def _op_kwargs(*, symmetry_on, preserve_sharp, preserve_boundary,
               preserve_attributes, smooth_normals, target_faces, seed,
               mesh_area=-1.0):
    return dict(
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


def _run_op(context, work_obj, **params):
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

    kwargs = _op_kwargs(**params)
    with context.temp_override(**_override(context, work_obj)):
        res = bpy.ops.object.quadriflow_remesh(**kwargs)
    if "CANCELLED" in res:
        raise BackendError(
            "QuadriFlow cancelled - the input is most likely non-manifold or "
            "too degenerate for the solver"
        )


def _separate_loose(context, work_obj):
    """Split work_obj into loose parts. Returns the list of part objects
    (work_obj itself is one of them)."""
    for o in context.view_layer.objects:
        try:
            o.select_set(False)
        except Exception:
            pass
    work_obj.select_set(True)
    context.view_layer.objects.active = work_obj
    before = set(bpy.data.objects)
    with context.temp_override(
        object=work_obj,
        active_object=work_obj,
        selected_objects=[work_obj],
        selected_editable_objects=[work_obj],
    ):
        bpy.ops.mesh.separate(type='LOOSE')
    parts = [work_obj] + [o for o in bpy.data.objects if o not in before]
    return parts


def _join_parts(context, work_obj, parts):
    """Join the part objects back into work_obj."""
    others = [p for p in parts if p is not work_obj]
    if not others:
        return
    for o in context.view_layer.objects:
        try:
            o.select_set(False)
        except Exception:
            pass
    for p in parts:
        p.select_set(True)
    context.view_layer.objects.active = work_obj
    with context.temp_override(
        object=work_obj,
        active_object=work_obj,
        selected_objects=list(parts),
        selected_editable_objects=list(parts),
    ):
        bpy.ops.object.join()


def _mesh_area(mesh) -> float:
    n = len(mesh.polygons)
    if not n:
        return 0.0
    areas = np.empty(n, dtype=np.float64)
    mesh.polygons.foreach_get("area", areas)
    return float(areas.sum())


def _fuse_stray_tris(mesh) -> int:
    """QuadriFlow emits a few triangles on small shells; fuse adjacent pairs
    into quads. Returns the number of triangles remaining."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    tris = [f for f in bm.faces if len(f.verts) == 3]
    if tris:
        bmesh.ops.join_triangles(
            bm, faces=tris,
            angle_face_threshold=3.15, angle_shape_threshold=3.15,
        )
        bm.to_mesh(mesh)
        mesh.update()
    left = sum(1 for f in bm.faces if len(f.verts) == 3)
    bm.free()
    return left


def _boundary_loops(mesh) -> int:
    """Number of connected boundary components (union-find over the vertices of
    edges carrying exactly one face). 0 means the mesh is watertight."""
    if mesh is None or len(mesh.polygons) == 0:
        return 0
    bm = bmesh.new()
    bm.from_mesh(mesh)
    parent = {}

    def find(x):
        while parent[x] is not x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in bm.edges:
        if len(e.link_faces) != 1:
            continue
        for v in e.verts:
            parent.setdefault(v, v)
        ra, rb = find(e.verts[0]), find(e.verts[1])
        if ra is not rb:
            parent[rb] = ra
    loops = len({find(v) for v in parent})
    bm.free()
    return loops


def _run_worker_round(part_objs, per_part_kwargs, timeout: float):
    """One child-Blender round over the given parts. Replaces the mesh of every
    part the worker finished (even when a later part stalled). Returns the list
    of parts that remain unsolved, or raises BackendError on infrastructure
    failure."""
    import os
    import subprocess
    import tempfile

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qf_worker.py")
    with tempfile.TemporaryDirectory(prefix="quadforge_") as td:
        in_blend = os.path.join(td, "in.blend")
        out_blend = os.path.join(td, "out.blend")
        bpy.data.libraries.write(in_blend, set(part_objs), fake_user=False)
        params = {
            "out": out_blend,
            "objects": {p.name: per_part_kwargs[p.name] for p in part_objs},
        }
        # params go via file: hundreds of parts overflow ARG_MAX as an argument
        params_path = os.path.join(td, "params.json")
        with open(params_path, "w") as fh:
            json.dump(params, fh)
        cmd = [
            bpy.app.binary_path, "--background", "--factory-startup", in_blend,
            "--python", worker, "--", params_path,
        ]
        stdout = ""
        timed_out = False
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            stdout = proc.stdout or ""
            if proc.returncode != 0 and "QF_WORKER_FINISHED" not in stdout:
                tail = (stdout + (proc.stderr or "")).strip()[-300:]
                raise BackendError(f"QuadriFlow worker failed: {tail}")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            out = exc.stdout
            stdout = out.decode(errors="replace") if isinstance(out, bytes) else (out or "")

        done = {}  # part object name -> solved mesh datablock name
        failed = set()  # parts the op deterministically refused this round
        for line in stdout.splitlines():
            if line.startswith("QF_PART_DONE"):
                fields = line.split()
                if len(fields) >= 3:
                    done[fields[1]] = fields[2]
            elif line.startswith("QF_PART_FAILED"):
                fields = line.split()
                if len(fields) >= 2:
                    failed.add(fields[1])
        if done and os.path.exists(out_blend):
            wanted = [done[p.name] for p in part_objs if p.name in done]
            with bpy.data.libraries.load(out_blend) as (src, dst):
                avail = set(src.meshes)
                request = [m for m in wanted if m in avail]
                request_names = list(request)  # load mutates the assigned list
                dst.meshes = request
            # dst.meshes preserves request order even when clashes rename them
            solved = {
                name: mesh for name, mesh in zip(request_names, dst.meshes)
                if mesh is not None
            }
            for part in part_objs:
                mesh_name = done.get(part.name)
                if mesh_name is None:
                    continue
                new_mesh = solved.get(mesh_name)
                if new_mesh is None:
                    done.pop(part.name, None)
                    continue
                old = part.data
                keep_name = old.name
                # QuadriFlow sometimes returns a part TORN: the operator
                # reports success but the quad mesh it hands back has holes the
                # input did not have. Two shapes of the same defect:
                #
                # * a shell far below its useful resolution (a 24-face hair
                #   card, a tooth, a button) comes back with open edges despite
                #   watertight input;
                # * a part that legitimately has a boundary — the bisected half
                #   the exact-symmetry path solves — comes back with an EXTRA
                #   boundary loop away from the cut (seen nondeterministically
                #   with many solver threads). The mirror then duplicates that
                #   hole on both sides, out of reach of the seam weld and the
                #   seam_open_edges count.
                #
                # The invariant covering both: the solver must never create new
                # boundary loops (for closed input, 0 loops in -> 0 loops out).
                # The cut ring may be resampled, but it stays one loop. A torn
                # part is treated exactly like a refusal, so the existing
                # escalation (jittered retry -> native rescue -> keep the
                # original topology) handles it.
                old_loops = _boundary_loops(old)
                part.data = new_mesh
                _fuse_stray_tris(new_mesh)
                if _boundary_loops(new_mesh) > old_loops:
                    part.data = old
                    try:
                        bpy.data.meshes.remove(new_mesh)
                    except Exception:
                        pass
                    done.pop(part.name, None)
                    failed.add(part.name)
                    continue
                try:
                    bpy.data.meshes.remove(old)
                except Exception:
                    pass
                new_mesh.name = keep_name

    return [p for p in part_objs if p.name not in done], failed


def _solve(context, work_obj, s, stats, **params):
    """One logical QuadriFlow solve.

    Isolation mode (default): the mesh is split into loose parts, each part is
    remeshed with an area-proportional face budget in a killable child Blender
    process, and the parts are joined back. A part whose solve stalls (rare
    upstream QuadriFlow non-convergence, chaotically input-dependent) is
    retried with rescale/target/seed jitter; after all retries only that part
    is left unremeshed and reported, instead of the whole mesh failing — and a
    stall can never freeze the session.
    """
    if not bool(getattr(s, "solver_isolation", True)):
        _run_op(context, work_obj, **params)
        return

    kwargs = _op_kwargs(**params)
    total_target = int(kwargs["target_faces"])

    parts = _separate_loose(context, work_obj)

    # Small separate shells (hair strands, feathers, piercings, teeth) are
    # usually hand-authored and far below the solver's useful resolution —
    # remeshing turns them into blobs. Keep them verbatim unless disabled.
    preserved = []
    if bool(getattr(s, "preserve_small_shells", True)) and len(parts) > 1:
        limit = int(getattr(s, "small_shell_limit", 0) or 0)
        if limit <= 0:
            total_in = sum(len(p.data.polygons) for p in parts)
            limit = max(64, int(0.02 * total_in))
        biggest = max(parts, key=lambda p: len(p.data.polygons))
        preserved = [p for p in parts
                     if p is not biggest and len(p.data.polygons) < limit]
        if preserved:
            stats["preserved_shells"] = len(preserved)
            stats["preserved_shell_faces"] = sum(
                len(p.data.polygons) for p in preserved)
    solve_parts = [p for p in parts if p not in preserved]

    # per-part autoscale: a tiny shell (whiskers, claws) can sit far below
    # QuadriFlow's absolute edge epsilon even after the global autoscale
    base_scale = {}
    for p in solve_parts:
        f = _autoscale_factor(p.data)
        if f != 1.0:
            _scale_mesh(p.data, f)
        base_scale[p.name] = f
    big_part = max(solve_parts, key=lambda p: len(p.data.polygons)) if solve_parts else None
    big_backup = (big_part.data.copy()
                  if big_part is not None and len(solve_parts) > 1 else None)
    try:
        areas = {p.name: _mesh_area(p.data) for p in solve_parts}
        total_area = sum(areas.values()) or 1.0
        per_part = {}
        for p in solve_parts:
            k = dict(kwargs)
            # below ~24 faces QuadriFlow cancels instead of solving
            k["target_faces"] = int(max(24, round(total_target * areas[p.name] / total_area)))
            per_part[p.name] = k
        floor_sum = sum(k["target_faces"] for k in per_part.values())
        if floor_sum > 1.3 * total_target:
            stats["floor_overshoot"] = floor_sum
            stats.setdefault("warnings", []).append(
                "target %d is below what %d separate shells can express; "
                "expect roughly %d+ faces" % (total_target, len(parts), floor_sum)
            )

        # (mesh rescale, target jitter, seed jitter) — each retry round changes
        # the solver's discretization completely, the most reliable stall escape
        plans = ((1.0, 1.0, 0), (1.31, 1.09, 977), (0.77, 0.93, 3251), (1.73, 1.17, 7919))
        remaining = list(solve_parts)
        fail_counts = {}
        gave_up = []
        for round_i, (rescale, t_jitter, s_jitter) in enumerate(plans):
            if not remaining:
                break
            n_faces = sum(len(p.data.polygons) for p in remaining)
            timeout = 60.0 + n_faces / 1000.0 + 0.3 * len(remaining)
            round_kwargs = {}
            scaled = list(remaining)
            for p in scaled:
                k = dict(per_part[p.name])
                k["target_faces"] = int(round(k["target_faces"] * t_jitter)) + (1 if round_i else 0)
                k["seed"] = int(k["seed"]) + s_jitter
                round_kwargs[p.name] = k
                if rescale != 1.0:
                    _scale_mesh(p.data, rescale)
            try:
                remaining, failed = _run_worker_round(scaled, round_kwargs, timeout)
                for name in failed:
                    fail_counts[name] = fail_counts.get(name, 0) + 1
                # a part the op refuses twice is degenerate for QuadriFlow —
                # keep its original geometry instead of burning more rounds
                gave_up.extend(p for p in remaining if fail_counts.get(p.name, 0) >= 2)
                remaining = [p for p in remaining if fail_counts.get(p.name, 0) < 2]
            except Exception as exc:
                if rescale != 1.0:
                    for p in scaled:
                        _scale_mesh(p.data, 1.0 / rescale)
                if isinstance(exc, BackendError):
                    raise
                # isolation infrastructure failed (no binary, sandbox, ...):
                # solve in-process rather than not at all, keeping originals
                # for parts the operator refuses
                stats["isolation_fallback"] = str(exc)[:120]
                kept = 0
                for p in remaining:
                    try:
                        _run_op(context, p, **params)
                    except Exception:
                        kept += 1
                if kept:
                    stats["unsolvable_parts"] = stats.get("unsolvable_parts", 0) + kept
                remaining = []
                break
            if rescale != 1.0:
                # both solved (mesh swapped, still at worker scale) and
                # unsolved parts of this round need unscaling
                for p in scaled:
                    _scale_mesh(p.data, 1.0 / rescale)
            if round_i and not remaining:
                stats["stall_retries"] = round_i

        # Shells QuadriFlow refuses (degenerate card stacks) get a second
        # chance with the native field solver before falling back to their
        # original topology — a remeshed card blends in, original tris don't.
        if gave_up:
            native = None
            try:
                from . import native as _native
                if hasattr(_native, "remesh"):
                    native = _native
            except Exception:
                native = None
            if native is not None:
                rescued = []
                for p in gave_up:
                    try:
                        native.remesh(context, p, s, per_part[p.name]["target_faces"])
                        _fuse_stray_tris(p.data)
                        rescued.append(p)
                    except Exception:
                        pass
                if rescued:
                    stats["native_rescued_parts"] = len(rescued)
                    gave_up = [p for p in gave_up if p not in rescued]

        if gave_up:
            stats["unsolvable_parts"] = len(gave_up)
            stats["unsolvable_part_faces"] = sum(len(p.data.polygons) for p in gave_up)
        if remaining:
            stats["stalled_parts"] = len(remaining)
            stats["stalled_part_faces"] = sum(len(p.data.polygons) for p in remaining)

        # Tiny shells can't go below QuadriFlow's per-part floor, so many-part
        # meshes overshoot the total; compensate by re-solving the biggest part
        # (from its original geometry) with the excess subtracted.
        #
        # The main body pays that bill, so it needs a floor of its own. With
        # Keep Small Shells off every hair card and tooth reaches the solver and
        # burns the 24-face per-part minimum, and the excess this produces could
        # cut the body's budget by more than half (measured on the plate
        # fixture: 565 -> 278), which both guts the silhouette and pushes the
        # solve into the coarse regime where QuadriFlow returns torn geometry.
        # Overshooting the requested count is the better trade - and it is
        # already what the floor_overshoot warning above promises the user.
        if not remaining and big_backup is not None:
            actual = sum(len(p.data.polygons) for p in parts)
            excess = actual - total_target
            big_share = per_part[big_part.name]["target_faces"]
            rebal_target = big_share - excess
            if excess > max(0.08 * total_target, 40) and \
                    rebal_target >= max(50, 0.5 * big_share):
                solved_mesh = big_part.data
                big_part.data = big_backup
                big_backup = None
                k = dict(per_part[big_part.name])
                k["target_faces"] = int(rebal_target)
                timeout = 60.0 + len(big_part.data.polygons) / 1000.0
                left, _ = _run_worker_round([big_part], {big_part.name: k}, timeout)
                if left:
                    orig = big_part.data
                    big_part.data = solved_mesh  # keep the first solve
                    try:
                        bpy.data.meshes.remove(orig)
                    except Exception:
                        pass
                else:
                    stats["rebalanced"] = int(rebal_target)
                    try:
                        bpy.data.meshes.remove(solved_mesh)
                    except Exception:
                        pass
    finally:
        if big_backup is not None:
            try:
                bpy.data.meshes.remove(big_backup)
            except Exception:
                pass
        for p in solve_parts:
            f = base_scale.get(p.name, 1.0)
            if f != 1.0:
                _scale_mesh(p.data, 1.0 / f)
        _join_parts(context, work_obj, parts)


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

    # QuadriFlow needs resolution headroom: when the target approaches the
    # input triangle count its refine/collapse cycle can stall forever.
    # Midpoint-subdivide (surface preserving) until the target is well below
    # the input density.
    subdiv_rounds = 0
    while subdiv_rounds < 2:
        tris = sum(max(0, len(p.vertices) - 2) for p in mesh.polygons)
        if face_target <= 0.4 * tris or tris > 400000:
            break
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1,
                                  use_grid_fill=True, smooth=0.0)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        subdiv_rounds += 1
    if subdiv_rounds:
        stats["headroom_subdiv"] = subdiv_rounds

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
                # the isolated solver may have swapped the mesh datablock
                mesh = work_obj.data
                backup.to_mesh(mesh)
                mesh.update()
                _apply_symmetry_flags(mesh, symmetry)
            stats["attempts"] = attempt + 1
            _solve(
                context, work_obj, s, stats,
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
