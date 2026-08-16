"""Regressions for the exact-symmetry seam and shell-orientation repairs.

Real-world trigger: multi-shell game meshes (dozens of loose parts, thin
walls). The bisect+mirror path used to leave holes along the seam, and
recalc_face_normals' outward heuristic flipped whole small shells relative
to the source.
"""

import bmesh
import bpy


def _boundary_edges(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    bm.free()
    return n


def _shell_orientations(obj):
    """[(center, mean outward dot)] per connected shell."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    seen = set()
    shells = []
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
        import mathutils
        center = mathutils.Vector((0, 0, 0))
        for f in comp:
            center += f.calc_center_median()
        center /= len(comp)
        score = 0.0
        for f in comp:
            d = f.calc_center_median() - center
            if d.length > 1e-9:
                score += f.normal.dot(d.normalized())
        shells.append((center, score / len(comp)))
    bm.free()
    return shells


def _two_shell_object(ctx, invert_small=False):
    """One big sphere + one small sphere crossing the X plane, single object."""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
    big = bpy.context.active_object
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.3,
                                         location=(0.0, 0.0, 1.15))
    small = bpy.context.active_object
    if invert_small:
        bm = bmesh.new()
        bm.from_mesh(small.data)
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
        bm.to_mesh(small.data)
        bm.free()
    bpy.ops.object.select_all(action='DESELECT')
    big.select_set(True)
    small.select_set(True)
    bpy.context.view_layer.objects.active = big
    bpy.ops.object.join()
    return big


def run(ctx):
    r = ctx.results()

    from quadforge import pipeline

    with r.case("closed_multishell_stays_closed") as c:
        ctx.fresh_scene()
        obj = _two_shell_object(ctx)
        s = ctx.settings(obj, target_count=1200, symmetry_x=True,
                         exact_symmetry=True, keep_original=False)
        res = pipeline.run_remesh(bpy.context, obj, s)
        c.require(res.get("ok"), "run failed: %r" % (res.get("error"),))
        out = res["object"]
        holes = _boundary_edges(out)
        c.require(holes == 0,
                  "closed 2-shell input came back with %d boundary edges" % holes)
        seam = (res.get("stats") or {}).get("seam_open_edges")
        c.note("faces=%d seam_open=%s" % (len(out.data.polygons), seam))

    with r.case("shell_orientation_follows_source") as c:
        ctx.fresh_scene()
        obj = _two_shell_object(ctx, invert_small=True)
        src_shells = sorted(_shell_orientations(obj), key=lambda t: -abs(t[1]))
        s = ctx.settings(obj, target_count=1200, symmetry_x=True,
                         exact_symmetry=True, keep_original=False)
        res = pipeline.run_remesh(bpy.context, obj, s)
        c.require(res.get("ok"), "run failed: %r" % (res.get("error"),))
        shells = _shell_orientations(res["object"])
        c.require(len(shells) >= 2, "expected 2 shells, got %d" % len(shells))
        big = max(shells, key=lambda t: abs(t[1]) if t[0].length < 0.5 else 0)
        small = max(shells, key=lambda t: abs(t[1]) if t[0].z > 0.8 else 0)
        c.require(big[1] > 0.3,
                  "outer shell should stay outward (score %.2f)" % big[1])
        c.require(small[1] < -0.3,
                  "inverted source shell should stay inverted (score %.2f)" % small[1])
        c.note("big=%.2f small=%.2f" % (big[1], small[1]))

    with r.case("in_process_solver_still_works") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere()
        s = ctx.settings(obj, target_count=600, solver_isolation=False,
                         keep_original=False)
        res = pipeline.run_remesh(bpy.context, obj, s)
        c.require(res.get("ok"), "run failed: %r" % (res.get("error"),))
        c.require(_boundary_edges(res["object"]) == 0, "sphere came back open")

    return r.list()
