"""Regressions for exact-symmetry seam debris.

Two defects lived here, both invisible to ``mesh.validate()``:

* the post-solve trim flattened every vertex within 0.2 edge lengths of the
  symmetry plane onto it, burying interior edges and whole faces *inside* the
  plane. ``bmesh.ops.mirror`` then folded that geometry onto its own reflection
  (four-face edges, zero-thickness double flaps) and the weld tore the seam
  open again — by a count that drifted run to run, because whether a given
  vertex fell inside the flatten band depended on QuadriFlow's output, which is
  not reproducible even for identical input.
* small shells are split aside before the bisect and rejoined verbatim after
  the mirror, so they were the one part of the mesh that never reached the
  backend's preclean. Authored coincident vertices — and the zero-length edges
  between them — were copied straight into the result and read as seam damage.
"""

import bmesh
import bpy


def _plane_stats(obj, axis=0, eps=1e-6):
    """(faces lying in the plane, non-manifold edges, open edges on the plane)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    on = lambda v: abs(v.co[axis]) <= eps  # noqa: E731
    flat = sum(1 for f in bm.faces if all(on(v) for v in f.verts))
    nonman = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    open_on = sum(1 for e in bm.edges
                  if len(e.link_faces) == 1 and on(e.verts[0]) and on(e.verts[1]))
    bm.free()
    return flat, nonman, open_on


def _zero_length_edges(obj, tol=1e-6):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n = sum(1 for e in bm.edges if e.calc_length() < tol)
    bm.free()
    return n


def _shell_with_degenerate_edges(ctx):
    """Big sphere + a small shell carrying authored degeneracy: the vertex ring
    below the pole is collapsed onto the pole without being merged, so its edges
    are zero-length and the tip triangles are zero-area. Real avatar hair cards
    and eyelash shells ship exactly this."""
    big = ctx.uv_sphere(segments=32, rings=16)
    small = ctx.uv_sphere(segments=12, rings=6, radius=0.25, name="Debris")
    small.location = (0.0, 0.0, 1.4)

    bm = bmesh.new()
    bm.from_mesh(small.data)
    top = max(v.co.z for v in bm.verts)
    below = max(v.co.z for v in bm.verts if v.co.z < top - 1e-6)
    for v in bm.verts:
        if abs(v.co.z - below) < 1e-6:
            v.co = (0.0, 0.0, top)
    bm.to_mesh(small.data)
    bm.free()
    small.data.update()

    bpy.ops.object.select_all(action='DESELECT')
    big.select_set(True)
    small.select_set(True)
    bpy.context.view_layer.objects.active = big
    bpy.ops.object.join()
    return big


def run(ctx):
    r = ctx.results()

    from quadforge import pipeline

    with r.case("mirror_weld_survives_inplane_interior") as c:
        # Reproduce the flatten band the post-solve trim used to apply: press
        # every vertex within half an edge length of the plane onto it, which
        # buries interior edges and whole faces inside the plane. mirror_weld
        # has to hand back a watertight, manifold mesh anyway.
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=32, rings=16)
        mean_edge = pipeline._mean_edge_length(obj.data)
        c.require(pipeline.bisect_to_half(obj, [0], mean_edge * 1e-3),
                  "bisect left no geometry")

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        for v in bm.verts:
            if abs(v.co.x) < 0.5 * mean_edge:
                v.co.x = 0.0
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()

        buried_faces, _nm, _oo = _plane_stats(obj)
        c.require(buried_faces > 0,
                  "fixture did not actually bury faces in the plane")

        leftover = pipeline.mirror_weld(obj, [0], mean_edge * 0.25)
        flat, nonman, open_on = _plane_stats(obj)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
        bm.free()
        c.require(nonman == 0,
                  "%d edge(s) ended up with more than two faces" % nonman)
        c.require(flat == 0,
                  "%d face(s) still lie inside the symmetry plane" % flat)
        c.require(open_on == 0,
                  "%d open edge(s) left lying on the seam" % open_on)
        c.require(boundary == 0,
                  "closed input came back with %d boundary edges" % boundary)
        c.require(leftover == 0, "mirror_weld reported %d leftovers" % leftover)
        c.note("buried=%d faces=%d" % (buried_faces, len(obj.data.polygons)))

    with r.case("preserved_shells_carry_no_zero_length_edges") as c:
        ctx.fresh_scene()
        obj = _shell_with_degenerate_edges(ctx)
        before = _zero_length_edges(obj)
        c.require(before > 0, "fixture should carry authored zero-length edges")
        c.note("fixture zero-length edges=%d" % before)
        s = ctx.settings(obj, target_count=1500, symmetry_x=True,
                         exact_symmetry=True, preserve_small_shells=True,
                         small_shell_limit=2000, keep_original=False)
        res = pipeline.run_remesh(bpy.context, obj, s)
        c.require(res.get("ok"), "run failed: %r" % (res.get("error"),))
        out = res["object"]
        zero = _zero_length_edges(out)
        c.require(zero == 0,
                  "%d zero-length edge(s) survived into the result" % zero)
        _flat, _nm, open_on = _plane_stats(out)
        c.require(open_on == 0, "%d open edge(s) left on the seam" % open_on)
        c.note("faces=%d" % len(out.data.polygons))

    with r.case("seam_stays_closed_across_repeats") as c:
        # QuadriFlow is not reproducible run to run, so a seam that only closes
        # for some solver outputs shows up here as a flapping count.
        counts = []
        for _ in range(3):
            ctx.fresh_scene()
            big = ctx.uv_sphere(segments=32, rings=16)
            small = ctx.uv_sphere(segments=16, rings=8, radius=0.3, name="Small")
            small.location = (0.0, 0.0, 1.15)
            bpy.ops.object.select_all(action='DESELECT')
            big.select_set(True)
            small.select_set(True)
            bpy.context.view_layer.objects.active = big
            bpy.ops.object.join()
            obj = big
            s = ctx.settings(obj, target_count=1200, symmetry_x=True,
                             exact_symmetry=True, keep_original=False)
            res = pipeline.run_remesh(bpy.context, obj, s)
            c.require(res.get("ok"), "run failed: %r" % (res.get("error"),))
            out = res["object"]
            bm = bmesh.new()
            bm.from_mesh(out.data)
            counts.append(sum(1 for e in bm.edges if len(e.link_faces) == 1))
            bm.free()
            bpy.data.objects.remove(out, do_unlink=True)
        c.require(counts == [0, 0, 0],
                  "closed input produced boundary edges %r across repeats" % counts)
        c.note("boundary counts=%r" % counts)

    return r.list()
