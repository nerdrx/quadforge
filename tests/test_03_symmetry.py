"""Exact symmetry: bisect + remesh half + mirror weld."""

import bpy

TOL = 1e-5
TARGET = 2000
AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def _run_axis(ctx, axis):
    ctx.fresh_scene()
    obj = ctx.symmetric_blob()
    ctx.activate(obj)
    kw = {"symmetry_x": False, "symmetry_y": False, "symmetry_z": False,
          "symmetry_" + axis: True}
    s = ctx.settings(obj, mode='FACES', target_count=TARGET,
                     backend='QUADRIFLOW', exact_symmetry=True, **kw)
    res = ctx.pipeline().run_remesh(bpy.context, obj, s)
    return obj, res


def run(ctx):
    r = ctx.results()

    with r.case("fixture_is_symmetric") as c:
        ctx.fresh_scene()
        blob = ctx.symmetric_blob()
        ex = ctx.symmetry_error(blob, 0)
        ey = ctx.symmetry_error(blob, 1)
        # float32 vertex storage puts a ~1e-6 floor under any "symmetric" mesh;
        # stay comfortably under the 1e-5 the exact-symmetry pass must hit.
        c.require(ex < 5e-6, "fixture X asymmetry %.3g" % ex)
        c.require(ey < 5e-6, "fixture Y asymmetry %.3g" % ey)
        c.note("verts=%d ex=%.2g ey=%.2g" % (len(blob.data.vertices), ex, ey))

    for axis in ("x", "y"):
        with r.case("exact_symmetry_" + axis) as c:
            src, res = _run_axis(ctx, axis)
            c.require(res.get("ok") is True,
                      "run_remesh failed: %r" % (res.get("error"),))
            out = res.get("object")
            c.require(ctx.is_mesh_valid(out), "no result mesh")
            err = ctx.symmetry_error(out, AXIS_INDEX[axis])
            c.require(err < TOL,
                      "max mirror mismatch on %s is %.3g (limit %.0e) over %d verts"
                      % (axis.upper(), err, TOL, len(out.data.vertices)))
            fs = ctx.face_stats(out)
            c.require(fs["faces"] > 0, "result has no faces")
            c.require(fs["quad_pct"] >= 95.0,
                      "symmetric result is only %.1f%% quads" % fs["quad_pct"])
            c.note("err=%.2g faces=%d quad_pct=%.1f%%"
                   % (err, fs["faces"], fs["quad_pct"]))

    with r.case("symmetry_reported") as c:
        src, res = _run_axis(ctx, "x")
        c.require(res.get("ok") is True, "run failed: %r" % (res.get("error"),))
        stats = res.get("stats") or {}
        key = None
        for k in ("symmetry_error_x", "symmetry_x_error", "sym_error_x"):
            if k in stats:
                key = k
                break
        if key is None:
            c.skip("stats carries no symmetry_error_x key (stats: %s)" % sorted(stats))
        c.require(float(stats[key]) < TOL,
                  "%s reported as %s" % (key, stats[key]))
        c.note("%s=%s" % (key, stats[key]))

    with r.case("no_symmetry_still_works") as c:
        ctx.fresh_scene()
        obj = ctx.symmetric_blob()
        ctx.activate(obj)
        s = ctx.settings(obj, target_count=TARGET, backend='QUADRIFLOW',
                         symmetry_x=False, symmetry_y=False, symmetry_z=False)
        res = ctx.pipeline().run_remesh(bpy.context, obj, s)
        c.require(res.get("ok") is True, "run failed: %r" % (res.get("error"),))
        c.require(ctx.is_mesh_valid(res.get("object")), "no result mesh")
        c.note("faces=%d" % len(res["object"].data.polygons))

    return r.list()
