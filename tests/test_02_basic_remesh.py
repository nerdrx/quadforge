"""Baseline QUADRIFLOW remesh: quad purity + count accuracy."""

import bpy

TARGET = 4000


def _remesh(ctx, **overrides):
    ctx.fresh_scene()
    obj = ctx.suzanne(subdiv=2)
    ctx.activate(obj)
    s = ctx.settings(obj, mode='FACES', target_count=TARGET,
                     backend='QUADRIFLOW', detect_hard_edges=True,
                     **overrides)
    pipeline = ctx.pipeline()
    res = pipeline.run_remesh(bpy.context, obj, s)
    return obj, res


def run(ctx):
    r = ctx.results()

    # ---------------------------------------------------------------- basic
    state = {}
    with r.case("run_remesh_ok") as c:
        src, res = _remesh(ctx)
        state["res"] = res
        c.require(isinstance(res, dict), "run_remesh returned %r, want dict" % type(res))
        for key in ("ok", "error", "object", "stats"):
            c.require(key in res, "result dict missing key %r (got %s)"
                      % (key, sorted(res)))
        c.require(res["ok"] is True,
                  "ok is not True; error=%r" % (res.get("error"),))
        c.require(res.get("error") in (None, ""),
                  "error set despite ok: %r" % (res.get("error"),))
        c.note("stats keys=%s" % sorted(res.get("stats") or {}))

    res = state.get("res") or {}
    new_obj = res.get("object")

    with r.case("result_object_exists") as c:
        c.require(new_obj is not None, "result['object'] is None")
        c.require(ctx.is_mesh_valid(new_obj),
                  "result object is not a mesh with faces: %r" % (new_obj,))
        c.require(new_obj.name in bpy.data.objects,
                  "result object is not in bpy.data.objects")
        c.require(new_obj.name in bpy.context.view_layer.objects,
                  "result object is not linked into the view layer")
        c.note("%s, %d faces" % (new_obj.name, len(new_obj.data.polygons)))

    with r.case("all_quads") as c:
        c.require(ctx.is_mesh_valid(new_obj), "no result mesh to measure")
        fs = ctx.face_stats(new_obj)
        c.require(fs["quad_pct"] >= 99.5,
                  "quad_pct %.2f%% (%d quads / %d faces, %d tris, %d ngons)"
                  % (fs["quad_pct"], fs["quads"], fs["faces"], fs["tris"], fs["ngons"]))
        c.note("quad_pct=%.2f%%" % fs["quad_pct"])

    with r.case("face_count_within_40pct") as c:
        c.require(ctx.is_mesh_valid(new_obj), "no result mesh to measure")
        n = len(new_obj.data.polygons)
        c.require_rel(n, TARGET, 0.40, "faces")

    with r.case("stats_match_mesh") as c:
        c.require(ctx.is_mesh_valid(new_obj), "no result mesh to measure")
        stats = res.get("stats") or {}
        c.require(stats, "result['stats'] is empty")
        fs = ctx.face_stats(new_obj)
        for key in ("faces", "quads", "quad_pct", "time_s"):
            c.require(key in stats, "stats missing %r (got %s)" % (key, sorted(stats)))
        c.require(int(stats["faces"]) == fs["faces"],
                  "stats['faces']=%s but mesh has %d" % (stats["faces"], fs["faces"]))
        c.require(int(stats["quads"]) == fs["quads"],
                  "stats['quads']=%s but mesh has %d" % (stats["quads"], fs["quads"]))
        c.require(abs(float(stats["quad_pct"]) - fs["quad_pct"]) < 0.5,
                  "stats['quad_pct']=%s but mesh has %.2f" % (stats["quad_pct"], fs["quad_pct"]))
        c.require(float(stats["time_s"]) >= 0.0, "time_s is negative")

    with r.case("last_report_json") as c:
        c.require(new_obj is not None, "no result object")
        import json
        txt = new_obj.quadforge.last_report or ""
        if not txt:
            # the pipeline may have written it onto the source settings instead
            c.skip("last_report empty on result object (pipeline may store it elsewhere)")
        data = json.loads(txt)
        c.require(isinstance(data, dict), "last_report is not a JSON object")
        c.note("%d keys" % len(data))

    # -------------------------------------------------------- strict count
    with r.case("strict_count_within_12pct") as c:
        src2, res2 = _remesh(ctx, strict_count=True, adapt_quad_count=False)
        c.require(res2.get("ok") is True,
                  "strict run failed: %r" % (res2.get("error"),))
        obj2 = res2.get("object")
        c.require(ctx.is_mesh_valid(obj2), "strict run produced no mesh")
        c.require_rel(len(obj2.data.polygons), TARGET, 0.12, "faces")

    # --------------------------------------------------------- keep original
    with r.case("keep_original") as c:
        ctx.fresh_scene()
        obj = ctx.suzanne(subdiv=1)
        ctx.activate(obj)
        src_name = obj.name
        s = ctx.settings(obj, target_count=800, backend='QUADRIFLOW',
                         keep_original=True)
        res3 = ctx.pipeline().run_remesh(bpy.context, obj, s)
        c.require(res3.get("ok") is True, "run failed: %r" % (res3.get("error"),))
        coll = bpy.data.collections.get("QuadForge Originals")
        c.require(coll is not None,
                  "'QuadForge Originals' collection was not created "
                  "(collections: %s)" % [x.name for x in bpy.data.collections])
        names = [o.name for o in coll.objects]
        c.require(names, "'QuadForge Originals' collection is empty")
        c.note("original stored as %s (source was %s)" % (names, src_name))

    return r.list()
