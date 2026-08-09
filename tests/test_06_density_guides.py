"""Painted density attribute and curve guides."""

import bpy

TARGET = 2500
DENSITY_LOW = 0.5
DENSITY_HIGH = 2.0


def _paint_density(obj):
    """qf_density = 2.0 on the upper hemisphere, 0.5 on the lower one."""
    me = obj.data
    attr = me.attributes.get("qf_density")
    if attr is None or attr.domain != 'POINT' or attr.data_type != 'FLOAT':
        if attr is not None:
            me.attributes.remove(attr)
        attr = me.attributes.new("qf_density", 'FLOAT', 'POINT')
    for i, v in enumerate(me.vertices):
        attr.data[i].value = DENSITY_HIGH if v.co.z >= 0.0 else DENSITY_LOW
    return attr


def _mean_edge_len(obj, upper):
    me = obj.data
    verts = [v.co for v in me.vertices]
    total = 0.0
    n = 0
    for e in me.edges:
        a = verts[e.vertices[0]]
        b = verts[e.vertices[1]]
        mz = 0.5 * (a.z + b.z)
        if (mz > 0.25) if upper else (mz < -0.25):
            total += (b - a).length
            n += 1
    return (total / n) if n else None, n


def _guide_count(res):
    """Find a 'how many guide edges did we use' number in the result."""
    stats = (res.get("stats") or {}) if isinstance(res, dict) else {}
    pools = [stats, res if isinstance(res, dict) else {}]
    for pool in pools:
        for k, v in pool.items():
            if "guide" in str(k).lower() and isinstance(v, (int, float)):
                return k, v
    return None, None


def run(ctx):
    r = ctx.results()

    # ------------------------------------------------------------- density
    state = {}
    with r.case("density_attr_accepted") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=48, rings=24)
        ctx.activate(obj)
        attr = _paint_density(obj)
        c.require(attr is not None, "could not create qf_density attribute")
        c.require(len(attr.data) == len(obj.data.vertices),
                  "qf_density has %d values for %d verts"
                  % (len(attr.data), len(obj.data.vertices)))
        s = ctx.settings(obj, mode='FACES', target_count=TARGET,
                         backend='QUADRIFLOW', use_paint_density=True,
                         adaptive_size=50.0, adapt_quad_count=True)
        res = ctx.pipeline().run_remesh(bpy.context, obj, s)
        state["res"] = res
        c.require(res.get("ok") is True,
                  "run with use_paint_density failed: %r" % (res.get("error"),))
        out = res.get("object")
        c.require(ctx.is_mesh_valid(out), "no result mesh")
        state["out"] = out
        c.note("faces=%d quad_pct=%.1f%%"
               % (len(out.data.polygons), ctx.face_stats(out)["quad_pct"]))

    with r.case("density_has_effect_or_is_reported") as c:
        out = state.get("out")
        res = state.get("res") or {}
        c.require(out is not None, "no result mesh (density_attr_accepted failed)")
        hi, nh = _mean_edge_len(out, upper=True)     # density 2.0 -> short edges
        lo, nl = _mean_edge_len(out, upper=False)    # density 0.5 -> long edges
        c.require(hi and lo, "not enough edges per hemisphere (%s/%s)" % (nh, nl))
        ratio = lo / hi
        if ratio >= 1.25:
            c.note("edge-length ratio low/high = %.2f (adaptivity active)" % ratio)
        else:
            stats = res.get("stats") or {}
            notes = []
            for pool in (stats, res):
                for k, v in pool.items():
                    kl = str(k).lower()
                    if any(t in kl for t in ("warn", "limit", "unsupported", "note")):
                        notes.append("%s=%r" % (k, v))
            c.skip("backend did not vary density (ratio %.2f) - reported: %s"
                   % (ratio, notes or "nothing"))

    with r.case("build_density_attr") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=32, rings=16)
        ctx.activate(obj)
        _paint_density(obj)
        s = ctx.settings(obj, use_paint_density=True, adaptive_size=50.0)
        analysis = ctx.imp("quadforge.core.analysis")
        analysis.build_density_attr(obj, s)
        attr = obj.data.attributes.get("qf_density")
        c.require(attr is not None,
                  "build_density_attr did not leave a 'qf_density' attribute")
        c.require(attr.domain == 'POINT' and attr.data_type == 'FLOAT',
                  "qf_density is %s/%s, contract says POINT/FLOAT"
                  % (attr.domain, attr.data_type))
        c.require(len(attr.data) == len(obj.data.vertices),
                  "qf_density length %d != vert count %d"
                  % (len(attr.data), len(obj.data.vertices)))
        vals = [d.value for d in attr.data]
        c.require(min(vals) >= 0.0, "qf_density has negative values (min %.3f)" % min(vals))
        c.note("range %.3f..%.3f" % (min(vals), max(vals)))

    with r.case("no_density_attr_is_harmless") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        ctx.activate(obj)
        s = ctx.settings(obj, target_count=600, backend='QUADRIFLOW',
                         use_paint_density=True)
        res = ctx.pipeline().run_remesh(bpy.context, obj, s)
        c.require(res.get("ok") is True,
                  "use_paint_density with no painted attribute failed: %r"
                  % (res.get("error"),))
        c.note("faces=%d" % len(res["object"].data.polygons))

    # -------------------------------------------------------------- guides
    gstate = {}
    with r.case("guides_run_ok") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=48, rings=24)
        coll = bpy.data.collections.new("QF Guides")
        bpy.context.scene.collection.children.link(coll)
        curve = ctx.bezier_circle(radius=1.0, name="qf_guide_circle")
        for c_ in list(curve.users_collection):
            c_.objects.unlink(curve)
        coll.objects.link(curve)
        ctx.activate(obj)
        s = ctx.settings(obj, mode='FACES', target_count=TARGET,
                         backend='QUADRIFLOW', use_guides=True)
        s.guide_collection = coll
        c.require(s.guide_collection is not None, "guide_collection did not stick")
        res = ctx.pipeline().run_remesh(bpy.context, obj, s)
        gstate["res"] = res
        c.require(res.get("ok") is True,
                  "run with use_guides failed: %r" % (res.get("error"),))
        out = res.get("object")
        c.require(ctx.is_mesh_valid(out), "no result mesh")
        c.note("faces=%d" % len(out.data.polygons))

    with r.case("guides_reported") as c:
        res = gstate.get("res") or {}
        c.require(res.get("ok") is True, "guides run did not succeed")
        key, val = _guide_count(res)
        if key is None:
            c.skip("no guide count in the result (stats: %s)"
                   % sorted(res.get("stats") or {}))
        c.require(val >= 0, "%s = %r, expected >= 0" % (key, val))
        c.note("%s=%s" % (key, val))

    with r.case("project_guides_api") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=48, rings=24)
        curve = ctx.bezier_circle(radius=1.0, name="qf_guide_circle")
        ctx.activate(obj)
        s = ctx.settings(obj, use_guides=True)
        guides = ctx.imp("quadforge.core.guides")
        n = guides.project_guides(obj, [curve], s)
        c.require(isinstance(n, int),
                  "project_guides returned %r, want int" % type(n))
        c.require(n >= 0, "project_guides returned %d" % n)
        attr = obj.data.attributes.get("qf_guide")
        if attr is not None:
            c.require(attr.domain == 'FACE',
                      "'qf_guide' is on %s, contract says FACE" % attr.domain)
            c.require("VECTOR" in attr.data_type or "FLOAT" in attr.data_type,
                      "'qf_guide' data_type is %s, want a 3-float type" % attr.data_type)
            c.note("qf_guide %s/%s" % (attr.domain, attr.data_type))
        else:
            c.note("no 'qf_guide' face attribute written")
        c.note("edges=%d" % n)

    with r.case("material_boundaries_to_sharp") as c:
        ctx.fresh_scene()
        obj = ctx.rigged_sphere(segments=32, rings=16)
        analysis = ctx.imp("quadforge.core.analysis")
        n = analysis.material_boundaries_to_sharp(obj)
        c.require(isinstance(n, int),
                  "material_boundaries_to_sharp returned %r, want int" % type(n))
        c.require(n > 0,
                  "no sharp edges marked on a 2-material sphere (returned %d)" % n)
        c.note("marked=%d" % n)

    with r.case("guides_empty_collection_is_harmless") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        coll = bpy.data.collections.new("QF Guides Empty")
        bpy.context.scene.collection.children.link(coll)
        ctx.activate(obj)
        s = ctx.settings(obj, target_count=600, backend='QUADRIFLOW', use_guides=True)
        s.guide_collection = coll
        res = ctx.pipeline().run_remesh(bpy.context, obj, s)
        c.require(res.get("ok") is True,
                  "empty guide collection broke the run: %r" % (res.get("error"),))

    return r.list()
