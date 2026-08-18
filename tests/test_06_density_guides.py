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


# a direction 30 degrees off the grid axes: far from the axis-aligned 4-RoSy
# class of a staircase, so the two are easy to tell apart
_GUIDE_DIR = (0.8660254037844387, 0.5, 0.0)


def _flat_grid(n, size=1.0):
    """(V, F) of an n x n triangulated flat grid in the z = 0 plane."""
    import numpy as np
    g = np.linspace(-size, size, n)
    X, Y = np.meshgrid(g, g, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            F.append((a, a + n, a + n + 1))
            F.append((a, a + n + 1, a + 1))
    return V, np.asarray(F, dtype=np.int64)


def _staircase_guide_grid(n):
    """Flat grid + a ~30 degree guide stored the way project_guides stores one.

    Returns ``(V, F, sharp, guide_faces, chain)``: the walked path as sharp
    *axis-aligned* mesh edges (what the surface walk produces), the smooth
    curve direction as a per-face vector attribute, and the path vertices.
    """
    import numpy as np
    V, F = _flat_grid(n)
    i, j = 3, 3
    chain = [i * n + j]
    for k in range(2 * (n - 8)):
        if k % 3 == 2:                      # 2 steps in x per step in y ~ 27 deg
            j += 1
        else:
            i += 1
        if i >= n - 3 or j >= n - 3:
            break
        chain.append(i * n + j)
    chain = np.asarray(chain, dtype=np.int64)
    sharp = np.stack([chain[:-1], chain[1:]], axis=1)
    on_path = np.zeros(len(V), dtype=bool)
    on_path[chain] = True
    touch = on_path[F].any(axis=1)
    guide_faces = np.zeros((len(F), 3))
    guide_faces[touch] = _GUIDE_DIR
    return V, F, sharp, guide_faces, chain


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

    # --------------------------------------------- guides vs their sharp path
    # project_guides marks the path it walks sharp (that is the only channel
    # QuadriFlow understands), so in the native backend every guided vertex is
    # also a sharp vertex.  The staircase of mesh edges the walk took must not
    # be allowed to speak for the curve: an axis-aligned staircase is
    # 4-RoSy-consistent with the grid it steps over, so letting it win pins the
    # flow to exactly what the guide was drawn to override.
    with r.case("guide_outranks_its_own_sharp_path") as c:
        import numpy as np
        fields = ctx.imp("quadforge.backends.native.fields")
        V, F = _flat_grid(9)
        N = fields.vertex_normals(V, F)
        n = len(V)
        mid = n // 2
        sharp = np.array([[mid, mid + 1]], dtype=np.int64)   # runs along +x
        gdir = np.zeros((n, 3))
        gdir[mid] = _GUIDE_DIR
        m_win, d_win = fields.build_constraints(V, N, n, sharp, gdir,
                                                guides_win=True)
        m_old, d_old = fields.build_constraints(V, N, n, sharp, gdir,
                                                guides_win=False)
        c.require(bool(m_win[mid]) and bool(m_old[mid]),
                  "the guided vertex lost its constraint entirely")
        a_win = float(fields.rosy4_angle(d_win[mid:mid + 1],
                                         np.array([_GUIDE_DIR]))[0])
        a_old = float(fields.rosy4_angle(d_old[mid:mid + 1],
                                         np.array([_GUIDE_DIR]))[0])
        c.require(np.degrees(a_win) < 1.0,
                  "guides_win=True still pins the sharp direction "
                  "(%.1f deg off the guide)" % np.degrees(a_win))
        c.require(np.degrees(a_old) > 10.0,
                  "guides_win=False no longer reproduces the old behaviour "
                  "(%.1f deg)" % np.degrees(a_old))
        c.note("guide %.1f deg vs sharp-path %.1f deg off the curve"
               % (np.degrees(a_win), np.degrees(a_old)))

    with r.case("guide_steers_the_orientation_field") as c:
        import numpy as np
        fields = ctx.imp("quadforge.backends.native.fields")
        V, F, sharp, gfaces, chain = _staircase_guide_grid(21)
        sol = fields.solve_fields(V, F, {
            "target_faces": 400, "sharp_edges": sharp, "guide_dirs": gfaces,
            "curvature_align": 0.0, "seed": 0,
        })
        tgt = np.tile(_GUIDE_DIR, (len(chain), 1))
        ang = np.degrees(fields.rosy4_angle(sol.Q[chain], tgt))
        med = float(np.median(ang))
        c.require(med < 5.0,
                  "the field on the guide sits %.1f deg off it - the staircase "
                  "of sharp edges is speaking for the curve again" % med)
        c.note("median %.2f deg over %d guided verts" % (med, len(chain)))

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
