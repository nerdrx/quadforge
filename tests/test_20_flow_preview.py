"""Flow preview: ``quadforge.preview_flow`` / ``quadforge.clear_preview``.

The preview runs only the Native field stage (``fields.solve_fields`` plus the
``rho`` shaping ``solver.solve`` applies to it) and turns the result into a
line mesh: one short stroke per sampled face, along the 4-RoSy principal
direction, as long as the quad that will land there.

Checked here, all headless:

* the object is created with the contracted name, tag and collection, and it
  rides the source object's transform;
* the stroke count is bounded by both the face count and the operator's cap;
* every stroke sits on the surface (BVH distance, in units of stroke length)
  and lies exactly in the plane of the face it was built on (to 1e-9), with a
  length of 0.85 x the local target quad size;
* the drawn direction is *combed*: which member of a 4-RoSy class sits in
  ``sol.Q`` is arbitrary, so the share of adjacent strokes more than 60 degrees
  apart has to drop once ``comb_field`` has chosen consistently (9.8% -> 4.1%
  on this fixture);
* stroke length tracks the sizing field: with Adaptive Size + Size Contrast up,
  the long/short ratio of the strokes opens up well past the uniform case;
* the field really is the solve's field: the strokes are compared against the
  edges of a *real* ``solver.solve`` on the same arrays and the same parameter
  dict, as a 4-RoSy angle, against two references measured on the same pairs -
  a random tangent field (chance) and the raw ``Q`` the preview draws (the
  floor: the extractor quantises the field onto a lattice and deviates from it
  near every irregular vertex).  Measured on Suzanne at 1500 faces: chance
  23.2, field 11.2, preview 11.9 degrees median - i.e. the preview is within
  ~1 deg of a perfect rendering of the field it claims to draw, and the whole
  remaining gap is the extraction, not the preview;
* re-running replaces the preview instead of accumulating one per run, the
  clear operator removes it, and ``ops.remesh.selected_meshes`` -- the single
  gate every remesh / batch / LOD operator goes through -- never hands a
  preview object to the solver;
* a degenerate input fails as a cancelled operator, not as a traceback.
"""

import numpy as np


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _seg_arrays(prev):
    """(k, 2, 3) stroke endpoints of a preview object, in its local space."""
    me = prev.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(nv, 3)
    ne = len(me.edges)
    ev = np.empty(ne * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(ne, 2)
    return co[ev]


def _bvh(obj):
    from mathutils.bvhtree import BVHTree
    me = obj.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    verts = co.reshape(nv, 3).tolist()
    faces = []
    for p in me.polygons:
        vs = list(p.vertices)
        for i in range(1, len(vs) - 1):
            faces.append((vs[0], vs[i], vs[i + 1]))
    return BVHTree.FromPolygons(verts, faces, all_triangles=True)


def _surface_fit(prev, src):
    """Endpoint/midpoint distances to the source surface, in stroke lengths.

    Returns ``(max, p99, p95)``.  A stroke is a straight chord of roughly one
    quad edge, deliberately lifted off the surface by 6 % of its length so it
    does not z-fight, so it can never be at distance zero: over a patch whose
    curvature radius is comparable to the quad size the chord alone accounts
    for ``L / 8R``.  The percentiles are the useful statistic; the max is a
    single stroke somewhere on Suzanne's ear rim.
    """
    from mathutils import Vector
    seg = _seg_arrays(prev)
    tree = _bvh(src)
    vals = []
    step = max(1, len(seg) // 400)          # 400 samples is plenty and quick
    for a, b in seg[::step]:
        ln = float(np.linalg.norm(b - a))
        if ln < 1e-12:
            continue
        for pt in (a, b, 0.5 * (a + b)):
            hit, _nrm, _idx, dist = tree.find_nearest(
                Vector((float(pt[0]), float(pt[1]), float(pt[2]))))
            if hit is None:
                continue
            vals.append(float(dist) / ln)
    v = np.asarray(vals) if vals else np.zeros(1)
    return float(v.max()), float(np.percentile(v, 99)), float(np.percentile(v, 95))


def _length_ratio(prev):
    seg = _seg_arrays(prev)
    ln = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
    ln = ln[ln > 1e-12]
    if len(ln) < 10:
        return 1.0
    lo = float(np.percentile(ln, 5))
    hi = float(np.percentile(ln, 95))
    return hi / max(lo, 1e-12)


def _rosy4_deg(a, b):
    """4-RoSy angle between two unit direction arrays, in degrees."""
    c = np.clip(np.abs(np.einsum("ij,ij->i", a, b)), 0.0, 1.0)
    ang = np.arccos(c)
    return np.degrees(np.minimum(ang, 0.5 * np.pi - ang))


# --------------------------------------------------------------------------

def run(ctx):
    r = ctx.results()
    P = ctx.try_imp("quadforge.ops.preview")
    remesh_ops = ctx.try_imp("quadforge.ops.remesh")
    shared = {}                 # one field solve, reused by the geometry cases

    import bpy

    # ---- one shared solve, reused by the geometry cases -------------------
    ctx.fresh_scene()
    src = ctx.suzanne(subdiv=2, name="Prev")
    ctx.activate(src)
    ctx.settings(src, backend='NATIVE', mode='FACES', target_count=1500,
                 adaptive_size=0.0, keep_original=True, seed=0)
    made = None
    if P is not None:
        try:
            made = bpy.ops.quadforge.preview_flow()
        except Exception as exc:                       # noqa: BLE001
            made = repr(exc)
    prev = bpy.data.objects.get("Prev Flow Preview")

    with r.case("operators_registered") as c:
        c.require(P is not None, "quadforge.ops.preview did not import")
        c.require(hasattr(bpy.types, "QUADFORGE_OT_preview_flow"),
                  "quadforge.preview_flow is not registered")
        c.require(hasattr(bpy.types, "QUADFORGE_OT_clear_preview"),
                  "quadforge.clear_preview is not registered")
        c.note("preview_flow + clear_preview registered")

    with r.case("preview_object") as c:
        c.require(made == {'FINISHED'}, "preview_flow returned %r" % (made,))
        c.require(prev is not None, "no object named 'Prev Flow Preview'")
        c.require(prev.type == 'MESH', "preview is a %s, not a mesh" % prev.type)
        c.require(len(prev.data.polygons) == 0,
                  "preview carries %d faces (it must be edges only)"
                  % len(prev.data.polygons))
        c.require(prev.get(P.PREVIEW_KEY) == src.name,
                  "preview is not tagged with its source (%r)"
                  % prev.get(P.PREVIEW_KEY))
        colls = [x.name for x in prev.users_collection]
        c.require(colls == [P.PREVIEWS_COLLECTION],
                  "preview lives in %s, expected only '%s'"
                  % (colls, P.PREVIEWS_COLLECTION))
        c.require(P.is_preview(prev), "is_preview() does not recognise it")
        c.require(not P.is_preview(src), "is_preview() claims the source")
        c.require(prev.hide_select and prev.hide_render,
                  "preview is selectable or renderable")
        c.note("%d strokes, %s, %.2fs solve"
               % (len(prev.data.edges), colls[0],
                  float(prev.get("qf_preview_time_s", 0.0))))

    with r.case("follows_source_transform") as c:
        src.location = (1.7, -0.4, 0.9)
        src.rotation_euler = (0.3, 0.0, 1.1)
        src.scale = (1.3, 1.3, 1.3)
        bpy.context.view_layer.update()
        c.require(prev.parent is src, "preview is not parented to the source")
        delta = max(abs(prev.matrix_world[i][j] - src.matrix_world[i][j])
                    for i in range(4) for j in range(4))
        c.require(delta < 1e-6,
                  "preview world matrix differs from the source by %.3g" % delta)
        src.location = (0.0, 0.0, 0.0)
        src.rotation_euler = (0.0, 0.0, 0.0)
        src.scale = (1.0, 1.0, 1.0)
        bpy.context.view_layer.update()
        c.note("parented, matrix matches to %.1e" % delta)

    with r.case("stroke_count_bounds") as c:
        nseg = len(prev.data.edges)
        nfaces = len(src.data.polygons)
        cap = 12000                                     # operator default
        c.require(nseg > 0, "no strokes at all")
        c.require(nseg <= nfaces,
                  "%d strokes for %d input faces (must not exceed)"
                  % (nseg, nfaces))
        c.require(nseg <= cap, "%d strokes exceeds the %d cap" % (nseg, cap))
        c.require(nseg >= 0.2 * min(nfaces, cap),
                  "only %d strokes for %d faces - too sparse to read"
                  % (nseg, nfaces))
        c.require(len(prev.data.vertices) == 2 * nseg,
                  "%d verts for %d strokes (segments must be disjoint)"
                  % (len(prev.data.vertices), nseg))
        c.note("%d strokes / %d faces (cap %d)" % (nseg, nfaces, cap))

    with r.case("strokes_on_surface") as c:
        worst, p99, p95 = _surface_fit(prev, src)
        # measured on this fixture: max 0.38, p99 0.21, p95 0.12
        c.require(p95 < 0.20,
                  "5%% of the strokes sit further than %.2f stroke-lengths off "
                  "the surface" % p95)
        c.require(p99 < 0.30, "p99 stroke offset is %.2f stroke-lengths" % p99)
        c.require(worst < 0.60,
                  "a stroke sits %.2f stroke-lengths off the surface" % worst)
        seg = _seg_arrays(prev)
        ln = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
        c.require(float(ln.min()) > 0.0, "a stroke has zero length")
        c.note("offset in stroke lengths: p95 %.3f, p99 %.3f, max %.3f over "
               "%d strokes" % (p95, p99, worst, len(seg)))

    with r.case("strokes_tangent_and_unit") as c:
        # exact, not sampled: every stroke is built in the plane of the face it
        # sits on, so its direction must be perpendicular to that face's normal
        # to machine precision.  (Measuring this against a BVH's nearest-face
        # normal instead would report the *neighbouring* face across a thin
        # feature -- Suzanne's ears -- and say nothing about the preview.)
        if P is None:
            c.skip("quadforge.ops.preview unavailable")
        fields = ctx.imp("quadforge.backends.native.fields")
        ctx.activate(src)
        V, F, sol, rho, info = P.solve_preview_field(
            bpy.context, src, src.quadforge)
        seg, sel = P.build_segments(V, F, sol, rho, max_segments=12000)
        shared["field"] = (V, F, sol, rho, info, seg, sel)
        c.require(len(seg) == len(sel), "segment/face-index mismatch")
        fn, _areas = fields.face_normals_areas(V, F[sel])
        fn = fields.normalize(fn)
        d = seg[:, 1] - seg[:, 0]
        ln = np.linalg.norm(d, axis=1)
        c.require(float(ln.min()) > 0.0, "a stroke has zero length")
        d = d / ln[:, None]
        dot = np.abs(np.einsum("ij,ij->i", d, fn))
        c.require(float(dot.max()) < 1e-9,
                  "a stroke leaves its own face plane by %.3g" % float(dot.max()))
        unit = np.abs(np.linalg.norm(d, axis=1) - 1.0)
        c.require(float(unit.max()) < 1e-9,
                  "a stroke direction is not unit length (%.3g off)"
                  % float(unit.max()))
        # length is the local target quad size (build_segments' default 0.85)
        ratio = ln / np.maximum(np.mean(rho[F[sel]], axis=1), 1e-12)
        c.require_close(float(np.mean(ratio)), 0.85, 1e-6, "length/rho")
        c.note("max |dir.n| %.1e over %d strokes, length = %.2f x rho"
               % (float(dot.max()), len(seg), float(np.mean(ratio))))

    with r.case("field_is_combed") as c:
        # A 4-RoSy class has four equally valid members; which one lands in
        # sol.Q is arbitrary, so drawing it raw leaves neighbouring strokes at
        # 90 degrees to each other and the picture reads as static.  comb_field
        # picks a consistent member without touching the field.
        if P is None or "field" not in shared:
            c.skip("no field solve available")
        fields = ctx.imp("quadforge.backends.native.fields")
        V, F, sol, rho, info, seg, sel = shared["field"]

        def _stroke_dirs(Q):
            fn, _a = fields.face_normals_areas(V, F)
            fn = fields.normalize(fn)
            ref = Q[F[:, 0]]
            acc = ref.copy()
            for k in (1, 2):
                acc = acc + fields.rosy4_representative(
                    Q[F[:, k]], np.asarray(sol.N)[F[:, k]], ref)
            d = acc - fn * np.einsum("ij,ij->i", acc, fn)[:, None]
            ln = np.linalg.norm(d, axis=1)
            ok = ln > 1e-9
            d[ok] /= ln[ok][:, None]
            return d, ok

        # adjacent face pairs (they share an edge, so their strokes should
        # agree unless the field really turns there)
        nv = int(F.max()) + 1
        a = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
        b = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
        fi = np.tile(np.arange(len(F)), 3)
        key = np.minimum(a, b) * np.int64(nv) + np.maximum(a, b)
        order = np.argsort(key, kind="stable")
        key, fi = key[order], fi[order]
        hit = np.nonzero(key[1:] == key[:-1])[0]
        p, q = fi[hit], fi[hit + 1]
        c.require(len(p) > 1000, "only %d adjacent face pairs" % len(p))

        def _crossing(Q):
            d, ok = _stroke_dirs(Q)
            m = ok[p] & ok[q]
            cs = np.clip(np.abs(np.einsum("ij,ij->i", d[p][m], d[q][m])), 0, 1)
            ang = np.degrees(np.arccos(cs))       # undirected: 0..90
            return float(np.mean(ang > 60.0))

        raw = _crossing(np.asarray(sol.Q))
        combed = _crossing(P.comb_field(V, F, sol))
        # measured on this fixture: 9.8% raw -> 4.1% combed
        c.require(combed < 0.06,
                  "%.1f%% of adjacent strokes still cross at >60 deg after "
                  "combing" % (100 * combed))
        c.require(combed <= raw + 1e-9,
                  "combing made it worse: %.1f%% -> %.1f%%"
                  % (100 * raw, 100 * combed))
        c.note("adjacent strokes >60 deg apart: %.1f%% raw -> %.1f%% combed, "
               "%d pairs" % (100 * raw, 100 * combed, len(p)))

    # ---- sizing responds to the adaptive settings ------------------------
    with r.case("sizing_follows_adaptive") as c:
        uniform = _length_ratio(prev)
        ctx.settings(src, adaptive_size=90.0, detail_range=10.0)
        ctx.activate(src)
        c.require(bpy.ops.quadforge.preview_flow() == {'FINISHED'},
                  "adaptive preview_flow failed")
        prev_a = bpy.data.objects.get("Prev Flow Preview")
        adaptive = _length_ratio(prev_a)
        c.require(adaptive > 1.6 * uniform,
                  "stroke length spread barely moved: %.2f uniform -> %.2f "
                  "adaptive" % (uniform, adaptive))
        span = (float(prev_a.get("qf_preview_rho_max", 0.0))
                / max(float(prev_a.get("qf_preview_rho_min", 1.0)), 1e-12))
        c.require(span > 3.0,
                  "rho span is only %.2f with Size Contrast at 10" % span)
        ctx.settings(src, adaptive_size=0.0, detail_range=3.0)
        c.note("length ratio %.2f -> %.2f, rho span %.2f"
               % (uniform, adaptive, span))

    # ---- the honesty check: does it match a real solve? -------------------
    with r.case("matches_the_real_solve") as c:
        solver = ctx.try_imp("quadforge.backends.native.solver")
        if P is None or solver is None or "field" not in shared:
            c.skip("native solver modules unavailable")
        V, F, sol, rho, info, seg, sel = shared["field"]
        c.require(len(seg) > 100, "only %d strokes to compare" % len(seg))

        # The real thing: same arrays, same parameter dict the preview solved
        # with, so the only difference left is the position solve + extraction
        # the preview skips.
        VQ, FQ = solver.solve(V, F, dict(info["params"]))
        VQ = np.asarray(VQ, dtype=np.float64)
        pairs = set()
        for f in FQ:
            k = len(f)
            for i in range(k):
                a, b = int(f[i]), int(f[(i + 1) % k])
                pairs.add((a, b) if a < b else (b, a))
        E = np.asarray(sorted(pairs), dtype=np.int64)
        emid = 0.5 * (VQ[E[:, 0]] + VQ[E[:, 1]])
        edir = VQ[E[:, 1]] - VQ[E[:, 0]]
        elen = np.linalg.norm(edir, axis=1)
        keep = elen > 1e-9
        emid, edir = emid[keep], edir[keep] / elen[keep][:, None]
        c.require(len(edir) > 100, "the control solve produced %d edges"
                  % len(edir))

        # nearest solved edge for each sampled stroke, then the 4-RoSy angle
        # (an edge running across the flow is as aligned as one running along
        # it: both belong to the same cross field)
        smid = 0.5 * (seg[:, 0] + seg[:, 1])
        sdir = seg[:, 1] - seg[:, 0]
        sdir = sdir / np.maximum(np.linalg.norm(sdir, axis=1), 1e-12)[:, None]
        take = np.arange(0, len(smid), max(1, len(smid) // 600))
        kd = ctx.kdtree(emid)
        from mathutils import Vector
        idx = np.asarray(
            [kd.find(Vector((float(p[0]), float(p[1]), float(p[2]))))[1]
             for p in smid[take]], dtype=np.int64)
        match = edir[idx]
        ang = _rosy4_deg(sdir[take], match)

        # Two references, measured on the very same pairs.
        # (a) chance: a random tangent direction in each stroke's face plane.
        fields = ctx.imp("quadforge.backends.native.fields")
        fn, _ar = fields.face_normals_areas(V, F[sel][take])
        fn = fields.normalize(fn)
        rng = np.random.default_rng(0)
        rd = rng.normal(size=(len(take), 3))
        rd = rd - fn * np.einsum("ij,ij->i", rd, fn)[:, None]
        rd = rd / np.maximum(np.linalg.norm(rd, axis=1), 1e-12)[:, None]
        chance = float(np.median(_rosy4_deg(rd, match)))
        # (b) the field itself, sampled at the input vertex nearest the same
        #     stroke: the floor this metric can reach, because the extractor
        #     quantises the field onto a lattice and deviates from it near
        #     every irregular vertex.
        kdv = ctx.kdtree(V)
        vi = np.asarray(
            [kdv.find(Vector((float(p[0]), float(p[1]), float(p[2]))))[1]
             for p in smid[take]], dtype=np.int64)
        Nv, Qv = np.asarray(sol.N)[vi], np.asarray(sol.Q)[vi]
        ep = match - Nv * np.einsum("ij,ij->i", match, Nv)[:, None]
        epl = np.linalg.norm(ep, axis=1)
        good = epl > 1e-9
        floor = float(np.median(_rosy4_deg(ep[good] / epl[good][:, None],
                                           Qv[good])))

        median = float(np.median(ang))
        mean = float(np.mean(ang))
        c.require(median < 0.65 * chance,
                  "preview strokes barely beat chance: median %.1f deg vs "
                  "%.1f for a random tangent field" % (median, chance))
        c.require(median < 13.0,
                  "preview strokes disagree with the solved edges: median "
                  "%.1f deg" % median)
        c.require(mean < 18.0, "mean disagreement %.1f deg" % mean)
        c.require(median < floor + 4.0,
                  "the preview adds %.1f deg on top of the raw field it draws "
                  "(field %.1f, preview %.1f)" % (median - floor, floor, median))
        c.note("preview %.1f deg median (mean %.1f) vs field %.1f, chance "
               "%.1f; %d quads / %d edges solved"
               % (median, mean, floor, chance, len(FQ), len(edir)))

    # ---- lifecycle --------------------------------------------------------
    with r.case("rerun_replaces") as c:
        ctx.activate(src)
        before = [o.name for o in bpy.data.objects if P.is_preview(o)]
        for _ in range(3):
            c.require(bpy.ops.quadforge.preview_flow() == {'FINISHED'},
                      "a repeat preview_flow failed")
        after = [o.name for o in bpy.data.objects if P.is_preview(o)]
        c.require(len(after) == 1,
                  "%d preview objects after 3 re-runs: %s" % (len(after), after))
        c.require(after == before or before == [],
                  "the preview was renamed across runs: %s -> %s"
                  % (before, after))
        meshes = [m.name for m in bpy.data.meshes
                  if m.name.startswith("Prev Flow Preview")]
        c.require(len(meshes) <= 1,
                  "%d orphaned preview meshes: %s" % (len(meshes), meshes))
        c.note("1 object, %d mesh datablock(s) after 4 runs" % len(meshes))

    with r.case("second_object_gets_its_own") as c:
        other = ctx.uv_sphere(segments=24, rings=12, name="Other")
        ctx.activate(other)
        ctx.settings(other, backend='NATIVE', target_count=300)
        c.require(bpy.ops.quadforge.preview_flow() == {'FINISHED'},
                  "preview_flow on the second object failed")
        names = sorted(o.name for o in bpy.data.objects if P.is_preview(o))
        c.require(names == ["Other Flow Preview", "Prev Flow Preview"],
                  "expected one preview per object, got %s" % (names,))
        c.note(", ".join(names))

    with r.case("remesh_ignores_previews") as c:
        c.require(remesh_ops is not None, "ops.remesh did not import")
        for o in bpy.data.objects:
            try:
                o.select_set(True)
            except Exception:
                pass
        bpy.context.view_layer.objects.active = src
        picked = [o.name for o in remesh_ops.selected_meshes(bpy.context)]
        leaked = [n for n in picked if n.endswith(P.PREVIEW_SUFFIX)]
        c.require(not leaked,
                  "selected_meshes() handed the solver %s" % (leaked,))
        c.require(src.name in picked,
                  "selected_meshes() lost the real object (%s)" % (picked,))
        # ...and with *only* a preview active, the operator must not run
        prev_obj = bpy.data.objects["Prev Flow Preview"]
        for o in bpy.data.objects:
            try:
                o.select_set(False)
            except Exception:
                pass
        prev_obj.select_set(True)
        bpy.context.view_layer.objects.active = prev_obj
        c.require(not remesh_ops.selected_meshes(bpy.context),
                  "a lone preview object is still offered to the solver")
        c.require(not bpy.ops.quadforge.remesh.poll(),
                  "quadforge.remesh polls True on a lone preview object")
        edges_before = len(prev_obj.data.edges)
        c.require(len(prev_obj.data.polygons) == 0 and edges_before > 0,
                  "the preview was modified")
        c.note("%d objects offered, preview excluded from all of "
               "remesh/batch/LODs (one gate)" % len(picked))

    with r.case("clear_removes") as c:
        ctx.activate(src)
        c.require(bpy.ops.quadforge.clear_preview() == {'FINISHED'},
                  "clear_preview failed")
        left = sorted(o.name for o in bpy.data.objects if P.is_preview(o))
        c.require(left == ["Other Flow Preview"],
                  "clear_preview took %s (it should only clear the active "
                  "object's)" % (left,))
        bpy.context.view_layer.objects.active = None
        bpy.ops.quadforge.clear_preview(all_objects=True)
        left = [o.name for o in bpy.data.objects if P.is_preview(o)]
        c.require(not left, "previews survived a clear-all: %s" % (left,))
        c.require(bpy.data.collections.get(P.PREVIEWS_COLLECTION) is None,
                  "the empty previews collection was left behind")
        c.require(not bpy.ops.quadforge.clear_preview.poll(),
                  "clear_preview still polls True with nothing to clear")
        c.note("per-object clear, then clear-all, collection removed")

    with r.case("degenerate_input") as c:
        ctx.fresh_scene()
        me = bpy.data.meshes.new("Tiny")
        me.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        me.update()
        tiny = ctx.link(bpy.data.objects.new("Tiny", me))
        ctx.activate(tiny)
        ctx.settings(tiny, backend='NATIVE', target_count=500)
        try:
            res = bpy.ops.quadforge.preview_flow()
        except RuntimeError as exc:
            res = {'CANCELLED'}                 # reported error, not a crash
            c.note("reported: %s" % str(exc).splitlines()[0][:60])
        c.require(res in ({'CANCELLED'}, {'FINISHED'}),
                  "preview_flow returned %r on a single triangle" % (res,))
        if res == {'FINISHED'}:
            p2 = bpy.data.objects.get("Tiny Flow Preview")
            c.require(p2 is not None and len(p2.data.edges) > 0,
                      "claimed success without producing strokes")

        # zero-area mesh: two coincident triangles
        flat = bpy.data.meshes.new("Flat")
        flat.from_pydata([(0, 0, 0)] * 4, [], [(0, 1, 2), (0, 2, 3)])
        flat.update()
        zero = ctx.link(bpy.data.objects.new("Flat", flat))
        ctx.activate(zero)
        ctx.settings(zero, backend='NATIVE', target_count=500)
        try:
            res2 = bpy.ops.quadforge.preview_flow()
        except RuntimeError:
            res2 = {'CANCELLED'}
        c.require(res2 in ({'CANCELLED'}, {'FINISHED'}),
                  "preview_flow returned %r on a zero-area mesh" % (res2,))
        c.note("single triangle -> %s, zero area -> %s"
               % (sorted(res)[0], sorted(res2)[0]))

    return r.list()
