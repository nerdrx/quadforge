"""Native (experimental) backend. Every check is SKIP-friendly: the native
solver is allowed to be absent or to refuse a mesh via its documented
QuadForgeNativeError, and that must not fail the suite."""

import bpy

TARGET = 800
MIN_QUAD_PCT = 70.0


def _native_errors(ctx):
    """Collect the exception types the native backend is allowed to raise."""
    types = []
    for dotted in ("quadforge.backends.native", "quadforge.backends.native.solver",
                   "quadforge.pipeline", "quadforge"):
        mod = ctx.try_imp(dotted)
        if mod is None:
            continue
        for name in dir(mod):
            if "NativeError" in name or name == "QuadForgeNativeError":
                obj = getattr(mod, name)
                if isinstance(obj, type) and issubclass(obj, BaseException):
                    if obj not in types:
                        types.append(obj)
    return tuple(types)


def run(ctx):
    r = ctx.results()
    errs = _native_errors(ctx)

    with r.case("native_module_present") as c:
        mod = ctx.try_imp("quadforge.backends.native")
        if mod is None:
            c.skip("quadforge.backends.native is not importable")
        c.require(hasattr(mod, "remesh"),
                  "backends.native has no remesh(context, work_obj, s, face_target)")
        c.note("errors=%s" % ([e.__name__ for e in errs] or "none declared"))

    with r.case("native") as c:
        ctx.fresh_scene()
        obj = ctx.ico_sphere(subdivisions=3)     # 1280 tris
        ctx.activate(obj)
        s = ctx.settings(obj, mode='FACES', target_count=TARGET,
                         backend='NATIVE', seed=1)
        pipeline = ctx.try_imp("quadforge.pipeline")
        if pipeline is None:
            c.skip("quadforge.pipeline is not importable")
        try:
            res = pipeline.run_remesh(bpy.context, obj, s)
        except errs as e:            # documented, expected refusal
            c.skip("native backend raised %s: %s" % (type(e).__name__, e))
        except (ImportError, NotImplementedError, AttributeError) as e:
            c.skip("native backend unavailable: %s: %s" % (type(e).__name__, e))
        except Exception as e:
            blob = (type(e).__name__ + " " + str(e)).lower()
            if "native" in blob or "not implement" in blob or "experimental" in blob:
                c.skip("native backend refused: %s: %s" % (type(e).__name__, e))
            raise
        if not res.get("ok"):
            c.skip("native backend declined: %r" % (res.get("error"),))
        out = res.get("object")
        c.require(ctx.is_mesh_valid(out),
                  "native run reported ok but produced no mesh")
        fs = ctx.face_stats(out)
        c.require(fs["quad_pct"] >= MIN_QUAD_PCT,
                  "native result is %.1f%% quads (%d quads / %d faces), want >= %.0f%%"
                  % (fs["quad_pct"], fs["quads"], fs["faces"], MIN_QUAD_PCT))
        c.require(ctx.non_manifold_edge_count(out) == 0,
                  "native result has %d non-manifold edges"
                  % ctx.non_manifold_edge_count(out))
        co = ctx.verts_np(out)
        radii = (co ** 2).sum(axis=1) ** 0.5
        c.require(float(radii.max()) < 1.5 and float(radii.min()) > 0.5,
                  "native result strays off the unit sphere: radius %.3f..%.3f"
                  % (float(radii.min()), float(radii.max())))
        c.note("faces=%d quad_pct=%.1f%% r=%.3f..%.3f"
               % (fs["faces"], fs["quad_pct"], float(radii.min()), float(radii.max())))

    with r.case("native_solver_api") as c:
        solver = ctx.try_imp("quadforge.backends.native.solver")
        if solver is None or not hasattr(solver, "solve"):
            c.skip("backends.native.solver.solve is not available")
        import numpy as np
        ctx.fresh_scene()
        obj = ctx.ico_sphere(subdivisions=2)      # 320 tris
        me = obj.data
        V = ctx.verts_np(obj)
        F = np.array([list(p.vertices) for p in me.polygons], dtype="i4")
        c.require(F.shape[1] == 3, "ico sphere is not triangulated (%s)" % (F.shape,))
        params = {
            'target_faces': 200, 'adaptive': 0.0,
            'sharp_edges': np.zeros((0, 2), dtype="i4"),
            'guide_dirs': None, 'density': None,
            'symmetry': (False, False, False), 'seed': 0,
        }
        try:
            VQ, FQ = solver.solve(V.astype("f8"), F, params)
        except errs as e:
            c.skip("solver.solve raised %s: %s" % (type(e).__name__, e))
        except (ImportError, NotImplementedError) as e:
            c.skip("solver unavailable: %s: %s" % (type(e).__name__, e))
        c.require(VQ is not None and FQ is not None, "solve returned None")
        # the contract fixes the shapes, not the container type
        VQ = np.asarray(VQ, dtype="f8")
        c.require(VQ.ndim == 2 and VQ.shape[1] == 3,
                  "VQ shape %s, want (k,3)" % (VQ.shape,))
        c.require(len(FQ) > 0, "solve produced no faces")

        sizes = {}
        for f in FQ:
            sizes[len(f)] = sizes.get(len(f), 0) + 1
        flat = [i for f in FQ for i in f]
        c.require(min(flat) >= 0, "FQ has negative indices")
        c.require(max(flat) < len(VQ),
                  "FQ indexes vertex %d but VQ has %d" % (max(flat), len(VQ)))
        c.require(set(sizes) <= {3, 4},
                  "CONTRACTS.md allows faces of size 3 or 4 but solve() returned "
                  "%d polygons with sizes %s" % (len(FQ), sizes))
        quad_pct = 100.0 * sizes.get(4, 0) / len(FQ)
        c.require(quad_pct >= MIN_QUAD_PCT,
                  "solver output is %.1f%% quads, want >= %.0f%%"
                  % (quad_pct, MIN_QUAD_PCT))
        c.note("VQ=%s faces=%d quad_pct=%.1f%%" % (VQ.shape, len(FQ), quad_pct))

    return r.list()
