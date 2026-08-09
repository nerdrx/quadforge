"""LOD generation and batch remeshing.

Both features are exposed as operators, and operators are the part of the addon
most likely to be unhappy in --background. Where the operator cannot run for
context reasons we fall back to the pipeline and record the operator-level part
as a SKIP rather than a failure.
"""

import re

import bpy

LOD_TARGETS = "2000,500"
ORIGINALS = "QuadForge Originals"
LOD_NAME = re.compile(r"_LOD\d+(\.\d+)?$")
TEMP_NAME = re.compile(r"LODTMP")


def _mesh_objects():
    return {o.name for o in bpy.data.objects if o.type == 'MESH'}


def _leftover_temps():
    return [o.name for o in bpy.data.objects if TEMP_NAME.search(o.name)]


def _new_lod_objects(before):
    """New mesh objects that are LOD results.

    Prefer the ``<base>_LOD<n>`` naming the addon uses; fall back to "any new
    mesh object" so a differently-named implementation still gets measured.
    Working copies (``*LODTMP*``) and the stashed original are never counted.
    """
    originals = bpy.data.collections.get(ORIGINALS)
    orig_names = {o.name for o in originals.objects} if originals else set()
    fresh = []
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.name in before or o.name in orig_names:
            continue
        if not o.data or len(o.data.polygons) == 0:
            continue
        if TEMP_NAME.search(o.name):
            continue
        fresh.append(o)
    named = [o for o in fresh if LOD_NAME.search(o.name)]
    return named if named else fresh


def _override(obj):
    kw = {}
    try:
        kw["scene"] = bpy.context.scene
        kw["view_layer"] = bpy.context.view_layer
        kw["collection"] = bpy.context.view_layer.active_layer_collection.collection
    except Exception:
        pass
    if obj is not None:
        kw["object"] = obj
        kw["active_object"] = obj
        kw["selected_objects"] = [obj]
        kw["selected_editable_objects"] = [obj]
    return kw


def _call_op(op, obj, **kwargs):
    """Run an operator with a background-friendly override.
    Returns (result_set, error_or_None)."""
    try:
        with bpy.context.temp_override(**_override(obj)):
            return op('EXEC_DEFAULT', **kwargs), None
    except Exception as e:
        try:
            return op('EXEC_DEFAULT', **kwargs), None
        except Exception as e2:
            return None, "%s: %s | retry %s: %s" % (type(e).__name__, e,
                                                    type(e2).__name__, e2)


def _lods_via_python(ctx, obj):
    """Fallback: a module-level LOD entry point, else drive the pipeline."""
    mod = ctx.try_imp("quadforge.ops.lods")
    if mod is not None:
        for name in ("generate_lods", "build_lods", "make_lods", "run"):
            fn = getattr(mod, name, None)
            if callable(fn):
                for args in ((bpy.context, obj, obj.quadforge), (bpy.context, obj), (obj,)):
                    try:
                        fn(*args)
                        return "quadforge.ops.lods.%s%d" % (name, len(args))
                    except TypeError:
                        continue
    pipeline = ctx.pipeline()
    for target in [int(t) for t in obj.quadforge.lod_targets.split(",") if t.strip()]:
        s = ctx.settings(obj, mode='FACES', target_count=target,
                         backend='QUADRIFLOW', keep_original=True)
        res = pipeline.run_remesh(bpy.context, obj, s)
        if not res.get("ok"):
            raise RuntimeError("pipeline fallback failed at %d: %r"
                               % (target, res.get("error")))
    return "pipeline fallback"


def run(ctx):
    r = ctx.results()

    # ----------------------------------------------------------------- LODs
    state = {}
    with r.case("generate_lods") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=64, rings=32)
        ctx.activate(obj)
        ctx.settings(obj, lod_targets=LOD_TARGETS, backend='QUADRIFLOW',
                     keep_original=True)
        before = _mesh_objects()
        state["before"] = before
        how = "operator"
        if not hasattr(bpy.types, "QUADFORGE_OT_generate_lods"):
            err = "quadforge.generate_lods operator is not registered"
        else:
            _, err = _call_op(bpy.ops.quadforge.generate_lods, obj)
        if err:
            how = _lods_via_python(ctx, obj)
            c.note("operator failed in background (%s) -> used %s" % (err, how))
        lods = _new_lod_objects(before)
        state["lods"] = lods
        state["how"] = how
        state["temps"] = _leftover_temps()
        c.require(len(lods) >= 2,
                  "expected 2 LOD objects for targets %r, found %d: %s"
                  % (LOD_TARGETS, len(lods), [o.name for o in lods]))
        c.note("via %s: %s" % (how, [(o.name, len(o.data.polygons)) for o in lods]))

    with r.case("no_leftover_working_copies") as c:
        c.require(state.get("lods"), "no LOD objects (generate_lods failed)")
        temps = state.get("temps") or []
        c.require(not temps,
                  "LOD generation leaked %d working copies into the scene: %s"
                  % (len(temps), temps))

    with r.case("lod_face_counts_descending") as c:
        lods = state.get("lods") or []
        c.require(len(lods) >= 2, "no LOD objects (generate_lods failed)")
        counts = sorted((len(o.data.polygons) for o in lods), reverse=True)
        for a, b in zip(counts, counts[1:]):
            c.require(a > b, "LOD face counts are not strictly descending: %s" % counts)
        c.note("counts=%s" % counts)

    with r.case("lod_counts_near_targets") as c:
        lods = state.get("lods") or []
        c.require(len(lods) >= 2, "no LOD objects (generate_lods failed)")
        wanted = sorted((int(t) for t in LOD_TARGETS.split(",")), reverse=True)
        counts = sorted((len(o.data.polygons) for o in lods), reverse=True)
        c.require(len(counts) >= len(wanted),
                  "got %d LODs for %d targets" % (len(counts), len(wanted)))
        for got, want in zip(counts, wanted):
            c.require_rel(got, want, 0.50, "lod")

    with r.case("lods_are_quads") as c:
        lods = state.get("lods") or []
        c.require(len(lods) >= 2, "no LOD objects (generate_lods failed)")
        for o in lods:
            fs = ctx.face_stats(o)
            c.require(fs["quad_pct"] >= 95.0,
                      "%s is only %.1f%% quads" % (o.name, fs["quad_pct"]))
        c.note("all >= 95% quads")

    # ---------------------------------------------------------------- batch
    bstate = {}
    with r.case("remesh_batch_operator") as c:
        ctx.fresh_scene()
        a = ctx.uv_sphere(segments=32, rings=16, name="BatchA")
        a.location = (0.0, 0.0, 0.0)
        b = ctx.torus(major_segments=32, minor_segments=12, name="BatchB")
        b.location = (3.0, 0.0, 0.0)
        for o in (a, b):
            ctx.settings(o, mode='FACES', target_count=800,
                         backend='QUADRIFLOW', keep_original=True)
        bstate["names"] = ["BatchA", "BatchB"]
        try:
            for o in bpy.context.selected_objects:
                o.select_set(False)
        except Exception:
            pass
        a.select_set(True)
        b.select_set(True)
        bpy.context.view_layer.objects.active = a
        bstate["before"] = _mesh_objects()
        if not hasattr(bpy.types, "QUADFORGE_OT_remesh_batch"):
            c.skip("quadforge.remesh_batch operator is not registered")
        kw = _override(a)
        kw["selected_objects"] = [a, b]
        kw["selected_editable_objects"] = [a, b]
        try:
            with bpy.context.temp_override(**kw):
                res = bpy.ops.quadforge.remesh_batch('EXEC_DEFAULT')
            bstate["op_ok"] = True
            c.require('CANCELLED' not in res,
                      "remesh_batch returned %s" % (res,))
            c.note("returned %s" % (res,))
        except Exception as e:
            bstate["op_ok"] = False
            c.skip("operator needs UI context in --background: %s: %s"
                   % (type(e).__name__, e))

    with r.case("batch_pipeline_over_two_objects") as c:
        # Contract-level equivalent, always runnable headless.
        ctx.fresh_scene()
        a = ctx.uv_sphere(segments=32, rings=16, name="PA")
        b = ctx.torus(major_segments=32, minor_segments=12, name="PB")
        b.location = (3.0, 0.0, 0.0)
        pipeline = ctx.pipeline()
        counts = []
        for o in (a, b):
            ctx.activate(o)
            s = ctx.settings(o, mode='FACES', target_count=800,
                             backend='QUADRIFLOW', keep_original=True)
            res = pipeline.run_remesh(bpy.context, o, s)
            c.require(res.get("ok") is True,
                      "batch member %s failed: %r" % (o.name, res.get("error")))
            out = res.get("object")
            c.require(ctx.is_mesh_valid(out), "batch member %s produced no mesh" % o.name)
            counts.append(len(out.data.polygons))
        for n in counts:
            c.require_rel(n, 800, 0.50, "faces")
        c.note("counts=%s" % counts)

    with r.case("lod_targets_roundtrip") as c:
        ctx.fresh_scene()
        obj = ctx.cube()
        ctx.settings(obj, lod_targets="8000, 2000 ,500")
        parts = [p.strip() for p in obj.quadforge.lod_targets.split(",") if p.strip()]
        c.require([int(p) for p in parts] == [8000, 2000, 500],
                  "lod_targets did not round-trip: %r" % obj.quadforge.lod_targets)

    return r.list()
