"""Data preservation: vertex groups, shape keys, materials, UVs."""

import bpy

TARGET = 3000
SHAPE_KEYS = {"qf_grow": 0.25, "qf_shift": 0.15}   # name -> expected max offset


def run(ctx):
    r = ctx.results()

    state = {}
    with r.case("remesh_ok") as c:
        ctx.fresh_scene()
        src = ctx.rigged_sphere()
        ctx.activate(src)
        state["src"] = src
        s = ctx.settings(src, mode='FACES', target_count=TARGET,
                         backend='QUADRIFLOW',
                         preserve_uvs=True, preserve_weights=True,
                         preserve_shape_keys=True, preserve_materials=True)
        res = ctx.pipeline().run_remesh(bpy.context, src, s)
        state["res"] = res
        c.require(res.get("ok") is True, "run failed: %r" % (res.get("error"),))
        out = res.get("object")
        c.require(ctx.is_mesh_valid(out), "no result mesh")
        state["out"] = out
        c.note("faces=%d" % len(out.data.polygons))

    out = state.get("out")

    # -------------------------------------------------------- vertex groups
    with r.case("vertex_group_survives") as c:
        c.require(out is not None, "no result mesh (remesh_ok failed)")
        names = [g.name for g in out.vertex_groups]
        c.require("qf_grad" in names,
                  "vertex group 'qf_grad' missing (groups: %s)" % names)
        vg = out.vertex_groups["qf_grad"]
        # the source gradient is w = (z + 1) / 2 on a unit sphere
        me = out.data
        top = max(me.vertices, key=lambda v: v.co.z)
        expected = (top.co.z + 1.0) / 2.0
        got = None
        for ge in top.groups:
            if ge.group == vg.index:
                got = ge.weight
                break
        c.require(got is not None,
                  "top vertex (z=%.3f) has no weight in 'qf_grad'" % top.co.z)
        c.require(abs(got - expected) <= 0.1,
                  "top-pole weight %.3f, source weight there is %.3f (tol 0.1)"
                  % (got, expected))
        c.note("top z=%.3f w=%.3f (want %.3f)" % (top.co.z, got, expected))

    with r.case("vertex_group_gradient_shape") as c:
        c.require(out is not None, "no result mesh (remesh_ok failed)")
        c.require("qf_grad" in out.vertex_groups, "vertex group missing")
        vg = out.vertex_groups["qf_grad"]
        worst = 0.0
        n = 0
        for v in out.data.vertices:
            w = None
            for ge in v.groups:
                if ge.group == vg.index:
                    w = ge.weight
                    break
            if w is None:
                continue
            n += 1
            worst = max(worst, abs(w - (v.co.z + 1.0) / 2.0))
        c.require(n > 0, "no vertex carries the group")
        c.require(n >= 0.9 * len(out.data.vertices),
                  "only %d/%d vertices carry the group" % (n, len(out.data.vertices)))
        c.require(worst <= 0.15,
                  "worst gradient deviation %.3f over %d verts (tol 0.15)" % (worst, n))
        c.note("n=%d worst=%.3f" % (n, worst))

    # ------------------------------------------------------------ shape keys
    with r.case("shape_keys_survive") as c:
        c.require(out is not None, "no result mesh (remesh_ok failed)")
        keys = out.data.shape_keys
        c.require(keys is not None, "result object has no shape keys at all")
        names = [k.name for k in keys.key_blocks]
        for want in SHAPE_KEYS:
            c.require(want in names, "shape key %r missing (have %s)" % (want, names))
        c.require(len(keys.key_blocks) >= 3,
                  "expected Basis + 2 keys, got %s" % names)
        c.note("keys=%s" % names)

    for key_name, expected in SHAPE_KEYS.items():
        with r.case("shape_key_magnitude_" + key_name) as c:
            c.require(out is not None, "no result mesh (remesh_ok failed)")
            got = ctx.max_shape_key_offset(out, key_name)
            c.require(got is not None, "shape key %r missing" % key_name)
            c.require_rel(got, expected, 0.20, "max_offset[%s]" % key_name)

    with r.case("shape_key_activation") as c:
        c.require(out is not None, "no result mesh (remesh_ok failed)")
        keys = out.data.shape_keys
        c.require(keys is not None, "no shape keys")
        c.require("qf_shift" in keys.key_blocks, "qf_shift missing")
        for kb in keys.key_blocks:
            kb.value = 0.0
        keys.key_blocks["qf_shift"].value = 1.0
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        ev = out.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
        try:
            base = ctx.verts_np(out)
            n = len(me.vertices)
            c.require(n == len(base),
                      "evaluated mesh has %d verts, base has %d" % (n, len(base)))
            import numpy as np
            a = np.empty(n * 3, dtype="f8")
            me.vertices.foreach_get("co", a)
            d = a.reshape(n, 3) - base
            got = float(np.max(np.sqrt((d * d).sum(axis=1))))
        finally:
            bpy.data.meshes.remove(me)
            for kb in keys.key_blocks:
                kb.value = 0.0
        c.require_rel(got, SHAPE_KEYS["qf_shift"], 0.20, "evaluated_offset")

    # ------------------------------------------------------------ materials
    with r.case("materials_survive") as c:
        c.require(out is not None, "no result mesh (remesh_ok failed)")
        mats = [m.name if m else None for m in out.data.materials]
        c.require(len(mats) == 2,
                  "expected 2 material slots, got %d: %s" % (len(mats), mats))
        c.require("qf_mat_low" in mats and "qf_mat_high" in mats,
                  "material names not preserved: %s" % mats)
        c.note("mats=%s" % mats)

    with r.case("material_boundary_at_equator") as c:
        c.require(out is not None, "no result mesh (remesh_ok failed)")
        mats = [m.name if m else None for m in out.data.materials]
        c.require(len(mats) == 2, "need 2 material slots to test the boundary")
        hi_idx = mats.index("qf_mat_high")
        lo_idx = mats.index("qf_mat_low")
        clear = 0
        right = 0
        band = 0.0
        for p in out.data.polygons:
            z = p.center.z
            if abs(z) <= 0.1:
                continue
            clear += 1
            want = hi_idx if z > 0 else lo_idx
            if p.material_index == want:
                right += 1
            else:
                band = max(band, abs(z))
        c.require(clear > 0, "no polygons outside the equatorial band")
        ratio = right / float(clear)
        c.require(ratio >= 0.95,
                  "%d/%d (%.1f%%) polygons on the correct side; worst misassignment "
                  "at |z|=%.3f" % (right, clear, ratio * 100.0, band))
        c.note("%.1f%% correct over %d polys" % (ratio * 100.0, clear))

    # ----------------------------------------------------------------- UVs
    with r.case("uvs_survive") as c:
        c.require(out is not None, "no result mesh (remesh_ok failed)")
        me = out.data
        c.require(len(me.uv_layers) >= 1,
                  "no UV layer on the result (the source sphere had one)")
        uv = me.uv_layers.active or me.uv_layers[0]
        nonzero = sum(1 for d in uv.data if abs(d.uv[0]) > 1e-6 or abs(d.uv[1]) > 1e-6)
        c.require(nonzero > 0,
                  "UV layer %r exists but all %d loops are at (0,0)"
                  % (uv.name, len(uv.data)))
        c.require(nonzero >= 0.5 * len(uv.data),
                  "only %d/%d UV loops are non-zero" % (nonzero, len(uv.data)))
        c.note("layer=%r nonzero=%d/%d" % (uv.name, nonzero, len(uv.data)))

    # ------------------------------------------------------- transfer.apply
    with r.case("transfer_api_shape") as c:
        transfer = ctx.imp("quadforge.core.transfer")
        c.require(hasattr(transfer, "capture"), "transfer.capture missing")
        c.require(hasattr(transfer, "apply"), "transfer.apply missing")
        ctx.fresh_scene()
        src = ctx.rigged_sphere(segments=24, rings=12)
        snap = transfer.capture(src)
        c.require(snap is not None, "capture returned None")
        dst = ctx.uv_sphere(segments=20, rings=10, name="Dst")
        report = transfer.apply(snap, dst, ctx.settings(dst))
        c.require(isinstance(report, dict),
                  "transfer.apply returned %r, want dict" % type(report))
        for key in ("uvs", "weights", "shape_keys", "materials"):
            c.require(key in report,
                      "apply report missing %r (got %s)" % (key, sorted(report)))
        c.require(int(report["weights"]) >= 1, "weights=%s, expected >=1" % report["weights"])
        c.require(int(report["shape_keys"]) >= 2,
                  "shape_keys=%s, expected >=2" % report["shape_keys"])
        c.require(int(report["materials"]) >= 2,
                  "materials=%s, expected >=2" % report["materials"])
        c.require(bool(report["uvs"]), "uvs reported as not transferred")
        c.note(str({k: report[k] for k in ("uvs", "weights", "shape_keys", "materials")}))

    return r.list()
