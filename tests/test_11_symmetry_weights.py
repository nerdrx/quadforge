"""Regression: exact symmetry must not leak the kept half's weights onto the
mirrored half.

The bmesh mirror duplicates deform layers un-swapped, so the mirrored side
arrived with the source side's groups still attached; transfer then added the
correct groups on top, leaving verts weighted 1.0 to BOTH sides — posed, the
mirrored limb only moved halfway (found on a real avatar's fingers)."""

import bpy
import numpy as np


def run(ctx):
    r = ctx.results()

    from quadforge import pipeline

    with r.case("no_double_side_weights") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=48, rings=24)
        me = obj.data
        gl = obj.vertex_groups.new(name="Side_L")
        gr = obj.vertex_groups.new(name="Side_R")
        for v in me.vertices:
            if v.co.x > 0.02:
                gl.add([v.index], 1.0, 'REPLACE')
            elif v.co.x < -0.02:
                gr.add([v.index], 1.0, 'REPLACE')
        s = ctx.settings(obj, target_count=1500, symmetry_x=True,
                         exact_symmetry=True, keep_original=False)
        res = pipeline.run_remesh(bpy.context, obj, s)
        c.require(res.get("ok"), "run failed: %r" % (res.get("error"),))
        new = res["object"]
        names = {g.index: g.name for g in new.vertex_groups}
        c.require("Side_L" in names.values() and "Side_R" in names.values(),
                  "groups missing: %s" % sorted(names.values()))
        both = 0
        wrong_side = 0
        for v in new.data.vertices:
            w = {}
            for ge in v.groups:
                w[names.get(ge.group, "?")] = ge.weight
            if w.get("Side_L", 0.0) > 0.5 and w.get("Side_R", 0.0) > 0.5:
                both += 1
            if v.co.x > 0.15 and w.get("Side_R", 0.0) > 0.5:
                wrong_side += 1
            if v.co.x < -0.15 and w.get("Side_L", 0.0) > 0.5:
                wrong_side += 1
        c.require(both == 0, "%d verts weighted >0.5 to BOTH sides" % both)
        c.require(wrong_side == 0, "%d verts carry the wrong side's weight" % wrong_side)
        c.note("verts=%d clean" % len(new.data.vertices))

    return r.list()
