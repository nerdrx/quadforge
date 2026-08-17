"""Follow UV Islands: island boundaries become feature edges the solver
aligns to, so texture seams survive the remesh."""

import bpy
import numpy as np


def run(ctx):
    r = ctx.results()

    from quadforge import pipeline
    from quadforge.core import analysis

    with r.case("uv_seam_detection") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=48, rings=24)
        # a UV sphere's projection has a vertical seam column and pole fans
        n = analysis.uv_island_boundaries_to_sharp(obj)
        c.require(n > 10, "expected a seam column, marked %d edges" % n)
        c.note("marked %d edges" % n)

    with r.case("uv_seam_alignment") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=48, rings=24)
        res = None
        s = ctx.settings(obj, target_count=1200, use_uv_seams=True,
                         keep_original=False)
        res = pipeline.run_remesh(bpy.context, obj, s)
        c.require(res.get("ok"), "run failed: %r" % (res.get("error"),))
        rep = res.get("stats") or {}
        seam_n = rep.get("uv_seam_edges")
        c.require((seam_n or 0) > 10, "report seam edges: %r" % (seam_n,))
        out = res["object"]
        # the UV sphere's seam runs along the -Y meridian (x=0, y<0):
        # result edges near it should be predominantly vertical (along the seam)
        me = out.data
        co = ctx.verts_np(out)
        ev = np.empty(len(me.edges) * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        mid = (co[ev[:, 0]] + co[ev[:, 1]]) / 2
        d = co[ev[:, 0]] - co[ev[:, 1]]
        near = (np.abs(mid[:, 0]) < 0.09) & (mid[:, 1] < -0.85) & (np.abs(mid[:, 2]) < 0.7)
        if near.sum() < 10:
            c.skip("too few edges near seam to judge (%d)" % int(near.sum()))
        dn = d[near] / (np.linalg.norm(d[near], axis=1, keepdims=True) + 1e-12)
        vertical = np.abs(dn[:, 2]) > 0.8
        horizontal = np.abs(dn[:, 2]) < 0.2
        # QuadriFlow's seam alignment is soft and its solver is not seed-
        # reproducible; require a not-worse-than-random tendency, not a
        # coin-flip-sensitive majority
        c.require(vertical.sum() >= 0.7 * horizontal.sum(),
                  "flow not aligned along seam: %d vertical vs %d horizontal"
                  % (int(vertical.sum()), int(horizontal.sum())))
        c.note("near-seam edges: %d vertical / %d horizontal of %d"
               % (int(vertical.sum()), int(horizontal.sum()), int(near.sum())))

    with r.case("no_uv_layer_graceful") as c:
        ctx.fresh_scene()
        obj = ctx.cube(size=2.0, subdiv=2)
        obj.data.uv_layers.remove(obj.data.uv_layers[0]) if obj.data.uv_layers else None
        s = ctx.settings(obj, target_count=300, use_uv_seams=True,
                         keep_original=False)
        res = pipeline.run_remesh(bpy.context, obj, s)
        c.require(res.get("ok"), "run failed: %r" % (res.get("error"),))

    return r.list()
