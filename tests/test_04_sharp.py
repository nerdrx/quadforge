"""Hard-edge detection and feature preservation on a subdivided cube."""

import math

import bpy
from mathutils import Vector

SUBDIV = 4          # simple subsurf: 6 * 4^4 / 4 -> 1536 quads, corners kept
TARGET = 1500
CORNER_TOL = 0.02
EDGE_TOL = 0.035
ANGLE_TOL_DEG = 10.0
FEATURE_RATIO = 0.90

CORNERS = [Vector((x, y, z))
           for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]

# (axis, other_axis_a, value_a, other_axis_b, value_b) for the 12 cube edges
CUBE_EDGES = []
for _a in (0, 1, 2):
    _o = [i for i in (0, 1, 2) if i != _a]
    for _va in (-1.0, 1.0):
        for _vb in (-1.0, 1.0):
            CUBE_EDGES.append((_a, _o[0], _va, _o[1], _vb))


def _dihedral_over(obj, angle):
    """Independent count of edges whose face angle exceeds `angle` (radians)."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n = 0
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > angle:
            n += 1
    bm.free()
    return n


def run(ctx):
    r = ctx.results()

    # ------------------------------------------------ analysis.mark_hard_edges
    with r.case("mark_hard_edges_count") as c:
        ctx.fresh_scene()
        obj = ctx.cube(size=2.0, subdiv=SUBDIV)
        ctx.activate(obj)
        s = ctx.settings(obj, detect_hard_edges=True,
                         hard_edge_angle=math.radians(40.0),
                         use_marked_sharp=False, use_materials=False)
        analysis = ctx.imp("quadforge.core.analysis")
        expected = _dihedral_over(obj, math.radians(40.0))
        c.require(expected > 0, "fixture has no >40deg edges (bad test cube)")
        got = analysis.mark_hard_edges(obj, s)
        c.require(isinstance(got, int), "mark_hard_edges returned %r, want int"
                  % type(got))
        slack = max(2, int(0.05 * expected))
        c.require(abs(got - expected) <= slack,
                  "marked %d hard edges, expected ~%d (+-%d)" % (got, expected, slack))
        sharp = obj.data.attributes.get("sharp_edge")
        if sharp is not None:
            marked = sum(1 for d in sharp.data if d.value)
            c.require(marked >= expected - slack,
                      "'sharp_edge' attribute has %d marked, expected ~%d"
                      % (marked, expected))
            c.note("attr marked=%d" % marked)
        c.note("returned=%d expected=%d" % (got, expected))

    # ---------------------------------------------------------------- remesh
    state = {}
    with r.case("remesh_ok") as c:
        ctx.fresh_scene()
        obj = ctx.cube(size=2.0, subdiv=SUBDIV)
        ctx.activate(obj)
        s = ctx.settings(obj, mode='FACES', target_count=TARGET,
                         backend='QUADRIFLOW', detect_hard_edges=True,
                         hard_edge_angle=math.radians(40.0),
                         preserve_boundaries=True)
        res = ctx.pipeline().run_remesh(bpy.context, obj, s)
        state["res"] = res
        c.require(res.get("ok") is True, "run failed: %r" % (res.get("error"),))
        out = res.get("object")
        c.require(ctx.is_mesh_valid(out), "no result mesh")
        state["out"] = out
        c.note("faces=%d quad_pct=%.1f%%"
               % (len(out.data.polygons), ctx.face_stats(out)["quad_pct"]))

    out = state.get("out")

    with r.case("corners_preserved") as c:
        c.require(out is not None, "no result mesh (remesh_ok failed)")
        co = ctx.verts_np(out)
        kd = ctx.kdtree(co)
        misses = []
        for corner in CORNERS:
            _, _, dist = kd.find(corner)
            if dist is None or dist > CORNER_TOL:
                misses.append("%s off by %s" % (tuple(corner),
                                                "inf" if dist is None else "%.4f" % dist))
        c.require(not misses,
                  "%d/8 cube corners not reproduced within %.3f: %s"
                  % (len(misses), CORNER_TOL, "; ".join(misses)))
        c.note("all 8 corners within %.3f" % CORNER_TOL)

    with r.case("feature_edges_axis_aligned") as c:
        c.require(out is not None, "no result mesh (remesh_ok failed)")
        me = out.data
        verts = [v.co for v in me.vertices]
        cos_limit = math.cos(math.radians(ANGLE_TOL_DEG))
        total = 0
        aligned = 0
        worst = 0.0
        for e in me.edges:
            a = verts[e.vertices[0]]
            b = verts[e.vertices[1]]
            for axis, oa, va, ob, vb in CUBE_EDGES:
                if (abs(a[oa] - va) <= EDGE_TOL and abs(b[oa] - va) <= EDGE_TOL
                        and abs(a[ob] - vb) <= EDGE_TOL and abs(b[ob] - vb) <= EDGE_TOL):
                    d = b - a
                    if d.length < 1e-9:
                        break
                    d = d.normalized()
                    total += 1
                    dot = abs(d[axis])
                    if dot >= cos_limit:
                        aligned += 1
                    else:
                        worst = max(worst, math.degrees(math.acos(min(1.0, dot))))
                    break
        c.require(total >= 8,
                  "only %d edges lie on the cube's feature lines - "
                  "features were not preserved at all" % total)
        ratio = aligned / float(total)
        c.require(ratio >= FEATURE_RATIO,
                  "%d/%d (%.1f%%) feature edges within %.0f deg of their axis "
                  "(worst off-axis %.1f deg), need %.0f%%"
                  % (aligned, total, ratio * 100.0, ANGLE_TOL_DEG, worst,
                     FEATURE_RATIO * 100.0))
        c.note("%d/%d feature edges aligned (%.1f%%)" % (aligned, total, ratio * 100.0))

    with r.case("bbox_preserved") as c:
        c.require(out is not None, "no result mesh (remesh_ok failed)")
        co = ctx.verts_np(out)
        lo = co.min(axis=0)
        hi = co.max(axis=0)
        for i, ax in enumerate("XYZ"):
            c.require(abs(lo[i] + 1.0) <= CORNER_TOL,
                      "min%s=%.4f, want -1.0" % (ax, lo[i]))
            c.require(abs(hi[i] - 1.0) <= CORNER_TOL,
                      "max%s=%.4f, want 1.0" % (ax, hi[i]))
        c.note("bbox %s..%s" % (tuple(round(float(x), 4) for x in lo),
                               tuple(round(float(x), 4) for x in hi)))

    return r.list()
