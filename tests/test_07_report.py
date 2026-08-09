"""core.report.mesh_report metrics."""

import bpy

REQUIRED_KEYS = [
    "faces", "quads", "tris", "ngons", "quad_pct",
    "poles_3", "poles_5plus", "non_manifold_edges",
    "symmetry_error_x", "symmetry_error_y", "symmetry_error_z", "area",
]


def run(ctx):
    r = ctx.results()

    state = {}
    with r.case("mesh_report_keys") as c:
        ctx.fresh_scene()
        cube = ctx.cube(size=2.0)
        state["cube"] = cube
        report = ctx.report_mod()
        rep = report.mesh_report(cube)
        state["rep"] = rep
        c.require(isinstance(rep, dict), "mesh_report returned %r, want dict" % type(rep))
        missing = [k for k in REQUIRED_KEYS if k not in rep]
        c.require(not missing,
                  "mesh_report missing keys %s (got %s)" % (missing, sorted(rep)))
        c.note("%d keys" % len(rep))

    rep = state.get("rep") or {}

    with r.case("cube_faces") as c:
        c.require(rep, "no report (mesh_report_keys failed)")
        c.require(rep.get("faces") == 6, "faces=%r, want 6" % rep.get("faces"))
        c.require(rep.get("quads") == 6, "quads=%r, want 6" % rep.get("quads"))
        c.require(rep.get("tris") == 0, "tris=%r, want 0" % rep.get("tris"))
        c.require(rep.get("ngons") == 0, "ngons=%r, want 0" % rep.get("ngons"))

    with r.case("cube_quad_pct") as c:
        c.require(rep, "no report (mesh_report_keys failed)")
        c.require(abs(float(rep.get("quad_pct", -1)) - 100.0) < 1e-6,
                  "quad_pct=%r, want 100" % rep.get("quad_pct"))

    with r.case("cube_poles") as c:
        c.require(rep, "no report (mesh_report_keys failed)")
        c.require(rep.get("poles_3") == 8,
                  "poles_3=%r, want 8 (every cube corner is a 3-pole)"
                  % rep.get("poles_3"))
        c.require(rep.get("poles_5plus") == 0,
                  "poles_5plus=%r, want 0" % rep.get("poles_5plus"))

    with r.case("cube_manifold") as c:
        c.require(rep, "no report (mesh_report_keys failed)")
        c.require(rep.get("non_manifold_edges") == 0,
                  "non_manifold_edges=%r, want 0" % rep.get("non_manifold_edges"))

    with r.case("cube_symmetry") as c:
        c.require(rep, "no report (mesh_report_keys failed)")
        for axis in "xyz":
            key = "symmetry_error_" + axis
            val = float(rep.get(key, 1.0))
            c.require(val < 1e-6, "%s=%.3g, want ~0 for a cube" % (key, val))
        c.note("x=%.2g y=%.2g z=%.2g" % tuple(
            float(rep["symmetry_error_" + a]) for a in "xyz"))

    with r.case("cube_area") as c:
        c.require(rep, "no report (mesh_report_keys failed)")
        c.require_close(float(rep.get("area", 0.0)), 24.0, 1e-4, "area")
        c.note("area=%.4f" % float(rep["area"]))

    # ------------------------------------------- second fixture: uv sphere
    with r.case("sphere_report") as c:
        ctx.fresh_scene()
        sphere = ctx.uv_sphere(segments=32, rings=16)
        rep2 = ctx.report_mod().mesh_report(sphere)
        fs = ctx.face_stats(sphere)
        c.require(rep2.get("faces") == fs["faces"],
                  "faces=%r, mesh has %d" % (rep2.get("faces"), fs["faces"]))
        c.require(rep2.get("tris") == fs["tris"],
                  "tris=%r, mesh has %d (the two pole rings)"
                  % (rep2.get("tris"), fs["tris"]))
        c.require(abs(float(rep2["quad_pct"]) - fs["quad_pct"]) < 0.01,
                  "quad_pct=%r, computed %.3f" % (rep2.get("quad_pct"), fs["quad_pct"]))
        c.require(rep2.get("non_manifold_edges") == 0,
                  "sphere reported %r non-manifold edges" % rep2.get("non_manifold_edges"))
        # the two poles are 32-valence, every other vertex is a regular 4-pole
        c.require(rep2.get("poles_5plus") == 2,
                  "poles_5plus=%r, want 2 (the UV-sphere poles)" % rep2.get("poles_5plus"))
        c.require_close(float(rep2["area"]), 4.0 * 3.141592653589793, 0.6, "area")
        c.note("faces=%d tris=%d area=%.3f"
               % (rep2["faces"], rep2["tris"], rep2["area"]))

    with r.case("non_manifold_detected") as c:
        # three quads fanning off one shared edge -> unambiguously non-manifold
        ctx.fresh_scene()
        verts = [(0, 0, 0), (1, 0, 0),
                 (1, 1, 0), (0, 1, 0),
                 (1, -1, 0), (0, -1, 0),
                 (1, 0, 1), (0, 0, 1)]
        faces = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 6, 7]]
        me = bpy.data.meshes.new("Fan")
        me.from_pydata(verts, [], faces)
        me.update()
        obj = bpy.data.objects.new("Fan", me)
        ctx.link(obj)
        rep3 = ctx.report_mod().mesh_report(obj)
        c.require(rep3.get("faces") == 3, "faces=%r, want 3" % rep3.get("faces"))
        c.require(int(rep3.get("non_manifold_edges", 0)) >= 1,
                  "non_manifold_edges=%r on a 3-faces-share-one-edge fan, want >= 1"
                  % rep3.get("non_manifold_edges"))
        c.note("fan -> %s non-manifold edges" % rep3["non_manifold_edges"])

    with r.case("open_mesh_boundary") as c:
        ctx.fresh_scene()
        cube = ctx.cube(size=2.0)
        me = cube.data
        verts = [tuple(v.co) for v in me.vertices]
        polys = [list(p.vertices) for p in me.polygons][1:]
        me.clear_geometry()
        me.from_pydata(verts, [], polys)
        me.update()
        rep5 = ctx.report_mod().mesh_report(cube)
        c.require(rep5.get("faces") == 5, "faces=%r, want 5" % rep5.get("faces"))
        got = int(rep5.get("non_manifold_edges", -1))
        if got == 0:
            c.skip("implementation excludes boundary edges from non_manifold_edges")
        c.require(got == 4,
                  "non_manifold_edges=%d for an open cube, want 4 boundary edges "
                  "(or 0 if boundaries are excluded)" % got)
        c.note("open cube -> %d boundary edges" % got)

    with r.case("asymmetric_detected") as c:
        ctx.fresh_scene()
        cube = ctx.cube(size=2.0)
        cube.data.vertices[0].co.x += 0.5
        rep4 = ctx.report_mod().mesh_report(cube)
        c.require(float(rep4.get("symmetry_error_x", 0.0)) > 0.1,
                  "symmetry_error_x=%r on a deliberately skewed cube, want > 0.1"
                  % rep4.get("symmetry_error_x"))
        c.note("symmetry_error_x=%.3f" % float(rep4["symmetry_error_x"]))

    return r.list()
