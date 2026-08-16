"""Edge cases: pathological inputs, hostile topology, extreme scales and
setting combinations.

Contract under test: run_remesh NEVER raises, hangs or crashes — every case
either succeeds with a valid mesh or returns {'ok': False, 'error': <str>}.
"""

import math

import bmesh
import bpy
import numpy as np


def _run(ctx, obj, **over):
    from quadforge import pipeline
    over.setdefault("keep_original", False)
    s = ctx.settings(obj, **over)
    return pipeline.run_remesh(bpy.context, obj, s)


def _ok_or_clean(c, res, want_faces=True):
    """Common invariant: success with a real mesh, or a clean string error."""
    if res.get("ok"):
        out = res.get("object")
        c.require(out is not None, "ok=True but no object")
        if want_faces:
            c.require(len(out.data.polygons) > 0, "ok=True but empty mesh")
        return out
    err = res.get("error")
    c.require(isinstance(err, str) and len(err) > 3,
              "not a clean error: %r" % (err,))
    return None


def _mesh_obj(name, verts, faces, edges=()):
    me = bpy.data.meshes.new(name)
    me.from_pydata(list(verts), list(edges), list(faces))
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def run(ctx):
    r = ctx.results()

    # ---------------- degenerate inputs ----------------------------------

    with r.case("empty_mesh") as c:
        ctx.fresh_scene()
        obj = _mesh_obj("Empty", [], [])
        res = _run(ctx, obj, target_count=100)
        c.require(not res.get("ok"), "empty mesh should fail cleanly")
        _ok_or_clean(c, res)

    with r.case("point_cloud") as c:
        ctx.fresh_scene()
        obj = _mesh_obj("Points", [(i * 0.1, 0, 0) for i in range(20)], [])
        res = _run(ctx, obj, target_count=100)
        c.require(not res.get("ok"), "point cloud should fail cleanly")
        _ok_or_clean(c, res)

    with r.case("wire_only") as c:
        ctx.fresh_scene()
        verts = [(math.cos(a), math.sin(a), 0) for a in np.linspace(0, 6.2, 12)]
        obj = _mesh_obj("Wire", verts, [], edges=[(i, (i + 1) % 12) for i in range(12)])
        res = _run(ctx, obj, target_count=100)
        c.require(not res.get("ok"), "wire mesh should fail cleanly")
        _ok_or_clean(c, res)

    with r.case("single_triangle") as c:
        ctx.fresh_scene()
        obj = _mesh_obj("Tri", [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
        res = _run(ctx, obj, target_count=50)
        _ok_or_clean(c, res)
        c.note("ok=%s" % res.get("ok"))

    with r.case("single_quad") as c:
        ctx.fresh_scene()
        obj = _mesh_obj("Quad", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                        [(0, 1, 2, 3)])
        res = _run(ctx, obj, target_count=64)
        out = _ok_or_clean(c, res)
        if out is not None:
            c.require(len(out.data.polygons) >= 4, "flat quad barely remeshed")

    with r.case("huge_ngon_disc") as c:
        ctx.fresh_scene()
        verts = [(math.cos(a), math.sin(a), 0.0) for a in np.linspace(0, 2 * math.pi, 96, endpoint=False)]
        obj = _mesh_obj("Disc", verts, [tuple(range(96))])
        res = _run(ctx, obj, target_count=200)
        out = _ok_or_clean(c, res)
        if out is not None:
            st = ctx.face_stats(out)
            c.note("faces=%d quad_pct=%.1f" % (st["faces"], st["quad_pct"]))

    with r.case("flat_grid_2d") as c:
        ctx.fresh_scene()
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=24, y_subdivisions=24)
        obj = bpy.context.active_object
        res = _run(ctx, obj, target_count=300, preserve_boundaries=True)
        out = _ok_or_clean(c, res)
        if out is not None:
            co = ctx.verts_np(out)
            c.require(float(np.abs(co[:, 2]).max()) < 1e-4,
                      "flat grid gained thickness: max |z| %.5f" % float(np.abs(co[:, 2]).max()))
            # boundary should still be the unit square
            c.require(abs(float(co[:, 0].max()) - 1.0) < 0.05, "boundary lost")

    # ---------------- hostile topology -----------------------------------

    with r.case("nonmanifold_T_edge") as c:
        ctx.fresh_scene()
        obj = _mesh_obj("Tee",
                        [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1),
                         (0, 1, 0), (1, 1, 0), (0, -1, 0.5), (1, -1, 0.5)],
                        [(0, 1, 3, 2), (0, 1, 5, 4), (0, 1, 7, 6)])
        res = _run(ctx, obj, target_count=50)
        _ok_or_clean(c, res)
        c.note("ok=%s err=%s" % (res.get("ok"), str(res.get("error"))[:60]))

    with r.case("duplicate_unwelded_shells") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.from_mesh(obj.data)  # exact coincident copy, unwelded
        bm.to_mesh(obj.data)
        obj.data.update()
        res = _run(ctx, obj, target_count=800)
        _ok_or_clean(c, res)
        c.note("ok=%s" % res.get("ok"))

    with r.case("zero_area_faces") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        me = obj.data
        co = ctx.verts_np(obj)
        co[10:16] = co[10]  # collapse a few verts onto one point
        me.vertices.foreach_set("co", co.ravel())
        me.update()
        res = _run(ctx, obj, target_count=500)
        _ok_or_clean(c, res)

    with r.case("inconsistent_normals_input") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.reverse_faces(bm, faces=[f for f in bm.faces if f.calc_center_median().x > 0])
        bm.to_mesh(obj.data)
        res = _run(ctx, obj, target_count=500)
        out = _ok_or_clean(c, res)
        if out is not None:
            st = ctx.face_stats(out)
            c.note("faces=%d quad_pct=%.1f" % (st["faces"], st["quad_pct"]))

    with r.case("mobius_strip") as c:
        ctx.fresh_scene()
        n = 48
        verts, faces = [], []
        for i in range(n):
            a = 2 * math.pi * i / n
            for s in (-0.2, 0.2):
                t = a / 2
                x = (1 + s * math.cos(t)) * math.cos(a)
                y = (1 + s * math.cos(t)) * math.sin(a)
                z = s * math.sin(t)
                verts.append((x, y, z))
        for i in range(n):
            j = (i + 1) % n
            if j == 0:  # the twist: reconnect flipped
                faces.append((2 * i, 2 * i + 1, 1, 0))
            else:
                faces.append((2 * i, 2 * i + 1, 2 * j + 1, 2 * j))
        obj = _mesh_obj("Mobius", verts, faces)
        res = _run(ctx, obj, target_count=200)
        _ok_or_clean(c, res)
        c.note("ok=%s (non-orientable input)" % res.get("ok"))

    with r.case("self_intersecting_cubes") as c:
        ctx.fresh_scene()
        obj = ctx.cube(size=2.0)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.create_cube(bm, size=2.0,
                              matrix=__import__("mathutils").Matrix.Translation((0.7, 0.6, 0.5)))
        bm.to_mesh(obj.data)
        res = _run(ctx, obj, target_count=600)
        _ok_or_clean(c, res)

    with r.case("high_genus_torus_grid") as c:
        ctx.fresh_scene()
        objs = []
        for i in range(3):
            bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.2,
                                             major_segments=24, minor_segments=8,
                                             location=(i * 1.2, 0, 0),
                                             rotation=(0, math.radians(90 * (i % 2)), 0))
            objs.append(bpy.context.active_object)
        for o in bpy.context.view_layer.objects:
            o.select_set(o in objs)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
        res = _run(ctx, objs[0], target_count=1500)
        out = _ok_or_clean(c, res)
        if out is not None:
            c.require(ctx.non_manifold_edge_count(out) == 0, "genus mesh non-manifold")

    # ---------------- scale / transform extremes --------------------------

    with r.case("tiny_scale_1e4") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12, radius=1.0)
        co = ctx.verts_np(obj) * 1e-4
        obj.data.vertices.foreach_set("co", co.ravel())
        obj.data.update()
        res = _run(ctx, obj, target_count=400)
        out = _ok_or_clean(c, res)
        if out is not None:
            rad = float(np.linalg.norm(ctx.verts_np(out), axis=1).mean())
            c.require(abs(rad - 1e-4) < 3e-5, "scale not restored: r=%.2e" % rad)

    with r.case("huge_scale_1e4") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12, radius=1.0)
        co = ctx.verts_np(obj) * 1e4
        obj.data.vertices.foreach_set("co", co.ravel())
        obj.data.update()
        res = _run(ctx, obj, target_count=400)
        out = _ok_or_clean(c, res)
        if out is not None:
            rad = float(np.linalg.norm(ctx.verts_np(out), axis=1).mean())
            c.require(abs(rad - 1e4) / 1e4 < 0.05, "scale wrong: r=%.1f" % rad)

    with r.case("far_from_origin") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        co = ctx.verts_np(obj) + np.array([5000.0, -3000.0, 800.0])
        obj.data.vertices.foreach_set("co", co.ravel())
        obj.data.update()
        res = _run(ctx, obj, target_count=400)
        out = _ok_or_clean(c, res)
        if out is not None:
            center = ctx.verts_np(out).mean(0)
            c.require(np.linalg.norm(center - [5000, -3000, 800]) < 1.0,
                      "mesh moved: center %s" % np.round(center, 1))

    with r.case("rotated_nonuniform_scaled_object") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        obj.rotation_euler = (0.5, 0.3, 1.1)
        obj.scale = (2.0, 0.5, 1.3)
        res = _run(ctx, obj, target_count=500)
        out = _ok_or_clean(c, res)
        if out is not None:
            c.require(tuple(np.round(out.scale, 3)) == (2.0, 0.5, 1.3),
                      "object transform not preserved: %r" % (tuple(out.scale),))

    with r.case("negative_scale_object") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        obj.scale = (-1.0, 1.0, 1.0)
        res = _run(ctx, obj, target_count=400)
        _ok_or_clean(c, res)
        c.note("ok=%s" % res.get("ok"))

    # ---------------- setting extremes ------------------------------------

    with r.case("minimum_target_12") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        res = _run(ctx, obj, target_count=12)
        out = _ok_or_clean(c, res)
        if out is not None:
            c.note("faces=%d" % len(out.data.polygons))

    with r.case("upsample_ratio_3x") as c:
        ctx.fresh_scene()
        obj = ctx.cube(size=2.0)
        res = _run(ctx, obj, mode='RATIO', target_ratio=3.0)
        out = _ok_or_clean(c, res)
        if out is not None:
            c.require(len(out.data.polygons) >= 12,
                      "upsample produced %d faces" % len(out.data.polygons))

    with r.case("edge_length_larger_than_mesh") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        res = _run(ctx, obj, mode='EDGE', target_edge_length=50.0)
        _ok_or_clean(c, res)

    with r.case("xyz_exact_symmetry_combo") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=32, rings=16)
        res = _run(ctx, obj, target_count=1200, symmetry_x=True, symmetry_y=True,
                   symmetry_z=True, exact_symmetry=True)
        out = _ok_or_clean(c, res)
        if out is not None:
            for ax in range(3):
                e = ctx.symmetry_error(out, ax)
                c.require(e < 1e-5, "axis %d error %.2e" % (ax, e))

    with r.case("everything_on_at_once") as c:
        ctx.fresh_scene()
        obj = ctx.suzanne(subdiv=1)
        curve = ctx.bezier_circle(radius=0.8)
        coll = bpy.data.collections.new("G")
        bpy.context.scene.collection.children.link(coll)
        coll.objects.link(curve)
        me = obj.data
        attr = me.attributes.new("qf_density", 'FLOAT', 'POINT')
        attr.data.foreach_set("value", np.random.default_rng(1).uniform(
            0.5, 1.5, len(me.vertices)).astype(np.float32))
        s = obj.quadforge
        s.guide_collection = coll
        res = _run(ctx, obj, target_count=2000, strict_count=True,
                   adaptive_size=70.0, use_paint_density=True,
                   detect_hard_edges=True, use_materials=True, use_guides=True,
                   symmetry_x=True, exact_symmetry=True, preserve_boundaries=True)
        out = _ok_or_clean(c, res)
        if out is not None:
            c.require(ctx.symmetry_error(out, 0) < 1e-5, "symmetry broke under combo")

    with r.case("paint_density_without_attribute") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        res = _run(ctx, obj, target_count=400, use_paint_density=True)
        _ok_or_clean(c, res)

    with r.case("guides_with_junk_collection") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        coll = bpy.data.collections.new("Junk")
        bpy.context.scene.collection.children.link(coll)
        cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
        empty = bpy.data.objects.new("Empty", None)
        for o in (cam, empty):
            coll.objects.link(o)
        s = obj.quadforge
        s.guide_collection = coll
        res = _run(ctx, obj, target_count=400, use_guides=True)
        _ok_or_clean(c, res)

    with r.case("junk_lod_targets_parse") as c:
        from quadforge.ops.lods import parse_lod_targets
        for junk in ("", "abc", "-5", "0", "100,,200", "1e4", " 500 , 200 "):
            try:
                got = parse_lod_targets(junk)
                c.require(all(isinstance(x, int) and x > 0 for x in got),
                          "junk %r parsed to %r" % (junk, got))
            except Exception as exc:
                c.require(isinstance(exc, (ValueError,)),
                          "junk %r raised %s" % (junk, type(exc).__name__))
        c.note("parser sane")

    with r.case("double_remesh_idempotent") as c:
        ctx.fresh_scene()
        obj = ctx.suzanne(subdiv=1)
        res1 = _run(ctx, obj, target_count=1500)
        c.require(res1.get("ok"), "first remesh failed: %r" % res1.get("error"))
        out1 = res1["object"]
        res2 = _run(ctx, out1, target_count=800)
        out2 = _ok_or_clean(c, res2)
        if out2 is not None:
            st = ctx.face_stats(out2)
            c.require(st["quad_pct"] > 95, "second pass %.1f%% quads" % st["quad_pct"])

    # ---------------- data-layer edge cases --------------------------------

    with r.case("multi_user_mesh_data") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        twin = obj.copy()  # shares the mesh datablock
        bpy.context.scene.collection.objects.link(twin)
        mesh_name = obj.data.name
        res = _run(ctx, obj, target_count=400)
        _ok_or_clean(c, res)
        c.require(twin.data is not None and twin.data.name == mesh_name
                  and len(twin.data.polygons) > 0,
                  "linked duplicate lost its mesh")

    with r.case("custom_split_normals_source") as c:
        ctx.fresh_scene()
        obj = ctx.cube(size=2.0, subdiv=1)
        try:
            obj.data.normals_split_custom_set_from_vertices(
                [v.normal for v in obj.data.vertices])
        except Exception:
            c.skip("custom normals API unavailable")
        res = _run(ctx, obj, target_count=200)
        _ok_or_clean(c, res)

    with r.case("material_index_out_of_range") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        mat = bpy.data.materials.new("OnlyMat")
        obj.data.materials.append(mat)
        idx = np.full(len(obj.data.polygons), 5, dtype=np.int32)  # out of range
        obj.data.polygons.foreach_set("material_index", idx)
        res = _run(ctx, obj, target_count=400)
        _ok_or_clean(c, res)

    with r.case("active_shape_key_not_double_applied") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=32, rings=16)
        obj.shape_key_add(name="Basis")
        k = obj.shape_key_add(name="Stretch")
        for i, v in enumerate(obj.data.vertices):
            k.data[i].co = (v.co.x, v.co.y, v.co.z * 2.0)
        k.value = 1.0   # deformation active at remesh time
        res = _run(ctx, obj, target_count=600)
        out = _ok_or_clean(c, res)
        if out is not None:
            co = ctx.verts_np(out)
            kb = out.data.shape_keys.key_blocks
            c.require(abs(float(co[:, 2].max()) - 1.0) < 0.1,
                      "basis is the deformed shape: zmax %.2f" % float(co[:, 2].max()))
            sk = np.array([d.co[:] for d in kb["Stretch"].data])
            c.require(abs(float(sk[:, 2].max()) - 2.0) < 0.2,
                      "key target wrong: zmax %.2f" % float(sk[:, 2].max()))
            c.require(abs(kb["Stretch"].value - 1.0) < 1e-6, "slider value lost")

    with r.case("t_junction_edges_auto_split") as c:
        ctx.fresh_scene()
        obj = _mesh_obj("Tee2",
                        [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1),
                         (0, 1, 0), (1, 1, 0), (0, -1, 0.5), (1, -1, 0.5),
                         (0, 1, 1), (1, 1, 1), (0, -1, 1.5), (1, -1, 1.5)],
                        [(0, 1, 3, 2), (0, 1, 5, 4), (0, 1, 7, 6),
                         (4, 5, 9, 8), (6, 7, 11, 10)])
        res = _run(ctx, obj, target_count=80)
        out = _ok_or_clean(c, res)
        c.note("ok=%s (card fans sharing a spine)" % res.get("ok"))

    with r.case("shape_key_only_basis") as c:
        ctx.fresh_scene()
        obj = ctx.uv_sphere(segments=24, rings=12)
        obj.shape_key_add(name="Basis")  # basis only, no other keys
        res = _run(ctx, obj, target_count=400)
        _ok_or_clean(c, res)

    return r.list()
