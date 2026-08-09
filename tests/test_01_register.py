"""Registration + settings contract."""

from math import radians

import bpy

# CONTRACTS.md "Ops (ops/*.py) - bl_idnames"
OP_NAMES = [
    "remesh",
    "paint_density",
    "clear_density",
    "remesh_batch",
    "generate_lods",
    "quality_report",
    "toggle_original",
    "guides_new",
    "symmetry_check",
]

# (field, expected default) - spot check of the settings contract
DEFAULTS = [
    ("target_count", 5000),
    ("mode", 'FACES'),
    ("target_ratio", 1.0),
    ("target_edge_length", 0.1),
    ("adaptive_size", 0.0),
    ("strict_count", False),
    ("use_paint_density", False),
    ("detect_hard_edges", True),
    ("hard_edge_angle", radians(40.0)),
    ("use_materials", False),
    ("use_guides", False),
    ("symmetry_x", False),
    ("symmetry_y", False),
    ("symmetry_z", False),
    ("exact_symmetry", True),
    ("preserve_boundaries", True),
    ("preserve_uvs", True),
    ("preserve_weights", True),
    ("preserve_shape_keys", True),
    ("preserve_materials", True),
    ("preserve_creases", True),
    ("preserve_bevel_weights", True),
    ("keep_original", True),
    ("backend", 'QUADRIFLOW'),
    ("seed", 0),
]

# fields whose existence (not default) is contractual
EXTRA_FIELDS = [
    "adapt_quad_count", "use_marked_sharp", "guide_collection",
    "lod_targets", "last_report",
]


def run(ctx):
    r = ctx.results()

    with r.case("addon_imported") as c:
        import quadforge
        c.require(hasattr(quadforge, "register"), "quadforge.register missing")
        c.require(hasattr(quadforge, "unregister"), "quadforge.unregister missing")
        if ctx.register_error:
            c.require(False, ctx.register_error)
        c.note("bl_info=%s" % (getattr(quadforge, "bl_info", {}).get("name"),))

    with r.case("object_has_quadforge") as c:
        c.require(hasattr(bpy.types.Object, "quadforge"),
                  "bpy.types.Object.quadforge not registered")
        ctx.fresh_scene()
        obj = ctx.cube()
        c.require(obj.quadforge is not None, "obj.quadforge is None")
        c.note(type(obj.quadforge).__name__)

    ctx.fresh_scene()
    obj = ctx.cube()
    s = obj.quadforge

    with r.case("defaults") as c:
        bad = []
        for field, want in DEFAULTS:
            if not hasattr(s, field):
                bad.append("%s: MISSING" % field)
                continue
            got = getattr(s, field)
            if isinstance(want, bool):
                same = bool(got) == want
            elif isinstance(want, float):
                same = abs(float(got) - want) < 1e-6
            else:
                same = got == want
            if not same:
                bad.append("%s: got %r want %r" % (field, got, want))
        c.require(not bad, "wrong defaults -> " + " | ".join(bad))
        c.note("%d fields checked" % len(DEFAULTS))

    with r.case("extra_fields_present") as c:
        missing = [f for f in EXTRA_FIELDS if not hasattr(s, f)]
        c.require(not missing, "missing settings fields: %s" % missing)

    with r.case("lod_targets_parseable") as c:
        txt = s.lod_targets
        c.require(isinstance(txt, str), "lod_targets not a string: %r" % (txt,))
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        c.require(parts, "lod_targets default is empty")
        for p in parts:
            c.require(p.isdigit(), "lod_targets entry %r is not an integer" % p)
        c.note("default=%r" % txt)

    with r.case("mode_enum_items") as c:
        items = [i.identifier for i in
                 s.bl_rna.properties["mode"].enum_items]
        c.require(items == ['FACES', 'RATIO', 'EDGE'],
                  "mode enum is %s, contract says ['FACES','RATIO','EDGE']" % items)

    with r.case("backend_enum_items") as c:
        items = [i.identifier for i in
                 s.bl_rna.properties["backend"].enum_items]
        c.require(items == ['QUADRIFLOW', 'NATIVE'],
                  "backend enum is %s, contract says ['QUADRIFLOW','NATIVE']" % items)

    with r.case("target_count_min") as c:
        prop = s.bl_rna.properties["target_count"]
        c.require(prop.hard_min == 12, "target_count min is %r, want 12" % prop.hard_min)

    with r.case("settings_writable") as c:
        ctx.settings(obj, target_count=1234, backend='NATIVE', symmetry_x=True)
        c.require(s.target_count == 1234, "target_count did not stick")
        c.require(s.backend == 'NATIVE', "backend did not stick")
        c.require(s.symmetry_x is True, "symmetry_x did not stick")
        ctx.settings(obj, target_count=5000, backend='QUADRIFLOW', symmetry_x=False)

    # --- operators --------------------------------------------------------
    for name in OP_NAMES:
        with r.case("op_" + name) as c:
            c.require(hasattr(bpy.ops, "quadforge"),
                      "bpy.ops.quadforge namespace does not exist")
            c.require(hasattr(bpy.ops.quadforge, name),
                      "bpy.ops.quadforge.%s is not registered" % name)
            idname = "QUADFORGE_OT_" + name
            c.require(hasattr(bpy.types, idname),
                      "bpy.types.%s missing" % idname)

    with r.case("panel_category") as c:
        # CONTRACTS.md: 'Panel: View3D sidebar, category "QuadForge"'.
        # The class name is not part of the contract, so match on bl_category.
        panels = []
        for n in dir(bpy.types):
            t = getattr(bpy.types, n, None)
            if (isinstance(t, type) and t is not bpy.types.Panel
                    and issubclass(t, bpy.types.Panel)
                    and getattr(t, "bl_category", None) == "QuadForge"):
                panels.append(t)
        c.require(panels,
                  "no registered Panel has bl_category 'QuadForge'")
        spaces = {getattr(p, "bl_space_type", None) for p in panels}
        regions = {getattr(p, "bl_region_type", None) for p in panels}
        c.require('VIEW_3D' in spaces,
                  "no QuadForge panel in VIEW_3D (found %s)" % spaces)
        c.require('UI' in regions,
                  "no QuadForge panel in the sidebar (region types %s)" % regions)
        c.note("%d panel(s): %s" % (len(panels), [p.__name__ for p in panels]))

    return r.list()
