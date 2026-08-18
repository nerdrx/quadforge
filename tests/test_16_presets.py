"""Preset enum + panel draw contract.

Two things are checked here:

* every preset in ``quadforge.properties.PRESETS`` writes exactly the values
  documented in docs/MANUAL.md (the table there is generated from this data),
  and the preset machinery never fights a manual edit;
* the sidebar panel's ``draw()`` runs headless against a fake layout, so a
  typo'd property or operator name in the UI fails the suite instead of only
  showing up as a red panel in the viewport.
"""

import json

import bpy

# The documented preset table. Deliberately spelled out again instead of
# imported from the addon: this is the contract, PRESETS is the implementation.
EXPECTED = {
    'GAME_AVATAR': {
        "backend": 'NATIVE', "mode": 'FACES', "target_count": 15000,
        "adaptive_size": 40.0, "strict_count": False, "use_paint_density": False,
        "detect_hard_edges": True, "use_marked_sharp": True,
        "use_materials": False, "use_uv_seams": True, "use_guides": False,
        "symmetry_x": True, "symmetry_y": False, "symmetry_z": False,
        "exact_symmetry": True, "preserve_small_shells": True,
    },
    'SCULPT_CLEANUP': {
        "backend": 'NATIVE', "mode": 'FACES', "target_count": 20000,
        "adaptive_size": 60.0, "strict_count": False, "use_paint_density": False,
        "detect_hard_edges": False, "use_marked_sharp": False,
        "use_materials": False, "use_uv_seams": False, "use_guides": False,
        "symmetry_x": False, "symmetry_y": False, "symmetry_z": False,
        "exact_symmetry": True, "preserve_small_shells": False,
    },
    'HARD_SURFACE': {
        "backend": 'QUADRIFLOW', "mode": 'FACES', "target_count": 8000,
        "adaptive_size": 0.0, "strict_count": False, "use_paint_density": False,
        "detect_hard_edges": True, "use_marked_sharp": True,
        "use_materials": True, "use_uv_seams": False, "use_guides": False,
        "symmetry_x": False, "symmetry_y": False, "symmetry_z": False,
        "exact_symmetry": True, "preserve_small_shells": True,
    },
    'QUICK_DRAFT': {
        "backend": 'QUADRIFLOW', "mode": 'FACES', "target_count": 4000,
        "adaptive_size": 0.0, "strict_count": False, "use_paint_density": False,
        "detect_hard_edges": False, "use_marked_sharp": False,
        "use_materials": False, "use_uv_seams": False, "use_guides": False,
        "symmetry_x": False, "symmetry_y": False, "symmetry_z": False,
        "exact_symmetry": False, "preserve_small_shells": True,
    },
}

# Properties no preset may touch (data preservation / safety / identity).
UNTOUCHED = [
    "preserve_boundaries", "preserve_uvs", "preserve_weights",
    "preserve_shape_keys", "preserve_materials", "preserve_creases",
    "preserve_bevel_weights", "keep_original", "seed", "small_shell_limit",
    "solver_isolation", "lod_targets",
]


def _mismatches(s, want):
    bad = []
    for key, value in want.items():
        got = getattr(s, key)
        if isinstance(value, float):
            same = abs(float(got) - value) < 1e-4
        elif isinstance(value, bool):
            same = bool(got) is value
        else:
            same = got == value
        if not same:
            bad.append("%s: got %r want %r" % (key, got, value))
    return bad


# ---------------------------------------------------------------------------
# fake layout: enough of the UILayout surface for panel.draw()
# ---------------------------------------------------------------------------

class FakeLayout:
    """Records prop/operator/label calls; every sub-layout shares the log."""

    def __init__(self, log=None):
        self.log = [] if log is None else log
        self.use_property_split = False
        self.use_property_decorate = False
        self.alignment = 'EXPAND'
        self.active = True
        self.enabled = True
        self.scale_y = 1.0
        self.scale_x = 1.0

    # -- containers --------------------------------------------------------
    def box(self):
        return FakeLayout(self.log)

    def row(self, align=False, heading="", **kw):
        return FakeLayout(self.log)

    def column(self, align=False, heading="", **kw):
        return FakeLayout(self.log)

    def column_flow(self, **kw):
        return FakeLayout(self.log)

    def split(self, factor=0.5, align=False):
        return FakeLayout(self.log)

    def separator(self, factor=1.0):
        return None

    # -- leaves ------------------------------------------------------------
    def label(self, text="", icon='NONE', **kw):
        self.log.append(("label", text))

    def prop(self, data, prop, **kw):
        # raises AttributeError for a property the settings group lacks
        getattr(data, prop)
        self.log.append(("prop", prop))

    def operator(self, idname, **kw):
        self.log.append(("operator", idname))
        return FakeOperatorProps()

    def menu(self, *args, **kw):
        self.log.append(("menu", args[0] if args else ""))

    def template_ID(self, *args, **kw):
        self.log.append(("template_ID", ""))


class FakeOperatorProps:
    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)


class FakeContext:
    def __init__(self, obj):
        self.object = obj
        self.active_object = obj
        self.scene = bpy.context.scene
        self.mode = 'OBJECT'


def _panel_class():
    """The registered QuadForge sidebar panel (class name is not contractual)."""
    for name in dir(bpy.types):
        t = getattr(bpy.types, name, None)
        if (isinstance(t, type) and t is not bpy.types.Panel
                and issubclass(t, bpy.types.Panel)
                and getattr(t, "bl_category", None) == "QuadForge"
                and getattr(t, "bl_space_type", None) == 'VIEW_3D'):
            return t
    return None


def _shim(panel_cls):
    """A plain-Python stand-in carrying the panel's draw methods."""
    members = {}
    for name in dir(panel_cls):
        if not name.startswith("draw"):
            continue
        fn = getattr(panel_cls, name, None)
        if callable(fn) and getattr(fn, "__module__", "").startswith("quadforge"):
            members[name] = fn
    shim = type("QFPanelShim", (object,), members)()
    return shim


def _draw(panel_cls, obj):
    shim = _shim(panel_cls)
    layout = FakeLayout()
    shim.layout = layout
    panel_cls.draw(shim, FakeContext(obj))
    return layout.log


# ---------------------------------------------------------------------------


def run(ctx):
    r = ctx.results()

    props = ctx.imp("quadforge.properties")

    ctx.fresh_scene()
    obj = ctx.cube()
    s = obj.quadforge

    with r.case("preset_property_exists") as c:
        c.require(hasattr(s, "preset"), "obj.quadforge has no 'preset' property")
        c.require(s.preset == 'CUSTOM',
                  "preset default is %r, want 'CUSTOM'" % s.preset)
        items = [i.identifier for i in s.bl_rna.properties["preset"].enum_items]
        want = ['CUSTOM'] + list(EXPECTED)
        c.require(items == want, "preset items are %s, want %s" % (items, want))

    with r.case("preset_table_matches_addon") as c:
        c.require(set(props.PRESETS) == set(EXPECTED),
                  "PRESETS keys %s != documented %s"
                  % (sorted(props.PRESETS), sorted(EXPECTED)))
        bad = []
        for name, want in EXPECTED.items():
            got = props.PRESETS[name]
            if set(got) != set(want):
                bad.append("%s: keys %s != %s" % (name, sorted(got), sorted(want)))
                continue
            for k, v in want.items():
                if got[k] != v:
                    bad.append("%s.%s: %r != %r" % (name, k, got[k], v))
        c.require(not bad, "PRESETS drifted from the manual: " + " | ".join(bad))

    with r.case("preset_keys_cover_every_preset") as c:
        for name, values in props.PRESETS.items():
            c.require(set(values) == set(props.PRESET_KEYS),
                      "%s does not set exactly PRESET_KEYS" % name)

    # --- each preset applies its documented values -------------------------
    for name, want in EXPECTED.items():
        with r.case("apply_" + name.lower()) as c:
            ctx.fresh_scene()
            o = ctx.cube()
            st = o.quadforge
            st.preset = name
            c.require(st.preset == name,
                      "enum did not stick (%r after setting %r)" % (st.preset, name))
            bad = _mismatches(st, want)
            c.require(not bad, "%s applied wrong values -> %s" % (name, " | ".join(bad)))
            c.note("%d properties" % len(want))

    with r.case("presets_leave_preserve_block_alone") as c:
        ctx.fresh_scene()
        o = ctx.cube()
        st = o.quadforge
        before = {k: getattr(st, k) for k in UNTOUCHED}
        for name in EXPECTED:
            st.preset = name
        after = {k: getattr(st, k) for k in UNTOUCHED}
        bad = ["%s: %r -> %r" % (k, before[k], after[k])
               for k in UNTOUCHED if before[k] != after[k]]
        c.require(not bad, "a preset touched a non-covered property: " + " | ".join(bad))

    with r.case("custom_applies_nothing") as c:
        ctx.fresh_scene()
        o = ctx.cube()
        st = o.quadforge
        st.preset = 'HARD_SURFACE'
        st.target_count = 777
        st.preset = 'CUSTOM'
        c.require(st.target_count == 777,
                  "CUSTOM changed target_count to %r" % st.target_count)
        c.require(st.backend == 'QUADRIFLOW', "CUSTOM changed the backend")

    with r.case("switching_presets_leaves_nothing_stale") as c:
        ctx.fresh_scene()
        o = ctx.cube()
        st = o.quadforge
        st.preset = 'GAME_AVATAR'
        st.preset = 'QUICK_DRAFT'
        bad = _mismatches(st, EXPECTED['QUICK_DRAFT'])
        c.require(not bad, "stale values after GAME_AVATAR -> QUICK_DRAFT: "
                           + " | ".join(bad))

    with r.case("manual_edit_survives") as c:
        # the preset is a label, not a state machine: editing a covered
        # property must not be reverted, and must not reset the enum either
        ctx.fresh_scene()
        o = ctx.cube()
        st = o.quadforge
        st.preset = 'GAME_AVATAR'
        st.target_count = 42000
        st.symmetry_x = False
        c.require(st.target_count == 42000, "manual target_count was reverted")
        c.require(st.symmetry_x is False, "manual symmetry_x was reverted")
        c.require(st.preset == 'GAME_AVATAR',
                  "preset label changed to %r on a manual edit" % st.preset)

    with r.case("apply_preset_helper") as c:
        ctx.fresh_scene()
        o = ctx.cube()
        st = o.quadforge
        c.require(props.apply_preset(st, 'HARD_SURFACE') is True,
                  "apply_preset returned False for a real preset")
        c.require(not _mismatches(st, EXPECTED['HARD_SURFACE']),
                  "apply_preset wrote wrong values")
        c.require(props.apply_preset(st, 'CUSTOM') is False,
                  "apply_preset('CUSTOM') should be a no-op returning False")
        c.require(props.apply_preset(st, 'NOPE') is False,
                  "apply_preset on an unknown name should return False")

    with r.case("copy_settings_does_not_reapply") as c:
        # settings copies (LODs, batch, result objects) must carry the edited
        # values, not the preset's originals
        ctx.fresh_scene()
        src = ctx.cube(name="Src")
        dst = ctx.cube(name="Dst")
        st = src.quadforge
        st.preset = 'QUICK_DRAFT'
        st.target_count = 31337
        st.symmetry_z = True
        ops_remesh = ctx.imp("quadforge.ops.remesh")
        ops_remesh.copy_settings(st, dst.quadforge)
        c.require(dst.quadforge.preset == 'QUICK_DRAFT',
                  "preset was not copied (%r)" % dst.quadforge.preset)
        c.require(dst.quadforge.target_count == 31337,
                  "copy re-applied the preset: target_count=%r"
                  % dst.quadforge.target_count)
        c.require(dst.quadforge.symmetry_z is True,
                  "copy re-applied the preset: symmetry_z was reset")

        pipeline = ctx.pipeline()
        dst2 = ctx.cube(name="Dst2")
        pipeline.copy_settings(st, dst2.quadforge)
        c.require(dst2.quadforge.target_count == 31337,
                  "pipeline.copy_settings re-applied the preset: %r"
                  % dst2.quadforge.target_count)

    with r.case("lod_settings_snapshot_survives_preset") as c:
        # ops/lods.py overrides target_count on top of a settings snapshot
        ctx.fresh_scene()
        src = ctx.cube(name="LodSrc")
        tmp = ctx.cube(name="LodTmp")
        st = src.quadforge
        st.preset = 'HARD_SURFACE'
        ops_remesh = ctx.imp("quadforge.ops.remesh")
        snap = ops_remesh.settings_to_dict(st)
        snap.update(mode='FACES', target_count=512)
        ops_remesh.apply_settings(tmp.quadforge, snap)
        c.require(tmp.quadforge.target_count == 512,
                  "LOD target was overwritten by the preset (%r)"
                  % tmp.quadforge.target_count)

    # --- panel ------------------------------------------------------------
    panel_cls = _panel_class()

    with r.case("panel_found") as c:
        c.require(panel_cls is not None,
                  "no registered VIEW_3D panel with bl_category 'QuadForge'")
        c.note(panel_cls.__name__)

    with r.case("panel_draws") as c:
        if panel_cls is None:
            c.skip("no panel")
        ctx.fresh_scene()
        o = ctx.cube()
        log = _draw(panel_cls, o)
        drawn = [name for kind, name in log if kind == "prop"]
        c.require(drawn, "draw() emitted no properties at all")
        c.require("preset" in drawn,
                  "the preset enum is not drawn in the panel")
        c.require(drawn[0] == "preset",
                  "preset is not the first property drawn (got %r)" % drawn[0])
        c.note("%d props, %d operators"
               % (len(drawn), sum(1 for k, _ in log if k == "operator")))

    with r.case("panel_draws_every_mode_and_preset") as c:
        if panel_cls is None:
            c.skip("no panel")
        ctx.fresh_scene()
        o = ctx.cube()
        st = o.quadforge
        seen = 0
        for preset in ['CUSTOM'] + list(EXPECTED):
            st.preset = preset
            for mode in ('FACES', 'RATIO', 'EDGE'):
                st.mode = mode
                for flag in (False, True):
                    st.use_paint_density = flag
                    st.use_guides = flag
                    _draw(panel_cls, o)
                    seen += 1
        c.note("%d draw passes" % seen)

    with r.case("panel_draws_with_a_report") as c:
        if panel_cls is None:
            c.skip("no panel")
        ctx.fresh_scene()
        o = ctx.cube()
        o.quadforge.last_report = json.dumps({
            "ok": True,
            "warnings": ["solver dropped 12 faces of interior/nested geometry"],
            "limitations": ["adaptive post-pass only runs on the QuadriFlow backend"],
            "stats": {"faces": 4096, "quad_pct": 99.8, "tris": 2, "ngons": 0,
                      "poles_3": 12, "poles_5plus": 9, "non_manifold_edges": 0,
                      "time_s": 3.25, "symmetry_error_x": 0.0,
                      "backend": "QUADRIFLOW"},
        })
        log = _draw(panel_cls, o)
        labels = [text for kind, text in log if kind == "label"]
        c.require(any("4096" in t for t in labels),
                  "the report's face count was not drawn (labels=%r)" % labels[:8])
        # a malformed report must not raise either
        o.quadforge.last_report = "{not json"
        _draw(panel_cls, o)
        o.quadforge.last_report = ""
        _draw(panel_cls, o)

    with r.case("panel_operators_are_registered") as c:
        if panel_cls is None:
            c.skip("no panel")
        ctx.fresh_scene()
        o = ctx.cube()
        log = _draw(panel_cls, o)
        missing = []
        for kind, idname in log:
            if kind != "operator":
                continue
            ns, _, op = idname.partition(".")
            if not hasattr(getattr(bpy.ops, ns, None), op):
                missing.append(idname)
        c.require(not missing, "panel draws unregistered operators: %s" % missing)

    with r.case("panel_handles_no_mesh") as c:
        if panel_cls is None:
            c.skip("no panel")
        ctx.fresh_scene()
        shim = _shim(panel_cls)
        shim.layout = FakeLayout()
        panel_cls.draw(shim, FakeContext(None))
        c.require(shim.layout.log, "draw() with no object emitted nothing")

    return r.list()
