"""QuadForge settings (``object.quadforge``) and the preset system.

Presets are a *convenience*, not a state machine: selecting one writes its
values into the covered properties once and then gets out of the way. The enum
keeps showing the chosen name even after the user edits a covered property --
nothing tracks or reverts manual edits, and the preset is never re-applied
behind the user's back.
"""

from contextlib import contextmanager
from math import radians

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

#: Every property a preset writes. Each preset sets all of them, so switching
#: presets can never leave a stale value behind from the previous one.
PRESET_KEYS = (
    "backend",
    "mode",
    "target_count",
    "adaptive_size",
    "strict_count",
    "use_paint_density",
    "detect_hard_edges",
    "use_marked_sharp",
    "use_materials",
    "use_uv_seams",
    "use_guides",
    "symmetry_x",
    "symmetry_y",
    "symmetry_z",
    "exact_symmetry",
    "preserve_small_shells",
)

#: Deliberately *not* touched by any preset: the Preserve box (UVs, weights,
#: shape keys, materials, creases, bevel weights, boundaries), keep_original,
#: seed, small_shell_limit, solver_isolation, guide_collection, lod_targets.
#: Those are about not losing the user's data / not hanging Blender, and a
#: workflow preset has no business flipping them.
PRESETS = {
    'GAME_AVATAR': {
        "backend": 'NATIVE',
        "mode": 'FACES',
        "target_count": 15000,
        "adaptive_size": 40.0,
        "strict_count": False,
        "use_paint_density": False,
        "detect_hard_edges": True,
        "use_marked_sharp": True,
        "use_materials": False,
        "use_uv_seams": True,
        "use_guides": False,
        "symmetry_x": True,
        "symmetry_y": False,
        "symmetry_z": False,
        "exact_symmetry": True,
        "preserve_small_shells": True,
    },
    'SCULPT_CLEANUP': {
        "backend": 'NATIVE',
        "mode": 'FACES',
        "target_count": 20000,
        "adaptive_size": 60.0,
        "strict_count": False,
        "use_paint_density": False,
        "detect_hard_edges": False,
        "use_marked_sharp": False,
        "use_materials": False,
        "use_uv_seams": False,
        "use_guides": False,
        "symmetry_x": False,
        "symmetry_y": False,
        "symmetry_z": False,
        "exact_symmetry": True,
        "preserve_small_shells": False,
    },
    'HARD_SURFACE': {
        "backend": 'QUADRIFLOW',
        "mode": 'FACES',
        "target_count": 8000,
        "adaptive_size": 0.0,
        "strict_count": False,
        "use_paint_density": False,
        "detect_hard_edges": True,
        "use_marked_sharp": True,
        "use_materials": True,
        "use_uv_seams": False,
        "use_guides": False,
        "symmetry_x": False,
        "symmetry_y": False,
        "symmetry_z": False,
        "exact_symmetry": True,
        "preserve_small_shells": True,
    },
    'QUICK_DRAFT': {
        "backend": 'QUADRIFLOW',
        "mode": 'FACES',
        "target_count": 4000,
        "adaptive_size": 0.0,
        "strict_count": False,
        "use_paint_density": False,
        "detect_hard_edges": False,
        "use_marked_sharp": False,
        "use_materials": False,
        "use_uv_seams": False,
        "use_guides": False,
        "symmetry_x": False,
        "symmetry_y": False,
        "symmetry_z": False,
        "exact_symmetry": False,
        "preserve_small_shells": True,
    },
}

PRESET_ITEMS = [
    ('CUSTOM', "Custom", "Your own settings — nothing is applied or reverted"),
    ('GAME_AVATAR', "Game Avatar",
     "15k quads, Native solver, exact X symmetry, moderate adaptivity, "
     "UV islands followed, authored hair/teeth/eye shells kept"),
    ('SCULPT_CLEANUP', "Sculpt Cleanup",
     "20k quads, Native solver, strong adaptivity, no symmetry and no feature "
     "edges — for a raw sculpt with no authored marks or detail shells"),
    ('HARD_SURFACE', "Hard Surface",
     "8k quads, QuadriFlow+, uniform density, hard edges from angle, existing "
     "marks and material boundaries"),
    ('QUICK_DRAFT', "Quick Draft",
     "4k quads, QuadriFlow+, every edge-flow, symmetry and density feature off "
     "— the fastest look at the result"),
]

_SUPPRESS = 0


@contextmanager
def suppress_preset_update():
    """Set ``preset`` without applying it (used when copying settings around)."""
    global _SUPPRESS
    _SUPPRESS += 1
    try:
        yield
    finally:
        _SUPPRESS -= 1


def apply_preset(s, name) -> bool:
    """Write preset ``name`` into ``s``. Returns False for CUSTOM/unknown."""
    values = PRESETS.get(name)
    if not values:
        return False
    with suppress_preset_update():
        for key, value in values.items():
            try:
                setattr(s, key, value)
            except Exception:
                continue
    return True


def _preset_update(self, context):
    if _SUPPRESS:
        return
    apply_preset(self, self.preset)


class QF_Settings(bpy.types.PropertyGroup):
    # Declared first so that a settings copy which walks the RNA in order
    # applies the preset before the individual values overwrite it.
    preset: EnumProperty(
        name="Preset",
        description="Apply a curated set of settings. Picking one writes them "
                    "once; editing anything afterwards is kept — the name just "
                    "stays as a label",
        items=PRESET_ITEMS,
        default='CUSTOM',
        update=_preset_update,
    )

    target_count: IntProperty(
        name="Quad Count",
        description="Approximate quad count of the finished object. Faces kept "
                    "verbatim by Keep Small Shells are subtracted from the "
                    "solver's budget so the total lands near this number; if "
                    "they alone exceed 60% of it the result overshoots and says so",
        default=5000, min=12, soft_max=200000,
    )
    mode: EnumProperty(
        name="Target Mode",
        items=[
            ('FACES', "Faces", "Use Quad Count as an absolute number of quads"),
            ('RATIO', "Ratio", "Quad count = input face count x Ratio"),
            ('EDGE', "Edge Length", "Quad count = surface area / Edge Length squared"),
        ],
        default='FACES',
    )
    target_ratio: FloatProperty(
        name="Ratio",
        description="Fraction of the input face count to aim for (0.25 = a quarter "
                    "as many faces as the input has)",
        default=1.0, min=0.001, soft_max=10.0,
    )
    target_edge_length: FloatProperty(
        name="Edge Length",
        description="Average quad edge length in world units, converted to a "
                    "quad count from the object's world-space surface area",
        default=0.1, min=0.0001, subtype='DISTANCE',
    )

    adaptive_size: FloatProperty(
        name="Adaptive Size",
        description="Spend more of the quad budget where the surface curves "
                    "(0 = uniform). Native reallocates quad size directly; "
                    "QuadriFlow+ takes no density input, so it is applied "
                    "afterwards as a density-weighted relax — quad size varies, "
                    "the count does not",
        default=0.0, min=0.0, max=100.0, subtype='PERCENTAGE',
    )
    detail_range: FloatProperty(
        name="Size Contrast",
        description="Native backend: the widest quad-size ratio adaptivity may "
                    "open up — 6 means the flattest regions may carry quads "
                    "with six times the edge length of the busiest ones (36x "
                    "the area). Reached at Adaptive Size 100%; lower adaptivity "
                    "scales the spread back. 3 is the classic band; raise it "
                    "when the face budget is too small to carry the whole "
                    "surface at one size. Above 3 the sizing field is also "
                    "gradient-limited so quad size ramps instead of stepping. "
                    "No effect on QuadriFlow+, which takes no density input",
        default=3.0, min=3.0, max=12.0, precision=1,
    )
    use_input_density: BoolProperty(
        name="Detail from Input",
        description="Native backend: treat the input mesh's own tessellation as "
                    "a detail hint — where the artist left big triangles, go "
                    "coarse there too, even where curvature is ambiguous "
                    "(a decimated flat panel and a smooth blob both read as "
                    "zero curvature). Bounded by Size Contrast, and a no-op on "
                    "evenly tessellated input. Measured before the solver's own "
                    "refinement, so it survives it. No effect on QuadriFlow+, "
                    "which takes no density input",
        default=False,
    )
    adapt_quad_count: BoolProperty(
        name="Adapt Quad Count",
        description="Ask the solver for up to 10% extra faces so the QuadriFlow+ "
                    "adaptive pass has slack to drop whole edge loops in the flat "
                    "regions. No effect at Adaptive Size 0 or on the Native backend",
        default=True,
    )
    strict_count: BoolProperty(
        name="Strict Count",
        description="Re-solve up to 3 more times, correcting the request each "
                    "round, until the count is within 10% of the target "
                    "(QuadriFlow+ only; costs a full solve per retry)",
        default=False,
    )
    use_paint_density: BoolProperty(
        name="Painted Density",
        description="Scale quad size by the density map painted with the Paint "
                    "button (mid grey neutral, brighter = denser, darker = "
                    "coarser). Multiplies with Adaptive Size",
        default=False,
    )

    detect_hard_edges: BoolProperty(
        name="Detect Hard Edges",
        description="Mark edges whose dihedral angle exceeds Hard Edge Angle as "
                    "hard, so the result keeps an edge loop there. Non-manifold "
                    "edges are always treated as hard",
        default=True,
    )
    hard_edge_angle: FloatProperty(
        name="Hard Edge Angle",
        description="Angle between two faces above which their shared edge counts "
                    "as hard. Lower catches softer creases (and produces more of them)",
        default=radians(40.0), min=0.0, max=radians(180.0), subtype='ANGLE',
    )
    use_marked_sharp: BoolProperty(
        name="Use Marked Edges",
        description="Also treat the mesh's existing sharp edges, seams and creases "
                    "above 0.5 as hard. When off, existing sharp flags are ignored "
                    "(cleared on the working copy) so only detection drives the flow",
        default=True,
    )
    use_materials: BoolProperty(
        name="Material Boundaries",
        description="Treat edges between different material slots as hard, so an "
                    "edge loop runs along every material boundary",
        default=False,
    )
    use_uv_seams: BoolProperty(
        name="Follow UV Islands",
        description="Treat UV island boundaries (and marked seams) as feature "
                    "edges: edge flow aligns along them and texture seams "
                    "survive the remesh",
        default=False,
    )

    use_guides: BoolProperty(
        name="Use Guides",
        description="Project curve / Grease Pencil objects from the guide "
                    "collection onto the surface and steer the orientation "
                    "field along them. Guides need the Native solver, so "
                    "guided QuadriFlow+ solves switch to it automatically",
        default=False,
    )
    use_opening_rings: BoolProperty(
        name="Opening Rings",
        description="Native backend: run concentric edge loops around small "
                    "closed holes — eye sockets, mouth rims, ear canals — "
                    "instead of letting the curvature flow run straight past "
                    "them, and spend a few extra quads there so the loops can "
                    "actually close (paid for by a marginally coarser rest of "
                    "the mesh, not by a bigger face count). Affects only a few "
                    "quads' width around each hole; large borders and the "
                    "symmetry cut are left alone. Holes on shells that "
                    "Preserve Small Shells keeps verbatim never reach the "
                    "solver and so cannot be ringed",
        default=False,
    )

    guide_collection: PointerProperty(
        name="Guide Collection",
        description="Collection whose curve and Grease Pencil objects are used as "
                    "flow guides (New Guide creates one)",
        type=bpy.types.Collection,
    )

    symmetry_x: BoolProperty(
        name="X", description="Symmetry about the object-space YZ plane (x = 0)",
        default=False)
    symmetry_y: BoolProperty(
        name="Y", description="Symmetry about the object-space XZ plane (y = 0)",
        default=False)
    symmetry_z: BoolProperty(
        name="Z", description="Symmetry about the object-space XY plane (z = 0)",
        default=False)
    exact_symmetry: BoolProperty(
        name="Exact",
        description="Bisect, solve one half and mirror-weld it for a "
                    "mathematically exact result. Off uses the solver's own "
                    "approximate symmetry mode, which is faster but not "
                    "vertex-for-vertex symmetric",
        default=True,
    )

    preserve_boundaries: BoolProperty(
        name="Preserve Boundaries",
        description="Pin open boundary edges (holes, mesh borders) so the "
                    "silhouette of an open mesh stays put",
        default=True,
    )
    preserve_uvs: BoolProperty(
        name="UVs",
        description="Re-project every UV layer onto the new topology, keeping "
                    "island borders crisp",
        default=True,
    )
    preserve_weights: BoolProperty(
        name="Vertex Groups",
        description="Rebuild all vertex groups and their weights on the new "
                    "topology (side-aware, so mirrored limbs stay separate)",
        default=True,
    )
    preserve_shape_keys: BoolProperty(
        name="Shape Keys",
        description="Rebuild the whole shape-key stack on the new topology; "
                    "slider values are restored afterwards",
        default=True,
    )
    preserve_materials: BoolProperty(
        name="Materials",
        description="Keep the material slots and re-assign every face to the "
                    "material it sat on in the source",
        default=True,
    )
    preserve_creases: BoolProperty(
        name="Creases",
        description="Transfer subdivision crease values onto the matching new edges",
        default=True,
    )
    preserve_bevel_weights: BoolProperty(
        name="Bevel Weights",
        description="Transfer bevel weights onto the matching new edges",
        default=True,
    )

    keep_original: BoolProperty(
        name="Keep Original",
        description="Move the source object to a hidden 'QuadForge Originals' "
                    "collection instead of deleting it (Toggle Original flips "
                    "between the two)",
        default=True,
    )
    backend: EnumProperty(
        name="Backend",
        items=[
            ('QUADRIFLOW', "QuadriFlow+",
             "Blender's bundled solver, hardened: per-shell isolated workers, "
             "T-junction repair, cavity restore. Fast and predictable on "
             "hard-surface and mechanical input"),
            ('NATIVE', "Native (experimental)",
             "QuadForge's own field-aligned solver: curvature-following flow, "
             "true density/adaptive reallocation, full guide steering, "
             "deterministic per seed. Best for organic and character work"),
        ],
        default='QUADRIFLOW',
    )
    seed: IntProperty(
        name="Seed",
        description="Randomisation seed. The Native solver is bit-identical for a "
                    "given seed; QuadriFlow is not reproducible even at a fixed seed",
        default=0, min=0,
    )
    preserve_small_shells: BoolProperty(
        name="Keep Small Shells",
        description="Leave small separate shells (hair strands, feathers, "
                    "piercings, teeth) at their original topology instead of "
                    "remeshing them into blobs — they are usually already "
                    "hand-authored and below the solver's useful resolution",
        default=True,
    )
    small_shell_limit: IntProperty(
        name="Small Shell Limit",
        description="Shells with fewer faces than this keep their original "
                    "topology (0 = automatic: 2% of the input face count, at "
                    "least 64). The largest shell is never preserved",
        default=0, min=0, soft_max=5000,
    )
    solver_isolation: BoolProperty(
        name="Hang-Safe Solver",
        description="Run QuadriFlow in a separate Blender process with a timeout "
                    "and retry with an adjusted target if it stalls (works around "
                    "a rare upstream QuadriFlow non-convergence bug). Costs ~1s "
                    "of process startup per solve",
        default=True,
    )
    lod_targets: StringProperty(
        name="LOD Targets",
        description="Comma-separated face counts for Generate LODs, e.g. 8000,2000,500",
        default="8000,2000,500",
    )
    last_report: StringProperty(default="")


def register():
    bpy.utils.register_class(QF_Settings)
    bpy.types.Object.quadforge = PointerProperty(type=QF_Settings)


def unregister():
    del bpy.types.Object.quadforge
    bpy.utils.unregister_class(QF_Settings)
