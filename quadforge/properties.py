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


class QF_Settings(bpy.types.PropertyGroup):
    target_count: IntProperty(
        name="Quad Count",
        description="Approximate number of quads in the result",
        default=5000, min=12, soft_max=200000,
    )
    mode: EnumProperty(
        name="Target Mode",
        items=[
            ('FACES', "Faces", "Target a face count"),
            ('RATIO', "Ratio", "Target a ratio of the input face count"),
            ('EDGE', "Edge Length", "Target an edge length"),
        ],
        default='FACES',
    )
    target_ratio: FloatProperty(name="Ratio", default=1.0, min=0.001, soft_max=10.0)
    target_edge_length: FloatProperty(name="Edge Length", default=0.1, min=0.0001, subtype='DISTANCE')

    adaptive_size: FloatProperty(
        name="Adaptive Size",
        description="Concentrate quads in curved areas (0 = uniform)",
        default=0.0, min=0.0, max=100.0, subtype='PERCENTAGE',
    )
    adapt_quad_count: BoolProperty(
        name="Adapt Quad Count",
        description="Allow the final count to drift for better adaptivity",
        default=True,
    )
    strict_count: BoolProperty(
        name="Strict Count",
        description="Re-run the solver up to 3 times to land within 10% of the target",
        default=False,
    )
    use_paint_density: BoolProperty(
        name="Painted Density",
        description="Use the painted 'qf_density' attribute to control local quad size",
        default=False,
    )

    detect_hard_edges: BoolProperty(
        name="Detect Hard Edges",
        description="Auto-detect hard edges by dihedral angle",
        default=True,
    )
    hard_edge_angle: FloatProperty(
        name="Hard Edge Angle",
        default=radians(40.0), min=0.0, max=radians(180.0), subtype='ANGLE',
    )
    use_marked_sharp: BoolProperty(
        name="Use Marked Edges",
        description="Respect existing sharp, crease and seam edges as hard edges",
        default=True,
    )
    use_materials: BoolProperty(
        name="Material Boundaries",
        description="Preserve edge loops along material boundaries",
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
        description="Steer edge flow along curve / Grease Pencil objects in the guide collection",
        default=False,
    )
    guide_collection: PointerProperty(
        name="Guide Collection",
        type=bpy.types.Collection,
    )

    symmetry_x: BoolProperty(name="X", default=False)
    symmetry_y: BoolProperty(name="Y", default=False)
    symmetry_z: BoolProperty(name="Z", default=False)
    exact_symmetry: BoolProperty(
        name="Exact",
        description="Bisect, remesh one half and mirror-weld for mathematically exact symmetry",
        default=True,
    )

    preserve_boundaries: BoolProperty(name="Preserve Boundaries", default=True)
    preserve_uvs: BoolProperty(name="UVs", default=True)
    preserve_weights: BoolProperty(name="Vertex Groups", default=True)
    preserve_shape_keys: BoolProperty(name="Shape Keys", default=True)
    preserve_materials: BoolProperty(name="Materials", default=True)
    preserve_creases: BoolProperty(name="Creases", default=True)
    preserve_bevel_weights: BoolProperty(name="Bevel Weights", default=True)

    keep_original: BoolProperty(
        name="Keep Original",
        description="Move the source object to a hidden 'QuadForge Originals' collection",
        default=True,
    )
    backend: EnumProperty(
        name="Backend",
        items=[
            ('QUADRIFLOW', "QuadriFlow+", "Blender's built-in solver with QuadForge pre/post passes"),
            ('NATIVE', "Native (experimental)", "QuadForge field-based solver with full guide support"),
        ],
        default='QUADRIFLOW',
    )
    seed: IntProperty(name="Seed", default=0, min=0)
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
        description="Shells with fewer input faces than this keep their "
                    "original topology (0 = automatic: 2% of the input face "
                    "count, at least 64)",
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
        description="Comma-separated face counts, e.g. 8000,2000,500",
        default="8000,2000,500",
    )
    last_report: StringProperty(default="")


def register():
    bpy.utils.register_class(QF_Settings)
    bpy.types.Object.quadforge = PointerProperty(type=QF_Settings)


def unregister():
    del bpy.types.Object.quadforge
    bpy.utils.unregister_class(QF_Settings)
