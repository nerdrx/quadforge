"""View3D sidebar panel -- category "QuadForge".

Layout mirrors Quad Remesher: one big action button on top, then Target /
Edge Loops / Symmetry / Preserve / Output boxes, and a Results box that reads
back the JSON stashed in ``obj.quadforge.last_report``.
"""

from __future__ import annotations

import json

import bpy

CATEGORY = "QuadForge"


def _box(layout, text, icon='NONE'):
    """A titled box; returns the content column (property-split enabled)."""
    box = layout.box()
    header = box.row(align=True)
    header.label(text=text, icon=icon)
    col = box.column()
    col.use_property_split = True
    col.use_property_decorate = False
    return col


def _stat(layout, label, value):
    """One right-aligned label / value line."""
    if value is None:
        return
    split = layout.split(factor=0.45)
    split.alignment = 'RIGHT'
    split.label(text=label)
    split.label(text=str(value))


def _fmt(value, digits=3):
    try:
        return ("%%.%dg" % digits) % float(value)
    except (TypeError, ValueError):
        return str(value)


def read_report(s):
    """Flatten obj.quadforge.last_report into a single dict of display values.

    The pipeline writes a nested report (``{'stats': {...}, 'warnings': [...]}``)
    while quadforge.quality_report writes flat metrics; both are accepted.
    """
    try:
        data = json.loads(s.last_report or "")
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    nested = data.get('stats')
    if isinstance(nested, dict):
        merged = {k: v for k, v in data.items() if k != 'stats'}
        merged.update(nested)
        return merged
    return data


class VIEW3D_PT_quadforge(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = CATEGORY
    bl_label = "QuadForge"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        obj = context.object
        if obj is None or obj.type != 'MESH':
            layout.label(text="Select a mesh object", icon='INFO')
            return
        s = obj.quadforge

        layout.prop(s, "preset")

        row = layout.row()
        row.scale_y = 1.7
        row.operator("quadforge.remesh", text="Remesh", icon='MOD_REMESH')

        self.draw_target(layout, s)
        self.draw_edge_loops(layout, s)
        self.draw_symmetry(layout, s)
        self.draw_preserve(layout, s)
        self.draw_output(layout, s)
        self.draw_results(layout, s)

    # -- boxes ---------------------------------------------------------------

    def draw_target(self, layout, s):
        col = _box(layout, "Target", icon='SNAP_VERTEX')
        col.prop(s, "mode")
        if s.mode == 'FACES':
            col.prop(s, "target_count")
        elif s.mode == 'RATIO':
            col.prop(s, "target_ratio")
        else:
            col.prop(s, "target_edge_length")
        col.prop(s, "strict_count")

        col.separator()
        col.prop(s, "adaptive_size")
        sub = col.column()
        sub.active = s.adaptive_size > 0.0
        sub.prop(s, "adapt_quad_count")

        col.separator()
        col.prop(s, "use_paint_density")
        paint = col.row(align=True)
        paint.use_property_split = False
        paint.active = s.use_paint_density
        paint.operator("quadforge.paint_density", text="Paint", icon='BRUSH_DATA')
        paint.operator("quadforge.clear_density", text="Clear", icon='X')

    def draw_edge_loops(self, layout, s):
        col = _box(layout, "Edge Loops", icon='EDGESEL')
        col.prop(s, "detect_hard_edges")
        sub = col.column()
        sub.active = s.detect_hard_edges
        sub.prop(s, "hard_edge_angle")
        col.prop(s, "use_marked_sharp")
        col.prop(s, "use_materials")
        col.prop(s, "use_uv_seams")

        col.separator()
        col.prop(s, "use_guides")
        sub = col.column(align=True)
        sub.active = s.use_guides
        sub.prop(s, "guide_collection", text="Collection")
        row = sub.row()
        row.use_property_split = False
        row.operator("quadforge.guides_new", text="New Guide", icon='CURVE_BEZCURVE')

    def draw_symmetry(self, layout, s):
        col = _box(layout, "Symmetry", icon='MOD_MIRROR')
        row = col.row(align=True)
        row.use_property_split = False
        row.prop(s, "symmetry_x", toggle=True)
        row.prop(s, "symmetry_y", toggle=True)
        row.prop(s, "symmetry_z", toggle=True)
        sub = col.column()
        sub.active = s.symmetry_x or s.symmetry_y or s.symmetry_z
        sub.prop(s, "exact_symmetry")

    def draw_preserve(self, layout, s):
        col = _box(layout, "Preserve", icon='CHECKMARK')
        col.prop(s, "preserve_boundaries")
        col.prop(s, "preserve_uvs")
        col.prop(s, "preserve_weights")
        col.prop(s, "preserve_shape_keys")
        col.prop(s, "preserve_materials")
        col.prop(s, "preserve_creases")
        col.prop(s, "preserve_bevel_weights")

    def draw_output(self, layout, s):
        col = _box(layout, "Output", icon='EXPORT')
        col.prop(s, "keep_original")
        col.prop(s, "backend")
        col.prop(s, "seed")
        col.prop(s, "solver_isolation")
        col.prop(s, "preserve_small_shells")
        sub = col.column()
        sub.active = s.preserve_small_shells
        sub.prop(s, "small_shell_limit")

        col.separator()
        col.prop(s, "lod_targets", text="LODs")
        row = col.row()
        row.use_property_split = False
        row.operator("quadforge.generate_lods", text="Generate LODs", icon='RENDERLAYERS')
        row = col.row()
        row.use_property_split = False
        row.operator("quadforge.remesh_batch", text="Batch Remesh", icon='DUPLICATE')

    def draw_results(self, layout, s):
        col = _box(layout, "Results", icon='INFO')
        rep = read_report(s)
        if rep:
            grid = col.column(align=True)
            grid.use_property_split = False
            _stat(grid, "Faces", rep.get('faces'))
            if 'quad_pct' in rep:
                _stat(grid, "Quads", "%.1f%%" % float(rep['quad_pct']))
            if rep.get('tris') or rep.get('ngons'):
                _stat(grid, "Tris / Ngons", "%s / %s" % (rep.get('tris', 0), rep.get('ngons', 0)))
            if 'poles_3' in rep or 'poles_5plus' in rep:
                _stat(grid, "Poles (3 / 5+)",
                      "%s / %s" % (rep.get('poles_3', 0), rep.get('poles_5plus', 0)))
            if rep.get('non_manifold_edges'):
                _stat(grid, "Non-manifold", rep['non_manifold_edges'])
            if 'time_s' in rep:
                _stat(grid, "Time", "%.2f s" % float(rep['time_s']))
            sym = [rep.get('symmetry_error_%s' % a) for a in "xyz"]
            if any(v is not None for v in sym):
                _stat(grid, "Sym X/Y/Z",
                      " / ".join(_fmt(v, 2) if v is not None else "-" for v in sym))
            if rep.get('backend'):
                _stat(grid, "Backend", rep['backend'])

            notes = col.column(align=True)
            notes.use_property_split = False
            for text in (rep.get('warnings') or [])[:4]:
                notes.label(text=str(text), icon='ERROR')
            for text in (rep.get('limitations') or [])[:4]:
                notes.label(text=str(text), icon='INFO')
        else:
            col.label(text="No results yet", icon='DOT')

        row = col.row(align=True)
        row.use_property_split = False
        row.operator("quadforge.quality_report", text="Quality Report", icon='SHADERFX')
        row.operator("quadforge.symmetry_check", text="Symmetry", icon='MOD_MIRROR')
        row = col.row()
        row.use_property_split = False
        row.operator("quadforge.toggle_original", text="Toggle Original", icon='HIDE_OFF')


CLASSES = (VIEW3D_PT_quadforge,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
