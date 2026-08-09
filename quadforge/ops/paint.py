"""Density painting operators.

QuadForge's pipeline reads a float point attribute ``qf_density`` (0..2,
1.0 = neutral) to locally scale quad size.  Blender cannot paint *float*
attributes in any interactive mode -- Vertex Paint and the sculpt Paint brush
both require a **colour** attribute -- so the paintable surface is a
``FLOAT_COLOR`` point attribute named ``qf_density_col``.

Convention (mirrored in both directions by :func:`sync_density_from_color`)::

    qf_density = red * DENSITY_MAX          # red 0.0 -> 0.0, 0.5 -> 1.0, 1.0 -> 2.0
    qf_density_col = (d, d, d, 1) with d = qf_density / DENSITY_MAX

Neutral is therefore mid grey (0.5).  Darker = bigger quads, brighter = denser.
The colour attribute wins when both exist: it is what the user actually painted.
Red is authoritative; green/blue are written to match so the viewport shows a
readable greyscale.

Colours are read/written through ``color_srgb`` where available so that the
value shown in Blender's colour picker is the value used.
"""

from __future__ import annotations

import bpy
import numpy as np

DENSITY_ATTR = "qf_density"
DENSITY_COLOR_ATTR = "qf_density_col"
DENSITY_MAX = 2.0
NEUTRAL_DENSITY = 1.0
NEUTRAL_CHANNEL = NEUTRAL_DENSITY / DENSITY_MAX  # 0.5


# ---------------------------------------------------------------------------
# attribute plumbing
# ---------------------------------------------------------------------------

def _color_prop(data):
    """'color_srgb' when the build exposes it, else 'color'."""
    if len(data) == 0:
        return "color"
    return "color_srgb" if hasattr(data[0], "color_srgb") else "color"


def ensure_float_attr(me, fill=NEUTRAL_DENSITY):
    attr = me.attributes.get(DENSITY_ATTR)
    if attr is not None and (attr.data_type != 'FLOAT' or attr.domain != 'POINT'):
        try:
            me.attributes.remove(attr)
        except Exception:
            pass
        attr = None
    created = attr is None
    if created:
        attr = me.attributes.new(name=DENSITY_ATTR, type='FLOAT', domain='POINT')
        attr = me.attributes[DENSITY_ATTR]
        nv = len(me.vertices)
        if nv:
            attr.data.foreach_set("value", np.full(nv, fill, dtype=np.float32))
    return me.attributes[DENSITY_ATTR], created


def ensure_color_attr(me, fill=NEUTRAL_CHANNEL):
    attr = me.color_attributes.get(DENSITY_COLOR_ATTR)
    if attr is not None and (attr.domain != 'POINT'):
        try:
            me.color_attributes.remove(attr)
        except Exception:
            pass
        attr = None
    created = attr is None
    if created:
        me.color_attributes.new(name=DENSITY_COLOR_ATTR, type='FLOAT_COLOR', domain='POINT')
        attr = me.color_attributes[DENSITY_COLOR_ATTR]
        nv = len(me.vertices)
        if nv:
            buf = np.tile(np.array([fill, fill, fill, 1.0], dtype=np.float32), nv)
            attr.data.foreach_set(_color_prop(attr.data), buf)
    return me.color_attributes[DENSITY_COLOR_ATTR], created


def set_active_color(me):
    try:
        idx = me.color_attributes.find(DENSITY_COLOR_ATTR)
        if idx >= 0:
            me.color_attributes.active_color_index = idx
    except Exception:
        pass
    for holder, prop in ((me.attributes, "active_color_name"),
                         (me.attributes, "default_color_name")):
        try:
            setattr(holder, prop, DENSITY_COLOR_ATTR)
        except Exception:
            pass


def read_density(obj):
    """Effective per-vertex density as a float32 array (colour attr wins)."""
    me = getattr(obj, "data", None)
    if me is None:
        return None
    nv = len(me.vertices)
    if not nv:
        return None
    col = me.color_attributes.get(DENSITY_COLOR_ATTR)
    if col is not None and col.domain == 'POINT':
        buf = np.empty(nv * 4, dtype=np.float32)
        col.data.foreach_get(_color_prop(col.data), buf)
        return np.clip(buf.reshape(-1, 4)[:, 0] * DENSITY_MAX, 0.0, DENSITY_MAX)
    attr = me.attributes.get(DENSITY_ATTR)
    if attr is not None and attr.data_type == 'FLOAT' and attr.domain == 'POINT':
        buf = np.empty(nv, dtype=np.float32)
        attr.data.foreach_get("value", buf)
        return buf
    return None


def sync_density_from_color(obj):
    """Push the painted colour attribute into the float ``qf_density`` attr.

    Returns True when a sync actually happened.
    """
    me = getattr(obj, "data", None)
    if me is None or getattr(obj, "type", None) != 'MESH':
        return False
    nv = len(me.vertices)
    col = me.color_attributes.get(DENSITY_COLOR_ATTR)
    if not nv or col is None or col.domain != 'POINT':
        return False
    buf = np.empty(nv * 4, dtype=np.float32)
    col.data.foreach_get(_color_prop(col.data), buf)
    dens = np.clip(buf.reshape(-1, 4)[:, 0] * DENSITY_MAX, 0.0, DENSITY_MAX)
    attr, _ = ensure_float_attr(me)
    attr.data.foreach_set("value", dens.astype(np.float32))
    me.update()
    return True


def sync_color_from_density(obj):
    """Inverse of :func:`sync_density_from_color`."""
    me = getattr(obj, "data", None)
    if me is None or getattr(obj, "type", None) != 'MESH':
        return False
    nv = len(me.vertices)
    attr = me.attributes.get(DENSITY_ATTR)
    if not nv or attr is None or attr.data_type != 'FLOAT' or attr.domain != 'POINT':
        return False
    dens = np.empty(nv, dtype=np.float32)
    attr.data.foreach_get("value", dens)
    chan = np.clip(dens / DENSITY_MAX, 0.0, 1.0)
    col, _ = ensure_color_attr(me)
    buf = np.empty((nv, 4), dtype=np.float32)
    buf[:, 0] = chan
    buf[:, 1] = chan
    buf[:, 2] = chan
    buf[:, 3] = 1.0
    col.data.foreach_set(_color_prop(col.data), buf.ravel())
    me.update()
    return True


def fill_neutral(me):
    nv = len(me.vertices)
    attr, _ = ensure_float_attr(me)
    if nv:
        attr.data.foreach_set("value", np.full(nv, NEUTRAL_DENSITY, dtype=np.float32))
    col, _ = ensure_color_attr(me)
    if nv:
        buf = np.tile(
            np.array([NEUTRAL_CHANNEL, NEUTRAL_CHANNEL, NEUTRAL_CHANNEL, 1.0], dtype=np.float32),
            nv,
        )
        col.data.foreach_set(_color_prop(col.data), buf)
    me.update()


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

class _ActiveMeshPoll:
    @classmethod
    def poll(cls, context):
        obj = getattr(context, "object", None)
        return obj is not None and obj.type == 'MESH'


class QUADFORGE_OT_paint_density(_ActiveMeshPoll, bpy.types.Operator):
    bl_idname = "quadforge.paint_density"
    bl_label = "Paint Density"
    bl_description = ("Create the density attributes and enter Vertex Paint. "
                      "Mid grey is neutral, bright = denser, dark = coarser")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.object
        me = obj.data

        _, made_float = ensure_float_attr(me)
        _, made_color = ensure_color_attr(me)
        if made_color and not made_float:
            # a float attr already existed -> seed the paintable colours from it
            sync_color_from_density(obj)
        set_active_color(me)

        try:
            obj.quadforge.use_paint_density = True
        except Exception:
            pass

        if bpy.app.background:
            self.report({'INFO'}, "Density attributes ready ('%s' / '%s')"
                        % (DENSITY_ATTR, DENSITY_COLOR_ATTR))
            return {'FINISHED'}

        if context.mode != 'OBJECT' and context.mode != 'PAINT_VERTEX':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
        if context.mode != 'PAINT_VERTEX':
            try:
                bpy.ops.object.mode_set(mode='VERTEX_PAINT')
            except Exception as exc:
                self.report({'WARNING'},
                            "Attributes ready, but Vertex Paint could not be entered: %s" % exc)
                return {'FINISHED'}

        try:
            brush = context.tool_settings.vertex_paint.brush
            if brush is not None:
                brush.color = (1.0, 1.0, 1.0)
                brush.secondary_color = (0.0, 0.0, 0.0)
        except Exception:
            pass

        self.report({'INFO'}, "Paint '%s': mid grey = neutral, white = dense, black = coarse"
                    % DENSITY_COLOR_ATTR)
        return {'FINISHED'}


class QUADFORGE_OT_clear_density(_ActiveMeshPoll, bpy.types.Operator):
    bl_idname = "quadforge.clear_density"
    bl_label = "Clear Density"
    bl_description = "Reset the painted density back to neutral (1.0) everywhere"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.object
        me = obj.data
        had = (me.attributes.get(DENSITY_ATTR) is not None
               or me.color_attributes.get(DENSITY_COLOR_ATTR) is not None)
        fill_neutral(me)
        self.report({'INFO'}, "Density reset to neutral" if had else "Density attributes created (neutral)")
        return {'FINISHED'}


CLASSES = (
    QUADFORGE_OT_paint_density,
    QUADFORGE_OT_clear_density,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
