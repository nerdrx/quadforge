"""Core QuadForge operators.

``quadforge.remesh``          -- run the pipeline on every selected mesh object
``quadforge.quality_report``  -- mesh metrics for the active object
``quadforge.toggle_original`` -- flip visibility between a result and its source
``quadforge.symmetry_check``  -- per-axis mirror mismatch of the active object
``quadforge.guides_new``      -- add a guide curve ready to draw

The pipeline module is owned by another agent and may not exist yet, so it is
imported lazily *inside* ``execute()``. These operators therefore register
cleanly even in a half-built checkout.
"""

from __future__ import annotations

import json
import time

import bpy

from ..core.report import mesh_report

#: Custom property the pipeline writes on a result object, holding the name of
#: the object it was generated from.
ORIGINAL_KEY = "quadforge_original"

ORIGINALS_COLLECTION = "QuadForge Originals"
GUIDES_COLLECTION = "QuadForge Guides"
LODS_COLLECTION = "QuadForge LODs"


# ---------------------------------------------------------------------------
# shared helpers (also used by ops/batch.py and ops/lods.py)
# ---------------------------------------------------------------------------

def get_pipeline():
    """Import the pipeline lazily. Returns the module or None."""
    try:
        from .. import pipeline
    except Exception:
        return None
    if not hasattr(pipeline, "run_remesh"):
        return None
    return pipeline


def alive(obj):
    """False once the underlying ID has been freed by Blender."""
    if obj is None:
        return False
    try:
        obj.name  # noqa: B018 -- raises ReferenceError on a freed ID
        return True
    except ReferenceError:
        return False


def selected_meshes(context):
    objs = [o for o in getattr(context, "selected_objects", []) or [] if o.type == 'MESH']
    active = getattr(context, "object", None)
    if not objs and active is not None and active.type == 'MESH':
        objs = [active]
    return objs


def ensure_collection(context, name):
    """Get or create ``name`` and make sure it is linked to the scene."""
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    scene = getattr(context, "scene", None) or (bpy.data.scenes[0] if bpy.data.scenes else None)
    if scene is not None:
        root = scene.collection
        if coll.name not in root.children and not _is_descendant(root, coll):
            root.children.link(coll)
    return coll


def _is_descendant(parent, coll):
    for child in parent.children:
        if child is coll or _is_descendant(child, coll):
            return True
    return False


def move_to_collection(obj, coll):
    """Unlink ``obj`` from every collection and link it to ``coll`` only."""
    for c in list(obj.users_collection):
        if c is not coll:
            try:
                c.objects.unlink(obj)
            except Exception:
                pass
    if obj.name not in coll.objects:
        try:
            coll.objects.link(obj)
        except Exception:
            pass


_SKIP_SETTINGS = {"rna_type", "name", "last_report"}


def settings_to_dict(s):
    """Snapshot QF_Settings into plain Python values.

    Essential before calling the pipeline: ``obj.quadforge`` is a pointer into
    the object's data, and Blender does *not* invalidate PropertyGroup wrappers
    when the object is freed -- reading one afterwards is a hard crash, not an
    exception. Everything we need after a run must be copied out first.
    """
    out = {}
    if s is None:
        return out
    for prop in s.bl_rna.properties:
        ident = prop.identifier
        if ident in _SKIP_SETTINGS or prop.is_readonly:
            continue
        try:
            out[ident] = getattr(s, ident)
        except Exception:
            pass
    return out


def apply_settings(dst, data):
    if dst is None or not data:
        return
    for ident, value in data.items():
        try:
            setattr(dst, ident, value)
        except Exception:
            pass


def copy_settings(src, dst):
    """Copy every QF_Settings value from ``src`` to ``dst``."""
    if src is None or dst is None or src == dst:
        return
    apply_settings(dst, settings_to_dict(src))


def sync_painted_density(obj):
    """Refresh the float 'qf_density' attribute from the painted colour attr.

    No-op when ops/paint.py is unavailable or nothing has been painted.
    """
    try:
        from . import paint
    except Exception:
        return False
    try:
        return paint.sync_density_from_color(obj)
    except Exception:
        return False


def stats_line(stats):
    if not stats:
        return ""
    bits = []
    if "faces" in stats:
        bits.append("%d faces" % stats["faces"])
    if "quad_pct" in stats:
        bits.append("%.1f%% quads" % stats["quad_pct"])
    elif "quads" in stats:
        bits.append("%d quads" % stats["quads"])
    if "time_s" in stats:
        bits.append("%.2fs" % stats["time_s"])
    return ", ".join(bits)


def store_report(obj, stats):
    """Write ``stats`` as JSON into obj.quadforge.last_report."""
    if obj is None or not stats:
        return
    try:
        obj.quadforge.last_report = json.dumps(stats)
    except Exception:
        pass


def run_pipeline_on(context, obj, pipeline=None):
    """Call pipeline.run_remesh and normalise the result dict.

    Returns a dict with at least ``ok``/``error``/``object``/``stats``.
    """
    if pipeline is None:
        pipeline = get_pipeline()
    if pipeline is None:
        return {'ok': False, 'error': "QuadForge pipeline module is not available",
                'object': None, 'stats': {}}
    s = obj.quadforge
    # The pipeline may free ``obj`` (keep_original=False), which leaves ``s``
    # dangling. Take everything we need out of it now.
    snapshot = settings_to_dict(s)
    source_name = obj.name
    if snapshot.get("use_paint_density"):
        sync_painted_density(obj)
    t0 = time.perf_counter()
    try:
        res = pipeline.run_remesh(context, obj, s)
    except Exception as exc:  # pipeline blew up -- never take the UI down with it
        return {'ok': False, 'error': "%s: %s" % (type(exc).__name__, exc),
                'object': None, 'stats': {}}
    if not isinstance(res, dict):
        return {'ok': False, 'error': "pipeline returned %r" % (type(res).__name__,),
                'object': None, 'stats': {}}
    res.setdefault('ok', False)
    res.setdefault('error', None)
    res.setdefault('object', None)
    stats = res.get('stats') or {}
    stats.setdefault('time_s', time.perf_counter() - t0)
    res['stats'] = stats

    # From here on ``obj``/``s`` may be dead -- use ``snapshot``/``source_name``.
    new_obj = res.get('object')
    if res['ok'] and alive(new_obj):
        if new_obj is not obj:
            apply_settings(getattr(new_obj, "quadforge", None), snapshot)
            # Tag the pair for quadforge.toggle_original. The pipeline is meant
            # to do this, but we know both ends here, so make sure it happens --
            # only when the source actually survived the run.
            if ORIGINAL_KEY not in new_obj.keys() and alive(obj):
                new_obj[ORIGINAL_KEY] = source_name
        if not getattr(new_obj.quadforge, "last_report", ""):
            store_report(new_obj, stats)
    return res


class _ObjectModeMeshPoll:
    """Mixin: object mode + at least one mesh object selected."""

    @classmethod
    def poll(cls, context):
        if getattr(context, "mode", 'OBJECT') != 'OBJECT':
            return False
        return bool(selected_meshes(context))


# ---------------------------------------------------------------------------
# quadforge.remesh
# ---------------------------------------------------------------------------

class QUADFORGE_OT_remesh(_ObjectModeMeshPoll, bpy.types.Operator):
    bl_idname = "quadforge.remesh"
    bl_label = "Remesh"
    bl_description = "Retopologise the selected mesh objects into quads"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        pipeline = get_pipeline()
        if pipeline is None:
            self.report({'ERROR'}, "QuadForge pipeline is not installed (quadforge/pipeline.py missing)")
            return {'CANCELLED'}

        objs = selected_meshes(context)
        if not objs:
            self.report({'ERROR'}, "No mesh object selected")
            return {'CANCELLED'}

        done, failed, last_stats, last_obj = 0, [], None, None
        for obj in objs:
            res = run_pipeline_on(context, obj, pipeline)
            if res['ok']:
                done += 1
                last_stats = res['stats']
                last_obj = res['object'] or last_obj
            else:
                failed.append("%s: %s" % (obj.name, res['error'] or "unknown error"))

        if last_obj is not None:
            try:
                context.view_layer.objects.active = last_obj
                last_obj.select_set(True)
            except Exception:
                pass

        for msg in failed:
            self.report({'WARNING'}, msg)

        if not done:
            self.report({'ERROR'}, "Remesh failed")
            return {'CANCELLED'}

        line = stats_line(last_stats)
        if len(objs) > 1:
            self.report({'INFO'}, "QuadForge: %d/%d objects remeshed%s"
                        % (done, len(objs), (" - last: " + line) if line else ""))
        else:
            self.report({'INFO'}, "QuadForge: " + (line or "done"))
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# quadforge.quality_report
# ---------------------------------------------------------------------------

class QUADFORGE_OT_quality_report(bpy.types.Operator):
    bl_idname = "quadforge.quality_report"
    bl_label = "Quality Report"
    bl_description = "Measure quad ratio, poles and symmetry of the active mesh"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "object", None)
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        obj = context.object
        t0 = time.perf_counter()
        rep = mesh_report(obj)
        rep['time_s'] = time.perf_counter() - t0
        store_report(obj, rep)
        self.report(
            {'INFO'},
            "%s: %d faces, %.1f%% quads, poles %d/%d, ngons %d, non-manifold %d"
            % (obj.name, rep['faces'], rep['quad_pct'], rep['poles_3'],
               rep['poles_5plus'], rep['ngons'], rep['non_manifold_edges']),
        )
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# quadforge.symmetry_check
# ---------------------------------------------------------------------------

class QUADFORGE_OT_symmetry_check(bpy.types.Operator):
    bl_idname = "quadforge.symmetry_check"
    bl_label = "Symmetry Check"
    bl_description = "Report the largest mirror mismatch on each axis"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "object", None)
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        obj = context.object
        rep = mesh_report(obj)
        ex, ey, ez = (rep['symmetry_error_x'], rep['symmetry_error_y'], rep['symmetry_error_z'])
        # merge into the stored report so the panel can show it too
        try:
            stored = json.loads(obj.quadforge.last_report or "{}")
        except Exception:
            stored = {}
        stored.update({k: rep[k] for k in
                       ("symmetry_error_x", "symmetry_error_y", "symmetry_error_z")})
        store_report(obj, stored)
        self.report({'INFO'}, "Symmetry error  X %.6g   Y %.6g   Z %.6g" % (ex, ey, ez))
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# quadforge.toggle_original
# ---------------------------------------------------------------------------

#: Suffix the pipeline appends to result objects; used only as a last-resort
#: heuristic when the ``quadforge_original`` tag is missing.
RESULT_SUFFIX = "_quad"


def _find_pair(obj):
    """Return (result_obj, original_obj) for ``obj``, either way round."""
    if not alive(obj):
        return None, None
    name = obj.get(ORIGINAL_KEY)
    if name:
        other = bpy.data.objects.get(name)
        if other is not None and other is not obj:
            return obj, other
    # obj may itself be the original -- look for a result pointing at it
    for other in bpy.data.objects:
        if other is obj:
            continue
        if other.get(ORIGINAL_KEY) == obj.name:
            return other, obj
    # untagged pipeline output: fall back on the naming convention
    if obj.name.endswith(RESULT_SUFFIX):
        other = bpy.data.objects.get(obj.name[:-len(RESULT_SUFFIX)])
        if other is not None and other.type == 'MESH':
            return obj, other
    other = bpy.data.objects.get(obj.name + RESULT_SUFFIX)
    if other is not None and other.type == 'MESH':
        return other, obj
    return None, None


def _set_visible(obj, visible, context=None):
    if obj is None:
        return
    try:
        obj.hide_viewport = not visible
    except Exception:
        pass
    try:
        obj.hide_render = not visible
    except Exception:
        pass
    try:
        obj.hide_set(not visible)
    except Exception:
        pass  # not in the view layer (headless / hidden collection)
    if visible:
        for coll in obj.users_collection:
            try:
                coll.hide_viewport = False
                coll.hide_render = False
            except Exception:
                pass
            if context is not None:
                lc = _layer_collection_for(context, coll)
                if lc is not None:
                    lc.hide_viewport = False
                    lc.exclude = False


def _layer_collection_for(context, coll, root=None):
    vl = getattr(context, "view_layer", None)
    if vl is None:
        return None
    if root is None:
        root = vl.layer_collection
    if root.collection is coll:
        return root
    for child in root.children:
        found = _layer_collection_for(context, coll, child)
        if found is not None:
            return found
    return None


class QUADFORGE_OT_toggle_original(bpy.types.Operator):
    bl_idname = "quadforge.toggle_original"
    bl_label = "Toggle Original"
    bl_description = "Swap visibility between the remeshed object and its original"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "object", None)
        if obj is None:
            return False
        return _find_pair(obj)[1] is not None

    def execute(self, context):
        obj = context.object
        result, original = _find_pair(obj)
        if original is None:
            self.report({'ERROR'}, "No stored original for '%s'" % (obj.name if obj else "?"))
            return {'CANCELLED'}
        if result is None:
            self.report({'ERROR'}, "No remeshed counterpart found")
            return {'CANCELLED'}

        show_original = bool(original.hide_viewport)
        _set_visible(original, show_original, context)
        _set_visible(result, not show_original, context)

        shown = original if show_original else result
        try:
            context.view_layer.objects.active = shown
            shown.select_set(True)
            hidden = result if show_original else original
            hidden.select_set(False)
        except Exception:
            pass

        self.report({'INFO'}, "Showing %s ('%s')"
                    % ("original" if show_original else "remesh", shown.name))
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# quadforge.guides_new
# ---------------------------------------------------------------------------

class QUADFORGE_OT_guides_new(bpy.types.Operator):
    bl_idname = "quadforge.guides_new"
    bl_label = "New Guide"
    bl_description = "Create a guide curve in the guide collection and start drawing it"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return getattr(context, "mode", 'OBJECT') == 'OBJECT'

    def execute(self, context):
        obj = getattr(context, "object", None)
        s = getattr(obj, "quadforge", None) if obj is not None else None

        coll = s.guide_collection if s is not None else None
        if coll is None:
            coll = ensure_collection(context, GUIDES_COLLECTION)
            if s is not None:
                s.guide_collection = coll
        elif getattr(context, "scene", None) is not None:
            # make sure a user-picked collection is actually in the scene
            root = context.scene.collection
            if coll.name not in root.children and not _is_descendant(root, coll):
                try:
                    root.children.link(coll)
                except Exception:
                    pass

        if s is not None:
            s.use_guides = True

        curve = bpy.data.curves.new("QF_Guide", 'CURVE')
        curve.dimensions = '3D'
        curve.bevel_depth = 0.0
        guide = bpy.data.objects.new("QF_Guide", curve)
        try:
            cursor = context.scene.cursor.location
            guide.location = (cursor[0], cursor[1], cursor[2])
        except Exception:
            pass
        coll.objects.link(guide)

        if bpy.app.background:
            self.report({'INFO'}, "Guide '%s' added to '%s'" % (guide.name, coll.name))
            return {'FINISHED'}

        try:
            for o in list(context.selected_objects):
                o.select_set(False)
            guide.select_set(True)
            context.view_layer.objects.active = guide
            bpy.ops.object.mode_set(mode='EDIT')
            try:
                bpy.ops.wm.tool_set_by_id(name="builtin.draw")
            except Exception:
                pass
        except Exception as exc:
            self.report({'WARNING'}, "Guide created, but edit mode failed: %s" % exc)
            return {'FINISHED'}

        self.report({'INFO'}, "Draw the guide, then Tab back to Object Mode")
        return {'FINISHED'}


CLASSES = (
    QUADFORGE_OT_remesh,
    QUADFORGE_OT_quality_report,
    QUADFORGE_OT_symmetry_check,
    QUADFORGE_OT_toggle_original,
    QUADFORGE_OT_guides_new,
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
