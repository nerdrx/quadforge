"""LOD generation.

Parses ``s.lod_targets`` ("8000,2000,500") and produces one remesh per target,
named ``<source>_LOD0``, ``<source>_LOD1``, ... in a 'QuadForge LODs'
collection.  Every LOD keeps the source object's transform, and the source
object itself is left completely untouched: each run happens on a throwaway
duplicate which is cleaned up afterwards.
"""

from __future__ import annotations

import bpy

from .remesh import (
    LODS_COLLECTION,
    ORIGINAL_KEY,
    alive as _alive,
    apply_settings,
    ensure_collection,
    get_pipeline,
    move_to_collection,
    run_pipeline_on,
    selected_meshes,
    settings_to_dict,
    store_report,
)


def parse_lod_targets(text, minimum=12):
    """"8000, 2000; 500" -> [8000, 2000, 500]. Invalid entries are dropped."""
    out = []
    for chunk in str(text or "").replace(";", ",").replace("\n", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(round(float(chunk)))
        except ValueError:
            continue
        if value < minimum:
            continue
        if value not in out:
            out.append(value)
    return out


def _remove_object(obj):
    if not _alive(obj):
        return
    data = obj.data
    try:
        bpy.data.objects.remove(obj, do_unlink=True)
    except Exception:
        return
    if data is not None and getattr(data, "users", 1) == 0:
        try:
            bpy.data.meshes.remove(data)
        except Exception:
            pass


class QUADFORGE_OT_generate_lods(bpy.types.Operator):
    bl_idname = "quadforge.generate_lods"
    bl_label = "Generate LODs"
    bl_description = "Remesh the active object once per LOD target into a 'QuadForge LODs' collection"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if getattr(context, "mode", 'OBJECT') != 'OBJECT':
            return False
        return bool(selected_meshes(context))

    def execute(self, context):
        pipeline = get_pipeline()
        if pipeline is None:
            self.report({'ERROR'}, "QuadForge pipeline is not installed (quadforge/pipeline.py missing)")
            return {'CANCELLED'}

        objs = selected_meshes(context)
        if not objs:
            self.report({'ERROR'}, "No mesh object selected")
            return {'CANCELLED'}
        source = context.object if (context.object in objs) else objs[0]

        # snapshot once: every run below may invalidate PropertyGroup pointers
        snapshot = settings_to_dict(source.quadforge)
        targets = parse_lod_targets(snapshot.get("lod_targets", ""))
        if not targets:
            self.report({'ERROR'}, "No valid LOD targets in '%s'"
                        % snapshot.get("lod_targets", ""))
            return {'CANCELLED'}

        lod_coll = ensure_collection(context, LODS_COLLECTION)
        scene_coll = context.scene.collection
        base_name = source.name
        matrix = source.matrix_world.copy()

        made, failed = [], []
        for i, target in enumerate(targets):
            temp = source.copy()
            temp.data = source.data.copy()
            temp.name = "%s_QF_LODTMP%d" % (base_name, i)
            scene_coll.objects.link(temp)

            lod_settings = dict(snapshot)
            lod_settings.update(mode='FACES', target_count=max(12, target),
                                keep_original=False)
            apply_settings(temp.quadforge, lod_settings)

            res = run_pipeline_on(context, temp, pipeline)
            new_obj = res.get('object')

            if not res['ok'] or new_obj is None or not _alive(new_obj):
                failed.append("LOD%d (%d faces): %s" % (i, target, res.get('error') or "no result"))
                _remove_object(temp)
                continue

            if _alive(temp) and new_obj is not temp:
                _remove_object(temp)

            new_obj.name = "%s_LOD%d" % (base_name, i)
            if new_obj.data is not None:
                new_obj.data.name = new_obj.name
            new_obj.matrix_world = matrix
            new_obj.parent = source.parent
            try:
                del new_obj[ORIGINAL_KEY]
            except (KeyError, TypeError):
                pass
            move_to_collection(new_obj, lod_coll)

            keep = dict(snapshot)
            keep.update(mode='FACES', target_count=max(12, target),
                        keep_original=snapshot.get("keep_original", True))
            apply_settings(new_obj.quadforge, keep)
            store_report(new_obj, res['stats'])
            made.append((new_obj, target, res['stats']))

        for msg in failed:
            self.report({'WARNING'}, msg)

        try:
            for o in list(context.selected_objects):
                o.select_set(False)
            source.select_set(True)
            context.view_layer.objects.active = source
        except Exception:
            pass

        if not made:
            self.report({'ERROR'}, "LOD generation failed for all %d targets" % len(targets))
            return {'CANCELLED'}

        summary = ", ".join(
            "%s=%d" % (o.name.rsplit("_", 1)[-1], int(st.get('faces', t) or t))
            for o, t, st in made
        )
        self.report({'INFO'}, "Generated %d/%d LODs in '%s' (%s)"
                    % (len(made), len(targets), lod_coll.name, summary))
        return {'FINISHED'}


CLASSES = (QUADFORGE_OT_generate_lods,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
