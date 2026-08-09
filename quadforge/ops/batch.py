"""Batch remesh: run the pipeline over every selected mesh object.

Each object keeps its own ``obj.quadforge`` settings, so a batch can mix
targets.  A failure on one object never aborts the rest -- everything is
collected and summarised at the end.
"""

from __future__ import annotations

import bpy

from .remesh import (
    get_pipeline,
    run_pipeline_on,
    selected_meshes,
    stats_line,
)


class QUADFORGE_OT_remesh_batch(bpy.types.Operator):
    bl_idname = "quadforge.remesh_batch"
    bl_label = "Batch Remesh"
    bl_description = ("Remesh every selected mesh object using its own settings, "
                      "continuing past failures")
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
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        results = []
        total_faces = 0
        for obj in objs:
            name = obj.name
            res = run_pipeline_on(context, obj, pipeline)
            results.append((name, res))
            if res['ok']:
                total_faces += int(res['stats'].get('faces', 0) or 0)

        ok = [(n, r) for n, r in results if r['ok']]
        bad = [(n, r) for n, r in results if not r['ok']]

        for name, res in bad:
            self.report({'WARNING'}, "%s: %s" % (name, res['error'] or "unknown error"))
        for name, res in ok:
            line = stats_line(res['stats'])
            if line:
                print("[QuadForge] %s -> %s" % (name, line))

        # re-select everything that came out of the batch
        try:
            for o in list(context.selected_objects):
                o.select_set(False)
            last = None
            for _n, res in ok:
                new_obj = res.get('object')
                if new_obj is not None:
                    new_obj.select_set(True)
                    last = new_obj
            if last is not None:
                context.view_layer.objects.active = last
        except Exception:
            pass

        if not ok:
            self.report({'ERROR'}, "Batch remesh: all %d objects failed" % len(objs))
            return {'CANCELLED'}

        self.report(
            {'INFO'},
            "Batch remesh: %d/%d succeeded, %d faces total%s"
            % (len(ok), len(objs), total_faces,
               (" (%d failed)" % len(bad)) if bad else ""),
        )
        return {'FINISHED'}


CLASSES = (QUADFORGE_OT_remesh_batch,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
