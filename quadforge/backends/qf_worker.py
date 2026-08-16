"""Standalone QuadriFlow worker.

Executed by the addon in a child ``blender --background --factory-startup``
process so a stalled solve (rare upstream QuadriFlow non-convergence) can be
killed by timeout instead of freezing the whole session. Not imported by the
addon.

The parent passes a JSON blob: ``{"out": path, "objects": {name: op_kwargs}}``.
Each named object in the .blend is remeshed with its own operator kwargs; the
whole file is re-saved after every object so the parent can recover completed
parts even if a later part stalls and the process is killed.
"""

import json
import sys

import bpy


def main():
    argv = sys.argv
    arg = argv[argv.index("--") + 1]
    if arg.lstrip().startswith("{"):
        params = json.loads(arg)
    else:
        with open(arg) as fh:
            params = json.load(fh)
    out_path = params["out"]
    jobs = params["objects"]

    scene_objs = bpy.context.scene.collection.objects
    for name in jobs:
        obj = bpy.data.objects.get(name)
        if obj is None:
            print("QF_PART_MISSING", name, flush=True)
            continue
        if obj.name not in scene_objs:
            scene_objs.link(obj)

    for name, kwargs in jobs.items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            print("QF_PART_MISSING", name, flush=True)
            continue
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        try:
            with bpy.context.temp_override(
                object=obj,
                active_object=obj,
                selected_objects=[obj],
                selected_editable_objects=[obj],
            ):
                result = bpy.ops.object.quadriflow_remesh(**kwargs)
        except RuntimeError as exc:
            # degenerate part (e.g. intersecting zero-volume card stacks):
            # the op raises instead of cancelling — keep the batch alive
            print("QF_PART_FAILED", name, str(exc).replace("\n", " ")[:120], flush=True)
            obj.select_set(False)
            continue
        obj.select_set(False)
        if "CANCELLED" in result:
            print("QF_PART_FAILED", name, "cancelled", flush=True)
            continue
        bpy.ops.wm.save_as_mainfile(filepath=out_path)
        print("QF_PART_DONE", name, obj.data.name, len(obj.data.polygons), flush=True)

    print("QF_WORKER_FINISHED", flush=True)


main()
