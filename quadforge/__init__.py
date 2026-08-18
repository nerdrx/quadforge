bl_info = {
    "name": "QuadForge",
    "author": "nerdrx + Claude",
    "version": (0, 5, 5),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > QuadForge",
    "description": "Free auto-retopology: adaptive quad remeshing with guides, symmetry and full data preservation",
    "category": "Mesh",
}

from . import properties

_MODULES = []


def _collect_modules():
    global _MODULES
    if _MODULES:
        return _MODULES
    mods = []
    from .ops import remesh as _ops_remesh
    mods.append(_ops_remesh)
    for name in ("paint", "batch", "lods"):
        try:
            mods.append(__import__(f"{__package__}.ops.{name}", fromlist=[name]))
        except ImportError:
            pass
    from .ui import panel as _panel
    mods.append(_panel)
    _MODULES = mods
    return mods


def register():
    properties.register()
    for mod in _collect_modules():
        if hasattr(mod, "register"):
            mod.register()


def unregister():
    for mod in reversed(_collect_modules()):
        if hasattr(mod, "unregister"):
            mod.unregister()
    properties.unregister()
