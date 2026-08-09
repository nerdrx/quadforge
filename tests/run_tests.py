"""QuadForge headless test runner.

Usage:
    blender --background --factory-startup --python tests/run_tests.py

Environment:
    QF_ONLY=<substring>   only run test modules whose filename contains <substring>

Each tests/test_*.py module must expose ``run(ctx) -> list[(name, ok, msg)]``.
The runner is deliberately defensive: a module that fails to import, crashes on
import, or explodes inside run() is reported as a FAIL line, never as an
uncaught traceback that aborts the whole suite.
"""

import contextlib
import importlib
import importlib.util
import math
import os
import sys
import time
import traceback

_DEFAULT_ROOT = "/run/media/nerdrx/Lex/claude/quadforge"
try:
    TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    TESTS_DIR = os.path.join(_DEFAULT_ROOT, "tests")
# QF_ADDON_ROOT lets the harness be pointed at a different copy of the addon
# (used to self-test behaviour against a half-finished tree).
ADDON_ROOT = os.environ.get("QF_ADDON_ROOT") or os.path.dirname(TESTS_DIR) or _DEFAULT_ROOT

if ADDON_ROOT not in sys.path:
    sys.path.insert(0, ADDON_ROOT)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402


def _p(*args):
    print(*args)
    sys.stdout.flush()


# --------------------------------------------------------------------------
# result plumbing
# --------------------------------------------------------------------------

class SkipTest(Exception):
    """Raise inside a case to record it as a passing SKIP."""


class CheckFailed(AssertionError):
    pass


class _Case:
    """Handle handed to the ``with results.case(name)`` body."""

    def __init__(self, name):
        self.name = name
        self.notes = []

    def note(self, msg):
        self.notes.append(str(msg))

    def require(self, cond, msg):
        if not cond:
            raise CheckFailed(msg)
        return True

    def require_close(self, got, want, tol, label="value"):
        if want is None or got is None:
            raise CheckFailed("%s: missing value (got=%r want=%r)" % (label, got, want))
        if abs(got - want) > tol:
            raise CheckFailed("%s: %.6g not within %.6g of %.6g" % (label, got, tol, want))
        return True

    def require_rel(self, got, want, frac, label="value"):
        """got must be within +-frac (0..1) of want."""
        if want in (None, 0) or got is None:
            raise CheckFailed("%s: bad values (got=%r want=%r)" % (label, got, want))
        rel = abs(got - want) / float(abs(want))
        if rel > frac:
            raise CheckFailed("%s: %.6g is %.1f%% off target %.6g (limit %.0f%%)"
                              % (label, got, rel * 100.0, want, frac * 100.0))
        self.note("%s=%.6g (%.1f%% off %.6g)" % (label, got, rel * 100.0, want))
        return True

    def skip(self, msg):
        raise SkipTest(msg)

    def message(self):
        return "; ".join(self.notes) if self.notes else "ok"


class Results:
    def __init__(self):
        self.items = []

    def add(self, name, ok, msg=""):
        self.items.append((str(name), bool(ok), str(msg)))

    def skipped(self, name, msg):
        self.add(name, True, "SKIP: %s" % msg)

    @contextlib.contextmanager
    def case(self, name):
        mark = len(self.items)
        c = _Case(name)
        t0 = time.time()
        try:
            yield c
        except SkipTest as e:
            del self.items[mark:]
            self.add(name, True, "SKIP: %s" % e)
        except CheckFailed as e:
            del self.items[mark:]
            extra = (" [" + c.message() + "]") if c.notes else ""
            self.add(name, False, "%s%s" % (e, extra))
        except ImportError as e:
            # A module another agent has not landed yet: a plain FAIL line says
            # everything; the traceback would just be noise.
            del self.items[mark:]
            self.add(name, False, "%s: %s" % (type(e).__name__, e))
        except Exception as e:  # noqa: BLE001 - a crash is a failure, not a suite abort
            del self.items[mark:]
            _p("      ! %s raised:\n%s" % (name, traceback.format_exc().rstrip()))
            self.add(name, False, "%s: %s" % (type(e).__name__, e))
        else:
            if len(self.items) == mark:
                self.add(name, True, "%s (%.2fs)" % (c.message(), time.time() - t0))

    def list(self):
        return list(self.items)


# --------------------------------------------------------------------------
# test context: scene helpers, mesh factories, geometry assertions
# --------------------------------------------------------------------------

class Ctx:
    SkipTest = SkipTest
    CheckFailed = CheckFailed

    def __init__(self, addon_root):
        self.addon_root = addon_root
        self.register_error = None

    # ---- results ---------------------------------------------------------
    def results(self):
        return Results()

    # ---- module access ---------------------------------------------------
    def imp(self, dotted):
        """Import a quadforge submodule, raising ImportError if an agent has
        not landed it yet (which the caller's case turns into a FAIL)."""
        return importlib.import_module(dotted)

    def try_imp(self, dotted):
        try:
            return importlib.import_module(dotted)
        except Exception:
            return None

    def pipeline(self):
        return self.imp("quadforge.pipeline")

    def report_mod(self):
        return self.imp("quadforge.core.report")

    # ---- scene -----------------------------------------------------------
    def fresh_scene(self):
        """Wipe to a genuinely empty scene."""
        try:
            bpy.ops.wm.read_factory_settings(use_empty=True)
        except Exception:
            # manual fallback
            for ob in list(bpy.data.objects):
                bpy.data.objects.remove(ob, do_unlink=True)
        for coll in (bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                     bpy.data.collections, bpy.data.armatures):
            for block in list(coll):
                if getattr(block, "users", 0) == 0:
                    try:
                        coll.remove(block)
                    except Exception:
                        pass
        return bpy.context.scene

    def link(self, obj):
        if obj.name not in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.link(obj)
        return obj

    def activate(self, obj):
        vl = bpy.context.view_layer
        try:
            for o in bpy.context.selected_objects:
                o.select_set(False)
        except Exception:
            pass
        try:
            obj.select_set(True)
            vl.objects.active = obj
        except Exception:
            pass
        return obj

    # ---- primitive factories --------------------------------------------
    def _new(self, add_call, name=None):
        add_call()
        obj = bpy.context.object
        if name:
            obj.name = name
            obj.data.name = name
        return obj

    def apply_subsurf(self, obj, levels, simple=False):
        """Headless-safe modifier bake (no bpy.ops.object.modifier_apply)."""
        if levels <= 0:
            return obj
        mod = obj.modifiers.new("qf_test_sub", 'SUBSURF')
        mod.levels = levels
        mod.render_levels = levels
        if simple:
            mod.subdivision_type = 'SIMPLE'
        dg = bpy.context.evaluated_depsgraph_get()
        new_me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), depsgraph=dg)
        old = obj.data
        obj.modifiers.clear()
        new_me.name = old.name
        obj.data = new_me
        if old.users == 0:
            bpy.data.meshes.remove(old)
        return obj

    def suzanne(self, subdiv=0, name="Suzanne"):
        obj = self._new(lambda: bpy.ops.mesh.primitive_monkey_add(), name)
        return self.apply_subsurf(obj, subdiv)

    def uv_sphere(self, segments=32, rings=16, radius=1.0, name="Sphere"):
        return self._new(
            lambda: bpy.ops.mesh.primitive_uv_sphere_add(
                segments=segments, ring_count=rings, radius=radius),
            name)

    def ico_sphere(self, subdivisions=3, radius=1.0, name="Icosphere"):
        return self._new(
            lambda: bpy.ops.mesh.primitive_ico_sphere_add(
                subdivisions=subdivisions, radius=radius),
            name)

    def cube(self, size=2.0, subdiv=0, name="Cube"):
        obj = self._new(lambda: bpy.ops.mesh.primitive_cube_add(size=size), name)
        return self.apply_subsurf(obj, subdiv, simple=True)

    def torus(self, major_segments=48, minor_segments=16,
              major_radius=1.0, minor_radius=0.3, name="Torus"):
        return self._new(
            lambda: bpy.ops.mesh.primitive_torus_add(
                major_segments=major_segments, minor_segments=minor_segments,
                major_radius=major_radius, minor_radius=minor_radius),
            name)

    def bezier_circle(self, radius=1.2, name="Guide"):
        return self._new(
            lambda: bpy.ops.curve.primitive_bezier_circle_add(radius=radius), name)

    def rigged_sphere(self, segments=48, rings=24, name="Rigged"):
        """UV sphere carrying: gradient vertex group, 2 shape keys (+Basis),
        2 materials split at the equator, and a UV layer."""
        obj = self.uv_sphere(segments=segments, rings=rings, name=name)
        me = obj.data

        # --- vertex group: linear gradient in Z, top pole == 1.0 ----------
        vg = obj.vertex_groups.new(name="qf_grad")
        zs = [v.co.z for v in me.vertices]
        zmin, zmax = min(zs), max(zs)
        span = max(zmax - zmin, 1e-9)
        for v in me.vertices:
            vg.add([v.index], (v.co.z - zmin) / span, 'REPLACE')

        # --- materials: bottom = mat 0, top (z > 0) = mat 1 ---------------
        m0 = bpy.data.materials.new("qf_mat_low")
        m1 = bpy.data.materials.new("qf_mat_high")
        me.materials.append(m0)
        me.materials.append(m1)
        for poly in me.polygons:
            poly.material_index = 1 if poly.center.z > 0.0 else 0

        # --- shape keys ---------------------------------------------------
        obj.shape_key_add(name="Basis", from_mix=False)
        k1 = obj.shape_key_add(name="qf_grow", from_mix=False)
        for i, v in enumerate(me.vertices):
            k1.data[i].co = v.co * 1.25          # uniform max displacement 0.25
        k2 = obj.shape_key_add(name="qf_shift", from_mix=False)
        for i, v in enumerate(me.vertices):
            k2.data[i].co = v.co + Vector((0.15, 0.0, 0.0))   # max displacement 0.15
        for kb in obj.data.shape_keys.key_blocks:
            kb.value = 0.0

        # --- UV layer is created by the primitive; make sure it is there --
        if not me.uv_layers:
            me.uv_layers.new(name="UVMap")
        return obj

    def symmetric_blob(self, segments=48, rings=24, name="Blob"):
        """Sphere with a bump mirrored onto both sides of X (and of Y), so the
        mesh is exactly symmetric about X and Y."""
        obj = self.uv_sphere(segments=segments, rings=rings, name=name)
        me = obj.data
        for v in me.vertices:
            x, y, z = v.co
            d = 0.0
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    dx = x - sx * 0.6
                    dy = y - sy * 0.35
                    d += 0.35 * math.exp(-((dx * dx + dy * dy) / 0.06))
            n = v.co.normalized() if v.co.length > 1e-9 else Vector((0, 0, 1))
            v.co = v.co + n * d
        return obj

    # ---- settings --------------------------------------------------------
    def settings(self, obj, **overrides):
        s = obj.quadforge
        for k, val in overrides.items():
            if not hasattr(s, k):
                raise AttributeError(
                    "obj.quadforge has no field %r (contract drift?)" % k)
            setattr(s, k, val)
        return s

    # ---- mesh assertions / measurements ---------------------------------
    @staticmethod
    def verts_np(obj):
        import numpy as np
        me = obj.data
        n = len(me.vertices)
        a = np.empty(n * 3, dtype="f8")
        me.vertices.foreach_get("co", a)
        return a.reshape(n, 3)

    @staticmethod
    def face_stats(obj):
        me = obj.data
        faces = len(me.polygons)
        quads = tris = ngons = 0
        for p in me.polygons:
            n = len(p.vertices)
            if n == 4:
                quads += 1
            elif n == 3:
                tris += 1
            elif n > 4:
                ngons += 1
        return {
            "faces": faces, "quads": quads, "tris": tris, "ngons": ngons,
            "quad_pct": (100.0 * quads / faces) if faces else 0.0,
        }

    @staticmethod
    def kdtree(points):
        from mathutils.kdtree import KDTree
        kd = KDTree(len(points))
        for i, p in enumerate(points):
            kd.insert(Vector((float(p[0]), float(p[1]), float(p[2]))), i)
        kd.balance()
        return kd

    def symmetry_error(self, obj, axis=0):
        """Max distance from every vertex's mirror image to the nearest real
        vertex. 0 for a perfectly symmetric mesh."""
        co = self.verts_np(obj)
        if len(co) == 0:
            return float("inf")
        kd = self.kdtree(co)
        worst = 0.0
        for p in co:
            m = [float(p[0]), float(p[1]), float(p[2])]
            m[axis] = -m[axis]
            _, _, dist = kd.find(Vector(m))
            if dist is None:
                return float("inf")
            worst = max(worst, dist)
        return worst

    @staticmethod
    def is_mesh_valid(obj):
        return (obj is not None and getattr(obj, "type", None) == 'MESH'
                and obj.data is not None and len(obj.data.polygons) > 0)

    def max_shape_key_offset(self, obj, key_name):
        import numpy as np
        keys = obj.data.shape_keys
        if keys is None or key_name not in keys.key_blocks:
            return None
        kb = keys.key_blocks[key_name]
        basis = keys.key_blocks[0]
        n = len(kb.data)
        a = np.empty(n * 3, dtype="f8")
        b = np.empty(n * 3, dtype="f8")
        kb.data.foreach_get("co", a)
        basis.data.foreach_get("co", b)
        d = (a - b).reshape(n, 3)
        return float(np.max(np.sqrt((d * d).sum(axis=1)))) if n else 0.0

    def non_manifold_edge_count(self, obj):
        counts = {}
        for p in obj.data.polygons:
            vs = list(p.vertices)
            for i in range(len(vs)):
                k = tuple(sorted((vs[i], vs[(i + 1) % len(vs)])))
                counts[k] = counts.get(k, 0) + 1
        return sum(1 for c in counts.values() if c != 2)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def register_addon(ctx):
    try:
        import quadforge
    except Exception as e:
        ctx.register_error = "import quadforge failed: %s: %s" % (type(e).__name__, e)
        _p("!! %s" % ctx.register_error)
        _p(traceback.format_exc().rstrip())
        return None
    try:
        quadforge.register()
        _p(".. quadforge.register() ok")
    except Exception as e:
        ctx.register_error = "quadforge.register() failed: %s: %s" % (type(e).__name__, e)
        _p("!! %s" % ctx.register_error)
        _p(traceback.format_exc().rstrip())
        # Best effort: at least get the settings PropertyGroup up so property
        # tests can still say something useful.
        if hasattr(bpy.types.Object, "quadforge"):
            _p(".. settings PropertyGroup is registered anyway; continuing")
        else:
            try:
                from quadforge import properties
                properties.register()
                _p(".. fallback: quadforge.properties.register() ok")
            except Exception as e2:
                _p("!! fallback properties.register() failed: %s: %s"
                   % (type(e2).__name__, e2))
    return sys.modules.get("quadforge")


# --------------------------------------------------------------------------
# discovery + driving
# --------------------------------------------------------------------------

def discover():
    if not os.path.isdir(TESTS_DIR):
        return []
    names = sorted(f for f in os.listdir(TESTS_DIR)
                   if f.startswith("test_") and f.endswith(".py"))
    only = os.environ.get("QF_ONLY", "").strip()
    if only:
        names = [n for n in names if only in n]
    return names


def load_module(fname):
    modname = "qf_tests_" + os.path.splitext(fname)[0]
    path = os.path.join(TESTS_DIR, fname)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    t_start = time.time()
    ctx = Ctx(ADDON_ROOT)
    _p("== QuadForge test suite ==")
    _p(".. blender %s" % bpy.app.version_string)
    _p(".. addon root %s" % ADDON_ROOT)
    register_addon(ctx)

    files = discover()
    if not files:
        _p("!! no test modules discovered in %s" % TESTS_DIR)
    only = os.environ.get("QF_ONLY", "").strip()
    if only:
        _p(".. QF_ONLY=%r -> %d module(s)" % (only, len(files)))

    total = 0
    ok_count = 0
    failed_lines = []

    for fname in files:
        label = os.path.splitext(fname)[0]
        # Marker so an external watchdog can name the module if Blender dies
        # hard (a segfault inside the addon cannot be caught from Python).
        _p("BEGIN %s" % label)
        t0 = time.time()
        results = []
        try:
            mod = load_module(fname)
            runner = getattr(mod, "run", None)
            if runner is None:
                results = [("module", False, "no run(ctx) function")]
            else:
                try:
                    ctx.fresh_scene()
                except Exception as e:
                    _p("   (fresh_scene before %s failed: %s)" % (label, e))
                out = runner(ctx)
                if out is None:
                    results = [("module", False, "run(ctx) returned None")]
                else:
                    results = list(out)
        except Exception as e:  # import error, syntax error, crash in run()
            _p(traceback.format_exc().rstrip())
            results = [("module", False,
                        "%s crashed: %s: %s" % (label, type(e).__name__, e))]

        for entry in results:
            try:
                name, ok, msg = entry[0], bool(entry[1]), (entry[2] if len(entry) > 2 else "")
            except Exception:
                name, ok, msg = repr(entry), False, "malformed result tuple"
            total += 1
            if ok:
                ok_count += 1
            line = "%s %s::%s — %s" % ("PASS" if ok else "FAIL", label, name, msg)
            _p(line)
            if not ok:
                failed_lines.append(line)
        _p("   (%s: %d checks in %.1fs)" % (label, len(results), time.time() - t0))

    if failed_lines:
        _p("-- failures --")
        for line in failed_lines:
            _p("   " + line)
    if ctx.register_error:
        _p("-- note: %s" % ctx.register_error)

    _p("TOTAL TIME %.1fs" % (time.time() - t_start))
    _p("RESULT %d/%d" % (ok_count, total))
    return 0 if (total > 0 and ok_count == total) else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        _p(traceback.format_exc().rstrip())
        _p("RESULT 0/0")
        code = 1
    sys.stdout.flush()
    sys.exit(code)
