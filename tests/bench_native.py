"""QuadForge quality benchmark: native solver vs the QuadriFlow backend.

Usage
-----
    blender --background --factory-startup --python tests/bench_native.py
    blender --background --factory-startup --python tests/bench_native.py -- --quick

This is the gate that decides when the native solver may become the primary
backend.  It measures *topology quality*, not just "did it not crash":

    faces / quad% / tris / ngons        - is the output actually quad-dominant
    boundary + non-manifold edges       - closed input must give closed output
    poles per 1k faces + worst valence  - irregular vertex budget
    face-count error vs target          - does the target mean anything
    surface deviation p95 / max         - did the shape survive
    flow alignment (median degrees)     - do the quads follow curvature
    density response                    - does painted density really reallocate
    wall time

Flow alignment is the interesting one.  Per input vertex we fit the shape
operator over the 1-ring (least squares, numpy only), smooth the resulting
world-space curvature tensor over the 1-ring a couple of times, and keep the
vertices whose anisotropy |k1-k2|/(|k1|+|k2|) exceeds 0.3.  Every result edge
whose midpoint lands (nearest-surface) on such a vertex contributes the angle
between its direction (projected into the tangent plane) and the nearest of
the four 4-RoSy representatives of the principal direction, so the value lives
in [0, 45] degrees.  Lower = the quads run along the curvature lines = the
loops a modeller would have drawn by hand.

Outputs
-------
    stdout                aligned comparison table + GATES section
    <SCRATCH>/bench_results.json
    <SCRATCH>/<fixture>_<backend>.png    Workbench wire close-ups

Exit code is always 0 for now (v1 native is expected to fail gates; this run
is the baseline measurement).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

REPO_ROOT = "/run/media/nerdrx/Lex/claude/quadforge"
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_DIR = os.environ.get(
    "QF_BENCH_OUT",
    "/tmp/claude-1000/-run-media-nerdrx-Lex-claude/"
    "cbcfcc9a-ddfd-4fbb-8582-1c3b129cb280/scratchpad/bench",
)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

BACKENDS = ("QUADRIFLOW", "NATIVE")

# NATIVE_V2.md gates
GATE_QUAD_PCT = 97.0
GATE_COUNT_ERR = 25.0        # percent
GATE_DEVIATION = 1.5         # percent of bbox diagonal
ANISO_MIN = 0.3              # flow-alignment vertex filter
# Anisotropy is scale-free: a near-planar patch whose two principal curvatures
# are both numerical noise scores ~1.0 and would flood the flow metric with
# random angles.  Require a real bend too: radius of curvature smaller than the
# bounding box, i.e. |k_max| * bbox_diag > 1.
CURV_MIN_KD = 1.0


def _p(*args):
    print(*args)
    sys.stdout.flush()


# ==========================================================================
# numpy geometry helpers
# ==========================================================================

def tri_fan(loop_starts, loop_totals, loop_verts):
    """Fan-triangulate an ngon soup into an (m, 3) int array."""
    tris = []
    for st, tot in zip(loop_starts, loop_totals):
        vs = loop_verts[st:st + tot]
        for k in range(1, tot - 1):
            tris.append((vs[0], vs[k], vs[k + 1]))
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def read_mesh(me):
    """-> (V (n,3) f8, T (m,3) i8 fan-triangulated, E (e,2) i8 unique edges)."""
    n = len(me.vertices)
    V = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V)
    V = V.reshape(n, 3)

    npoly = len(me.polygons)
    ls = np.empty(npoly, dtype=np.int32)
    me.polygons.foreach_get("loop_start", ls)
    nloop = len(me.loops)
    try:
        lt = np.empty(npoly, dtype=np.int32)
        me.polygons.foreach_get("loop_total", lt)
    except Exception:
        lt = np.diff(np.append(ls.astype(np.int64), nloop)).astype(np.int32)
    lv = np.empty(nloop, dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    T = tri_fan(ls.tolist(), lt.tolist(), lv.tolist())

    ne = len(me.edges)
    E = np.empty(ne * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", E)
    E = E.reshape(ne, 2).astype(np.int64)
    return V, T, E


def vertex_normals(V, T):
    N = np.zeros_like(V)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    fn = np.cross(b - a, c - a)          # area-weighted (not normalised)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    ln = np.linalg.norm(N, axis=1)
    bad = ln < 1e-12
    N[bad] = (0.0, 0.0, 1.0)
    ln[bad] = 1.0
    return N / ln[:, None]


def tangent_frames(N):
    """Arbitrary but stable orthonormal (t1, t2) per vertex, t1,t2 _|_ N."""
    helper = np.tile(np.array([1.0, 0.0, 0.0]), (len(N), 1))
    flip = np.abs(N[:, 0]) > 0.9
    helper[flip] = (0.0, 1.0, 0.0)
    t1 = np.cross(N, helper)
    ln = np.linalg.norm(t1, axis=1)
    ln[ln < 1e-12] = 1.0
    t1 = t1 / ln[:, None]
    t2 = np.cross(N, t1)
    return t1, t2


def principal_curvature(V, T, E, smooth_iters=2):
    """Per-vertex principal curvature via a least-squares shape-operator fit.

    Returns (dir_max (n,3) unit tangent, aniso (n,) in [0,1], k_abs (n,), N (n,3)).
    The 4-RoSy metric only cares about the *cross*, and the two principal
    directions are orthogonal, so picking |k|max vs |k|min is irrelevant.
    """
    n = len(V)
    N = vertex_normals(V, T)
    t1, t2 = tangent_frames(N)

    # directed 1-ring (both orientations of every edge)
    i = np.concatenate([E[:, 0], E[:, 1]])
    j = np.concatenate([E[:, 1], E[:, 0]])

    d = V[j] - V[i]
    L2 = (d * d).sum(axis=1)
    dn = (d * N[i]).sum(axis=1)
    dt = d - dn[:, None] * N[i]
    lt = np.linalg.norm(dt, axis=1)

    ok = (L2 > 1e-16) & (lt > 1e-9)
    i, j, d, L2, dn, dt, lt = (i[ok], j[ok], d[ok], L2[ok], dn[ok], dt[ok], lt[ok])

    kappa = 2.0 * dn / L2                     # normal curvature along d
    cu = (dt * t1[i]).sum(axis=1) / lt
    cv = (dt * t2[i]).sum(axis=1) / lt
    A = np.stack([cu * cu, 2.0 * cu * cv, cv * cv], axis=1)

    ATA = np.zeros((n, 3, 3))
    ATb = np.zeros((n, 3))
    np.add.at(ATA, i, A[:, :, None] * A[:, None, :])
    np.add.at(ATb, i, A * kappa[:, None])
    ATA += 1e-9 * np.eye(3)[None, :, :]
    try:
        x = np.linalg.solve(ATA, ATb[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        x = (np.linalg.pinv(ATA) @ ATb[:, :, None])[:, :, 0]
    x = np.nan_to_num(x)

    # 2x2 shape operator -> world-space symmetric tensor
    S = np.zeros((n, 2, 2))
    S[:, 0, 0] = x[:, 0]
    S[:, 0, 1] = x[:, 1]
    S[:, 1, 0] = x[:, 1]
    S[:, 1, 1] = x[:, 2]
    R = np.stack([t1, t2], axis=2)            # (n,3,2)
    Tw = R @ S @ np.transpose(R, (0, 2, 1))   # (n,3,3)

    eye = np.eye(3)[None, :, :]
    for _ in range(max(0, smooth_iters)):
        acc = Tw.copy()
        cnt = np.ones(n)
        np.add.at(acc, i, Tw[j])
        np.add.at(cnt, i, 1.0)
        Tw = acc / cnt[:, None, None]
        P = eye - N[:, :, None] * N[:, None, :]
        Tw = P @ Tw @ P

    S2 = np.transpose(R, (0, 2, 1)) @ Tw @ R  # back into the tangent basis
    S2 = 0.5 * (S2 + np.transpose(S2, (0, 2, 1)))
    w, vec = np.linalg.eigh(S2)               # w ascending, vec columns
    k_lo, k_hi = w[:, 0], w[:, 1]
    denom = np.abs(k_lo) + np.abs(k_hi)
    aniso = np.where(denom > 1e-12, np.abs(k_hi - k_lo) / np.maximum(denom, 1e-12), 0.0)

    pick = np.argmax(np.abs(w), axis=1)
    k_abs = np.abs(w[np.arange(n), pick])
    e2d = vec[np.arange(n), :, pick]          # (n,2) in the (t1,t2) basis
    dir_max = t1 * e2d[:, 0:1] + t2 * e2d[:, 1:2]
    ln = np.linalg.norm(dir_max, axis=1)
    ln[ln < 1e-12] = 1.0
    return dir_max / ln[:, None], aniso, k_abs, N


# ==========================================================================
# input reference (captured before the fixture is consumed by run_remesh)
# ==========================================================================

class InputRef:
    def __init__(self, me):
        self.V, self.T, self.E = read_mesh(me)
        lo = self.V.min(axis=0)
        hi = self.V.max(axis=0)
        self.bbox_diag = float(np.linalg.norm(hi - lo)) or 1.0
        self.faces = len(me.polygons)
        self.verts = len(me.vertices)
        self.bvh = BVHTree.FromPolygons(
            [tuple(map(float, v)) for v in self.V],
            [tuple(map(int, t)) for t in self.T],
            all_triangles=True, epsilon=0.0,
        )
        self.dir_max, self.aniso, self.k_abs, self.N = principal_curvature(
            self.V, self.T, self.E)
        # vertices that count towards the flow metric: genuinely anisotropic
        # AND genuinely curved (see CURV_MIN_KD)
        self.flow_mask = ((self.aniso > ANISO_MIN)
                          & (self.k_abs * self.bbox_diag > CURV_MIN_KD))
        self.aniso_frac = float((self.aniso > ANISO_MIN).mean())
        self.flow_frac = float(self.flow_mask.mean())

    def nearest(self, co):
        """-> (dist, nearest input vertex index) for a world-space point."""
        loc, _nrm, idx, dist = self.bvh.find_nearest(Vector(co))
        if loc is None:
            return None, None
        tri = self.T[idx]
        p = np.asarray(loc)
        k = int(np.argmin(((self.V[tri] - p) ** 2).sum(axis=1)))
        return float(dist), int(tri[k])


# ==========================================================================
# fixtures
# ==========================================================================

def fresh_scene():
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    except Exception:
        for ob in list(bpy.data.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.objects, bpy.data.cameras,
                 bpy.data.materials, bpy.data.collections):
        for block in list(coll):
            if getattr(block, "users", 0) == 0:
                try:
                    coll.remove(block)
                except Exception:
                    pass
    return bpy.context.scene


def _active():
    return bpy.context.object


def bake_subsurf(obj, levels, simple=False):
    """Headless-safe modifier bake (no bpy.ops.object.modifier_apply)."""
    if levels <= 0:
        return obj
    mod = obj.modifiers.new("qf_bench_sub", 'SUBSURF')
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


def mark_sharp_by_angle(obj, deg=30.0):
    """Flag every edge whose dihedral angle exceeds `deg` as sharp."""
    me = obj.data
    V, T, E = read_mesh(me)
    # face normals per polygon (use the fan triangles' first tri per poly)
    normals = {}
    for p in me.polygons:
        normals[p.index] = np.asarray(p.normal)
    edge_faces = {}
    for p in me.polygons:
        vs = list(p.vertices)
        for a in range(len(vs)):
            key = (min(vs[a], vs[(a + 1) % len(vs)]), max(vs[a], vs[(a + 1) % len(vs)]))
            edge_faces.setdefault(key, []).append(p.index)
    thresh = math.cos(math.radians(deg))
    flags = np.zeros(len(me.edges), dtype=bool)
    for idx, e in enumerate(me.edges):
        a, b = e.vertices
        fs = edge_faces.get((min(a, b), max(a, b)), [])
        if len(fs) == 2:
            if float(np.dot(normals[fs[0]], normals[fs[1]])) < thresh:
                flags[idx] = True
        elif len(fs) == 1:
            flags[idx] = True
    me.edges.foreach_set("use_edge_sharp", flags)
    me.update()
    return int(flags.sum())


def paint_density(obj, high=2.0, low=0.6):
    """qf_density point attribute: `high` on +z, `low` on -z."""
    me = obj.data
    attr = me.attributes.get("qf_density")
    if attr is None or attr.domain != 'POINT' or attr.data_type != 'FLOAT':
        if attr is not None:
            me.attributes.remove(attr)
        attr = me.attributes.new("qf_density", 'FLOAT', 'POINT')
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    z = co.reshape(n, 3)[:, 2]
    vals = np.where(z >= 0.0, high, low).astype(np.float32)
    attr.data.foreach_set("value", vals)
    me.update()
    return attr


def fx_sphere():
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
    ob = _active()
    ob.name = ob.data.name = "sphere"
    return ob


def fx_cube():
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    ob = _active()
    ob.name = ob.data.name = "cube"
    bake_subsurf(ob, 3, simple=True)
    mark_sharp_by_angle(ob, 30.0)
    return ob


def fx_torus():
    bpy.ops.mesh.primitive_torus_add(major_segments=48, minor_segments=24,
                                     major_radius=1.0, minor_radius=0.45)
    ob = _active()
    ob.name = ob.data.name = "torus"
    return ob


def fx_suzanne():
    bpy.ops.mesh.primitive_monkey_add()
    ob = _active()
    ob.name = ob.data.name = "suzanne"
    bake_subsurf(ob, 2)
    return ob


def fx_ellipsoid():
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0)
    ob = _active()
    ob.name = ob.data.name = "ellipsoid"
    me = ob.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3) * np.array([1.0, 0.6, 0.4])
    me.vertices.foreach_set("co", co.reshape(-1))
    me.update()
    return ob


def fx_density_sphere():
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
    ob = _active()
    ob.name = ob.data.name = "density_sphere"
    paint_density(ob, high=2.0, low=0.6)
    return ob


FIXTURES = [
    dict(key="sphere", build=fx_sphere, target=1500,
         organic=True, isotropic=True, density=False,
         note="uv-sphere 64x32"),
    dict(key="cube_sharp", build=fx_cube, target=600,
         organic=False, isotropic=False, density=False,
         note="cube size 2, simple subdiv 3, sharp edges marked"),
    dict(key="torus", build=fx_torus, target=1200,
         organic=True, isotropic=False, density=False,
         note="torus 48x24, R=1 r=0.45"),
    dict(key="suzanne", build=fx_suzanne, target=2500,
         organic=True, isotropic=False, density=False,
         note="Suzanne, catmull-clark subdiv 2"),
    dict(key="ellipsoid", build=fx_ellipsoid, target=1200,
         organic=True, isotropic=False, density=False,
         note="icosphere subdiv 4 scaled (1, 0.6, 0.4)"),
    dict(key="density_sphere", build=fx_density_sphere, target=1500,
         organic=True, isotropic=True, density=True,
         note="uv-sphere 64x32, qf_density 2.0 (+z) / 0.6 (-z)"),
]
QUICK_KEYS = ("sphere", "suzanne")
DENSITY_IDEAL = math.sqrt(2.0 / 0.6)


# ==========================================================================
# measurement
# ==========================================================================

def topology_stats(me):
    faces = len(me.polygons)
    quads = tris = ngons = 0
    counts = {}
    for p in me.polygons:
        vs = list(p.vertices)
        k = len(vs)
        if k == 4:
            quads += 1
        elif k == 3:
            tris += 1
        elif k > 4:
            ngons += 1
        for a in range(k):
            key = (min(vs[a], vs[(a + 1) % k]), max(vs[a], vs[(a + 1) % k]))
            counts[key] = counts.get(key, 0) + 1

    boundary = sum(1 for c in counts.values() if c == 1)
    nonmanifold = sum(1 for c in counts.values() if c > 2)

    nverts = len(me.vertices)
    valence = np.zeros(nverts, dtype=np.int64)
    on_boundary = np.zeros(nverts, dtype=bool)
    for (a, b), c in counts.items():
        valence[a] += 1
        valence[b] += 1
        if c != 2:
            on_boundary[a] = True
            on_boundary[b] = True

    interior = ~on_boundary & (valence > 0)
    poles_mask = interior & (valence != 4)
    poles = int(poles_mask.sum())
    worst = int(valence[interior].max()) if interior.any() else 0
    return {
        "faces": faces, "quads": quads, "tris": tris, "ngons": ngons,
        "quad_pct": (100.0 * quads / faces) if faces else 0.0,
        "verts": nverts,
        "boundary_edges": boundary,
        "non_manifold_edges": nonmanifold,
        "poles": poles,
        "poles_per_1k": (1000.0 * poles / faces) if faces else 0.0,
        "worst_valence": worst,
    }


def surface_deviation(me, ref):
    V, _T, _E = read_mesh(me)
    if not len(V):
        return None, None
    d = np.empty(len(V))
    for k, co in enumerate(V):
        dist, _vi = ref.nearest(co)
        d[k] = dist if dist is not None else np.nan
    d = d[np.isfinite(d)]
    if not len(d):
        return None, None
    scale = 100.0 / ref.bbox_diag
    return float(np.percentile(d, 95) * scale), float(d.max() * scale)


def flow_alignment(me, ref):
    """Median angle (deg, 0..45) between result edges and the 4-RoSy class of
    the principal curvature direction, over anisotropic input regions."""
    V, _T, E = read_mesh(me)
    if not len(E):
        return None, 0
    angles = []
    for a, b in E:
        pa, pb = V[a], V[b]
        e = pb - pa
        el = float(np.linalg.norm(e))
        if el < 1e-12:
            continue
        mid = 0.5 * (pa + pb)
        _dist, vi = ref.nearest(mid)
        if vi is None or not ref.flow_mask[vi]:
            continue
        nv = ref.N[vi]
        et = e - float(np.dot(e, nv)) * nv
        etl = float(np.linalg.norm(et))
        if etl < 1e-9:
            continue
        et /= etl
        c = abs(float(np.dot(et, ref.dir_max[vi])))
        ang = math.degrees(math.acos(min(1.0, max(0.0, c))))   # 0..90
        angles.append(min(ang, 90.0 - ang))                    # 4-RoSy fold
    if not angles:
        return None, 0
    return float(np.median(angles)), len(angles)


def density_response(me):
    """Mean result edge length on -z vs +z (ratio low-density / high-density)."""
    V, _T, E = read_mesh(me)
    if not len(E):
        return None
    a, b = V[E[:, 0]], V[E[:, 1]]
    lens = np.linalg.norm(b - a, axis=1)
    zmid = 0.5 * (a[:, 2] + b[:, 2])
    up = zmid > 0.02
    dn = zmid < -0.02
    if up.sum() < 8 or dn.sum() < 8:
        return None
    hi = float(lens[up].mean())
    if hi < 1e-12:
        return None
    return float(lens[dn].mean() / hi)


# ==========================================================================
# rendering
# ==========================================================================

def setup_render(scene):
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    sh = scene.display.shading
    sh.light = 'MATCAP'
    try:
        sh.studio_light = 'clay_studio.exr'
    except Exception:
        pass
    sh.color_type = 'OBJECT'
    sh.show_object_outline = False
    sh.show_shadows = False
    try:
        scene.display.render_aa = '16'
    except Exception:
        pass


def render_wire(obj, path):
    """Workbench close-up of `obj` with a black object-colour wire copy."""
    scene = bpy.context.scene
    setup_render(scene)

    # World-space vertices, straight from the mesh: obj.bound_box is lazily
    # evaluated and is stale right after run_remesh swaps the mesh in, which
    # silently wrecks the framing.
    n = len(obj.data.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    Vw = co @ mw[:3, :3].T + mw[:3, 3]
    lo, hi = Vw.min(axis=0), Vw.max(axis=0)
    center = Vector(0.5 * (lo + hi))
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    bb = np.array([[x, y, z] for x in (lo[0], hi[0])
                   for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    obj.color = (0.72, 0.73, 0.78, 1.0)

    wire = obj.copy()
    wire.data = obj.data.copy()
    wire.name = obj.name + "_wire"
    scene.collection.objects.link(wire)
    mod = wire.modifiers.new("qf_wire", 'WIREFRAME')
    mod.thickness = diag * 0.0018
    mod.use_even_offset = False          # even offset explodes on poles
    mod.use_replace = True
    mod.use_boundary = True
    mod.offset = 0.0
    wire.color = (0.02, 0.02, 0.03, 1.0)
    wire.scale = (1.0025, 1.0025, 1.0025)

    d = Vector((0.65, -1.0, 0.42)).normalized()
    cam_data = bpy.data.cameras.new("qf_bench_cam")
    cam_data.type = 'ORTHO'
    cam = bpy.data.objects.new("qf_bench_cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = center + d * (diag * 4.0)
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam

    # fit the bbox corners in camera space, then tighten for a close-up.
    # matrix_world is lazily evaluated: without this the camera transform we
    # just assigned is not visible yet and the framing comes out cropped.
    bpy.context.view_layer.update()
    mat = cam.matrix_world.inverted()
    ex = ey = 1e-9
    for corner in bb:
        v = mat @ Vector(corner)
        ex = max(ex, abs(v.x))
        ey = max(ey, abs(v.y))
    aspect = scene.render.resolution_x / float(scene.render.resolution_y)
    cam_data.ortho_scale = 2.0 * max(ex, ey * aspect) * 0.92

    scene.render.filepath = path
    try:
        bpy.ops.render.render(write_still=True)
    except Exception as exc:
        _p("   !! render failed for %s: %s" % (os.path.basename(path), exc))
        return None
    finally:
        for blk in (wire, cam):
            try:
                bpy.data.objects.remove(blk, do_unlink=True)
            except Exception:
                pass
    return path if os.path.exists(path) else None


# ==========================================================================
# one (fixture, backend) run
# ==========================================================================

def run_case(pipeline, fx, backend):
    key = fx["key"]
    row = {
        "fixture": key, "backend": backend, "target": fx["target"],
        "note": fx["note"], "organic": fx["organic"], "isotropic": fx["isotropic"],
        "error": None, "render": None,
    }

    fresh_scene()
    obj = fx["build"]()
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    except Exception:
        pass

    ref = InputRef(obj.data)
    row["input_faces"] = ref.faces
    row["input_verts"] = ref.verts
    row["bbox_diag"] = round(ref.bbox_diag, 6)
    row["aniso_frac"] = round(ref.aniso_frac, 4)
    row["flow_region_frac"] = round(ref.flow_frac, 4)

    s = obj.quadforge
    s.mode = 'FACES'
    s.target_count = int(fx["target"])
    s.strict_count = False
    s.keep_original = False
    s.backend = backend
    s.seed = 0
    s.adaptive_size = 0.0
    s.use_guides = False
    s.symmetry_x = s.symmetry_y = s.symmetry_z = False
    s.use_paint_density = bool(fx["density"])

    t0 = time.perf_counter()
    try:
        res = pipeline.run_remesh(bpy.context, obj, s)
    except Exception as exc:
        row["time_s"] = round(time.perf_counter() - t0, 3)
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
        _p("   !! %s/%s raised:\n%s" % (key, backend, traceback.format_exc().rstrip()))
        return row
    row["time_s"] = round(time.perf_counter() - t0, 3)

    if not res or not res.get("ok"):
        row["error"] = str((res or {}).get("error") or "run_remesh returned not-ok")
        return row
    out = res.get("object")
    if out is None or out.type != 'MESH' or not len(out.data.polygons):
        row["error"] = "backend produced no mesh"
        return row

    rep = res.get("report") or {}
    row["backend_used"] = rep.get("backend", backend)
    if row["backend_used"] != backend:
        row["error"] = "fell back to %s" % row["backend_used"]

    me = out.data
    row.update(topology_stats(me))
    row["count_err_pct"] = 100.0 * (row["faces"] - fx["target"]) / float(fx["target"])

    p95, dmax = surface_deviation(me, ref)
    row["dev_p95_pct"] = p95
    row["dev_max_pct"] = dmax

    if fx["isotropic"]:
        row["flow_deg"] = None
        row["flow_samples"] = 0
        row["flow_skip"] = "isotropic"
    else:
        fdeg, fn = flow_alignment(me, ref)
        row["flow_deg"] = fdeg
        row["flow_samples"] = fn

    if fx["density"]:
        row["density_ratio"] = density_response(me)
        row["density_ideal"] = DENSITY_IDEAL

    png = os.path.join(OUT_DIR, "%s_%s.png" % (key, backend))
    row["render"] = render_wire(out, png)
    if row["render"]:
        row["render_bytes"] = os.path.getsize(png)
    return row


# ==========================================================================
# reporting
# ==========================================================================

COLUMNS = [
    ("fixture", 15, "%-15s"),
    ("backend", 11, "%-11s"),
    ("faces", 6, "%6s"),
    ("err%", 7, "%7s"),
    ("quad%", 6, "%6s"),
    ("tri", 5, "%5s"),
    ("ngon", 5, "%5s"),
    ("bnd", 4, "%4s"),
    ("nonmf", 5, "%5s"),
    ("pol/1k", 7, "%7s"),
    ("wval", 4, "%4s"),
    ("dev95%", 7, "%7s"),
    ("devmx%", 7, "%7s"),
    ("flow", 6, "%6s"),
    ("sec", 6, "%6s"),
]


def _fmt(v, spec="%.2f"):
    if v is None:
        return "-"
    try:
        return spec % v
    except Exception:
        return str(v)


def print_table(rows):
    head = "  ".join(f % name for name, _w, f in COLUMNS)
    _p(head)
    _p("-" * len(head))
    last_fixture = None
    for r in rows:
        if last_fixture is not None and r["fixture"] != last_fixture:
            _p("")
        last_fixture = r["fixture"]
        if r.get("error") and "faces" not in r:
            line = ("%-15s  %-11s  " % (r["fixture"], r["backend"])) + \
                   ("FAILED: %s" % r["error"])[:110]
            _p(line)
            continue
        cells = [
            r["fixture"], r["backend"],
            "%d" % r["faces"],
            "%+.1f" % r["count_err_pct"],
            "%.1f" % r["quad_pct"],
            "%d" % r["tris"],
            "%d" % r["ngons"],
            "%d" % r["boundary_edges"],
            "%d" % r["non_manifold_edges"],
            "%.1f" % r["poles_per_1k"],
            "%d" % r["worst_valence"],
            _fmt(r["dev_p95_pct"], "%.3f"),
            _fmt(r["dev_max_pct"], "%.3f"),
            ("iso" if r.get("flow_skip") else _fmt(r.get("flow_deg"), "%.1f")),
            "%.1f" % r["time_s"],
        ]
        _p("  ".join(f % c for c, (_n, _w, f) in zip(cells, COLUMNS)))
        if r.get("error"):
            _p("%-15s  %-11s  (note: %s)" % ("", "", r["error"]))


def print_density(rows):
    dens = [r for r in rows if r.get("density_ratio") is not None]
    if not dens:
        return
    _p("")
    _p("DENSITY RESPONSE  (mean edge length -z / +z; ideal sqrt(2.0/0.6) = %.3f)"
       % DENSITY_IDEAL)
    _p("  %-11s  %8s  %10s" % ("backend", "ratio", "of ideal"))
    for r in dens:
        _p("  %-11s  %8.3f  %9.0f%%"
           % (r["backend"], r["density_ratio"],
              100.0 * r["density_ratio"] / DENSITY_IDEAL))


def evaluate_gates(rows):
    """NATIVE_V2.md acceptance gates, evaluated for the NATIVE backend."""
    _p("")
    _p("GATES  (NATIVE_V2.md, native backend)")
    _p("  closed: 0 boundary + 0 non-manifold edges | quads >= %.0f%% (organic) | "
       "count within %.0f%% | max deviation <= %.1f%% bbox diag"
       % (GATE_QUAD_PCT, GATE_COUNT_ERR, GATE_DEVIATION))
    _p("")
    hdr = "  %-15s  %-22s  %-24s  %-22s  %-22s  %s" % (
        "fixture", "closed", "quad%", "count", "deviation", "verdict")
    _p(hdr)
    _p("  " + "-" * (len(hdr) - 2))

    summary = []
    for r in rows:
        if r["backend"] != "NATIVE":
            continue
        key = r["fixture"]
        if r.get("error") and "faces" not in r:
            _p("  %-15s  %s" % (key, "FAILED — %s" % r["error"][:80]))
            summary.append((key, False, ["run failed"]))
            continue

        fails = []
        closed_ok = (r["boundary_edges"] == 0 and r["non_manifold_edges"] == 0)
        if not closed_ok:
            fails.append("not closed")
        c_txt = "%s bnd=%d nonmf=%d" % ("PASS" if closed_ok else "FAIL",
                                        r["boundary_edges"], r["non_manifold_edges"])

        if r["organic"]:
            q_ok = r["quad_pct"] >= GATE_QUAD_PCT
            q_txt = "%s %.1f%% (>=%.0f)" % ("PASS" if q_ok else "FAIL",
                                            r["quad_pct"], GATE_QUAD_PCT)
            if not q_ok:
                fails.append("quad%% %.1f" % r["quad_pct"])
        else:
            q_ok = True
            q_txt = "n/a  %.1f%% (hard-surf)" % r["quad_pct"]

        n_ok = abs(r["count_err_pct"]) <= GATE_COUNT_ERR
        n_txt = "%s %+.1f%% (<=%.0f)" % ("PASS" if n_ok else "FAIL",
                                         r["count_err_pct"], GATE_COUNT_ERR)
        if not n_ok:
            fails.append("count %+.1f%%" % r["count_err_pct"])

        dmax = r.get("dev_max_pct")
        d_ok = dmax is not None and dmax <= GATE_DEVIATION
        d_txt = "%s max %s%% (<=%.1f)" % ("PASS" if d_ok else "FAIL",
                                          _fmt(dmax, "%.3f"), GATE_DEVIATION)
        if not d_ok:
            fails.append("deviation %s%%" % _fmt(dmax, "%.2f"))

        verdict = "PASS" if not fails else "FAIL (%s)" % ", ".join(fails)
        _p("  %-15s  %-22s  %-24s  %-22s  %-22s  %s"
           % (key, c_txt, q_txt, n_txt, d_txt, verdict))
        summary.append((key, not fails, fails))

    if summary:
        npass = sum(1 for _k, ok, _f in summary if ok)
        _p("")
        _p("  NATIVE gate score: %d/%d fixtures pass all gates" % (npass, len(summary)))
        if npass < len(summary):
            _p("  (native v1 is expected to fail here; this run is the baseline "
               "the v2 work must beat)")
    return summary


def print_flow_summary(rows):
    interesting = [r for r in rows
                   if not r.get("isotropic") and r.get("flow_deg") is not None]
    if not interesting:
        return
    _p("")
    _p("FLOW ALIGNMENT  (median deg between result edges and the 4-RoSy class of the")
    _p("                principal curvature direction, sampled where anisotropy > %.1f"
       % ANISO_MIN)
    _p("                and |k_max| * bbox_diag > %.1f (i.e. genuinely bent, not flat);"
       % CURV_MIN_KD)
    _p("                0 = perfectly curvature-aligned, 22.5 = random, 45 = worst)")
    _p("  %-15s  %-11s  %8s  %9s  %10s"
       % ("fixture", "backend", "median", "edges", "input reg."))
    for r in interesting:
        _p("  %-15s  %-11s  %7.1f°  %9d  %9.1f%%"
           % (r["fixture"], r["backend"], r["flow_deg"], r["flow_samples"],
              100.0 * r.get("flow_region_frac", 0.0)))


def check_determinism(keys):
    """Build every fixture twice and compare the input topology."""
    _p("")
    _p("FIXTURE DETERMINISM  (two builds, same process)")
    ok_all = True
    out = {}
    for fx in FIXTURES:
        if fx["key"] not in keys:
            continue
        sig = []
        for _ in range(2):
            fresh_scene()
            ob = fx["build"]()
            V, T, E = read_mesh(ob.data)
            sig.append((len(ob.data.polygons), len(ob.data.vertices), len(E),
                        round(float(V.sum()), 9), round(float((V * V).sum()), 9)))
        same = sig[0] == sig[1]
        ok_all &= same
        out[fx["key"]] = {"ok": same, "faces": sig[0][0], "verts": sig[0][1],
                          "edges": sig[0][2]}
        _p("  %-15s  %s  faces=%d verts=%d edges=%d"
           % (fx["key"], "OK " if same else "DIFFER", sig[0][0], sig[0][1], sig[0][2]))
    _p("  -> %s" % ("all fixtures deterministic" if ok_all
                    else "!! NON-DETERMINISTIC FIXTURES"))
    return ok_all, out


# ==========================================================================
# main
# ==========================================================================

def parse_args():
    argv = sys.argv
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    return {"quick": "--quick" in extra}


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    _p("== QuadForge native quality benchmark ==")
    _p(".. blender %s   numpy %s" % (bpy.app.version_string, np.__version__))
    _p(".. repo    %s" % REPO_ROOT)
    _p(".. out     %s" % OUT_DIR)
    if args["quick"]:
        _p(".. --quick: fixtures %s only" % (", ".join(QUICK_KEYS)))

    try:
        import quadforge
        quadforge.register()
        _p(".. quadforge.register() ok")
    except Exception as exc:
        _p("!! quadforge.register() failed: %s: %s" % (type(exc).__name__, exc))
        _p(traceback.format_exc().rstrip())
        return 0

    import quadforge.pipeline as pipeline

    keys = QUICK_KEYS if args["quick"] else tuple(f["key"] for f in FIXTURES)
    fixtures = [f for f in FIXTURES if f["key"] in keys]

    det_ok, det_info = check_determinism(keys)

    t_start = time.time()
    rows = []
    for fx in fixtures:
        for backend in BACKENDS:
            _p("")
            _p(">> %s / %s  (target %d, %s)"
               % (fx["key"], backend, fx["target"], fx["note"]))
            try:
                row = run_case(pipeline, fx, backend)
            except Exception as exc:
                _p(traceback.format_exc().rstrip())
                row = {"fixture": fx["key"], "backend": backend,
                       "target": fx["target"], "note": fx["note"],
                       "organic": fx["organic"], "isotropic": fx["isotropic"],
                       "error": "bench harness: %s: %s" % (type(exc).__name__, exc),
                       "time_s": None}
            rows.append(row)
            if row.get("error"):
                _p("   error: %s" % row["error"])
            else:
                _p("   faces=%d quad=%.1f%% dev95=%s%% t=%.1fs"
                   % (row["faces"], row["quad_pct"],
                      _fmt(row.get("dev_p95_pct"), "%.3f"), row["time_s"]))
    elapsed = time.time() - t_start

    _p("")
    _p("=" * 118)
    _p("COMPARISON TABLE")
    _p("  err%    = face count vs target       pol/1k = interior verts with valence != 4, per 1000 faces")
    _p("  bnd    = boundary edges (want 0)     nonmf  = edges shared by > 2 faces (want 0)")
    _p("  dev*   = distance to the input surface, % of bbox diagonal")
    _p("  flow   = median 4-RoSy misalignment in degrees (0 best, 22.5 random, 45 worst); 'iso' = isotropic fixture")
    _p("  sec    = wall time incl. QuadForge pre/post passes (QuadriFlow runs isolated: ~1s process overhead)")
    _p("=" * 118)
    print_table(rows)
    print_flow_summary(rows)
    print_density(rows)
    summary = evaluate_gates(rows)

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "blender": bpy.app.version_string,
        "numpy": np.__version__,
        "repo": REPO_ROOT,
        "quick": args["quick"],
        "elapsed_s": round(elapsed, 2),
        "gates": {"quad_pct": GATE_QUAD_PCT, "count_err_pct": GATE_COUNT_ERR,
                  "deviation_pct": GATE_DEVIATION, "aniso_min": ANISO_MIN},
        "density_ideal": DENSITY_IDEAL,
        "fixture_determinism": {"ok": det_ok, "fixtures": det_info},
        "gate_summary": [{"fixture": k, "pass": ok, "failures": f}
                         for k, ok, f in summary],
        "rows": rows,
    }
    json_path = os.path.join(OUT_DIR, "bench_results.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)

    _p("")
    _p("renders:")
    for r in rows:
        if r.get("render"):
            _p("  %-40s %8.1f KB" % (os.path.basename(r["render"]),
                                     r["render_bytes"] / 1024.0))
        else:
            _p("  %-40s (none: %s)" % ("%s_%s.png" % (r["fixture"], r["backend"]),
                                       r.get("error") or "render failed"))
    _p("json:    %s" % json_path)
    _p("BENCH TOTAL %.1fs" % elapsed)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        _p(traceback.format_exc().rstrip())
        code = 0          # gate logic is advisory for now
    sys.stdout.flush()
    sys.exit(code)
