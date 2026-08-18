"""QuadForge guide-quality benchmark: does the flow actually follow the guide?

Usage
-----
    blender --background --factory-startup --python tests/bench_guides.py
    blender --background --factory-startup --python tests/bench_guides.py -- --quick

``tests/test_06_density_guides.py`` only proves the guide path *runs* (edges get
marked, nothing raises).  This measures whether the resulting edge flow is
actually steered, and how far the influence reaches.

Fixtures (the desired flow is unambiguous by construction)
----------------------------------------------------------
    sphere_gc     UV sphere + one great-circle bezier guide tilted 30 deg off
                  the UV grid.  Curvature is isotropic, so nothing competes
                  with the guide: this is the clean signal.
    grid_scurve   flat grid + an S-curve guide.  Zero curvature, a strongly
                  turning target: tests whether the guide bends the field.
    cyl_helix     open cylinder + a 45-degree helical guide.  Adversarial: the
                  principal-curvature flow (axis / circumference) is exactly
                  45 degrees away from the guide, i.e. maximally far in the
                  4-RoSy metric, so guide and curvature alignment fight.

Metric
------
Every result edge is projected into the surface tangent plane and compared with
the guide tangent at the nearest point of the guide polyline; the angle is
folded into ``[0, 45]`` degrees (4-RoSy: a quad grid is invariant under 90-deg
rotations).  0 = perfectly on the guide, 22.5 = a random field, 45 = rotated
exactly halfway, the worst a quad field can do.

Result edges are bucketed by *graph distance on the result mesh* from the quad
ring that sits on the guide (bands 0-2 / 3-5 / 6+), which is the honest way to
ask "how far does the influence reach" - it is measured in output quad rings,
not input triangles.  ``nat`` reports the same angle against the fixture's
natural flow direction (parallels / grid axis / cylinder axis), so on the
adversarial fixture one can see *what wins where*.

Every fixture runs on both backends, with and without guides (the no-guide
control column), and renders a Workbench wire close-up with the guide curve
drawn in the QuadForge accent colour.

Outputs
-------
    stdout                              banded comparison table
    <OUT>/bench_guides.json
    <OUT>/<fixture>_<backend>_<guides|control>.png
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

REPO_ROOT = "/run/media/nerdrx/Lex/claude/quadforge"
HERE = os.path.join(REPO_ROOT, "tests")
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(HERE)
except NameError:
    pass
for _p_ in (REPO_ROOT, HERE):
    if _p_ not in sys.path:
        sys.path.insert(0, _p_)

OUT_DIR = os.environ.get(
    "QF_GUIDE_BENCH_OUT",
    "/tmp/claude-1000/-run-media-nerdrx-Lex-claude/"
    "cbcfcc9a-ddfd-4fbb-8582-1c3b129cb280/scratchpad/guides",
)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import bench_native as bn  # noqa: E402  (sibling: scene/render/topology helpers)

BACKENDS = ("QUADRIFLOW", "NATIVE")
BANDS = (("0-2", 0, 2), ("3-5", 3, 5), ("6+", 6, 1 << 30))

# a result vertex counts as "on the guide" when it is closer to the guide
# polyline than this many result-edge lengths
SEED_FRAC = 0.75
MIN_SEEDS = 6

# acceptance thresholds for the native backend (task brief)
GATE_BAND02 = 8.0
GATE_BAND35 = 15.0
GATE_FIXTURES = ("sphere_gc", "grid_scurve")

ACCENT = (0.467, 0.0, 1.0, 1.0)     # #7700FF, the guide curve in the renders


def _p(*args):
    print(*args)
    sys.stdout.flush()


# ==========================================================================
# geometry helpers
# ==========================================================================

def rosy4_deg(edge, target, normal):
    """4-RoSy angle in degrees (0..45) between two directions in a tangent plane.

    Both vectors are projected onto the plane of ``normal`` first.  Returns
    ``None`` when either projection degenerates.
    """
    n = normal / max(float(np.linalg.norm(normal)), 1e-12)
    e = edge - n * float(np.dot(edge, n))
    t = target - n * float(np.dot(target, n))
    el = float(np.linalg.norm(e))
    tl = float(np.linalg.norm(t))
    if el < 1e-9 or tl < 1e-9:
        return None
    c = abs(float(np.dot(e / el, t / tl)))
    ang = math.degrees(math.acos(min(1.0, max(0.0, c))))     # 0..90
    return min(ang, 90.0 - ang)                              # 4-RoSy fold


def densify(points, step):
    """Resample a world-space polyline and return ``(P (m,3), T (m,3) unit)``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = [pts[0]]
    for i in range(1, len(pts)):
        a, b = out[-1], pts[i]
        d = float(np.linalg.norm(b - a))
        if d <= 1e-12:
            continue
        k = int(d / step)
        for j in range(1, k + 1):
            out.append(a + (b - a) * (j / (k + 1.0)))
        out.append(b)
    P = np.asarray(out, dtype=np.float64)
    if len(P) < 2:
        return P, np.zeros_like(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    ln = np.sqrt(np.einsum("ij,ij->i", T, T))
    T /= np.maximum(ln, 1e-12)[:, None]
    return P, T


class GuideRef:
    """Dense sampling of the guide polylines + nearest-point tangent lookup."""

    def __init__(self, curve_objs, step):
        import quadforge.core.guides as qf_guides
        dg = bpy.context.evaluated_depsgraph_get()
        lines = qf_guides.sample_guides(list(curve_objs), dg)
        if not lines:
            raise RuntimeError("guide curve produced no polyline")
        Ps, Ts = [], []
        for ln in lines:
            P, T = densify([tuple(v) for v in ln], step)
            if len(P) >= 2:
                Ps.append(P)
                Ts.append(T)
        if not Ps:
            raise RuntimeError("guide polyline too short")
        self.P = np.concatenate(Ps, axis=0)
        self.T = np.concatenate(Ts, axis=0)
        self.length = float(sum(
            np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in Ps))

    def floor(self, step, normal_fn, mask_fn):
        """Best score a quad ring of edge length ``step`` could possibly get.

        A quad edge is a *chord* of the guide, so a guide that turns by more
        than a few degrees per quad cannot be followed exactly by any quad
        mesh.  Walking the guide in ``step``-long chords and scoring them with
        the same metric gives the discretisation floor of the fixture, which
        is what the measured numbers have to be read against.
        """
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        if arc[-1] < 2.0 * step:
            return None
        picks = np.searchsorted(arc, np.arange(0.0, arc[-1], step))
        picks = np.unique(np.clip(picks, 0, len(self.P) - 1))
        if len(picks) < 3:
            return None
        a, b = self.P[picks[:-1]], self.P[picks[1:]]
        mid = 0.5 * (a + b)
        keep = mask_fn(mid)
        if not keep.any():
            return None
        a, b, mid = a[keep], b[keep], mid[keep]
        _d, tang = self.nearest(mid)
        nrm = normal_fn(mid)
        vals = [rosy4_deg(b[k] - a[k], tang[k], nrm[k]) for k in range(len(mid))]
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    def nearest(self, Q):
        """(k,3) query points -> (dist (k,), tangent (k,3)), chunked."""
        Q = np.asarray(Q, dtype=np.float64).reshape(-1, 3)
        idx = np.empty(len(Q), dtype=np.int64)
        dst = np.empty(len(Q), dtype=np.float64)
        chunk = max(1, int(4_000_000 / max(len(self.P), 1)))
        for i in range(0, len(Q), chunk):
            q = Q[i:i + chunk]
            d = ((q[:, None, :] - self.P[None, :, :]) ** 2).sum(axis=2)
            j = np.argmin(d, axis=1)
            idx[i:i + chunk] = j
            dst[i:i + chunk] = np.sqrt(d[np.arange(len(q)), j])
        return dst, self.T[idx]


def band_of_verts(V, E, seed_mask, max_ring=64):
    """Graph distance (in edges) from ``seed_mask``; -1 where unreachable."""
    nv = len(V)
    band = np.full(nv, -1, dtype=np.int64)
    if not seed_mask.any() or not len(E):
        return band
    band[seed_mask] = 0
    cur = seed_mask.copy()
    for k in range(1, max_ring + 1):
        m = cur[E[:, 0]] | cur[E[:, 1]]
        if not m.any():
            break
        nxt = np.zeros(nv, dtype=bool)
        nxt[E[m].ravel()] = True
        newly = nxt & (band < 0)
        if not newly.any():
            break
        band[newly] = k
        cur = newly
    return band


# ==========================================================================
# fixtures
# ==========================================================================

def render_guides(curves, diag):
    """Give the guide curves a visible radius for the render only."""
    for cu in curves:
        try:
            cu.data.bevel_depth = diag * 0.004
        except Exception:
            pass


def _link_guides(objs, name="QF Guides"):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    for ob in objs:
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        coll.objects.link(ob)
        ob.color = ACCENT
    return coll


def _poly_curve(name, pts):
    # NO bevel: a bevelled curve's `to_mesh()` is a *tube*, and both
    # core.guides.sample_guides and this file's ground truth walk that mesh's
    # edges - the polyline then spirals around the tube and every tangent
    # estimate picks up the bevel radius as noise (measured: it put the
    # discretisation floor of the grid fixture at 14.7 deg instead of 1.4).
    # The renders bevel the curve afterwards, see render_guides().
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 0.0
    cu.resolution_u = 1
    sp = cu.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
    ob = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _mesh_object(name, V, F):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in V],
                   [], [tuple(map(int, f)) for f in F])
    me.update(calc_edges=True)
    me.validate(verbose=False)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


# --- (a) sphere + tilted great circle --------------------------------------

SPHERE_TILT = math.radians(30.0)


def fx_sphere_gc():
    # built by hand rather than with bpy.ops: the primitive operators emit
    # identical coordinates but a *different edge / loop / polygon ordering*
    # on every call, and float summation order alone moves the solver's face
    # count by a couple of percent.  Deterministic fixtures, deterministic run.
    nu, nv = 64, 32
    th = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    ph = np.linspace(0.0, np.pi, nv + 1)[1:-1]         # interior latitude rings
    ct, st = np.cos(th), np.sin(th)
    V = [(0.0, 0.0, 1.0)]
    for p in ph:
        sp, cp = math.sin(p), math.cos(p)
        for i in range(nu):
            V.append((sp * ct[i], sp * st[i], cp))
    V.append((0.0, 0.0, -1.0))
    nring = len(ph)
    top, bot = 0, len(V) - 1

    def vid(r, i):
        return 1 + r * nu + (i % nu)

    F = [(top, vid(0, i + 1), vid(0, i)) for i in range(nu)]
    for r in range(nring - 1):
        F += [(vid(r, i), vid(r, i + 1), vid(r + 1, i + 1), vid(r + 1, i))
              for i in range(nu)]
    F += [(bot, vid(nring - 1, i), vid(nring - 1, i + 1)) for i in range(nu)]
    ob = _mesh_object("sphere_gc", np.asarray(V, dtype=np.float64), F)

    bpy.ops.curve.primitive_bezier_circle_add(radius=1.0, enter_editmode=False,
                                              location=(0.0, 0.0, 0.0),
                                              rotation=(SPHERE_TILT, 0.0, 0.0))
    cu = bpy.context.object
    cu.name = cu.data.name = "guide_great_circle"
    cu.data.resolution_u = 24
    cu.data.bevel_depth = 0.0
    coll = _link_guides([cu])
    bpy.context.view_layer.objects.active = ob
    return ob, [cu], coll


def sphere_normal(P):
    return P / np.maximum(np.linalg.norm(P, axis=1), 1e-12)[:, None]


def sphere_natural(P):
    """Direction of the UV parallels (z x p) - the sphere's own grid flow."""
    z = np.array([0.0, 0.0, 1.0])
    d = np.cross(np.broadcast_to(z, P.shape), P)
    return d


def sphere_mask(P):
    return np.ones(len(P), dtype=bool)


# --- (b) flat grid + S curve ------------------------------------------------

GRID_HALF = 2.0
SCURVE_AMP = 0.9
SCURVE_HALF = 1.7


def fx_grid_scurve():
    n = 64                                    # quads per side (see fx_sphere_gc)
    g = np.linspace(-GRID_HALF, GRID_HALF, n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    F = [(i * (n + 1) + j, (i + 1) * (n + 1) + j,
          (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1)
         for i in range(n) for j in range(n)]
    ob = _mesh_object("grid_scurve", V, F)
    t = np.linspace(-SCURVE_HALF, SCURVE_HALF, 240)
    pts = np.stack([t,
                    SCURVE_AMP * np.sin(np.pi * t / SCURVE_HALF),
                    np.full_like(t, 0.15)], axis=1)
    cu = _poly_curve("guide_s_curve", pts)
    coll = _link_guides([cu])
    bpy.context.view_layer.objects.active = ob
    return ob, [cu], coll


def grid_normal(P):
    return np.broadcast_to(np.array([0.0, 0.0, 1.0]), P.shape).copy()


def grid_natural(P):
    return np.broadcast_to(np.array([1.0, 0.0, 0.0]), P.shape).copy()


def grid_mask(P):
    lim = GRID_HALF * 0.94
    return (np.abs(P[:, 0]) < lim) & (np.abs(P[:, 1]) < lim)


# --- (c) open cylinder + 45-degree helix ------------------------------------

CYL_R = 1.0
CYL_HALF = 2.0
HELIX_TURN = 1.75           # radians either side of theta = 0


def fx_cyl_helix():
    nu, nv = 64, 48
    th = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    z = np.linspace(-CYL_HALF, CYL_HALF, nv)
    TH, Z = np.meshgrid(th, z, indexing="ij")
    V = np.stack([CYL_R * np.cos(TH).ravel(),
                  CYL_R * np.sin(TH).ravel(), Z.ravel()], axis=1)

    def vid(i, j):
        return (i % nu) * nv + j

    F = [(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1))
         for i in range(nu) for j in range(nv - 1)]
    ob = _mesh_object("cyl_helix", V, F)
    # 45 degrees: moving by d(theta) travels R*d(theta) around and dz = R*d(theta) up
    t = np.linspace(-HELIX_TURN, HELIX_TURN, 260)
    pts = np.stack([CYL_R * np.cos(t), CYL_R * np.sin(t), CYL_R * t], axis=1)
    cu = _poly_curve("guide_helix", pts)
    coll = _link_guides([cu])
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    return ob, [cu], coll


def cyl_normal(P):
    d = P.copy()
    d[:, 2] = 0.0
    return d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]


def cyl_natural(P):
    return np.broadcast_to(np.array([0.0, 0.0, 1.0]), P.shape).copy()


def cyl_mask(P):
    # stay clear of the open rims: their boundary constraint owns the flow there
    return np.abs(P[:, 2]) < CYL_HALF * 0.82


FIXTURES = [
    dict(key="sphere_gc", build=fx_sphere_gc, target=1600,
         normal=sphere_normal, natural=sphere_natural, mask=sphere_mask,
         note="uv-sphere 64x32 + great-circle bezier tilted 30 deg"),
    dict(key="grid_scurve", build=fx_grid_scurve, target=1500,
         normal=grid_normal, natural=grid_natural, mask=grid_mask,
         note="flat grid 64x64 + S-curve guide"),
    dict(key="cyl_helix", build=fx_cyl_helix, target=1800,
         normal=cyl_normal, natural=cyl_natural, mask=cyl_mask,
         note="open cylinder 64x48 + 45-deg helix (fights curvature flow)"),
]
QUICK_KEYS = ("sphere_gc", "grid_scurve")


# ==========================================================================
# measurement
# ==========================================================================

def measure(me, fx, guide):
    """Banded 4-RoSy misalignment of the result edges against the guide."""
    V, _T, E = bn.read_mesh(me)
    out = {"bands": {}, "seed_radius": None, "seeds": 0}
    if not len(E):
        return out

    elen = np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1)
    mean_edge = float(elen.mean()) or 1.0

    dv, _tv = guide.nearest(V)
    radius = SEED_FRAC * mean_edge
    for _ in range(8):
        seed = dv < radius
        if int(seed.sum()) >= MIN_SEEDS:
            break
        radius *= 1.5
    out["seed_radius"] = round(radius, 5)
    out["seeds"] = int(seed.sum())
    if not seed.any():
        return out

    band_v = band_of_verts(V, E, seed)
    band_e = np.minimum(band_v[E[:, 0]], band_v[E[:, 1]])
    both = (band_v[E[:, 0]] >= 0) & (band_v[E[:, 1]] >= 0)
    band_e = np.where(both, band_e, np.maximum(band_v[E[:, 0]], band_v[E[:, 1]]))

    mid = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
    keep = fx["mask"](mid) & (band_e >= 0) & (elen > 1e-9)
    if not keep.any():
        return out
    idx = np.nonzero(keep)[0]

    _dist, tang = guide.nearest(mid[idx])
    nrm = fx["normal"](mid[idx])
    nat = fx["natural"](mid[idx])
    edir = V[E[idx, 1]] - V[E[idx, 0]]

    per_band = {name: {"guide": [], "nat": [], "d": []} for name, _, _ in BANDS}
    for k in range(len(idx)):
        b = int(band_e[idx[k]])
        for name, lo, hi in BANDS:
            if lo <= b <= hi:
                g = rosy4_deg(edir[k], tang[k], nrm[k])
                a = rosy4_deg(edir[k], nat[k], nrm[k])
                if g is not None:
                    per_band[name]["guide"].append(g)
                    per_band[name]["d"].append(float(_dist[k]))
                if a is not None:
                    per_band[name]["nat"].append(a)
                break

    for name, _lo, _hi in BANDS:
        g = per_band[name]["guide"]
        a = per_band[name]["nat"]
        out["bands"][name] = {
            "n": len(g),
            "guide_med": float(np.median(g)) if g else None,
            "guide_p75": float(np.percentile(g, 75)) if g else None,
            "nat_med": float(np.median(a)) if a else None,
            "dist_med": float(np.median(per_band[name]["d"])) if g else None,
        }
    out["mean_edge"] = round(mean_edge, 5)
    out["floor_deg"] = guide.floor(mean_edge, fx["normal"], fx["mask"])
    return out


# ==========================================================================
# one (fixture, backend, guides on/off) run
# ==========================================================================

def run_case(pipeline, fx, backend, use_guides, render=True, seed=0):
    key = fx["key"]
    variant = "guides" if use_guides else "control"
    row = {"fixture": key, "backend": backend, "variant": variant,
           "target": fx["target"], "note": fx["note"], "error": None,
           "render": None, "seed": seed}

    bn.fresh_scene()
    obj, curves, coll = fx["build"]()
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    except Exception:
        pass

    V, _T, E = bn.read_mesh(obj.data)
    step = float(np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1).mean()) * 0.25
    guide = GuideRef(curves, step)
    row["guide_len"] = round(guide.length, 4)
    row["input_faces"] = len(obj.data.polygons)
    row["input_verts"] = len(obj.data.vertices)

    s = obj.quadforge
    s.mode = 'FACES'
    s.target_count = int(fx["target"])
    s.strict_count = False
    s.keep_original = False
    s.backend = backend
    s.seed = int(seed)
    s.adaptive_size = 0.0
    s.symmetry_x = s.symmetry_y = s.symmetry_z = False
    s.use_paint_density = False
    s.use_guides = bool(use_guides)
    if use_guides:
        s.guide_collection = coll

    t0 = time.perf_counter()
    try:
        res = pipeline.run_remesh(bpy.context, obj, s)
    except Exception as exc:
        row["time_s"] = round(time.perf_counter() - t0, 3)
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
        _p("   !! %s/%s/%s raised:\n%s"
           % (key, backend, variant, traceback.format_exc().rstrip()))
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
    row["guide_edges"] = rep.get("guide_edges")
    # guided QuadriFlow solves are deliberately routed to the native solver
    # (QuadriFlow has no constraint channel); show that, don't fail on it
    row["rerouted"] = bool(rep.get("guides_rerouted"))
    if row["backend_used"] != backend and not row["rerouted"]:
        row["error"] = "fell back to %s" % row["backend_used"]

    me = out.data
    st = bn.topology_stats(me)
    row["faces"] = st["faces"]
    row["quad_pct"] = st["quad_pct"]
    row["poles_per_1k"] = st["poles_per_1k"]
    row.update(measure(me, fx, guide))

    if render:
        render_guides(curves, float(np.linalg.norm(
            V.max(axis=0) - V.min(axis=0))))
        png = os.path.join(OUT_DIR, "%s_%s_%s.png" % (key, backend, variant))
        row["render"] = bn.render_wire(out, png)
        if row["render"]:
            row["render_bytes"] = os.path.getsize(png)
    return row


# ==========================================================================
# reporting
# ==========================================================================

def _f(v, spec="%.1f"):
    return "-" if v is None else (spec % v)


def print_table(rows):
    head = ("%-12s %-11s %-8s %6s %6s %5s | %-18s | %-18s | %-18s"
            % ("fixture", "backend", "variant", "faces", "quad%", "floor",
               "band 0-2  n /guide/nat", "band 3-5  n /guide/nat",
               "band 6+   n /guide/nat"))
    _p(head)
    _p("-" * len(head))
    for r in rows:
        if r.get("error"):
            _p("%-12s %-11s %-8s  ERROR: %s"
               % (r["fixture"], r["backend"], r["variant"], r["error"]))
            continue
        cells = []
        for name, _lo, _hi in BANDS:
            b = (r.get("bands") or {}).get(name) or {}
            cells.append("%5s %5s %5s" % (b.get("n", 0),
                                          _f(b.get("guide_med")),
                                          _f(b.get("nat_med"))))
        label = "QF>NATIVE" if r.get("rerouted") else r["backend"]
        _p("%-12s %-11s %-8s %6d %6.1f %5s | %-18s | %-18s | %-18s"
           % (r["fixture"], label, r["variant"], r["faces"],
              r["quad_pct"], _f(r.get("floor_deg")),
              cells[0], cells[1], cells[2]))


def print_delta(rows):
    """guide-vs-control improvement per fixture/backend/band."""
    _p("")
    _p("GUIDE EFFECT (control median - guided median, degrees; + = the guide helped)")
    by = {}
    for r in rows:
        if r.get("error"):
            continue
        by[(r["fixture"], r["backend"], r["variant"])] = r
    for fx in FIXTURES:
        for backend in BACKENDS:
            g = by.get((fx["key"], backend, "guides"))
            c = by.get((fx["key"], backend, "control"))
            if not g or not c:
                continue
            parts = []
            for name, _lo, _hi in BANDS:
                gb = (g.get("bands") or {}).get(name) or {}
                cb = (c.get("bands") or {}).get(name) or {}
                if gb.get("guide_med") is None or cb.get("guide_med") is None:
                    parts.append("%s: -" % name)
                else:
                    parts.append("%s: %+5.1f" % (name, cb["guide_med"] - gb["guide_med"]))
            _p("  %-12s %-11s  %s" % (fx["key"], backend, "   ".join(parts)))


def evaluate_gates(rows):
    _p("")
    _p("GATES (native, guided: band 0-2 <= %.0f deg, band 3-5 <= %.0f deg)"
       % (GATE_BAND02, GATE_BAND35))
    out = []
    for r in rows:
        if r["backend"] != "NATIVE" or r["variant"] != "guides":
            continue
        if r["fixture"] not in GATE_FIXTURES:
            continue
        fails = []
        if r.get("error"):
            fails.append(r["error"])
        b02 = ((r.get("bands") or {}).get("0-2") or {}).get("guide_med")
        b35 = ((r.get("bands") or {}).get("3-5") or {}).get("guide_med")
        if b02 is None or b02 > GATE_BAND02:
            fails.append("band 0-2 = %s (want <= %.0f)" % (_f(b02), GATE_BAND02))
        if b35 is None or b35 > GATE_BAND35:
            fails.append("band 3-5 = %s (want <= %.0f)" % (_f(b35), GATE_BAND35))
        ok = not fails
        _p("  %-12s %s%s" % (r["fixture"], "PASS" if ok else "FAIL",
                             "" if ok else "  (" + "; ".join(fails) + ")"))
        out.append((r["fixture"], ok, fails))
    return out


# ==========================================================================
# main
# ==========================================================================

def parse_args():
    argv = sys.argv
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    return {"quick": "--quick" in extra, "norender": "--no-render" in extra,
            "legacy": "--legacy-guides" in extra}


def use_legacy_guides():
    """Restore the pre-fix guide handling (sharp path wins, no soft halo).

    Only for the before/after column of this benchmark - it flips the two
    documented knobs in ``fields.FIELD_DEFAULTS``, nothing else.
    """
    from quadforge.backends.native import fields as _f
    _f.FIELD_DEFAULTS["guides_win"] = False
    _f.FIELD_DEFAULTS["guide_falloff"] = 0.0


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    _p("== QuadForge guide-quality benchmark ==")
    _p(".. blender %s   numpy %s" % (bpy.app.version_string, np.__version__))
    _p(".. repo    %s" % REPO_ROOT)
    _p(".. out     %s" % OUT_DIR)

    try:
        import quadforge
        quadforge.register()
        _p(".. quadforge.register() ok")
    except Exception as exc:
        _p("!! quadforge.register() failed: %s: %s" % (type(exc).__name__, exc))
        _p(traceback.format_exc().rstrip())
        return 1

    import quadforge.pipeline as pipeline

    if args["legacy"]:
        use_legacy_guides()
        _p(".. --legacy-guides: pre-fix guide handling")

    keys = QUICK_KEYS if args["quick"] else tuple(f["key"] for f in FIXTURES)
    fixtures = [f for f in FIXTURES if f["key"] in keys]

    t_start = time.time()
    rows = []
    for fx in fixtures:
        for backend in BACKENDS:
            for use_guides in (True, False):
                _p("")
                _p(">> %s / %s / %s  (target %d)"
                   % (fx["key"], backend, "guides" if use_guides else "control",
                      fx["target"]))
                try:
                    row = run_case(pipeline, fx, backend, use_guides,
                                   render=not args["norender"])
                except Exception as exc:
                    _p(traceback.format_exc().rstrip())
                    row = {"fixture": fx["key"], "backend": backend,
                           "variant": "guides" if use_guides else "control",
                           "target": fx["target"], "note": fx["note"],
                           "error": "bench harness: %s: %s"
                                    % (type(exc).__name__, exc)}
                rows.append(row)
                if row.get("error"):
                    _p("   error: %s" % row["error"])
                else:
                    b = (row.get("bands") or {}).get("0-2") or {}
                    _p("   faces=%d quad=%.1f%% guide_edges=%s band0-2=%s deg "
                       "(n=%s) t=%.1fs"
                       % (row["faces"], row["quad_pct"], row.get("guide_edges"),
                          _f(b.get("guide_med")), b.get("n"), row["time_s"]))

    # determinism: the same seed must give the same guided result
    det = {"ok": None}
    try:
        fx0 = fixtures[0]
        a = run_case(pipeline, fx0, "NATIVE", True, render=False, seed=3)
        b = run_case(pipeline, fx0, "NATIVE", True, render=False, seed=3)
        ka = (a.get("faces"), ((a.get("bands") or {}).get("0-2") or {}).get("guide_med"))
        kb = (b.get("faces"), ((b.get("bands") or {}).get("0-2") or {}).get("guide_med"))
        det = {"ok": ka == kb, "a": ka, "b": kb, "fixture": fx0["key"]}
    except Exception as exc:
        det = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    elapsed = time.time() - t_start

    _p("")
    _p("=" * 100)
    _p("GUIDE ALIGNMENT TABLE   (median 4-RoSy angle in degrees: 0 = on the")
    _p("guide, 22.5 = random field, 45 = worst.  'nat' = same angle against the")
    _p("fixture's natural flow.  Bands are result-mesh graph rings from the guide.)")
    _p("=" * 100)
    print_table(rows)
    print_delta(rows)
    gates = evaluate_gates(rows)
    _p("")
    _p("determinism (native, guided, seed 3, x2): %s  %s"
       % ("OK" if det.get("ok") else "FAIL", det))

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "blender": bpy.app.version_string,
        "repo": REPO_ROOT,
        "quick": args["quick"],
        "elapsed_s": round(elapsed, 2),
        "gates": {"band_0_2": GATE_BAND02, "band_3_5": GATE_BAND35,
                  "fixtures": list(GATE_FIXTURES)},
        "gate_summary": [{"fixture": k, "pass": ok, "failures": f}
                         for k, ok, f in gates],
        "determinism": det,
        "rows": rows,
    }
    json_path = os.path.join(OUT_DIR, "bench_guides.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    _p("")
    _p("renders:")
    for r in rows:
        if r.get("render"):
            _p("  %-44s %8.1f KB" % (os.path.basename(r["render"]),
                                     r.get("render_bytes", 0) / 1024.0))
    _p("json:    %s" % json_path)
    _p("BENCH TOTAL %.1fs" % elapsed)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        _p(traceback.format_exc().rstrip())
        code = 0
    sys.stdout.flush()
    sys.exit(code)
