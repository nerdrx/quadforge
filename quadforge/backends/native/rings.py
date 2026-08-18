"""Concentric edge loops around mesh openings ("opening rings").

Why this module exists
----------------------
The v2 orientation field aligns to *principal curvature*, which is the right
answer almost everywhere and the wrong answer around a hole with a flat rim.
An eye socket cut into a head, the rim of a mouth bag, an ear canal: the
surface around them is a nearly developable collar whose principal directions
are either undefined (isotropic) or dictated by the larger form the collar
sits on, so the field walks straight past the hole and the extractor produces
a grid that is merely *clipped* by the opening instead of *ringing* it.

Artists ring those openings on purpose: concentric loops around an eye or a
mouth are what makes the shape deform correctly and what makes the model read
as hand-made.  This module produces exactly that instruction as data:

    detect_openings(...)      which boundary loops are openings
    opening_ring_field(...)   per-vertex ring direction + a decaying weight

The direction at a vertex is the tangent of the *offset ring* through it -
the iso-curve of the geodesic distance to the opening - i.e. perpendicular to
the distance gradient.  That is the direction a hand-placed loop would take.

The result is handed to :func:`fields.solve_fields` through the machinery the
guide system already owns (4-RoSy accumulation, the soft alignment channel
that is restricted onto every hierarchy level, and the hard constraints of the
rim itself), so nothing here duplicates the field solver.

Nothing in this module runs unless ``use_opening_rings`` is on; with the flag
off ``solve_fields`` never imports it and the field is bit-identical.
"""

from __future__ import annotations

import numpy as np

from . import fields as _f

__all__ = [
    "Opening",
    "detect_openings",
    "boundary_loops",
    "opening_ring_field",
    "RING_DEFAULTS",
]

EPS = 1e-12

RING_DEFAULTS = {
    # band width, in output-quad widths, over which the ring instruction
    # decays to nothing
    "ring_falloff": 6.0,
    # ...held at full strength for this many quads first.  A pure decay from
    # the rim spends most of the band already weak: measured on the disc
    # fixture at a 6-quad falloff, a 2-quad plateau moves the 4-quad-band
    # alignment from ~11 to ~7.8 degrees for free.
    "ring_plateau": 2.0,
    # Weight at the rim, on the same 0..1 scale as the curvature alignment
    # (1 = snap the field onto the ring, 0 = ignore it).  Deliberately below
    # 1: the ring count *has* to change as the band widens, which needs
    # irregular vertices, which the field can only place where it is free to
    # deviate from the ideal polar direction.  0.6 measured best across the
    # two fixtures and three seeds and is also the least brittle.
    "ring_strength": 0.6,
    # an opening is a *closed* boundary loop shorter than this fraction of
    # 4*sqrt(area) - i.e. shorter than the perimeter of a square patch
    # covering that fraction of the surface.  A plane's outer border, an open
    # cylinder's ends on a big shell and the symmetry-bisect cut are all well
    # above it; eye sockets, mouth rims and ear holes are well below.
    "ring_max_perimeter": 0.15,
    # ...and long enough to be worth ringing, in output quads
    "ring_min_perimeter_quads": 3.0,
    # a vertex within this distance of a symmetry plane counts as "on" it
    # (relative to the bounding-box diagonal)
    "ring_sym_eps": 1e-5,
}


class Opening:
    """One detected opening.  ``verts`` is the boundary cycle in order,
    ``seed`` the subset used as the distance source (the symmetry-cut part of
    a bisected rim is excluded), ``perimeter`` is in world units."""

    __slots__ = ("verts", "seed", "perimeter", "center")

    def __init__(self, verts, seed, perimeter, center):
        self.verts = verts
        self.seed = seed
        self.perimeter = float(perimeter)
        self.center = center

    def __repr__(self):  # pragma: no cover - debugging aid
        return ("Opening(n=%d, perim=%.4f, center=[%.3f %.3f %.3f])"
                % (len(self.verts), self.perimeter, *self.center))


# --------------------------------------------------------------------------
# boundary loop tracing
# --------------------------------------------------------------------------

def boundary_loops(F, n):
    """Trace the mesh boundary into connected components.

    Returns a list of ``(vert_indices, edge_rows, closed)``.  ``closed`` is
    True only for a component in which every vertex has exactly two boundary
    edges, i.e. a clean cycle; ``vert_indices`` is then in cycle order.  A
    non-manifold boundary junction (a vertex with 4+ boundary edges, which
    happens on self-touching sculpts) yields ``closed=False`` and is skipped
    by the caller rather than guessed at.
    """
    be = _f.boundary_edges(F)
    out = []
    if not len(be):
        return out

    bv = np.unique(be)
    # local numbering over boundary vertices only
    loc = np.full(n, -1, dtype=np.int64)
    loc[bv] = np.arange(len(bv))
    a = loc[be[:, 0]]
    b = loc[be[:, 1]]
    m = len(bv)

    deg = np.bincount(np.concatenate([a, b]), minlength=m)

    # connected components over the boundary graph (iterative label pushing;
    # boundary graphs are tiny compared to the mesh)
    lab = np.arange(m, dtype=np.int64)
    while True:
        nl = lab.copy()
        np.minimum.at(nl, a, lab[b])
        np.minimum.at(nl, b, lab[a])
        # path-compress one step
        nl = np.minimum(nl, nl[nl])
        if np.array_equal(nl, lab):
            break
        lab = nl

    order = np.argsort(lab, kind="stable")
    lsorted = lab[order]
    starts = np.flatnonzero(np.r_[True, lsorted[1:] != lsorted[:-1]])
    bounds = np.r_[starts, len(order)]

    # adjacency for cycle ordering
    nbr = [[] for _ in range(m)]
    for i, j in zip(a.tolist(), b.tolist()):
        nbr[i].append(j)
        nbr[j].append(i)

    for k in range(len(starts)):
        comp = order[bounds[k]:bounds[k + 1]]
        if len(comp) < 3:
            continue
        closed = bool(np.all(deg[comp] == 2))
        if closed:
            ring = [int(comp[0])]
            prev = -1
            cur = int(comp[0])
            ok = True
            for _ in range(len(comp) - 1):
                nb = nbr[cur]
                nxt = nb[0] if nb[0] != prev else nb[1]
                if nxt == ring[0]:
                    ok = False       # closed early -> not a single cycle
                    break
                ring.append(nxt)
                prev, cur = cur, nxt
            if ok and len(ring) == len(comp):
                out.append((bv[np.asarray(ring, dtype=np.int64)], None, True))
                continue
            closed = False
        out.append((bv[comp], None, False))
    return out


# --------------------------------------------------------------------------
# opening detection
# --------------------------------------------------------------------------

def detect_openings(V, F, symmetry=(False, False, False), params=None,
                    rho_mean=None):
    """Boundary loops that look like openings (eyes, mouth, ear holes).

    ``symmetry`` marks the axes on which the pipeline may have bisected the
    mesh before the backend ran.  A loop lying *entirely* in such a plane is
    the bisect cut, not an opening, and is dropped; a loop that merely touches
    the plane (a mouth bag rim cut in half) is kept, but the part of it that
    lies in the plane is excluded from the distance source so that the ring
    field is measured from the real rim and stays mirror-symmetric.
    """
    p = dict(RING_DEFAULTS)
    for k, v in (params or {}).items():
        if k in p and v is not None:
            p[k] = v

    V = np.asarray(V, dtype=np.float64)
    n = V.shape[0]
    _, areas = _f.face_normals_areas(V, F)
    area = float(np.sum(areas))
    if not np.isfinite(area) or area <= 0.0:
        return []
    max_perim = float(p["ring_max_perimeter"]) * 4.0 * np.sqrt(area)

    lo = V.min(axis=0)
    hi = V.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    seps = max(float(p["ring_sym_eps"]) * max(diag, EPS), 1e-12)

    min_perim = 0.0
    if rho_mean is not None and np.isfinite(rho_mean) and rho_mean > 0.0:
        min_perim = float(p["ring_min_perimeter_quads"]) * float(rho_mean)

    out = []
    for verts, _e, closed in boundary_loops(F, n):
        if not closed:
            continue
        P = V[verts]
        seg = P - np.roll(P, 1, axis=0)
        perim = float(np.sum(np.sqrt(np.einsum("ij,ij->i", seg, seg))))
        if perim > max_perim or perim < min_perim:
            continue

        on_plane = np.zeros(len(verts), dtype=bool)
        for ax in range(3):
            if not symmetry[ax]:
                continue
            on_plane |= np.abs(P[:, ax]) <= seps
        if on_plane.all():
            continue                      # the bisect cut itself
        seed = verts[~on_plane]
        if len(seed) < 3:
            seed = verts
        out.append(Opening(verts, seed, perim, P.mean(axis=0)))
    return out


# --------------------------------------------------------------------------
# geodesic-ish distance and its iso-curve tangent
# --------------------------------------------------------------------------

def _band_distance(V, edges, seed_mask, dmax, max_sweeps=512):
    """Approximate geodesic distance from ``seed_mask``, cut off at ``dmax``.

    Bellman-Ford sweeps over the whole (undirected, length-weighted) edge set,
    each sweep a single segmented ``np.minimum.reduceat``, with everything
    beyond ``dmax`` clamped back to infinity so that the wavefront - and the
    cost - stays inside the band.  It converges in as many sweeps as the band
    is graph hops wide and then stops.

    The obvious cheaper thing, a hop-limited BFS, is what this replaced: the
    hop budget has to be guessed from a *mean* edge length, and the mesh
    around an opening is exactly where a sculpt is finest, so the guess ran
    the band short - measured on the disc fixture, a 6-quad band stopped at
    3.7 quads and the outer half of it silently got no ring instruction.
    """
    n = V.shape[0]
    d = np.full(n, np.inf)
    if not seed_mask.any() or not len(edges) or not np.isfinite(dmax):
        return d, np.zeros(n, dtype=bool)

    e0 = np.concatenate([edges[:, 0], edges[:, 1]])
    e1 = np.concatenate([edges[:, 1], edges[:, 0]])
    dv = V[e1] - V[e0]
    w = np.sqrt(np.einsum("ij,ij->i", dv, dv))

    # group the directed edges by target so a sweep is a segmented min
    order = np.argsort(e1, kind="stable")
    src = e0[order]
    wt = w[order]
    tgt = e1[order]
    starts = np.flatnonzero(np.r_[True, tgt[1:] != tgt[:-1]])
    tv = tgt[starts]

    d[seed_mask] = 0.0
    big = 1e30
    dd = np.where(np.isfinite(d), d, big)
    for _ in range(int(max_sweeps)):
        cand = np.minimum.reduceat(dd[src] + wt, starts)
        new = np.minimum(dd[tv], cand)
        new = np.where(new > dmax, big, new)
        upd = new < dd[tv] - 1e-15
        if not upd.any():
            break
        dd[tv[upd]] = new[upd]
    d = np.where(dd >= big * 0.5, np.inf, dd)
    return d, np.isfinite(d)


def _smooth_scalar(d, region, seed_mask, edges, iters):
    """A few Jacobi passes on ``d`` inside ``region`` (seeds stay pinned).

    The BFS/Bellman-Ford distance is piecewise-linear with kinks at the wave
    fronts; the gradient of the raw field is therefore noisy at exactly the
    scale we care about.  Smoothing first costs nothing and takes the median
    ring error down by several degrees.
    """
    n = len(d)
    fin = np.isfinite(d) & region
    if iters <= 0 or not fin.any():
        return d
    e = edges[fin[edges[:, 0]] & fin[edges[:, 1]]]
    if not len(e):
        return d
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(src, minlength=n).astype(np.float64)
    free = fin & ~seed_mask & (deg > 0)
    if not free.any():
        return d
    out = d.copy()
    out[~fin] = 0.0
    for _ in range(int(iters)):
        acc = np.bincount(src, weights=out[dst], minlength=n)
        avg = (acc + out * 2.0) / (deg + 2.0)
        out[free] = avg[free]
    out[~fin] = np.inf
    return out


def _iso_tangent(V, F, N, d, region):
    """Tangent of the iso-distance curve of ``d`` at every band vertex.

    The gradient of the piecewise-linear ``d`` is constant per triangle
    (``grad = sum_i d_i * (n x e_i) / 2A`` with ``e_i`` the edge opposite to
    corner ``i``); it is accumulated onto the vertices with the triangle
    areas, projected into the vertex tangent plane and rotated by 90 degrees.

    Returns ``(dirs (n,3), conf (n,))``; ``conf`` is the gradient magnitude
    normalised by its band median, which collapses on the medial ridge
    between two nearby openings - exactly where no ring direction exists.
    """
    n = V.shape[0]
    dirs = np.zeros((n, 3))
    conf = np.zeros(n)
    ok_f = region[F[:, 0]] & region[F[:, 1]] & region[F[:, 2]]
    ok_f &= np.isfinite(d[F[:, 0]]) & np.isfinite(d[F[:, 1]]) & np.isfinite(d[F[:, 2]])
    if not ok_f.any():
        return dirs, conf
    Ff = F[ok_f]
    p0, p1, p2 = V[Ff[:, 0]], V[Ff[:, 1]], V[Ff[:, 2]]
    cr = np.cross(p1 - p0, p2 - p0)
    a2 = np.sqrt(np.einsum("ij,ij->i", cr, cr))     # = 2A
    good = a2 > 1e-18
    if not good.any():
        return dirs, conf
    Ff, p0, p1, p2, cr, a2 = Ff[good], p0[good], p1[good], p2[good], cr[good], a2[good]
    nf = cr / a2[:, None]
    g = (np.cross(nf, p2 - p1) * d[Ff[:, 0]][:, None]
         + np.cross(nf, p0 - p2) * d[Ff[:, 1]][:, None]
         + np.cross(nf, p1 - p0) * d[Ff[:, 2]][:, None]) / a2[:, None]

    wgt = 0.5 * a2
    acc = np.zeros((n, 3))
    wsum = np.zeros(n)
    for k in range(3):
        for c in range(3):
            acc[:, c] += np.bincount(Ff[:, k], weights=g[:, c] * wgt, minlength=n)
        wsum += np.bincount(Ff[:, k], weights=wgt, minlength=n)
    live = wsum > 0.0
    acc[live] /= wsum[live][:, None]
    acc -= N * np.einsum("ij,ij->i", acc, N)[:, None]
    gl = np.sqrt(np.einsum("ij,ij->i", acc, acc))

    band = live & region & (gl > 1e-12)
    if not band.any():
        return dirs, conf
    med = float(np.median(gl[band]))
    if med <= 1e-12:
        return dirs, conf
    conf[band] = np.clip(gl[band] / med, 0.0, 1.0)
    # rotate the gradient 90 degrees inside the tangent plane -> iso tangent
    t = np.cross(N[band], acc[band] / gl[band][:, None])
    tl = np.sqrt(np.einsum("ij,ij->i", t, t))
    fine = tl > 1e-9
    idx = np.nonzero(band)[0][fine]
    dirs[idx] = t[fine] / tl[fine][:, None]
    conf[np.nonzero(band)[0][~fine]] = 0.0
    return dirs, conf


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def opening_ring_field(V, F, N, edges, rho, symmetry=(False, False, False),
                       params=None):
    """Per-vertex ring direction and weight around every detected opening.

    Returns ``(dirs (n,3), w (n,), openings)``.  ``w`` is 0 outside the band
    and decays from ``ring_strength`` at the rim to 0 at ``ring_falloff``
    output-quad widths away.  ``dirs`` is a unit tangent wherever ``w > 0``.
    """
    p = dict(RING_DEFAULTS)
    for k, v in (params or {}).items():
        if k in p and v is not None:
            p[k] = v

    V = np.ascontiguousarray(np.asarray(V, dtype=np.float64).reshape(-1, 3))
    F = np.ascontiguousarray(np.asarray(F, dtype=np.int64).reshape(-1, 3))
    n = V.shape[0]
    rho = np.asarray(rho, dtype=np.float64).reshape(n)
    dirs = np.zeros((n, 3))
    w = np.zeros(n)

    rho_mean = float(np.mean(rho)) if n else 0.0
    openings = detect_openings(V, F, symmetry=symmetry, params=p,
                               rho_mean=rho_mean)
    if not openings:
        return dirs, w, openings

    seed_mask = np.zeros(n, dtype=bool)
    for op in openings:
        seed_mask[op.seed] = True
    if not seed_mask.any():
        return dirs, w, []

    band = float(p["ring_falloff"])
    # the band is `band` output quads wide and rho is per-vertex, so the world
    # cutoff has to use the largest rho in play; 1.25x of that keeps the
    # clamped edge of the distance field (where the gradient is one-sided)
    # outside the part of the band that actually carries weight
    fin = np.isfinite(rho) & (rho > 0.0)
    dmax = 1.25 * band * (float(np.max(rho[fin])) if fin.any() else rho_mean)

    d, region = _band_distance(V, edges, seed_mask, dmax)
    d = _smooth_scalar(d, region, seed_mask, edges, iters=4)
    dirs, conf = _iso_tangent(V, F, N, d, region)

    # distance in output-quad widths, so the falloff is resolution independent
    dq = np.where(np.isfinite(d), d / np.maximum(rho, EPS), np.inf)
    plateau = float(np.clip(p["ring_plateau"], 0.0, max(band - 0.25, 0.0)))
    t = np.clip((dq - plateau) / max(band - plateau, EPS), 0.0, 1.0)
    strength = float(np.clip(p["ring_strength"], 0.0, 1.0))
    w = strength * (1.0 - _f.smoothstep(t)) * conf
    w[~np.isfinite(dq)] = 0.0
    w[np.einsum("ij,ij->i", dirs, dirs) < 0.5] = 0.0
    return dirs, np.clip(w, 0.0, 1.0), openings
