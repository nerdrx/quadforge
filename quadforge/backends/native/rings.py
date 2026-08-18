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
    ring_bands(...)           the geodesic band around every opening
    ring_density(...)         local sample-density boost inside those bands
    ring_weights(...)         per-vertex ring direction + a decaying weight
    opening_ring_field(...)   the last two in one call (compatibility wrapper)

The direction at a vertex is the tangent of the *offset ring* through it -
the iso-curve of the geodesic distance to the opening - i.e. perpendicular to
the distance gradient.  That is the direction a hand-placed loop would take.

A direction on its own is not enough at a game-avatar budget.  An eye socket
on a 12k-face head is only ten-odd quads around, and a ring the field cannot
subdivide is a ring the extractor renders as a lumpy pentagon: the instruction
is followed and still reads as noise.  ``ring_density`` therefore also asks
for *resolution* - it raises the sample density inside the band until the
opening carries ``ring_min_quads`` quads around, capped, and pays for it out
of the same cell budget the rest of the field is drawn from (the interior
goes very slightly coarser; see :func:`fields.budget_scale`).

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
    "RingBands",
    "detect_openings",
    "boundary_loops",
    "ring_bands",
    "ring_density",
    "ring_weights",
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

    # ---- band density boost (ring_density) -------------------------------
    # How many output quads an opening should have around it before the ring
    # instruction is worth giving.  Below roughly this many, a "ring" is a
    # polygon: the extractor cannot place the irregular vertices that let the
    # ring count grow outwards, so the loops it does produce are visibly
    # faceted and read as noise rather than as eyelid loops.  Measured on the
    # Dinasty head at a 12k budget the socket carries ~16 quads before the
    # boost and the loops do not read; the boost buys the rest.
    "ring_min_quads": 32.0,
    # ...but never finer than this multiple of the surrounding rho.  An
    # uncapped rule would try to give a 3-quad ear hole a 26-quad collar and
    # spend the whole budget on it.
    "ring_density_max": 2.0,
    # profile of the boost, in *pre-boost* output quads from the rim: full
    # strength for `plateau`, then an exponential decay of length `decay`.
    # Same shape as the feature-density boost in solver.solve, and for the
    # same reason: a hard step in rho stitches a dense collar onto a coarse
    # interior and the seam shows up as a scraggly band of extra poles.
    "ring_density_plateau": 1.5,
    "ring_density_decay": 2.5,
    # ...and the bands together may never claim more than this fraction of
    # the cell budget on top of what they already had.  The budget is
    # conserved (fields.budget_scale), so this is really a cap on how much
    # coarser the rest of the mesh is allowed to get: 0.30 costs the interior
    # sqrt(1.3) = 14% in edge length, which is invisible; a head with nine
    # openings and no cap would cost it a lot more.
    "ring_density_budget": 0.30,

    # ---- extraction pinning (ring_pin_segments) --------------------------
    # Hand the extractor the iso-geodesic contour this many output quads out
    # from each rim as a feature curve, so the first offset ring is a closed
    # loop *by construction* instead of by hoping the field alignment survives
    # extraction.  Measured on the Dinasty eye socket it takes first-ring loop
    # purity from 0.70 to 0.81 at no cost to the face count.  1.5 rather than
    # the 1.0 the first lattice line nominally wants: the rim is pinned too,
    # so a contour a bare quad out asks the extractor to fit a row of cells
    # into a strip thinner than one, and on the disc fixture (seed 5) it duly
    # collapsed one into a valence-18 fan.
    "ring_pin_offset": 1.5,
    # ...and however the offset is set, a contour that comes closer than this
    # many quads to the rim anywhere is dropped rather than squeezed: the
    # snapped contour drifts inwards where the band is coarse, so the offset
    # on its own is not a guarantee.
    "ring_pin_min_gap": 0.9,
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

def boundary_loops(F, n, be=None):
    """Trace the mesh boundary into connected components.

    Returns a list of ``(vert_indices, edge_rows, closed)``.  ``closed`` is
    True only for a component in which every vertex has exactly two boundary
    edges, i.e. a clean cycle; ``vert_indices`` is then in cycle order.  A
    non-manifold boundary junction (a vertex with 4+ boundary edges, which
    happens on self-touching sculpts) yields ``closed=False`` and is skipped
    by the caller rather than guessed at.

    ``be`` may be supplied when the caller already knows the boundary edges
    (the pipeline reads them straight off a Blender mesh), in which case ``F``
    is not looked at.
    """
    if be is None:
        be = _f.boundary_edges(F)
    be = np.asarray(be, dtype=np.int64).reshape(-1, 2)
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
                    rho_mean=None, be=None, area=None):
    """Boundary loops that look like openings (eyes, mouth, ear holes).

    ``symmetry`` marks the axes on which the pipeline may have bisected the
    mesh before the backend ran.  A loop lying *entirely* in such a plane is
    the bisect cut, not an opening, and is dropped; a loop that merely touches
    the plane (a mouth bag rim cut in half) is kept, but the part of it that
    lies in the plane is excluded from the distance source so that the ring
    field is measured from the real rim and stays mirror-symmetric.

    ``be`` (boundary edges) and ``area`` may be supplied by a caller that
    already has them - the pipeline counts openings on a Blender mesh without
    triangulating it - in which case ``F`` is never looked at.
    """
    p = dict(RING_DEFAULTS)
    for k, v in (params or {}).items():
        if k in p and v is not None:
            p[k] = v

    V = np.asarray(V, dtype=np.float64)
    n = V.shape[0]
    if area is None:
        _, areas = _f.face_normals_areas(V, F)
        area = float(np.sum(areas))
    area = float(area)
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
    for verts, _e, closed in boundary_loops(F, n, be=be):
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

class RingBands:
    """The geodesic collar around every detected opening.

    ``dist`` is the (smoothed) geodesic distance to the nearest opening rim in
    world units, ``inf`` outside the band; ``dq0`` is the same distance in
    *pre-boost* output quads; ``dirs``/``conf`` are the iso-curve tangent and
    its trustworthiness; ``own`` names the opening each band vertex belongs to
    (``-1`` outside).  ``empty`` is True when nothing was detected, which is
    the overwhelmingly common case and the only one that has to stay free.
    """

    __slots__ = ("dirs", "conf", "dist", "dq0", "region", "own", "openings",
                 "seed_mask", "params")

    def __init__(self, dirs, conf, dist, dq0, region, own, openings,
                 seed_mask, params):
        self.dirs = dirs
        self.conf = conf
        self.dist = dist
        self.dq0 = dq0
        self.region = region
        self.own = own
        self.openings = openings
        self.seed_mask = seed_mask
        self.params = params

    @property
    def empty(self):
        return not self.openings


def _merged_params(params):
    p = dict(RING_DEFAULTS)
    for k, v in (params or {}).items():
        if k in p and v is not None:
            p[k] = v
    return p


def ring_bands(V, F, N, edges, rho, symmetry=(False, False, False),
               params=None):
    """Detect the openings and measure the geodesic collar around them.

    This is everything that does not depend on the *final* ``rho``, so that
    :func:`ring_density` may change ``rho`` inside the band and
    :func:`ring_weights` can then be evaluated against the changed value
    without the distance field being solved twice.
    """
    p = _merged_params(params)

    V = np.ascontiguousarray(np.asarray(V, dtype=np.float64).reshape(-1, 3))
    F = np.ascontiguousarray(np.asarray(F, dtype=np.int64).reshape(-1, 3))
    n = V.shape[0]
    rho = np.asarray(rho, dtype=np.float64).reshape(n)

    empty = RingBands(np.zeros((n, 3)), np.zeros(n), np.full(n, np.inf),
                      np.full(n, np.inf), np.zeros(n, dtype=bool),
                      np.full(n, -1, dtype=np.int64), [],
                      np.zeros(n, dtype=bool), p)

    rho_mean = float(np.mean(rho)) if n else 0.0
    openings = detect_openings(V, F, symmetry=symmetry, params=p,
                               rho_mean=rho_mean)
    if not openings:
        return empty

    seed_mask = np.zeros(n, dtype=bool)
    for op in openings:
        seed_mask[op.seed] = True
    if not seed_mask.any():
        return empty

    band = float(p["ring_falloff"])
    # The band is `band` output quads wide and rho is per-vertex, so the world
    # cutoff has to use the largest rho in play; 1.25x of that keeps the
    # clamped edge of the distance field (where the gradient is one-sided)
    # outside the part of the band that actually carries weight.  The density
    # boost can only ever make rho *smaller*, so a cutoff measured on the
    # pre-boost rho still contains the post-boost band.
    fin = np.isfinite(rho) & (rho > 0.0)
    dmax = 1.25 * band * (float(np.max(rho[fin])) if fin.any() else rho_mean)

    d, region = _band_distance(V, edges, seed_mask, dmax)
    d = _smooth_scalar(d, region, seed_mask, edges, iters=4)
    dirs, conf = _iso_tangent(V, F, N, d, region)
    dq0 = np.where(np.isfinite(d), d / np.maximum(rho, EPS), np.inf)

    # Which opening owns a band vertex.  Nearest *centre* rather than a second
    # labelled wavefront: the assignment only picks which opening's quad quota
    # applies, two openings close enough for the choice to be ambiguous have
    # near-identical quotas anyway, and Euclidean nearest-centre is exact,
    # cheap and order independent (a labelled Bellman-Ford is none of those).
    own = np.full(n, -1, dtype=np.int64)
    live = np.isfinite(d)
    if live.any():
        C = np.asarray([op.center for op in openings], dtype=np.float64)
        P = V[live]
        dd = np.einsum("ijk,ijk->ij", P[:, None, :] - C[None, :, :],
                       P[:, None, :] - C[None, :, :])
        own[live] = np.argmin(dd, axis=1)
    return RingBands(dirs, conf, d, dq0, region, own, openings, seed_mask, p)


def ring_density(bands, V, F, rho, areas=None, params=None):
    """Raise the sample density inside the ring bands, budget-conserving.

    Every opening gets a quota: enough quads around its rim for a ring to be
    a ring rather than a polygon (``ring_min_quads``).  The factor needed for
    that is ``quota * rho_rim / perimeter``, clamped to ``ring_density_max``,
    and it is applied over the band with the same plateau-then-decay profile
    the feature-density boost uses.

    The result is then rescaled onto the pre-boost cell budget
    ``sum(A_v / rho_v**2)``, so the extra socket quads are paid for by a
    marginally coarser interior instead of by a bigger mesh - the requested
    face count survives the profile.  ``ring_density_budget`` caps how much
    may be moved, by shrinking the whole boost if the bands ask for too much.

    Returns ``(rho_new, info)``; ``rho_new is rho`` (the same object) when
    nothing was boosted.
    """
    n = len(rho)
    info = {"ring_boost_max": 1.0, "ring_boost_demand": 1.0, "ring_boost_verts": 0}
    if bands is None or bands.empty:
        return rho, info
    p = _merged_params(params) if params is not None else bands.params

    quota = float(p["ring_min_quads"])
    cap = float(p["ring_density_max"])
    if quota <= 0.0 or cap <= 1.0:
        return rho, info

    rho = np.asarray(rho, dtype=np.float64).reshape(n)
    # per-opening factor, measured against the rho that already applies at its
    # own rim (so an opening the adaptive/paint density already refined does
    # not get refined twice)
    fac = np.ones(len(bands.openings))
    for i, op in enumerate(bands.openings):
        rr = rho[op.verts]
        rr = rr[np.isfinite(rr) & (rr > 0.0)]
        if not len(rr) or op.perimeter <= EPS:
            continue
        have = op.perimeter / float(np.mean(rr))       # quads around today
        if have <= 0.0:
            continue
        fac[i] = float(np.clip(quota / have, 1.0, cap))
    if not np.any(fac > 1.0 + 1e-9):
        return rho, info

    plateau = max(float(p["ring_density_plateau"]), 0.0)
    decay = max(float(p["ring_density_decay"]), 1e-3)
    prof = np.zeros(n)
    live = np.isfinite(bands.dq0) & (bands.own >= 0)
    if not live.any():
        return rho, info
    prof[live] = np.exp(-np.maximum(bands.dq0[live] - plateau, 0.0) / decay)
    gain = np.ones(n)
    gain[live] = 1.0 + (fac[bands.own[live]] - 1.0) * prof[live]

    if areas is None:
        _, areas = _f.face_normals_areas(V, F)
    wa = _f.vertex_areas(V, F, n, areas=areas)
    rr = np.maximum(rho, EPS)
    pre = float(np.sum(wa / rr ** 2))
    demand = float(np.sum(wa * gain ** 2 / rr ** 2))
    if not np.isfinite(pre) or pre <= 0.0 or not np.isfinite(demand):
        return rho, info

    # cap how much of the budget the bands may pull towards themselves
    budget = max(float(p["ring_density_budget"]), 0.0)
    ratio = demand / pre
    if ratio > 1.0 + budget and ratio > 1.0:
        # shrink the boost until the demand fits.  Solving exactly needs a
        # root find; a bisection on the single scalar that scales (gain - 1)
        # is deterministic, converges in a fixed number of steps and never
        # overshoots.
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            g = 1.0 + (gain - 1.0) * mid
            if float(np.sum(wa * g ** 2 / rr ** 2)) > (1.0 + budget) * pre:
                hi = mid
            else:
                lo = mid
        gain = 1.0 + (gain - 1.0) * lo
        demand = float(np.sum(wa * gain ** 2 / rr ** 2))

    boosted = rho / np.maximum(gain, EPS)
    rho_new = boosted * _f.budget_scale(boosted, wa, pre)
    info["ring_boost_max"] = float(np.max(gain))
    info["ring_boost_demand"] = float(demand / pre)
    info["ring_boost_verts"] = int(np.count_nonzero(gain > 1.001))
    return rho_new, info


def ring_weights(bands, rho, params=None):
    """Per-vertex ring direction and soft-alignment weight.

    ``w`` is 0 outside the band and decays from ``ring_strength`` at the rim
    to 0 at ``ring_falloff`` output-quad widths away, measured against the
    ``rho`` handed in - so calling this *after* :func:`ring_density` makes the
    band ``ring_falloff`` quads wide in the quads that will actually exist.
    """
    n = len(rho)
    if bands is None or bands.empty:
        return np.zeros((n, 3)), np.zeros(n)
    p = _merged_params(params) if params is not None else bands.params

    rho = np.asarray(rho, dtype=np.float64).reshape(n)
    dirs = bands.dirs
    band = float(p["ring_falloff"])
    # distance in output-quad widths, so the falloff is resolution independent
    dq = np.where(np.isfinite(bands.dist),
                  bands.dist / np.maximum(rho, EPS), np.inf)
    plateau = float(np.clip(p["ring_plateau"], 0.0, max(band - 0.25, 0.0)))
    t = np.clip((dq - plateau) / max(band - plateau, EPS), 0.0, 1.0)
    strength = float(np.clip(p["ring_strength"], 0.0, 1.0))
    w = strength * (1.0 - _f.smoothstep(t)) * bands.conf
    w[~np.isfinite(dq)] = 0.0
    w[np.einsum("ij,ij->i", dirs, dirs) < 0.5] = 0.0
    return dirs, np.clip(w, 0.0, 1.0)


def _contour_cycles(V, F, s, level, min_verts=6):
    """Ordered vertex cycles tracing the level set ``s == level``.

    Marching triangles: a triangle straddling the level crosses exactly two of
    its edges, which links those two crossings, so the crossed-edge graph has
    degree at most two and falls apart into cycles and open chains that can be
    walked directly.  Every crossing is then snapped to whichever endpoint of
    its mesh edge is nearer the level, and repeats are dropped.

    The "every vertex at most once" rule is the whole point.  The extractor
    treats a feature vertex whose feature-degree is not exactly two as a
    *corner* that cannot slide (see ``extract._feature_corners``), so a
    contour handed over as an unordered bag of snapped pairs - which is what
    a per-triangle emit produces - manufactures junctions and dead ends by the
    dozen and freezes the lattice onto the staircase instead of laying a loop
    along it.  Measured on the Dinasty eye socket that cost first-ring loop
    purity 0.68 -> 0.68 (i.e. bought nothing) and visibly scarred the collar;
    the ordered cycle below takes it to 0.81.
    """
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(s[F]).all(axis=1)
    Ff = F[ok]
    if not len(Ff):
        return []
    a = s[Ff] - float(level)
    hit = (a.min(axis=1) < 0.0) & (a.max(axis=1) > 0.0)
    Ff, a = Ff[hit], a[hit]
    if not len(Ff):
        return []

    eid = {}
    snap = []
    links = []
    for t, av in zip(Ff.tolist(), a.tolist()):
        ee = []
        for i in range(3):
            j = (i + 1) % 3
            if (av[i] < 0.0) != (av[j] < 0.0):
                u, v = int(t[i]), int(t[j])
                k = (u, v) if u < v else (v, u)
                q = eid.get(k)
                if q is None:
                    q = eid[k] = len(snap)
                    snap.append(u if abs(av[i]) <= abs(av[j]) else v)
                ee.append(q)
        if len(ee) == 2 and ee[0] != ee[1]:
            links.append((ee[0], ee[1]))
    if not links:
        return []

    ne = len(snap)
    adj = [[] for _ in range(ne)]
    for i0, i1 in links:
        adj[i0].append(i1)
        adj[i1].append(i0)

    seen = np.zeros(ne, dtype=bool)
    # start from open ends first so a chain is walked whole, then from any
    # remaining (necessarily closed) component - deterministic either way
    order = [i for i in range(ne) if len(adj[i]) == 1] + list(range(ne))
    loops = []
    for st in order:
        if seen[st] or not adj[st]:
            continue
        chain = []
        prev, cur = -1, st
        while cur >= 0 and not seen[cur]:
            seen[cur] = True
            chain.append(cur)
            nxt = -1
            for w in adj[cur]:
                if w != prev and not seen[w]:
                    nxt = w
                    break
            prev, cur = cur, nxt
        if len(chain) < min_verts:
            continue
        vs = []
        used = set()
        for c in chain:
            v = int(snap[c])
            if v in used:
                continue
            vs.append(v)
            used.add(v)
        if len(vs) >= min_verts:
            loops.append(vs)
    return loops


def ring_pin_segments(V, F, dist, rho, params=None):
    """Feature segments tracing the first offset ring around every opening.

    ``dist`` is :attr:`RingBands.dist` (world geodesic distance to the nearest
    rim, ``inf`` outside the band) and ``rho`` the *final* target edge length,
    so the contour is taken at ``ring_pin_offset`` output quads out - i.e.
    where the first lattice line after the rim wants to be anyway.

    Handing that curve to the extractor as a feature makes the loop exist by
    construction rather than by hoping the field alignment survives
    extraction: the position field pins its samples onto the (faired) curve
    and the lattice lays an edge run along it.  Returns an ``(e, 2)`` array of
    input-vertex index pairs, possibly empty.
    """
    p = _merged_params(params)
    n = len(rho)
    empty = np.zeros((0, 2), dtype=np.int64)
    if dist is None or not np.any(np.isfinite(dist)):
        return empty
    off = float(p["ring_pin_offset"])
    if off <= 0.0:
        return empty
    rho = np.asarray(rho, dtype=np.float64).reshape(n)
    dq = np.where(np.isfinite(dist), dist / np.maximum(rho, EPS), np.inf)
    loops = _contour_cycles(V, F, dq, off,
                            min_verts=int(max(6, p["ring_min_perimeter_quads"] * 2)))
    if not loops:
        return empty
    gap = float(p["ring_pin_min_gap"])
    out = []
    for vs in loops:
        if gap > 0.0 and float(np.min(dq[np.asarray(vs, dtype=np.int64)])) < gap:
            continue                  # too close to the pinned rim to fit a row
        k = len(vs)
        for i in range(k):
            a, b = vs[i], vs[(i + 1) % k]
            if a != b:
                out.append((a, b) if a < b else (b, a))
    if not out:
        return empty
    return np.unique(np.asarray(out, dtype=np.int64), axis=0)


def opening_ring_field(V, F, N, edges, rho, symmetry=(False, False, False),
                       params=None):
    """Per-vertex ring direction and weight around every detected opening.

    Returns ``(dirs (n,3), w (n,), openings)``.  Convenience wrapper over
    :func:`ring_bands` + :func:`ring_weights` that does *not* touch ``rho``;
    :func:`fields.solve_fields` calls the two halves separately so that it can
    slot :func:`ring_density` in between.
    """
    bands = ring_bands(V, F, N, edges, rho, symmetry=symmetry, params=params)
    dirs, w = ring_weights(bands, rho, params=params)
    return dirs, w, bands.openings
