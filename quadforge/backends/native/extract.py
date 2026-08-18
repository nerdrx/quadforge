"""Quad extraction from a 4-RoSy orientation field - v2.

Public entry points
-------------------
``extract(V, F, sol, params) -> (VQ, FQ)``   (NATIVE_V2 contract)
    ``sol`` carries only ``N`` (vertex normals), ``Q`` (4-RoSy representative
    tangent) and ``rho`` (per-vertex target edge length); the *position* field
    is therefore solved here, together with the extraction and repair.

``extract_quads(O, Q, N, rho, edges, bnd_verts=None) -> (VQ, faces)``  (v1)
    Kept for ``solver.py``: the caller supplies an already-solved position
    field ``O``.  Routed through the same core, minus surface reprojection
    (which needs the input triangles).

Pipeline
--------
1. **Position field** (v2 path only).  A randomised vertex-clustering
   multiresolution hierarchy is built from the input graph and a
   lattice-compatible position field ``O`` is relaxed coarse -> fine, exactly
   as in v1 - creases and boundaries may only slide along themselves.

2. **Graph collapse.**  For every input edge (i, j) the offset ``O[j] - O[i]``
   is expressed in both endpoints' tangent frames and rounded to integers.
   ``(0, 0)`` in both frames - or a separation below ``WELD_EPS * rho`` - is a
   *collapse*; ``|da| + |db| == 1`` in both frames is a lattice *step*.
   Collapsing runs through union-find and only ever follows graph edges, so
   coincident-but-disconnected shells (card stacks) never weld together.

3. **Faces from the rotation system.**  Neighbours of every extracted vertex
   are sorted by angle in its tangent frame and the orbits of
   ``next(u->v) = v->prev_ccw(u)`` are traced.  Clean lattice regions give
   4-cycles; 5..8-cycles are split into quads; longer orbits are dropped and
   left to the repair stage.

4. **Repair** - this is what v1 got wrong and what the v2 gates are about:

   * de-duplicate faces, drop degenerate ones,
   * enforce ``<= 2`` faces per edge (hard manifoldness gate),
   * propagate a consistent winding through each connected component,
   * trace *every* hole with a half-edge walk (pinch-safe: a figure-eight
     hole loop is decomposed into simple cycles) and fill it quad-dominantly.
     Loops whose vertices all sit on a genuine input boundary are preserved,
     everything else is closed - so a closed input yields a closed output,
     * n <= 6: split without adding a vertex (rejects chords that already
       exist, which would break manifoldness),
     * n >= 7: centroid fan - two boundary edges per quad.  Only ever touches
       edges with exactly one face, so it can never create a non-manifold
       edge,
     A geometrically collinear 3/4-loop cannot be closed by a face at all
     (``mesh.validate()`` would drop the zero-area result and re-open the
     hole), so its shortest side is welded instead,
   * drop dangling faces whose every edge is a boundary edge,
   * remove 2-valence doublets (two quads sharing two edges -> one quad),
   * fuse adjacent triangle pairs into quads, then **annihilate the rest
     pairwise**: a leftover triangle is walked through the quads separating
     it from another triangle (merge tri+quad into a pentagon, re-split so
     the triangle lands on the far side) until the two meet and fuse.  This
     is what takes the closed fixtures from ~96% to 100% quads,
   * repeat until no hole and no over-used edge remains.

5. **Face-count search** (v2 path only).  Steps 1-4 are repeated for a few
   uniform rescalings of ``rho``, driven by a secant iteration on
   ``log(count)`` against ``log(scale)``, until the count is within
   ``COUNT_TOL`` of the request.  The search runs *unpolished* (step 6 moves
   points and never changes a face), so an attempt costs about half a full
   extraction and only the winner is finished.

6. **Relax + reproject.**  3-5 tangent-space Laplacian iterations, each
   followed by a projection back onto the nearest input triangle through a
   uniform spatial hash, so repair patches blend in and no output vertex
   drifts off the input surface.  Creases, boundaries and pinned vertices are
   frozen.
"""

from __future__ import annotations

import math
import os
import time as _time

import numpy as np

EPS = 1e-12

# set QF_EXTRACT_DEBUG=1 to trace the repair stage
_DEBUG = bool(os.environ.get("QF_EXTRACT_DEBUG"))

# collapse an input edge whose two position-field samples are closer than
# this fraction of the local target edge length, even when the two tangent
# frames disagree about the integer offset
WELD_EPS = 0.30

# orbits longer than this are treated as extraction failures and left to the
# hole filler
MAX_ORBIT = 8

# a hole loop longer than MAX_ORBIT whose corners are at least this share
# boundary samples is a real opening, not a lattice defect (see _fill_holes)
FILL_BOUNDARY_FRAC = 0.90

# The position lattice is read off the input graph, so an input edge longer
# than ~1.5 rho cannot produce a lattice step at all: it rounds to an offset
# of 2+ and is discarded, which strips every face off the under-sampled part
# of the surface.  Real sculpts are wildly non-uniform (a Dinasty-style avatar
# has 15% of its edges longer than rho while the hair plates are 5x finer), so
# the mesh is conformingly refined until every edge is short enough.  Midpoint
# splits leave the piecewise-linear surface bit-for-bit identical.
SPLIT_RATIO = 0.55
MAX_REFINE_ROUNDS = 8
MAX_REFINE_VERTS = 400_000

# A connected shell smaller than a couple of lattice cells cannot carry a quad
# ring and extracts to nothing - on a real avatar the eyes, teeth, buttons and
# hair cards are dozens of such shells and can be 40% of the surface.  rho is
# clamped per shell so every shell keeps at least this many quads.
MIN_SHELL_QUADS = 16.0
# ...but that floor is a promise the face budget has to pay for: N shells cost
# N * MIN_SHELL_QUADS quads before a single quad is spent on shape.  With Keep
# Small Shells off, a 170-shell avatar reaches the solver whole, and at an
# economy target the promise exceeds the entire budget.  Honouring it anyway
# drove rho on the tiny shells 18x below the global value, which exploded the
# conforming refinement (2.7x the triangles, 6x the solve time) and starved the
# body (measured on Dinasty at target 2000: main shell 1290 -> 546 faces).
# The floor may therefore claim at most this share of the target; past that
# every shell gets an equal cut of what is actually available.
SHELL_FLOOR_SHARE = 0.5

# Output polish.  The extracted lattice is topologically clean but visually
# jittery (neighbouring quads varying ~2x in area); these drive the fairing
# pass that evens edge lengths and squares corners.  Overridable through
# params: regularize_iters / regularize_step / w_square / w_even / w_laplace /
# max_drift.
REGULARIZE_ITERS = 120
REGULARIZE_STEP = 0.8
W_SQUARE = 1.2
W_EVEN = 1.0
W_LAPLACE = 0.35
MAX_DRIFT = 1.25
SIZE_LOCK = 0.5
REST_SMOOTH = 6
REST_RHO = 0.25
PROJECT_EVERY = 6
# stop early once the pass stops moving anything (relative to local rho)
REGULARIZE_TOL = 6e-4

# escalate triangle-annihilation walk length only below this quad ratio
QUAD_FLOOR = 0.975

# ---- face-count adherence -------------------------------------------------
# The extracted count answers a uniform rho rescale as count ~ scale**-e.  For
# an ideal lattice e is exactly 2, but the measured value on real sculpts is
# 2.2-2.5: the repair stage (doublets, tri-pair fusion, triangle annihilation)
# deletes a *fraction* of the faces and that fraction grows with the lattice
# density, so the count falls faster than the cell area.  The old loop assumed
# e = 2 (scale *= sqrt(ratio)), overshot every correction by 20-25%, and then
# accepted anything inside a 0.82..1.22 band - so a request could land
# anywhere in -18%..+22% and exactly where it landed was a function of the
# density field.  These drive a secant iteration on log(count) vs log(scale)
# with a tolerance worth the name.
COUNT_TOL = 0.08              # accept while |count / target - 1| <= this
COUNT_ATTEMPTS = 4
# Opening step, before two samples exist to measure e from.  The excess over
# the ideal 2 is repair loss, and repair loss is exactly what makes the first
# attempt fall short of the cells its rho field promised - so the shortfall
# predicts the exponent.  A mesh that realises every predicted cell loses
# nothing to repair and answers with e = 2; the Dinasty head realises 82% and
# measures 2.25; the Rexouium body realises 56% and measures 2.58.
COUNT_E0 = 2.0
COUNT_E_YIELD = 1.3
COUNT_E_MIN, COUNT_E_MAX = 1.2, 4.0
# below this measured exponent the count is not answering the scale at all
COUNT_E_DEAD = 0.3
# the count search must not buy faces with surface: an attempt is eligible
# only if its coverage is within this of the best coverage seen.  Coverage
# above 1.0 carries no information (repair patches and fan-triangulated
# n-gons push it past the input area), so it is clamped before comparing.
COVER_SLACK = 0.02

# Feature-curve fairing.  Pinned crease/boundary chains are smoothed ALONG
# their own polyline and snapped back onto the input feature curve; a
# junction or a turn sharper than this is a genuine corner and is held.
FEATURE_CORNER_DEG = 120.0
FEATURE_STEP = 0.5
FEATURE_MAX_DRIFT = 0.6
FEATURE_MIN_SEGS = 3
FEATURE_MIN_LEN = 2.0
FEATURE_CORNER_SPAN = 3
# The detected feature polyline hops between triangulation vertices, so it
# zigzags at tessellation scale.  Snapping output chains onto it with perfect
# fidelity reproduces that zigzag, so the TARGET curve is faired first: uniform
# arc-length resample + Laplacian along the curve, held within this fraction of
# the local rho of the raw polyline so the rim shape survives.
FEATURE_FAIR_ITERS = 12
FEATURE_FAIR_LAMBDA = 0.5
FEATURE_FAIR_DRIFT = 0.35
FEATURE_RESAMPLE = 0.5
# a detection dropout this many mesh edges wide is bridged so a rim does not
# alternate faired / unfaired sections
# Bridging detection dropouts is implemented but OFF by default: measured on
# the Dinasty head it joins chains the sculpt did not intend, and the faired
# chain turning angle got worse (median 9.2 -> 10.8, p95 29.3 -> 39.2).  Set
# params["feature_bridge"]=True to enable.
FEATURE_BRIDGE = False
FEATURE_BRIDGE_GAP = 3.0
FEATURE_BRIDGE_COS = 0.5
# output vertices this close to the feature curve join its chain even if the
# collapse never gave them a pinned input sample
FEATURE_CAPTURE = 0.25

__all__ = ["extract", "extract_quads", "make_solution"]


# --------------------------------------------------------------------------
# small numpy helpers (self-contained: extract.py must keep working while
# fields.py is being rewritten)
# --------------------------------------------------------------------------

def _dot(A, B):
    return np.einsum("ij,ij->i", A, B)


def normalize(A, eps=EPS):
    """Row-wise normalisation of an (n, 3) array (safe against zeros)."""
    n = np.sqrt(np.einsum("ij,ij->i", A, A))[:, None]
    return A / np.maximum(n, eps)


# above this vertex count the packed edge key would overflow int64 and the
# (slow) row-wise np.unique has to be used instead
_KEY_MAX = 3_037_000_499


def _unique_edge_rows(a, b, n, counts=False):
    """Sorted unique undirected edges ``(min, max)`` of the pairs ``(a, b)``.

    Identical output to ``np.unique(np.sort(np.stack([a, b], 1), 1), axis=0)``
    - the packed key ``lo * n + hi`` is strictly monotone in the lexicographic
    row order - but through one int64 sort instead of a void-dtype row sort.
    ``np.unique(..., axis=0)`` on edge arrays was the single largest numpy
    cost in the solve (8.5 s of 42 s on a 57k-tri shell).
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    m = np.int64(max(int(n), 1))
    if m > _KEY_MAX:                                    # pragma: no cover
        e = np.stack([lo, hi], axis=1)
        if counts:
            return np.unique(e, axis=0, return_counts=True)
        return np.unique(e, axis=0), None
    if counts:
        key, cnt = np.unique(lo * m + hi, return_counts=True)
    else:
        key, cnt = np.unique(lo * m + hi), None
    out = np.empty((len(key), 2), dtype=np.int64)
    np.floor_divide(key, m, out=out[:, 0])
    np.subtract(key, out[:, 0] * m, out=out[:, 1])
    return out, cnt


def _tri_edge_pairs(F):
    """The three edges of every triangle, as two flat index arrays."""
    F = np.asarray(F, dtype=np.int64)
    a = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    b = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    return a, b


def _build_edges(F):
    F = np.asarray(F, dtype=np.int64)
    if F.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    a, b = _tri_edge_pairs(F)
    e, _ = _unique_edge_rows(a, b, int(F.max()) + 1)
    return e[e[:, 0] != e[:, 1]]


def _build_csr(edges, n):
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    src = src[order]
    dst = dst[order]
    deg = np.bincount(src, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    return indptr, dst, src


def _boundary_edges(F):
    F = np.asarray(F, dtype=np.int64)
    if F.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    a, b = _tri_edge_pairs(F)
    uniq, counts = _unique_edge_rows(a, b, int(F.max()) + 1, counts=True)
    return uniq[counts == 1]


def _tri_areas(V, F):
    a = V[F[:, 1]] - V[F[:, 0]]
    b = V[F[:, 2]] - V[F[:, 0]]
    cr = np.cross(a, b)
    return 0.5 * np.sqrt(np.einsum("ij,ij->i", cr, cr))


def _poly_area(P, faces):
    """Total area of a 3/4-gon soup (fan triangulation)."""
    if not len(faces):
        return 0.0
    tris, quads = _tri_quad_arrays(faces)
    s = 0.0
    if len(tris):
        s += float(_tri_areas(P, tris).sum())
    if len(quads):
        s += float(_tri_areas(P, quads[:, [0, 1, 2]]).sum())
        s += float(_tri_areas(P, quads[:, [0, 2, 3]]).sum())
    return s


def _any_tangent(N):
    alt = np.zeros((len(N), 3))
    pick = np.abs(N[:, 0]) < 0.9
    alt[pick] = (1.0, 0.0, 0.0)
    alt[~pick] = (0.0, 1.0, 0.0)
    return normalize(alt - N * _dot(alt, N)[:, None])


_MISSING = object()


def make_solution(N, Q, rho):
    """Tiny stand-in for ``fields.FieldSolution`` (handy in tests)."""
    from types import SimpleNamespace
    return SimpleNamespace(N=np.asarray(N, dtype=np.float64),
                           Q=np.asarray(Q, dtype=np.float64),
                           rho=np.asarray(rho, dtype=np.float64))


def _sol_get(sol, name, default=_MISSING):
    if isinstance(sol, dict):
        if default is _MISSING:
            return sol[name]
        return sol.get(name, default)
    if default is _MISSING:
        return getattr(sol, name)
    return getattr(sol, name, default)


# --------------------------------------------------------------------------
# adaptive refinement: make the input resolvable by the lattice
# --------------------------------------------------------------------------

def _edge_lookup(e, n):
    """``(a, b) -> row in e`` for the undirected edge array ``e`` (i<j)."""
    key = e[:, 0] * np.int64(n) + e[:, 1]
    order = np.argsort(key, kind="stable")
    skey = key[order]

    def eid(x, y):
        lo = np.minimum(x, y)
        hi = np.maximum(x, y)
        pos = np.searchsorted(skey, lo * np.int64(n) + hi)
        return order[np.clip(pos, 0, len(skey) - 1)]
    return eid


def _feature_corners(V, segs, n, fdeg, corner_deg=120.0, span=3):
    """Genuine corners of a feature polyline.

    The turn is measured over a ``span``-segment baseline on each side, not
    between adjacent segments: a crease traced across a sculpt's triangulation
    zig-zags, and an adjacent-segment test reads that tessellation noise as a
    corner on every second vertex (154 of 404 on the Dinasty head), which
    freezes the chain solid.
    """
    corner = fdeg != 2
    adj = {}
    for a, b in segs:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    def walk(v, first):
        prev, cur = v, first
        for _ in range(span - 1):
            nxt = [w for w in adj.get(cur, ()) if w != prev]
            if len(nxt) != 1:
                break
            prev, cur = cur, nxt[0]
        return cur

    lim = np.cos(np.radians(180.0 - corner_deg))
    for v, nbv in adj.items():
        if len(nbv) != 2:
            continue
        a = walk(v, nbv[0])
        b = walk(v, nbv[1])
        e0 = V[a] - V[v]
        e1 = V[b] - V[v]
        l0 = float(np.linalg.norm(e0))
        l1 = float(np.linalg.norm(e1))
        if l0 < 1e-15 or l1 < 1e-15:
            continue
        # straight run -> e0 and e1 are opposite -> cos = -1
        if float(e0 @ e1) / (l0 * l1) > -lim:
            corner[v] = True
    return corner


def _bridge_feature_gaps(V, segs, indptr, indices, rho, gap=FEATURE_BRIDGE_GAP,
                         min_cos=FEATURE_BRIDGE_COS, max_hops=32):
    """Close short dropouts in the detected feature curve.

    A rim whose dihedral dips under the detection threshold for a few edges
    arrives as two chains with loose ends, so the rim alternates faired and
    unfaired sections.  The search is bounded by *distance* (a multiple of the
    local rho), not by hop count: the mesh is adaptively refined before this
    runs, so a fixed hop budget reaches a different physical distance in every
    region and never fired at all on dense sculpt detail.
    """
    segs = np.asarray(segs, dtype=np.int64).reshape(-1, 2)
    if not len(segs):
        return segs
    adj = {}
    for a, b in segs:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    ends = sorted(v for v, l in adj.items() if len(l) == 1)
    if len(ends) < 2:
        return segs
    endset = set(ends)
    feat = set(adj)

    def tangent(v):
        d = V[v] - V[adj[v][0]]
        ln = np.linalg.norm(d)
        return d / ln if ln > 1e-15 else None

    used = set()
    added = []
    for a in ends:
        if a in used:
            continue
        ta = tangent(a)
        if ta is None:
            continue
        budget = gap * float(rho[a])
        prev = {a: -1}
        dist = {a: 0.0}
        frontier = [a]
        found = None
        for _hop in range(max_hops):
            nxt = []
            for u in frontier:
                du = dist[u]
                for k in range(indptr[u], indptr[u + 1]):
                    w = int(indices[k])
                    dw = du + float(np.linalg.norm(V[w] - V[u]))
                    if dw > budget or w in prev:
                        continue
                    prev[w] = u
                    dist[w] = dw
                    if w in endset:
                        if w != a and w not in used:
                            found = w
                            break
                        continue
                    if w in feat:
                        continue
                    nxt.append(w)
                if found is not None:
                    break
            if found is not None or not nxt:
                break
            frontier = nxt
        if found is None:
            continue
        b = found
        tb = tangent(b)
        if tb is None:
            continue
        d = V[b] - V[a]
        ln = np.linalg.norm(d)
        if ln < 1e-15:
            continue
        d = d / ln
        # both chains must run *into* the gap
        if float(-ta @ d) < min_cos or float(tb @ d) < min_cos:
            continue
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        for i in range(len(path) - 1):
            u, w = path[i], path[i + 1]
            added.append((min(u, w), max(u, w)))
        used.add(a)
        used.add(b)
    if not added:
        if _DEBUG:
            print("   [bridge] loose ends=%d bridged=0" % len(ends))
        return segs
    out = np.unique(np.concatenate(
        [segs, np.asarray(added, dtype=np.int64)], axis=0), axis=0)
    if _DEBUG:
        print("   [bridge] loose ends=%d bridged pairs=%d segments %d -> %d"
              % (len(ends), len(used) // 2, len(segs), len(out)))
    return out


def _feature_chains(segs, corner):
    """Split the feature graph into polylines at corners and junctions.

    Returns a list of ``(vertex list, closed?)``.
    """
    adj = {}
    for a, b in segs:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    done = set()

    def key(u, v):
        return (u, v) if u < v else (v, u)

    def walk(start, first):
        chain = [start]
        prev, cur = start, first
        while True:
            done.add(key(prev, cur))
            chain.append(cur)
            if cur == start:
                return chain
            if corner[cur] or len(adj[cur]) != 2:
                return chain
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt:
                return chain
            prev, cur = cur, nxt[0]

    chains = []
    for v in sorted(x for x in adj if corner[x] or len(adj[x]) != 2):
        for nb in sorted(adj[v]):
            if key(v, nb) in done:
                continue
            chains.append((walk(v, nb), False))
    for v in sorted(adj):                       # leftovers are closed loops
        for nb in sorted(adj[v]):
            if key(v, nb) in done:
                continue
            c = walk(v, nb)
            if len(c) > 1 and c[0] == c[-1]:
                chains.append((c[:-1], True))
            else:
                chains.append((c, False))
    return chains


def _resample_polyline(P, R, h, closed):
    """Uniform arc-length resample of a polyline (carrying a scalar R)."""
    Q = np.concatenate([P, P[:1]], axis=0) if closed else P
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-15 or h <= 0.0:
        return P.copy(), R.copy()
    if closed:
        m = max(4, int(round(total / h)))
        t = np.linspace(0.0, total, m, endpoint=False)
    else:
        m = max(2, int(round(total / h)) + 1)
        t = np.linspace(0.0, total, m)
    Rr = np.concatenate([R, R[:1]]) if closed else R
    out = np.empty((len(t), 3))
    for c in range(3):
        out[:, c] = np.interp(t, cum, Q[:, c])
    return out, np.interp(t, cum, Rr)


def _fair_feature_curves(V, segs, rho, corner,
                         iters=FEATURE_FAIR_ITERS, lam=FEATURE_FAIR_LAMBDA,
                         drift=FEATURE_FAIR_DRIFT, resample=FEATURE_RESAMPLE):
    """Smoothed snap target for the output feature chains.

    Each chain is resampled to uniform arc length, then Laplacian-smoothed with
    corners and endpoints held.  Every sample is kept within ``drift * rho`` of
    where it started - and it started *on* the raw polyline - so the smoothed
    curve provably never leaves a ``drift * rho`` tube around the input feature.
    """
    segs = np.asarray(segs, dtype=np.int64).reshape(-1, 2)
    if not len(segs):
        return None, None, {}
    chains = _feature_chains(segs, corner)
    pts = []
    out = []
    base = 0
    worst = 0.0
    nsample = 0
    for chain, closed in chains:
        idx = np.asarray(chain, dtype=np.int64)
        if len(idx) < 2:
            continue
        P = V[idx]
        R = rho[idx]
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        seg = seg[seg > 1e-15]
        if not len(seg):
            continue
        h = min(float(np.median(seg)), resample * float(np.mean(R)))
        Q, Rq = _resample_polyline(P, R, h, closed)
        k = len(Q)
        if k >= 3:
            Q0 = Q.copy()
            cap = drift * Rq
            for _ in range(iters):
                if closed:
                    lap = 0.5 * (np.roll(Q, 1, axis=0) + np.roll(Q, -1, axis=0)) - Q
                else:
                    lap = np.zeros_like(Q)
                    lap[1:-1] = 0.5 * (Q[:-2] + Q[2:]) - Q[1:-1]
                Q = Q + lam * lap
                d = Q - Q0
                dl = np.sqrt(np.einsum("ij,ij->i", d, d))
                over = dl > cap
                if over.any():
                    Q[over] = Q0[over] + d[over] * (
                        cap[over] / np.maximum(dl[over], 1e-18))[:, None]
            dl = np.linalg.norm(Q - Q0, axis=1) / np.maximum(Rq, EPS)
            worst = max(worst, float(dl.max()))
            nsample += k
        pts.append(Q)
        e = np.stack([np.arange(k - 1), np.arange(1, k)], axis=1) + base
        if closed:
            e = np.concatenate([e, [[base + k - 1, base]]], axis=0)
        out.append(e)
        base += k
    if not pts:
        return None, None, {}
    FV = np.concatenate(pts, axis=0)
    FS = np.concatenate(out, axis=0).astype(np.int64)
    info = dict(chains=len(chains), samples=nsample, max_drift_rho=worst)
    if _DEBUG:
        print("   [fair-target] chains=%d samples=%d max drift=%.3f rho"
              % (len(chains), nsample, worst))
    return FV, FS, info


def _filter_feature_fragments(V, segs, rho, min_segs=FEATURE_MIN_SEGS,
                              min_len=FEATURE_MIN_LEN):
    """Drop feature components too small to be a real crease."""
    segs = np.asarray(segs, dtype=np.int64).reshape(-1, 2)
    if not len(segs):
        return segs
    roots = _union_find(len(V), segs)
    comp = roots[segs[:, 0]]
    uniq, inv = np.unique(comp, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uniq))
    L = np.linalg.norm(V[segs[:, 1]] - V[segs[:, 0]], axis=1)
    tot = np.bincount(inv, weights=L, minlength=len(uniq))
    keep = (cnt >= min_segs) & (tot >= min_len * max(rho, EPS))
    out = segs[keep[inv]]
    if _DEBUG:
        print("   [feat] components=%d kept=%d segments %d -> %d"
              % (len(uniq), int(keep.sum()), len(segs), len(out)))
    return out


def _clamp_rho_per_shell(V, F, rho, edges, min_quads=MIN_SHELL_QUADS,
                         target=0, floor_share=SHELL_FLOOR_SHARE):
    """Cap ``rho`` per connected shell so no shell extracts to nothing.

    ``target`` is the requested face count.  When the mesh carries more shells
    than the budget can give ``min_quads`` each, the per-shell floor is scaled
    down to what the budget can actually afford (see SHELL_FLOOR_SHARE) - an
    unaffordable floor buys nothing (the shells still cannot be expressed) and
    costs the rest of the mesh its resolution.
    """
    n = len(V)
    roots = _union_find(n, edges)
    uniq, comp = np.unique(roots, return_inverse=True)
    if len(uniq) < 2:
        return rho
    min_quads = float(min_quads)
    if target > 0:
        afford = floor_share * float(target) / float(len(uniq))
        min_quads = min(min_quads, max(1.0, afford))
    ar = _tri_areas(V, F)
    carea = np.bincount(comp[F[:, 0]], weights=ar, minlength=len(uniq))
    cap = np.sqrt(np.maximum(carea, 0.0) / min_quads)
    cap = np.where(cap > 0.0, cap, np.inf)
    out = np.minimum(rho, cap[comp])
    if _DEBUG:
        hit = int((out < rho - 1e-15).sum())
        print("   [shells] components=%d min_quads=%.2f verts_clamped=%d"
              % (len(uniq), min_quads, hit))
    return out


def _refine_for_lattice(V, F, N, Q, rho, sharp, ratio=SPLIT_RATIO,
                        max_verts=MAX_REFINE_VERTS,
                        rounds=MAX_REFINE_ROUNDS):
    """Conformingly split every input edge longer than ``ratio * rho``.

    Red-green refinement (1/2/3 marked edges per triangle -> 2/3/4 triangles),
    so no hanging nodes appear.  New vertices sit on edge midpoints, which
    keeps the surface exactly as it was; ``N``/``Q``/``rho`` are interpolated
    (``Q`` through a 4-RoSy match so the two endpoint crosses are averaged in
    the same rotational class).  ``sharp`` edges are split along with them.
    """
    for _ in range(rounds):
        n = len(V)
        e = _build_edges(F)
        d = V[e[:, 1]] - V[e[:, 0]]
        L = np.sqrt(np.einsum("ij,ij->i", d, d))
        rho_e = np.maximum(0.5 * (rho[e[:, 0]] + rho[e[:, 1]]), EPS)
        over = L / rho_e
        mark = over > ratio
        if not mark.any():
            break
        budget = max_verts - n
        if budget < 8:
            break
        if int(mark.sum()) > budget:
            # split the worst offenders first
            cand = np.nonzero(mark)[0]
            keep = cand[np.argsort(-over[cand])[:budget]]
            mark = np.zeros(len(e), dtype=bool)
            mark[keep] = True

        nm = int(mark.sum())
        new_id = np.full(len(e), -1, dtype=np.int64)
        new_id[mark] = n + np.arange(nm, dtype=np.int64)
        ea, eb = e[mark, 0], e[mark, 1]

        Vm = 0.5 * (V[ea] + V[eb])
        Nm = normalize(N[ea] + N[eb])
        Qm = Q[ea] + _match_4rosy(Q[eb], N[eb], Q[ea])
        Qm = Qm - Nm * _dot(Qm, Nm)[:, None]
        ql = np.sqrt(np.einsum("ij,ij->i", Qm, Qm))
        bad = ql < 1e-9
        if bad.any():
            Qm[bad] = _any_tangent(Nm[bad])
        Qm = normalize(Qm)
        rhom = 0.5 * (rho[ea] + rho[eb])

        eid = _edge_lookup(e, n)
        M = np.stack([new_id[eid(F[:, 0], F[:, 1])],
                      new_id[eid(F[:, 1], F[:, 2])],
                      new_id[eid(F[:, 2], F[:, 0])]], axis=1)
        hit = M >= 0
        cnt = hit.sum(axis=1)
        arange3 = np.arange(3)[None, :]
        out = [F[cnt == 0]]

        sel = cnt == 1
        if sel.any():
            k = np.argmax(hit[sel], axis=1)          # rotate marked edge to 0
            idx = (arange3 + k[:, None]) % 3
            T = np.take_along_axis(F[sel], idx, axis=1)
            m0 = np.take_along_axis(M[sel], idx, axis=1)[:, 0]
            out.append(np.stack([T[:, 0], m0, T[:, 2]], axis=1))
            out.append(np.stack([m0, T[:, 1], T[:, 2]], axis=1))

        sel = cnt == 2
        if sel.any():
            u = np.argmin(hit[sel], axis=1)          # the unmarked edge
            idx = (arange3 + ((u + 1) % 3)[:, None]) % 3
            T = np.take_along_axis(F[sel], idx, axis=1)
            Ms = np.take_along_axis(M[sel], idx, axis=1)
            m0, m1 = Ms[:, 0], Ms[:, 1]
            out.append(np.stack([T[:, 0], m0, T[:, 2]], axis=1))
            out.append(np.stack([m0, T[:, 1], m1], axis=1))
            out.append(np.stack([m0, m1, T[:, 2]], axis=1))

        sel = cnt == 3
        if sel.any():
            T, Ms = F[sel], M[sel]
            m0, m1, m2 = Ms[:, 0], Ms[:, 1], Ms[:, 2]
            out.append(np.stack([T[:, 0], m0, m2], axis=1))
            out.append(np.stack([m0, T[:, 1], m1], axis=1))
            out.append(np.stack([m1, T[:, 2], m2], axis=1))
            out.append(np.stack([m0, m1, m2], axis=1))

        if sharp is not None and len(sharp):
            sm = new_id[eid(sharp[:, 0], sharp[:, 1])]
            spl = sm >= 0
            if spl.any():
                sharp = np.concatenate([
                    sharp[~spl],
                    np.stack([sharp[spl, 0], sm[spl]], axis=1),
                    np.stack([sm[spl], sharp[spl, 1]], axis=1)], axis=0)

        F = np.concatenate(out, axis=0)
        V = np.concatenate([V, Vm], axis=0)
        N = np.concatenate([N, Nm], axis=0)
        Q = np.concatenate([Q, Qm], axis=0)
        rho = np.concatenate([rho, rhom], axis=0)
        if _DEBUG:
            print("   [refine] verts=%d tris=%d (split %d edges, worst "
                  "L/rho=%.2f)" % (len(V), len(F), nm, over.max()))
    return V, F, N, Q, rho, sharp


# --------------------------------------------------------------------------
# position field (owned by extract.py in the v2 split)
# --------------------------------------------------------------------------

def _round_to_cell(O, Q, N, P, rho):
    T = np.cross(N, Q)
    d = P - O
    a = np.round(_dot(Q, d) / rho)
    b = np.round(_dot(T, d) / rho)
    return O + Q * (a * rho)[:, None] + T * (b * rho)[:, None]


def _smooth_positions(O, P, Q, N, rho, src, dst, iters, self_weight=1.0,
                      con_mask=None, con_dir=None):
    """Jacobi relaxation of the lattice-compatible position field.

    The hot loop of the whole solver (2.1 M directed edges x 20 iterations x
    one pass per count-search attempt on a 230k-tri shell).  Three things make
    it faster without changing a single rounded bit:

    * ``src`` arrives in CSR order, so ``O[src]`` is a *repeat*, not a random
      gather - 6x cheaper for the same values;
    * the per-edge replicas are accumulated in a (3, m) scratch, so
      ``np.bincount`` reads a contiguous row instead of copying a strided
      column three times per iteration;
    * the per-vertex tangent and the edge scratch buffers are hoisted out of
      the loop.

    Arithmetic order is untouched everywhere, so the result is bit-identical
    to the straightforward version.
    """
    n = O.shape[0]
    has_con = con_mask is not None and bool(np.any(con_mask))
    O = np.array(O, dtype=np.float64, copy=True)
    m = len(src)
    deg_i = np.bincount(src, minlength=n)
    deg = deg_i.astype(np.float64)
    denom = deg + self_weight
    rho_e = 0.5 * (rho[src] + rho[dst])
    inv_e = 1.0 / np.maximum(rho_e, EPS)
    Tj = np.cross(N[dst], Q[dst])
    Qj = Q[dst]
    QjT = np.ascontiguousarray(Qj.T)
    TjT = np.ascontiguousarray(Tj.T)
    T0 = np.cross(N, Q)
    # sorted src <=> src == repeat(arange(n), deg), so the source gather can
    # be a sequential expansion
    csr = bool(m == 0 or np.all(src[1:] >= src[:-1]))
    dbuf = np.empty((m, 3))
    repT = np.empty((3, m))
    tmp = np.empty(m)
    a = np.empty(m)
    b = np.empty(m)
    acc = np.empty((n, 3))

    for _ in range(iters):
        Od = O[dst]
        np.subtract(np.repeat(O, deg_i, axis=0) if csr else O[src], Od,
                    out=dbuf)
        np.einsum("ij,ij->i", Qj, dbuf, out=a)
        np.multiply(a, inv_e, out=a)
        np.round(a, out=a)
        np.multiply(a, rho_e, out=a)
        np.einsum("ij,ij->i", Tj, dbuf, out=b)
        np.multiply(b, inv_e, out=b)
        np.round(b, out=b)
        np.multiply(b, rho_e, out=b)
        OdT = Od.T
        for c in range(3):
            np.multiply(QjT[c], a, out=repT[c])
            np.add(OdT[c], repT[c], out=repT[c])
            np.multiply(TjT[c], b, out=tmp)
            np.add(repT[c], tmp, out=repT[c])
        for c in range(3):
            acc[:, c] = np.bincount(src, weights=repT[c], minlength=n)
        acc += O * self_weight
        acc /= denom[:, None]
        acc -= N * _dot(acc - P, N)[:, None]
        # _round_to_cell, with the tangent hoisted out of the loop
        cd0 = P - acc
        ra = np.round(_dot(Q, cd0) / rho)
        rb = np.round(_dot(T0, cd0) / rho)
        O = acc + Q * (ra * rho)[:, None] + T0 * (rb * rho)[:, None]
        if has_con:
            cd = con_dir[con_mask]
            rel = O[con_mask] - P[con_mask]
            O[con_mask] = P[con_mask] + cd * _dot(rel, cd)[:, None]
    return O


def _greedy_match(n, indptr, indices, rng):
    parent = np.full(n, -1, dtype=np.int64)
    order = rng.permutation(n)
    nc = 0
    ip = indptr
    idx = indices
    for i in order:
        i = int(i)
        if parent[i] != -1:
            continue
        partner = -1
        for k in range(ip[i], ip[i + 1]):
            j = int(idx[k])
            if j != i and parent[j] == -1:
                partner = j
                break
        parent[i] = nc
        if partner >= 0:
            parent[partner] = nc
        nc += 1
    return parent, nc


def _match_4rosy(Qf, Nf, ref):
    """Representative of the 4-RoSy class of ``Qf`` closest to ``ref``."""
    perp = np.cross(Nf, Qf)
    d0 = _dot(Qf, ref)
    d1 = _dot(perp, ref)
    use1 = np.abs(d1) > np.abs(d0)
    rep = np.where(use1[:, None], perp, Qf)
    sg = np.where(use1, d1, d0)
    return rep * np.where(sg < 0.0, -1.0, 1.0)[:, None]


def _build_pos_hierarchy(P, N, Q, rho, indptr, indices, pin_mask, pin_dir,
                         rng, min_verts=48, max_levels=12):
    """Vertex-clustering hierarchy carrying everything the position solve
    needs (``P N Q rho src indices pin_mask pin_dir parent n``)."""
    levels = []
    cur = dict(P=P, N=N, Q=Q, rho=rho, indptr=indptr, indices=indices,
               src=np.repeat(np.arange(len(P), dtype=np.int64),
                             np.diff(indptr)),
               pin_mask=pin_mask, pin_dir=pin_dir, parent=None, n=len(P))
    levels.append(cur)

    while len(levels) < max_levels and cur["n"] > min_verts:
        n = cur["n"]
        parent, nc = _greedy_match(n, cur["indptr"], cur["indices"], rng)
        if nc >= n or nc < 3:
            break
        cnt = np.bincount(parent, minlength=nc).astype(np.float64)
        cP = np.zeros((nc, 3))
        cN = np.zeros((nc, 3))
        for d in range(3):
            cP[:, d] = np.bincount(parent, weights=cur["P"][:, d], minlength=nc)
            cN[:, d] = np.bincount(parent, weights=cur["N"][:, d], minlength=nc)
        cP /= cnt[:, None]
        nl = np.sqrt(np.einsum("ij,ij->i", cN, cN))
        deg = nl < 1e-12
        if deg.any():
            cN[deg] = (0.0, 0.0, 1.0)
        cN = normalize(cN)
        crho = np.bincount(parent, weights=cur["rho"], minlength=nc) / cnt

        # 4-RoSy restriction of the orientation field
        ref = np.zeros((nc, 3))
        rev = np.arange(n)[::-1]
        ref[parent[rev]] = cur["Q"][rev]
        ref = ref - cN * _dot(ref, cN)[:, None]
        rl = np.sqrt(np.einsum("ij,ij->i", ref, ref))
        bad = rl < 1e-9
        if bad.any():
            ref[bad] = _any_tangent(cN[bad])
        ref = normalize(ref)
        rep = _match_4rosy(cur["Q"], cur["N"], ref[parent])
        acc = np.empty((nc, 3))
        for c in range(3):
            acc[:, c] = np.bincount(parent, weights=rep[:, c], minlength=nc)
        acc -= cN * _dot(acc, cN)[:, None]
        al = np.sqrt(np.einsum("ij,ij->i", acc, acc))
        cQ = ref.copy()
        good = al > 1e-9
        cQ[good] = acc[good] / al[good][:, None]

        pi = parent[cur["src"]]
        pj = parent[cur["indices"]]
        keep = pi != pj
        pi, pj = pi[keep], pj[keep]
        if len(pi) == 0:
            break
        ce, _ = _unique_edge_rows(pi, pj, nc)
        cip, cidx, csrc = _build_csr(ce, nc)

        cpin = np.zeros(nc, dtype=bool)
        cdir = np.zeros((nc, 3))
        ci = np.nonzero(cur["pin_mask"])[0]
        if len(ci):
            cdir[parent[ci[::-1]]] = cur["pin_dir"][ci[::-1]]
            cpin[parent[ci]] = True
            cd = cdir - cN * _dot(cdir, cN)[:, None]
            cl = np.sqrt(np.einsum("ij,ij->i", cd, cd))
            ok = cpin & (cl > 1e-7)
            cdir[ok] = cd[ok] / cl[ok][:, None]
            cpin = ok

        cur = dict(P=cP, N=cN, Q=cQ, rho=crho, indptr=cip, indices=cidx,
                   src=csrc, pin_mask=cpin, pin_dir=cdir, parent=None, n=nc)
        levels[-1]["parent"] = parent
        levels.append(cur)
    return levels


def _solve_positions(levels, scale, iters):
    """Coarse -> fine lattice-compatible position field."""
    top = len(levels) - 1
    O = levels[top]["P"].copy()
    for li in range(top, -1, -1):
        lv = levels[li]
        rho = lv["rho"] * scale
        it = iters if li == 0 else max(5, iters // 2)
        O = _smooth_positions(O, lv["P"], lv["Q"], lv["N"], rho,
                              lv["src"], lv["indices"], it,
                              con_mask=lv["pin_mask"], con_dir=lv["pin_dir"])
        if li > 0:
            par = levels[li - 1]["parent"]
            lf = levels[li - 1]
            Of = O[par]
            Of = Of - lf["N"] * _dot(Of - lf["P"], lf["N"])[:, None]
            O = _round_to_cell(Of, lf["Q"], lf["N"], lf["P"],
                               lf["rho"] * scale)
    return O


# --------------------------------------------------------------------------
# graph collapse
# --------------------------------------------------------------------------

def _union_find(n, pairs):
    """Connected-component label per vertex: the lowest index in the component.

    Vectorised hooking + pointer jumping.  Union by lower root gives exactly
    the same labels as the scalar disjoint-set version it replaces (the root
    of a component is its minimum index either way), but the Python-level
    ``find`` was called ~5 million times per solve.
    """
    n = int(n)
    lab = np.arange(n, dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if n == 0 or len(pairs) == 0:
        return lab

    # CSR over both directions, built once: every round is then a gather plus
    # a segmented minimum, with no ufunc.at scatter anywhere
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    order = np.argsort(src, kind="stable")
    src = src[order]
    dst = dst[order]
    deg = np.bincount(src, minlength=n)
    starts = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=starts[1:])
    live = deg > 0
    st = starts[:-1][live]
    if len(st) == 0:
        return lab

    for _ in range(64):
        nxt = lab.copy()
        nxt[live] = np.minimum(nxt[live], np.minimum.reduceat(lab[dst], st))
        # pointer jumping: collapse the label forest to its roots.  Labels
        # only ever decrease and lab[i] <= i holds by induction, so this
        # halves the depth every pass; the bound is belt and braces.
        for _ in range(64):
            jump = nxt[nxt]
            if np.array_equal(jump, nxt):
                break
            nxt = jump
        if np.array_equal(nxt, lab):
            break
        lab = nxt
    return lab


def _prune_low_degree(nv, e0, e1, min_deg=2):
    alive = np.ones(nv, dtype=bool)
    keep = np.ones(len(e0), dtype=bool)
    for _ in range(64):
        deg = np.bincount(np.concatenate([e0[keep], e1[keep]]), minlength=nv)
        dead = alive & (deg < min_deg)
        if not dead.any():
            break
        alive &= ~dead
        keep &= alive[e0] & alive[e1]
    return alive, keep


def _collapse(O, Q, N, rho, edges, pin=None):
    """Input graph -> extracted lattice graph.

    Returns ``(cluster, CP, CN, CQ, ce)``: the input-vertex -> extracted-vertex
    map, the extracted vertex positions / normals / tangents and the extracted
    undirected edge list.
    """
    n = O.shape[0]
    i = edges[:, 0]
    j = edges[:, 1]
    rho_e = np.maximum(0.5 * (rho[i] + rho[j]), EPS)
    inv = 1.0 / rho_e

    d = O[j] - O[i]
    Ti = np.cross(N[i], Q[i])
    ai = np.round(_dot(Q[i], d) * inv)
    bi = np.round(_dot(Ti, d) * inv)
    Tj = np.cross(N[j], Q[j])
    aj = np.round(_dot(Q[j], -d) * inv)
    bj = np.round(_dot(Tj, -d) * inv)

    zi = (ai == 0) & (bi == 0)
    zj = (aj == 0) & (bj == 0)
    near = np.sqrt(np.einsum("ij,ij->i", d, d)) < WELD_EPS * rho_e
    collapse = (zi & zj) | near

    si = np.abs(ai) + np.abs(bi)
    sj = np.abs(aj) + np.abs(bj)
    unit = (si == 1) & (sj == 1) & ~collapse

    roots = _union_find(n, edges[collapse])
    uniq, cluster = np.unique(roots, return_inverse=True)
    nc = len(uniq)
    cnt = np.bincount(cluster, minlength=nc).astype(np.float64)

    CP = np.empty((nc, 3))
    CN = np.empty((nc, 3))
    for c in range(3):
        CP[:, c] = np.bincount(cluster, weights=O[:, c], minlength=nc)
        CN[:, c] = np.bincount(cluster, weights=N[:, c], minlength=nc)
    CP /= cnt[:, None]
    if pin is not None and np.any(pin):
        # A cluster that contains crease/boundary samples must sit on the
        # feature, not halfway between it and the neighbouring surface
        # samples - plain averaging is what stair-steps a sharp rim.
        w = np.asarray(pin, dtype=np.float64)
        wsum = np.bincount(cluster, weights=w, minlength=nc)
        keep = wsum > 0.0
        if keep.any():
            CPp = np.empty((nc, 3))
            for c in range(3):
                CPp[:, c] = np.bincount(cluster, weights=O[:, c] * w,
                                        minlength=nc)
            CPp[keep] /= wsum[keep][:, None]
            CP = np.where(keep[:, None], CPp, CP)
    ln = np.sqrt(np.einsum("ij,ij->i", CN, CN))
    bad = ln < 1e-12
    if bad.any():
        CN[bad] = (0.0, 0.0, 1.0)
    CN = normalize(CN)

    rep_order = np.arange(n)[::-1]
    CQ = np.zeros((nc, 3))
    CQ[cluster[rep_order]] = Q[rep_order]
    CQ = CQ - CN * _dot(CQ, CN)[:, None]
    ql = np.sqrt(np.einsum("ij,ij->i", CQ, CQ))
    qbad = ql < 1e-9
    if qbad.any():
        CQ[qbad] = _any_tangent(CN[qbad])
    CQ = normalize(CQ)

    ei = cluster[i[unit]]
    ej = cluster[j[unit]]
    m = ei != ej
    ei, ej = ei[m], ej[m]
    if len(ei) == 0:
        return cluster, CP, CN, CQ, np.zeros((0, 2), dtype=np.int64)
    ce, _ = _unique_edge_rows(ei, ej, nc)
    return cluster, CP, CN, CQ, ce


# --------------------------------------------------------------------------
# faces from the rotation system
# --------------------------------------------------------------------------

def _rotation_faces(CP, CN, CQ, ce, nc, cbnd):
    """Trace the orbits of the rotation system.  Returns a list of cycles."""
    alive, keep = _prune_low_degree(nc, ce[:, 0], ce[:, 1], min_deg=2)
    ce = ce[keep]
    if len(ce) == 0:
        return []

    src = np.concatenate([ce[:, 0], ce[:, 1]])
    dst = np.concatenate([ce[:, 1], ce[:, 0]])
    dv = CP[dst] - CP[src]
    ang = np.arctan2(_dot(dv, np.cross(CN[src], CQ[src])), _dot(dv, CQ[src]))
    order = np.lexsort((ang, src))
    src = src[order]
    dst = dst[order]

    ne = len(src)
    deg = np.bincount(src, minlength=nc)
    start = np.zeros(nc + 1, dtype=np.int64)
    np.cumsum(deg, out=start[1:])

    keys = src * np.int64(nc) + dst
    korder = np.argsort(keys, kind="stable")
    skeys = keys[korder]
    rkeys = dst * np.int64(nc) + src
    pos = np.clip(np.searchsorted(skeys, rkeys), 0, ne - 1)
    rev = korder[pos]
    if not np.array_equal(keys[rev], rkeys):
        good = keys[rev] == rkeys
        rev = np.where(good, rev, np.arange(ne))

    dv_local = rev - start[dst]
    nxt = start[dst] + (dv_local - 1) % np.maximum(deg[dst], 1)

    visited = np.zeros(ne, dtype=bool)
    nxt_l = nxt.tolist()
    src_l = src.tolist()
    cycles = []
    for s0 in range(ne):
        if visited[s0]:
            continue
        cyc = []
        s = s0
        while not visited[s]:
            visited[s] = True
            cyc.append(src_l[s])
            s = nxt_l[s]
            if len(cyc) > MAX_ORBIT + 1:
                break
        if not (3 <= len(cyc) <= MAX_ORBIT):
            continue
        if len(set(cyc)) != len(cyc):
            continue
        if len(cyc) >= 5 and cbnd is not None and all(cbnd[v] for v in cyc):
            # the outer face of an open patch - not a real face
            continue
        cycles.append(cyc)
    return cycles


# --------------------------------------------------------------------------
# face-level geometry helpers
# --------------------------------------------------------------------------

def _quad_quality(P, quad):
    """0 (degenerate / reflex) .. 1 (square).

    Scalar arithmetic on Python floats: the repair loop calls this tens of
    thousands of times on single quads, and ``np.cross`` / ``np.dot`` on
    3-vectors is almost all dispatch overhead there.  Every expression is the
    one numpy evaluates, so the value is unchanged bit for bit.
    """
    p = P[list(quad)].tolist()
    nx = ny = nz = 0.0
    for k in range(4):
        q0, q1, q2 = p[k], p[(k + 1) % 4], p[(k + 2) % 4]
        a0 = q1[0] - q0[0]; a1 = q1[1] - q0[1]; a2 = q1[2] - q0[2]
        b0 = q2[0] - q1[0]; b1 = q2[1] - q1[1]; b2 = q2[2] - q1[2]
        nx += a1 * b2 - a2 * b1
        ny += a2 * b0 - a0 * b2
        nz += a0 * b1 - a1 * b0
    ln = nx * nx + ny * ny + nz * nz
    if ln < 1e-24:
        return 0.0
    s = math.sqrt(ln)
    nx /= s; ny /= s; nz /= s
    worst = 1.0
    for k in range(4):
        q0, qm, q1 = p[k], p[(k - 1) % 4], p[(k + 1) % 4]
        e00 = q0[0] - qm[0]; e01 = q0[1] - qm[1]; e02 = q0[2] - qm[2]
        e10 = q1[0] - q0[0]; e11 = q1[1] - q0[1]; e12 = q1[2] - q0[2]
        cx = e01 * e12 - e02 * e11
        cy = e02 * e10 - e00 * e12
        cz = e00 * e11 - e01 * e10
        if cx * nx + cy * ny + cz * nz <= 0.0:
            return 0.0
        l0 = math.sqrt(e00 * e00 + e01 * e01 + e02 * e02)
        l1 = math.sqrt(e10 * e10 + e11 * e11 + e12 * e12)
        if l0 < 1e-12 or l1 < 1e-12:
            return 0.0
        w = 1.0 - abs((e00 * e10 + e01 * e11 + e02 * e12) / (l0 * l1))
        if w < worst:
            worst = w
    return worst


def _face_area(P, f):
    p = P[list(f)].tolist()
    p0 = p[0]
    a = 0.0
    for k in range(1, len(f) - 1):
        q, r = p[k], p[k + 1]
        a0 = q[0] - p0[0]; a1 = q[1] - p0[1]; a2 = q[2] - p0[2]
        b0 = r[0] - p0[0]; b1 = r[1] - p0[1]; b2 = r[2] - p0[2]
        cx = a1 * b2 - a2 * b1
        cy = a2 * b0 - a0 * b2
        cz = a0 * b1 - a1 * b0
        a += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
    return a


def _fan_parts(g):
    """Zig-zag split of a cycle into quads (+ at most one triangle)."""
    k = len(g)
    parts = []
    i = 1
    while k - i >= 3:
        parts.append((g[0], g[i], g[i + 1], g[i + 2]))
        i += 2
    if i < k - 1:
        parts.append((g[0], g[i], g[i + 1]))
    return parts


def _split_ngon(P, f, forbidden=None, skip=None):
    """Split an n-gon (n >= 5) into quads plus at most one triangle.

    ``forbidden`` is a set of undirected edges that must not be *created*
    (they already exist elsewhere in the mesh and a second use would break
    manifoldness); ``skip`` lists the polygon's own edges, which are of course
    allowed.  Returns ``None`` when every rotation needs a forbidden chord.
    """
    k = len(f)
    best = None
    for r in range(k):
        g = list(f[r:]) + list(f[:r])
        parts = _fan_parts(g)
        if forbidden:
            bad = False
            for part in parts:
                m = len(part)
                for a in range(m):
                    u, v = part[a], part[(a + 1) % m]
                    e = (u, v) if u < v else (v, u)
                    if e in forbidden and not (skip and e in skip):
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
        worst = min([_quad_quality(P, q) for q in parts if len(q) == 4],
                    default=0.0)
        if best is None or worst > best[0]:
            best = (worst, parts)
    return None if best is None else best[1]


# --------------------------------------------------------------------------
# repair primitives
# --------------------------------------------------------------------------

def _edge_use(faces):
    """``{(lo, hi): [face index, ...]}`` for a 3/4-gon soup.

    Called ~45 times per solve over the whole face list by the repair passes,
    so the inner loop is unrolled per face size and the dict is poked once per
    edge instead of going through ``setdefault`` (which builds a throwaway
    list for every edge that already exists).
    """
    use = {}
    get = use.get
    for i, f in enumerate(faces):
        if len(f) == 4:
            a, b, c, d = f
            pairs = ((a, b), (b, c), (c, d), (d, a))
        elif len(f) == 3:
            a, b, c = f
            pairs = ((a, b), (b, c), (c, a))
        else:
            k = len(f)
            pairs = tuple((f[j], f[(j + 1) % k]) for j in range(k))
        for u, v in pairs:
            e = (u, v) if u < v else (v, u)
            lst = get(e)
            if lst is None:
                use[e] = [i]
            else:
                lst.append(i)
    return use


def _dedupe(P, faces, drop_degenerate=True):
    """Drop faces with repeated indices, duplicates (Blender's ``validate()``
    removes them, which would silently re-open a hole) and - optionally -
    zero-area faces.

    Vectorised: the scalar version called ``np.cross`` once per face, which on
    a 12k-quad extraction (repeated ~20x per solve by the repair loop) was
    375k tiny numpy calls.  The semantics are unchanged - a degenerate face is
    dropped *without* claiming its vertex-set key, so the surviving set is
    exactly "drop invalid and degenerate, then first occurrence of each key
    wins" in face order.
    """
    m = len(faces)
    if m == 0:
        return []
    lens = np.fromiter((len(f) for f in faces), dtype=np.int64, count=m)
    if int(lens.max()) > 4 or int(lens.min()) < 3:
        # n-gons never reach here (they are split first); fall back rather
        # than silently mangling them
        return _dedupe_scalar(P, faces, drop_degenerate)
    # (m, 4) index block; triangles pad their 4th slot with -1
    idx = np.full((m, 4), -1, dtype=np.int64)
    quad = lens == 4
    if quad.any():
        idx[quad] = np.asarray([f for f in faces if len(f) == 4],
                               dtype=np.int64)
    tri = ~quad
    if tri.any():
        idx[tri, :3] = np.asarray([f for f in faces if len(f) == 3],
                                  dtype=np.int64)

    # repeated indices inside one face (the -1 pad is unique, so it never
    # triggers on a triangle)
    srt = np.sort(idx, axis=1)
    keep = ~np.any((srt[:, :3] == srt[:, 1:]) & (srt[:, :3] >= 0), axis=1)

    if drop_degenerate:
        p0 = P[idx[:, 0]]
        c = np.cross(P[idx[:, 1]] - p0, P[idx[:, 2]] - p0)
        keep &= ~(np.einsum("ij,ij->i", c, c) < 1e-26)

    # first occurrence of each sorted vertex set wins, in face order
    live = np.nonzero(keep)[0]
    if len(live) > 1:
        ks = srt[live]
        # lexsort is stable, so the first row of each equal-key run carries
        # the lowest original face index - exactly "first occurrence wins"
        order = np.lexsort((ks[:, 3], ks[:, 2], ks[:, 1], ks[:, 0]))
        s = ks[order]
        grp = np.empty(len(order), dtype=bool)
        grp[0] = True
        np.any(s[1:] != s[:-1], axis=1, out=grp[1:])
        keep[live] = False
        keep[live[order[grp]]] = True
    return [tuple(f) for f, k in zip(faces, keep.tolist()) if k]


def _dedupe_scalar(P, faces, drop_degenerate=True):
    """Reference implementation of :func:`_dedupe` (n-gon safe)."""
    seen = set()
    out = []
    for f in faces:
        if len(f) < 3 or len(set(f)) != len(f):
            continue
        key = tuple(sorted(f))
        if key in seen:
            continue
        if drop_degenerate:
            pts = P[list(f)]
            c = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            if float(c @ c) < 1e-26:
                continue
        seen.add(key)
        out.append(tuple(f))
    return out


def _enforce_edge_manifold(P, faces):
    """Drop faces until no edge is used by more than two of them."""
    for _ in range(8):
        use = _edge_use(faces)
        bad = [e for e, l in use.items() if len(l) > 2]
        if not bad:
            return faces
        drop = set()
        for e in bad:
            fl = [i for i in use[e] if i not in drop]
            if len(fl) <= 2:
                continue
            # keep the two best: quads before triangles, larger before smaller
            fl.sort(key=lambda i: (len(faces[i]) == 4, _face_area(P, faces[i])),
                    reverse=True)
            drop.update(fl[2:])
        faces = [f for i, f in enumerate(faces) if i not in drop]
    return faces


def _orient(P, NRM, faces):
    """Propagate a consistent winding through every connected component and
    flip each component so it agrees with the reference normals."""
    n = len(faces)
    if n == 0:
        return faces
    # directed edge -> list of (face, forward?)
    dmap = {}
    dget = dmap.get
    for i, f in enumerate(faces):
        k = len(f)
        for a in range(k):
            u, v = f[a], f[(a + 1) % k]
            fw = u < v
            e = (u, v) if fw else (v, u)
            lst = dget(e)
            if lst is None:
                dmap[e] = [(i, fw)]
            else:
                lst.append((i, fw))

    # plain Python flags: the traversal touches these once per half-edge and
    # numpy scalar indexing costs more than the walk itself
    flip = [False] * n
    seen = [False] * n
    Pl = P.tolist()
    NRMl = NRM.tolist()
    for s in range(n):
        if seen[s]:
            continue
        comp = [s]
        seen[s] = True
        stack = [s]
        while stack:
            i = stack.pop()
            f = faces[i]
            k = len(f)
            fi = flip[i]
            for a in range(k):
                u, v = f[a], f[(a + 1) % k]
                fw = u < v
                e = (u, v) if fw else (v, u)
                for (j, fwd) in dmap[e]:
                    if j == i or seen[j]:
                        continue
                    # i traverses e as (u<v) == (u < v); j must be opposite
                    fwd_i = fw != fi
                    flip[j] = (fwd == fwd_i)
                    seen[j] = True
                    comp.append(j)
                    stack.append(j)
        # component-wide sign vote against the reference normals
        votes = 0
        for i in comp:
            f = faces[i] if not flip[i] else tuple(reversed(faces[i]))
            p0 = Pl[f[0]]; p1 = Pl[f[1]]; p2 = Pl[f[2]]
            a0 = p1[0] - p0[0]; a1 = p1[1] - p0[1]; a2 = p1[2] - p0[2]
            b0 = p2[0] - p0[0]; b1 = p2[1] - p0[1]; b2 = p2[2] - p0[2]
            s0 = s1 = s2 = 0.0
            for vtx in f:
                nv = NRMl[vtx]
                s0 += nv[0]; s1 += nv[1]; s2 += nv[2]
            dp = ((a1 * b2 - a2 * b1) * s0 + (a2 * b0 - a0 * b2) * s1
                  + (a0 * b1 - a1 * b0) * s2)
            votes += 1 if dp < 0.0 else (-1 if dp > 0.0 else 0)
        if votes > 0:
            for i in comp:
                flip[i] = not flip[i]

    return [tuple(reversed(f)) if flip[i] else tuple(f)
            for i, f in enumerate(faces)]


def _make_injective(faces):
    """Ensure every directed edge is used by at most one face (required by the
    half-edge hole walk).  Drops the offenders; the holes they leave behind are
    filled by the repair stage."""
    seen = {}
    drop = set()
    for i, f in enumerate(faces):
        k = len(f)
        ok = True
        for a in range(k):
            de = (f[a], f[(a + 1) % k])
            if de in seen:
                ok = False
                break
        if not ok:
            drop.add(i)
            continue
        for a in range(k):
            seen[(f[a], f[(a + 1) % k])] = i
    if not drop:
        return faces
    return [f for i, f in enumerate(faces) if i not in drop]


def _drop_isolated(faces, min_keep=8):
    """Remove dangling faces whose every edge is a boundary edge.

    Such a face is a floating flap: trying to *fill* the hole around it would
    only duplicate it.  Dropping it makes its edges disappear entirely.
    """
    for _ in range(4):
        if len(faces) <= min_keep:
            return faces
        use = _edge_use(faces)
        drop = set()
        for i, f in enumerate(faces):
            k = len(f)
            nb = 0
            for a in range(k):
                u, v = f[a], f[(a + 1) % k]
                if len(use[(u, v) if u < v else (v, u)]) == 1:
                    nb += 1
            if nb >= k:
                drop.add(i)
        if not drop:
            return faces
        faces = [f for i, f in enumerate(faces) if i not in drop]
    return faces


def _simple_cycles(loop):
    """Decompose a (possibly pinched) closed walk into simple cycles."""
    out = []
    stack = []
    pos = {}
    for v in loop:
        if v in pos:
            i = pos[v]
            cyc = stack[i:]
            if len(cyc) >= 3:
                out.append(cyc)
            for w in cyc[1:]:
                pos.pop(w, None)
            del stack[i + 1:]
        else:
            pos[v] = len(stack)
            stack.append(v)
    if len(stack) >= 3:
        out.append(stack)
    return out


def _hole_loops(faces):
    """All hole loops of the current face set, as vertex sequences whose
    consecutive pairs are the hole's directed edges."""
    dmap = {}
    prev_of = {}
    for i, f in enumerate(faces):
        k = len(f)
        for a in range(k):
            u, v = f[a], f[(a + 1) % k]
            dmap[(u, v)] = i
            prev_of[(u, v)] = (f[(a - 1) % k], u)

    hole = [(v, u) for (u, v) in dmap if (v, u) not in dmap]
    if not hole:
        return []
    hole_set = set(hole)
    loops = []
    visited = set()
    for h0 in hole:
        if h0 in visited:
            continue
        seq = []
        h = h0
        ok = True
        for _ in range(1 << 22):
            if h in visited:
                ok = False
                break
            visited.add(h)
            seq.append(h[0])
            u, v = h
            e = (v, u)
            nxt = None
            for _r in range(256):
                z = prev_of[e][0]
                if (v, z) in dmap:
                    e = (v, z)
                else:
                    nxt = (v, z)
                    break
            if nxt is None or nxt not in hole_set:
                ok = False
                break
            h = nxt
            if h == h0:
                break
        if ok and len(seq) >= 3:
            loops.extend(_simple_cycles(seq))
    return loops


def _collapse_vertex(faces, keep, drop):
    """Weld ``drop`` onto ``keep`` (used to close geometrically degenerate
    slivers, e.g. three collinear samples on a crease)."""
    out = []
    for f in faces:
        g = [keep if v == drop else v for v in f]
        k = len(g)
        h = [g[a] for a in range(k) if g[a] != g[(a + 1) % k]]
        if len(h) >= 3 and len(set(h)) == len(h):
            out.append(tuple(h))
    return out


def _fill_holes(P, faces, cbnd):
    """Close every hole that is not a genuine input boundary.

    Returns ``(faces, new_points, welds, changed)``.  ``new_points`` are the
    centroid vertices of the fan-filled loops (their indices continue after
    ``len(P)``); ``welds`` are ``(keep, drop)`` vertex pairs for degenerate
    slivers that cannot be closed by a face.
    """
    loops = _hole_loops(faces)
    if _DEBUG:
        nb = sum(1 for e, l in _edge_use(faces).items() if len(l) == 1)
        print("   [fill] boundary_edges=%d loops=%d sizes=%s"
              % (nb, len(loops), sorted(len(l) for l in loops)[:40]))
    if not loops:
        return faces, [], [], False
    existing = set(_edge_use(faces))
    added = []
    new_pts = []
    welds = []
    welded = set()
    changed = False
    skipped = []
    for loop in loops:
        k = len(loop)
        if k < 3 or len(set(loop)) != k:
            skipped.append(("repeated", k))
            continue
        pts = P[list(loop)]
        c = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if k <= 4 and float(c @ c) < 1e-26:
            # collinear sliver: no face can close it - weld the closest pair
            d = pts - np.roll(pts, -1, axis=0)
            a = int(np.argmin(np.einsum("ij,ij->i", d, d)))
            u, v = loop[a], loop[(a + 1) % k]
            keep, drop = (u, v) if u < v else (v, u)
            if keep not in welded and drop not in welded:
                welds.append((keep, drop))
                welded.add(keep)
                welded.add(drop)
                changed = True
            continue
        if cbnd is not None and all(
                v < len(cbnd) and cbnd[v] for v in loop):
            skipped.append(("boundary", k))
            continue                     # real opening of an open input mesh
        if cbnd is not None and k > MAX_ORBIT:
            # ...and "all" is too brittle a test for a *long* loop.  A hole
            # the filler exists to close is a lattice defect: a missing cell,
            # never more than an orbit long, and its corners are interior
            # samples.  A hundred-vertex loop is an opening the extractor
            # traced correctly, and one corner of it whose cluster happened
            # not to catch an input boundary vertex is enough to make `all`
            # fail - at which point the filler fans a centroid across the
            # entire opening and the mesh is destroyed (measured on a disc
            # with a hole, orientation field ringed: the outer border came
            # out 130 boundary corners plus 3 interior ones and was filled
            # with a 130-spoke fan).  Long loops therefore go by majority.
            nb = sum(1 for v in loop if v < len(cbnd) and cbnd[v])
            if nb >= FILL_BOUNDARY_FRAC * k:
                skipped.append(("boundary~", k))
                continue
        parts = None
        own = set()
        for b in range(k):
            u, v = loop[b], loop[(b + 1) % k]
            own.add((u, v) if u < v else (v, u))
        if k in (3, 4):
            parts = [tuple(loop)]
        elif k <= 6:
            parts = _split_ngon(P, tuple(loop), forbidden=existing, skip=own)
        if parts is None:
            # centroid fan: consumes two boundary edges per quad and only ever
            # touches edges that currently have a single face, so it can never
            # produce a non-manifold edge
            cen = P[list(loop)].mean(axis=0)
            ci = len(P) + len(new_pts)
            new_pts.append(cen)
            parts = []
            a = 0
            while a + 2 <= k:
                parts.append((loop[a], loop[(a + 1) % k],
                              loop[(a + 2) % k], ci))
                a += 2
            if a < k:                    # odd loop -> one closing triangle
                parts.append((loop[a], loop[(a + 1) % k], ci))
        for part in parts:
            m = len(part)
            for b in range(m):
                u, v = part[b], part[(b + 1) % m]
                existing.add((u, v) if u < v else (v, u))
        added.extend(parts)
        changed = True
    if _DEBUG and (skipped or welds):
        print("   [fill] skipped=%s welds=%s" % (skipped[:40], welds[:20]))
    return (faces + added if added else faces), new_pts, welds, changed


def _remove_doublets(P, faces):
    """Two faces meeting at a valence-2 interior vertex -> one face."""
    for _ in range(4):
        use = _edge_use(faces)
        # Only vertices of edge-valence exactly 2 can be doublet centres, and
        # they are a handful out of tens of thousands - so count first and
        # build the (much more expensive) per-vertex lists for those alone.
        val = {}
        vget = val.get
        for (u, v) in use:
            val[u] = vget(u, 0) + 1
            val[v] = vget(v, 0) + 1
        cand = {v for v, c in val.items() if c == 2}
        if not cand:
            return faces
        vert_edges = {}
        for e, fl in use.items():
            u, v = e
            if u in cand:
                vert_edges.setdefault(u, []).append((e, len(fl)))
            if v in cand:
                vert_edges.setdefault(v, []).append((e, len(fl)))
        vert_faces = {}
        for i, f in enumerate(faces):
            for v in f:
                if v in cand:
                    vert_faces.setdefault(v, []).append(i)

        dead = set()
        new = {}
        touched = set()
        for v, el in vert_edges.items():
            if len(el) != 2 or any(c != 2 for _e, c in el):
                continue
            fl = vert_faces.get(v, [])
            if len(fl) != 2 or fl[0] == fl[1]:
                continue
            i, j = fl
            if i in touched or j in touched:
                continue
            f1, f2 = list(faces[i]), list(faces[j])
            if v not in f1 or v not in f2:
                continue
            a1 = f1.index(v)
            b = f1[(a1 + 1) % len(f1)]
            r1 = f1[(a1 + 1) % len(f1):] + f1[:(a1 + 1) % len(f1)]
            r1 = [x for x in r1 if x != v]          # starts at b, ends at a
            a = f1[(a1 - 1) % len(f1)]
            if v not in f2:
                continue
            a2 = f2.index(v)
            if f2[(a2 + 1) % len(f2)] != a or f2[(a2 - 1) % len(f2)] != b:
                continue
            r2 = f2[(a2 + 1) % len(f2):] + f2[:(a2 + 1) % len(f2)]
            r2 = [x for x in r2 if x != v]          # starts at a, ends at b
            merged = r1 + r2[1:-1]
            if len(merged) not in (3, 4) or len(set(merged)) != len(merged):
                continue
            dead.add(i)
            dead.add(j)
            touched.add(i)
            touched.add(j)
            new[i] = tuple(merged)
        if not new:
            return faces
        out = []
        for i, f in enumerate(faces):
            if i in new:
                out.append(new[i])
            elif i in dead:
                continue
            else:
                out.append(f)
        faces = out
    return faces


def _merge_tri_pairs(P, faces, min_quality=-1.0):
    tri = [i for i, f in enumerate(faces) if len(f) == 3]
    if len(tri) < 2:
        return faces
    use = _edge_use(faces)
    is_tri = [len(f) == 3 for f in faces]
    existing = set(use)
    cand = []
    for e, fl in use.items():
        if len(fl) != 2:
            continue
        i, j = fl
        if not (is_tri[i] and is_tri[j]):
            continue
        fi, fj = faces[i], faces[j]
        a, b = e
        ai = fi.index(a)
        forward_i = fi[(ai + 1) % 3] == b
        aj = fj.index(a)
        forward_j = fj[(aj + 1) % 3] == b
        if forward_i == forward_j:
            continue
        if not forward_i:
            i, j = j, i
            fi, fj = fj, fi
        c = [v for v in fi if v not in (a, b)][0]
        d = [v for v in fj if v not in (a, b)][0]
        quad = (b, c, a, d)
        if len(set(quad)) != 4:
            continue
        key = (min(c, d), max(c, d))
        if key in existing:
            continue
        q = _quad_quality(P, quad)
        if q < min_quality:
            continue
        cand.append((-q, i, j, quad))
    if not cand:
        return faces
    cand.sort(key=lambda t: (t[0], t[1], t[2]))
    taken = set()
    merged = {}
    for _, i, j, quad in cand:
        if i in taken or j in taken:
            continue
        taken.add(i)
        taken.add(j)
        merged[i] = quad
    out = []
    for i, f in enumerate(faces):
        if i in merged:
            out.append(merged[i])
        elif i in taken:
            continue
        else:
            out.append(f)
    return out


def _face_edges(f):
    k = len(f)
    return [((f[a], f[(a + 1) % k]) if f[a] < f[(a + 1) % k]
             else (f[(a + 1) % k], f[a])) for a in range(k)]


def _merge_along(f1, f2):
    """Merge two consistently-wound faces sharing exactly one edge.

    Returns ``(cycle, shared_edge)`` or ``None``.
    """
    e1 = set(_face_edges(f1))
    k1, k2 = len(f1), len(f2)
    shared = []
    for a in range(k2):
        u, v = f2[a], f2[(a + 1) % k2]
        if ((u, v) if u < v else (v, u)) in e1:
            shared.append((u, v))
    if len(shared) != 1:
        return None
    b, a = shared[0]                      # f2 traverses b -> a
    if a not in f1 or b not in f1:
        return None
    i = f1.index(a)
    if f1[(i + 1) % k1] != b:
        return None
    j = f2.index(b)
    if f2[(j + 1) % k2] != a:
        return None
    r1 = list(f1[(i + 1) % k1:]) + list(f1[:(i + 1) % k1])
    r2 = list(f2[(j + 1) % k2:]) + list(f2[:(j + 1) % k2])
    merged = r1 + r2[1:-1]
    if len(set(merged)) != len(merged) or len(merged) != k1 + k2 - 2:
        return None
    return merged, ((a, b) if a < b else (b, a))


def _split_pentagon(P, pent, e_next, edge_set, min_quality=-1.0):
    """Split a pentagon into quad + triangle so that the triangle carries
    ``e_next``.  Returns ``(quad, tri, chord)`` or ``None``."""
    j = -1
    for a in range(5):
        u, v = pent[a], pent[(a + 1) % 5]
        if ((u, v) if u < v else (v, u)) == e_next:
            j = a
            break
    if j < 0:
        return None
    opts = []
    for which in (0, 1):
        if which == 0:
            tri = (pent[(j - 1) % 5], pent[j], pent[(j + 1) % 5])
            quad = (pent[(j + 1) % 5], pent[(j + 2) % 5],
                    pent[(j + 3) % 5], pent[(j - 1) % 5])
        else:
            tri = (pent[j], pent[(j + 1) % 5], pent[(j + 2) % 5])
            quad = (pent[(j + 2) % 5], pent[(j + 3) % 5],
                    pent[(j + 4) % 5], pent[j])
        c0, c1 = tri[0], tri[2]
        chord = (c0, c1) if c0 < c1 else (c1, c0)
        if chord in edge_set:
            continue
        q = _quad_quality(P, quad)
        if q < min_quality:
            continue
        opts.append((q, quad, tri, chord))
    if not opts:
        return None
    opts.sort(key=lambda t: -t[0])
    return opts[0][1], opts[0][2], opts[0][3]


def _bfs_to_tri(start, adj, tri_set, blocked, max_depth):
    """Shortest face path from ``start`` to another triangle through quads."""
    prev = {start: None}
    frontier = [start]
    for _d in range(max_depth):
        nxt = []
        for i in frontier:
            for j in adj.get(i, ()):
                if j in prev or j in blocked:
                    continue
                prev[j] = i
                if j in tri_set:
                    path = [j]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
                nxt.append(j)
        if not nxt:
            return None
        frontier = nxt
    return None


def _annihilate_triangles(P, faces, max_depth=16, rounds=16,
                          min_quality=-1.0):
    """Cancel triangles pairwise.

    A triangle is walked through the quads separating it from another triangle
    (merge tri+quad -> pentagon, re-split so the triangle lands on the far
    side); when the two meet they fuse into a quad.  Every step is checked
    against the existing edge set, so manifoldness is preserved.
    """
    faces = [tuple(f) for f in faces]
    for _r in range(rounds):
        alive = [f for f in faces if f is not None]
        if sum(1 for f in alive if len(f) == 3) < 2:
            break
        faces = alive
        use = _edge_use(faces)
        edge_set = set(use)
        adj = {}
        adj_edge = {}
        for e, fl in use.items():
            if len(fl) == 2:
                i, j = fl
                adj.setdefault(i, []).append(j)
                adj.setdefault(j, []).append(i)
                adj_edge[(i, j)] = e
                adj_edge[(j, i)] = e
        tri_ids = [i for i, f in enumerate(faces) if len(f) == 3]
        tri_set = set(tri_ids)
        blocked = set()
        progress = False
        for t0 in tri_ids:
            if t0 in blocked:
                continue
            path = _bfs_to_tri(t0, adj, tri_set - {t0}, blocked, max_depth)
            if path is None or len(path) < 2:
                continue
            pedge = [adj_edge.get((path[i], path[i + 1]))
                     for i in range(len(path) - 1)]
            if any(e is None for e in pedge):
                continue
            upd = {}
            cur = path[0]
            cur_face = faces[cur]
            # The walk speculatively edits the edge set and throws the edits
            # away when a step fails.  Copying the whole set per candidate
            # (~76k entries, once per triangle pair) dominated this pass, so
            # the edits are applied in place and journalled for rollback.
            eset = edge_set
            undo = []
            ok = True
            for si in range(1, len(path) - 1):
                step = path[si]
                m = _merge_along(cur_face, faces[step])
                if m is None:
                    ok = False
                    break
                pent, shared = m
                if len(pent) != 5:
                    ok = False
                    break
                if shared in eset:
                    eset.discard(shared)
                    undo.append((True, shared))
                sp = _split_pentagon(P, pent, pedge[si], eset,
                                     min_quality)
                if sp is None:
                    ok = False
                    break
                quad, tri, chord = sp
                if chord not in eset:
                    eset.add(chord)
                    undo.append((False, chord))
                upd[cur] = quad
                upd[step] = tri
                cur = step
                cur_face = tri
            if ok:
                last = path[-1]
                m = _merge_along(cur_face, faces[last])
                if m is None or len(m[0]) != 4:
                    ok = False
                else:
                    quad = tuple(m[0])
                    if _quad_quality(P, quad) < min_quality:
                        ok = False
            if not ok:
                for restore, e in reversed(undo):
                    if restore:
                        eset.add(e)
                    else:
                        eset.discard(e)
                continue
            upd[cur] = quad
            upd[last] = None
            for i, v in upd.items():
                faces[i] = v
            edge_set = eset
            edge_set.discard(m[1])
            blocked.update(path)
            progress = True
        faces = [f for f in faces if f is not None]
        if not progress:
            break
    return [f for f in faces if f is not None]


# --------------------------------------------------------------------------
# surface projection (uniform spatial hash over the input triangles)
# --------------------------------------------------------------------------

def _closest_on_tri(P, A, B, C):
    """Closest point on each triangle (Ericson's region test), vectorised.

    Region overrides are evaluated on the masked subset only - building six
    full-size candidate arrays was the single hottest line in the profile.
    """
    AB = B - A
    AC = C - A
    AP = P - A
    d1 = _dot(AB, AP)
    d2 = _dot(AC, AP)
    BP = P - B
    d3 = _dot(AB, BP)
    d4 = _dot(AC, BP)
    CP = P - C
    d5 = _dot(AB, CP)
    d6 = _dot(AC, CP)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    flat = np.abs(denom) < 1e-30
    inv = 1.0 / np.where(flat, 1.0, denom)
    res = A + AB * (vb * inv)[:, None] + AC * (vc * inv)[:, None]
    if flat.any():
        res[flat] = A[flat]

    def _safe(num, den):
        return num / np.where(np.abs(den) < 1e-30, 1.0, den)

    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)                 # edge AB
    if m.any():
        k = np.nonzero(m)[0]
        res[k] = A[k] + AB[k] * _safe(d1[k], d1[k] - d3[k])[:, None]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)                 # edge AC
    if m.any():
        k = np.nonzero(m)[0]
        res[k] = A[k] + AC[k] * _safe(d2[k], d2[k] - d6[k])[:, None]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)   # edge BC
    if m.any():
        k = np.nonzero(m)[0]
        num = d4[k] - d3[k]
        t = _safe(num, num + (d5[k] - d6[k]))
        res[k] = B[k] + (C[k] - B[k]) * t[:, None]

    m = (d1 <= 0) & (d2 <= 0)                             # corner A
    if m.any():
        res[m] = A[m]
    m = (d3 >= 0) & (d4 <= d3)                            # corner B
    if m.any():
        res[m] = B[m]
    m = (d6 >= 0) & (d5 <= d6)                            # corner C
    if m.any():
        res[m] = C[m]
    return res


def _closest_on_seg(P, A, B):
    """Closest point on each segment (A, B)."""
    AB = B - A
    denom = np.einsum("ij,ij->i", AB, AB)
    t = np.einsum("ij,ij->i", P - A, AB) / np.where(denom < 1e-30, 1.0, denom)
    t = np.clip(np.where(denom < 1e-30, 0.0, t), 0.0, 1.0)
    return A + AB * t[:, None]


class _Projector:
    """Nearest point on a triangle mesh - or a segment soup - via a uniform grid.

    ``prim`` is (m,3) triangles or (m,2) segments.  Segments are NOT expressible
    as degenerate triangles: with ``B == C`` the barycentric region test for
    edge BC is satisfied unconditionally and every query would snap to an
    endpoint, so they get their own closest-point routine.

    Fully batched: one ragged gather builds every (query, candidate triangle)
    pair, then a single ``_closest_on_tri`` call and a segment-min pick the
    winners.  Looping per grid cell instead cost ~25 s on a 91 k-triangle
    avatar, almost all of it small-array numpy overhead.
    """

    MAX_PAIRS = 4_000_000

    def __init__(self, V, F, max_dims=512):
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.F = np.ascontiguousarray(F, dtype=np.int64)
        self.k = self.F.shape[1] if self.F.ndim == 2 and len(self.F) else 3
        self.lo = self.V.min(axis=0)
        ext = np.maximum(self.V.max(axis=0) - self.lo, 1e-12)
        self.hi = self.lo + ext
        self.diag = float(np.linalg.norm(ext))
        if len(F):
            tri = self.V[self.F]
            roll = list(range(1, self.k)) + [0]
            # median, not mean: sculpts mix 5x size ranges and the mean would
            # leave hundreds of tiny triangles in every dense cell
            elen = np.median(np.linalg.norm(tri[:, roll] - tri, axis=2))
        else:
            elen = self.diag
        cell = max(1.5 * float(elen), self.diag / float(max_dims))
        self.dims = np.clip(np.ceil(ext / cell).astype(np.int64), 1, max_dims)
        self.cell = ext / self.dims

        if len(F) == 0:
            self.uniq_key = np.zeros(0, dtype=np.int64)
            self.start = np.zeros(0, dtype=np.int64)
            self.count = np.zeros(0, dtype=np.int64)
            self.tri_sorted = np.zeros(0, dtype=np.int64)
            return
        key, tid = self._register()
        order = np.argsort(key, kind="stable")
        skey = key[order]
        stid = tid[order]
        # Collapse (cell, primitive) duplicates.  The cell is ~1.5 median edge
        # lengths across, so a primitive usually registers all k+1 of its
        # sample points in the SAME cell and every candidate gather then hands
        # it to the distance test k+1 times - 2.4x more pairs than necessary
        # on a real sculpt.  Only the *first* occurrence is kept, so the order
        # of the survivors inside a cell is untouched and the distance test's
        # tie-break (lowest candidate index wins) picks exactly what it picked
        # before.
        m = np.int64(max(len(F), 1))
        _, firsts = np.unique(skey * m + stid, return_index=True)
        keep = np.zeros(len(order), dtype=bool)
        keep[firsts] = True
        self.tri_sorted = np.ascontiguousarray(stid[keep])
        self.uniq_key, self.start, self.count = np.unique(
            skey[keep], return_index=True, return_counts=True)
        # corner arrays, so a candidate gather indexes coordinates directly
        # instead of gathering the index triple first
        self.corner = [np.ascontiguousarray(self.V[self.F[:, c]])
                       for c in range(self.k)]

    def _register(self):
        """``(cell key, primitive id)`` pairs: corners plus centroid."""
        F, V = self.F, self.V
        pts = V[F]                                        # (m, k, 3)
        allp = np.concatenate([V[F[:, c]] for c in range(self.k)]
                              + [pts.mean(axis=1)], axis=0)
        return (self._key(self._cell_of(allp)),
                np.tile(np.arange(len(F), dtype=np.int64), self.k + 1))

    def _cell_of(self, P):
        c = np.floor((P - self.lo) / self.cell).astype(np.int64)
        return np.clip(c, 0, self.dims - 1)

    def _key(self, c):
        return (c[:, 0] * self.dims[1] + c[:, 1]) * self.dims[2] + c[:, 2]

    def _gather(self, cid, radius):
        """Ragged (query, triangle) pair lists for a neighbourhood radius."""
        o = np.arange(-radius, radius + 1, dtype=np.int64)
        w = len(o)
        nq = len(cid)
        # The cell key is affine in the cell coordinates, so a neighbourhood
        # key is the query's own key plus a constant per offset - no need to
        # build the (nq, K, 3) coordinate block or re-derive K keys from it.
        noff = ((o[:, None, None] * self.dims[1] + o[None, :, None])
                * self.dims[2] + o[None, None, :]).reshape(-1)
        keys = (self._key(cid)[:, None] + noff[None, :]).reshape(-1)
        # in-bounds is a product over the three axes, so it is built from
        # three (nq, w) tests instead of an (nq, K, 3) comparison
        ax = []
        for c in range(3):
            cc = cid[:, c, None] + o[None, :]
            ax.append((cc >= 0) & (cc < self.dims[c]))
        inb = (ax[0][:, :, None, None] & ax[1][:, None, :, None]
               & ax[2][:, None, None, :]).reshape(nq, w ** 3)
        pos = np.clip(np.searchsorted(self.uniq_key, keys), 0,
                      max(len(self.uniq_key) - 1, 0))
        hit = inb.ravel() & (self.uniq_key[pos] == keys) if len(
            self.uniq_key) else np.zeros(len(keys), dtype=bool)
        cnt = np.where(hit, self.count[pos], 0)
        st = np.where(hit, self.start[pos], 0)
        tot = int(cnt.sum())
        if tot == 0:
            return None, None, np.zeros(len(cid), dtype=np.int64)
        off = np.zeros(len(cnt) + 1, dtype=np.int64)
        np.cumsum(cnt, out=off[1:])
        run = np.arange(tot) - np.repeat(off[:-1], cnt)
        tri_idx = self.tri_sorted[np.repeat(st, cnt) + run]
        qrep = np.repeat(np.arange(nq, dtype=np.int64), w ** 3)
        pt_idx = np.repeat(qrep, cnt)
        return pt_idx, tri_idx, cnt.reshape(len(cid), -1).sum(axis=1)

    def _solve(self, P, pt_idx, tri_idx, out, best):
        """Segment-min over the candidate pairs; keeps the running best.

        ``pt_idx`` always arrives grouped (both callers emit it in query
        order), so the winner of each group is found with two segmented
        reductions instead of an O(m log m) lexsort.  The tie-break is the
        same one a stable lexsort gives: lowest candidate index wins.
        """
        m = len(pt_idx)
        if m == 0:
            return
        Pq = P[pt_idx]
        if self.k == 2:
            cp = _closest_on_seg(Pq, self.corner[0][tri_idx],
                                 self.corner[1][tri_idx])
        else:
            cp = _closest_on_tri(Pq, self.corner[0][tri_idx],
                                 self.corner[1][tri_idx],
                                 self.corner[2][tri_idx])
        d = cp - Pq
        d2 = np.einsum("ij,ij->i", d, d)
        starts = np.empty(m, dtype=bool)
        starts[0] = True
        np.not_equal(pt_idx[1:], pt_idx[:-1], out=starts[1:])
        starts = np.flatnonzero(starts)
        gmin = np.minimum.reduceat(d2, starts)
        cnt = np.diff(np.append(starts, m))
        pos = np.where(d2 == np.repeat(gmin, cnt), np.arange(m), m)
        win = np.minimum.reduceat(pos, starts)
        tgt = pt_idx[starts]
        better = gmin < best[tgt]
        out[tgt[better]] = cp[win[better]]
        best[tgt[better]] = gmin[better]

    def project(self, P):
        """Exact nearest point on the input surface for every row of ``P``.

        A radius-``r`` neighbourhood only proves a hit when the distance found
        is below ``r * min(cell)``; anything further out is re-queried with a
        wider ring and finally brute-forced, so the result never silently
        depends on the grid resolution.
        """
        P = np.ascontiguousarray(np.asarray(P, dtype=np.float64).reshape(-1, 3))
        out = P.copy()
        if len(self.F) == 0 or len(P) == 0:
            return out
        self._search(P, out, np.full(len(P), np.inf), None)
        return out

    def project_within(self, P, cap):
        """Nearest primitive point, but only for the points inside ``cap``.

        Returns ``(closest, found)``: ``closest[i]`` is exactly what
        :meth:`project` would return whenever ``found[i]`` is true, and
        ``found[i]`` is false only when the nearest primitive is provably
        farther than ``cap[i]`` (in which case ``closest[i]`` is ``P[i]``).

        A narrow-band membership test ("is this point on the feature curve?")
        does not need the exact distance of the points that are nowhere near
        it - and paying for those is ruinous against a *sparse* soup: the
        boundary-segment projector covers a curve, so almost every query walks
        the whole radius ladder and then falls into the brute force, which is
        one pass over every segment for every point.  ``cap`` (per point) lets
        the search stop as soon as it has proved the answer is bigger, and the
        caller reads ``inf`` as "further away than you cared about".
        """
        P = np.ascontiguousarray(np.asarray(P, dtype=np.float64).reshape(-1, 3))
        out = P.copy()
        found = np.zeros(len(P), dtype=bool)
        if len(self.F) == 0 or len(P) == 0:
            return out, found
        cap = np.asarray(cap, dtype=np.float64).reshape(-1)
        if cap.size == 1:
            cap = np.full(len(P), float(cap[0]))
        # everything lives inside the primitive bounding box, so a point
        # farther than `cap` from that box needs no search at all - this is
        # what a curve-shaped soup makes of most of a mesh's vertices
        gap = np.maximum(np.maximum(self.lo - P, P - self.hi), 0.0)
        idx = np.nonzero(np.einsum("ij,ij->i", gap, gap) <= cap * cap)[0]
        if len(idx) == 0:
            return out, found
        sub_out = P[idx].copy()
        sub_best = np.full(len(idx), np.inf)
        self._search(P[idx], sub_out, sub_best, cap[idx])
        hit = np.isfinite(sub_best)
        out[idx[hit]] = sub_out[hit]
        found[idx[hit]] = True
        return out, found

    def _search(self, P, out, best, cap):
        """Radius-escalating nearest-primitive search, then a brute force.

        ``cap`` (or ``None``) prunes queries the caller has already declared
        uninteresting: once the searched cube provably reaches past ``cap``,
        the true distance is past it too and the query can be dropped.
        """
        todo = np.arange(len(P), dtype=np.int64)
        for radius in (0, 1, 2, 4, 8):
            if len(todo) == 0:
                break
            cid = self._cell_of(P[todo])
            pt_idx, tri_idx, _per = self._gather(cid, radius)
            if pt_idx is not None:
                nchunk = max(1, int(np.ceil(len(pt_idx) / self.MAX_PAIRS)))
                bnd = np.linspace(0, len(pt_idx), nchunk + 1).astype(int)
                for a, b in zip(bnd[:-1], bnd[1:]):
                    self._solve(P, todo[pt_idx[a:b]], tri_idx[a:b], out, best)
            # exact guarantee: the searched cube reaches this far from P
            lo_b = self.lo + (cid - radius) * self.cell
            hi_b = self.lo + (cid + radius + 1) * self.cell
            safe = np.maximum(
                np.minimum(P[todo] - lo_b, hi_b - P[todo]).min(axis=1), 0.0)
            live = best[todo] > safe ** 2
            if cap is not None:
                # a capped query may also stop as soon as the ring it has
                # searched provably reaches past the caller's band
                live &= safe < cap[todo]
            todo = todo[live]
        if len(todo):
            allt = np.arange(len(self.F), dtype=np.int64)
            step = max(1, int(self.MAX_PAIRS // max(len(allt), 1)))
            for a in range(0, len(todo), step):
                blk = todo[a:a + step]
                self._solve(P, np.repeat(blk, len(allt)),
                            np.tile(allt, len(blk)), out, best)

# --------------------------------------------------------------------------
# relaxation
# --------------------------------------------------------------------------

def _tri_quad_arrays(faces):
    tris = np.asarray([f for f in faces if len(f) == 3],
                      dtype=np.int64).reshape(-1, 3)
    quads = np.asarray([f for f in faces if len(f) == 4],
                       dtype=np.int64).reshape(-1, 4)
    return tris, quads


def _vertex_normals_poly(P, tris, quads, n):
    N = np.zeros((n, 3))

    def _acc(idx, nrm):
        for k in range(idx.shape[1]):
            for c in range(3):
                N[:, c] += np.bincount(idx[:, k], weights=nrm[:, c],
                                       minlength=n)

    if len(tris):
        p = P[tris]
        _acc(tris, np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
    if len(quads):
        p = P[quads]
        _acc(quads, np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]))
    ln = np.sqrt(np.einsum("ij,ij->i", N, N))
    bad = ln < 1e-14
    if bad.any():
        N[bad] = (0.0, 0.0, 1.0)
    return normalize(N)


def _turn_report(tag, P, nb, move):
    """Turning angle along the faired chains (exact indices, no proxy)."""
    idx = np.nonzero(move)[0]
    if len(idx) < 3:
        print("   [turn] %s n<3" % tag)
        return
    e0 = P[idx] - P[nb[idx, 0]]
    e1 = P[nb[idx, 1]] - P[idx]
    l0 = np.sqrt(np.einsum("ij,ij->i", e0, e0))
    l1 = np.sqrt(np.einsum("ij,ij->i", e1, e1))
    ok = (l0 > 1e-15) & (l1 > 1e-15)
    c = np.einsum("ij,ij->i", e0[ok], e1[ok]) / (l0[ok] * l1[ok])
    a = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    print("   [turn] %s n=%d median=%.2f p95=%.2f max=%.2f"
          % (tag, len(a), np.median(a), np.percentile(a, 95), a.max()))


def _build_chains(faces, pinned, P, corner_pts, rho_v):
    """Feature chains through the pinned output vertices.

    Returns ``(nb (n,2) int64, move (n,) bool, snap_idx, snap_pos)``.  A pinned
    vertex is faired only when it has exactly two pinned neighbours - a
    junction cannot slide anywhere sensible.  Each corner of the input feature
    curve is *anchored*: the nearest pinned vertex is moved exactly onto it and
    held, which both keeps the corner sharp and gives the movable run between
    two anchors a clean, correctly-placed endpoint to straighten against.
    """
    n = len(P)
    nb = np.full((n, 2), -1, dtype=np.int64)
    move = np.zeros(n, dtype=bool)
    empty = (np.zeros(0, dtype=np.int64), np.zeros((0, 3)))
    if not np.any(pinned):
        return nb, move, *empty
    adj = {}
    seen = set()
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            if not (pinned[a] and pinned[b]):
                continue
            e = (a, b) if a < b else (b, a)
            if e in seen:
                continue
            seen.add(e)
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    cand = np.array(sorted(v for v, l in adj.items() if len(l) == 2),
                    dtype=np.int64)
    if len(cand):
        for v in cand:
            nb[v] = adj[int(v)]
        move[cand] = True

    snap_idx = np.zeros(0, dtype=np.int64)
    snap_pos = np.zeros((0, 3))
    if corner_pts is not None and len(corner_pts):
        pv = np.array(sorted(adj), dtype=np.int64)
        if len(pv):
            cp = np.asarray(corner_pts, dtype=np.float64)
            # nearest pinned output vertex per corner (deterministic ties)
            best = np.full(len(cp), -1, dtype=np.int64)
            bd = np.full(len(cp), np.inf)
            step = max(1, int(4_000_000 // max(len(pv), 1)))
            for a in range(0, len(cp), step):
                blk = cp[a:a + step]
                d = np.linalg.norm(blk[:, None, :] - P[pv][None, :, :], axis=2)
                j = np.argmin(d, axis=1)
                best[a:a + step] = pv[j]
                bd[a:a + step] = d[np.arange(len(blk)), j]
            order = np.lexsort((bd, best))
            keep = np.ones(len(order), dtype=bool)
            bs = best[order]
            keep[1:] = bs[1:] != bs[:-1]          # one corner per vertex
            sel = order[keep]
            snap_idx = best[sel]
            snap_pos = cp[sel]
            move[snap_idx] = False
    return nb, move, snap_idx, snap_pos


def _square_targets(P, quads, n, rest=None, size_lock=0.0):
    """Per-vertex target from each quad's best-fit square.

    In the quad's own plane the four corners are complex numbers ``z_k``; a
    perfect square is ``z_k = a * i**k`` for one complex ``a``, so the
    least-squares fit is just ``a = mean(z_k * conj(i**k))`` - rotation and
    scale come out free.  Pulling corners toward ``a * i**k`` is the
    "squarify" half of the polish (cf. Instant Meshes' output smoothing).
    """
    acc = np.zeros((n, 3))
    cnt = np.zeros(n)
    if not len(quads):
        return acc, cnt
    p = P[quads]                                   # (m,4,3)
    c = p.mean(axis=1)
    d = p - c[:, None, :]
    nrm = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    nl = np.sqrt(np.einsum("ij,ij->i", nrm, nrm))
    ok = nl > 1e-14
    nrm = nrm / np.maximum(nl, 1e-14)[:, None]
    u = d[:, 0] - nrm * _dot(d[:, 0], nrm)[:, None]
    ul = np.sqrt(np.einsum("ij,ij->i", u, u))
    deg = ul < 1e-12
    if deg.any():
        u[deg] = _any_tangent(nrm[deg])
        ul = np.sqrt(np.einsum("ij,ij->i", u, u))
    u = u / np.maximum(ul, 1e-14)[:, None]
    w = np.cross(nrm, u)

    x = np.einsum("mkc,mc->mk", d, u)
    y = np.einsum("mkc,mc->mk", d, w)
    # a = 1/4 * sum_k (x_k + i y_k) * conj(i**k),  conj(i**k) = 1, -i, -1, i
    ar = 0.25 * (x[:, 0] + y[:, 1] - x[:, 2] - y[:, 3])
    ai = 0.25 * (y[:, 0] - x[:, 1] - y[:, 2] + x[:, 3])
    if size_lock > 0.0 and rest is not None:
        # pull the square's *size* toward the local target edge length too, so
        # one term delivers both squareness and even quad areas
        mag = np.sqrt(ar * ar + ai * ai)
        want = rest[quads].mean(axis=1) * (0.5 * np.sqrt(2.0))
        newmag = (1.0 - size_lock) * mag + size_lock * want
        sc = newmag / np.maximum(mag, 1e-14)
        ar = ar * sc
        ai = ai * sc
    zx = np.stack([ar, -ai, -ar, ai], axis=1)      # Re(a * i**k)
    zy = np.stack([ai, ar, -ai, -ar], axis=1)      # Im(a * i**k)
    tgt = (c[:, None, :] + zx[:, :, None] * u[:, None, :]
           + zy[:, :, None] * w[:, None, :])
    # extract the live rows once: `tgt[ok, k, cc]` ran a boolean take over an
    # (m, 4, 3) block twelve times per call, 120 times per solve
    tok = tgt[ok]
    qok = quads[ok]
    for k in range(4):
        idx = qok[:, k]
        for cc in range(3):
            acc[:, cc] += np.bincount(idx, weights=tok[:, k, cc], minlength=n)
        cnt += np.bincount(idx, minlength=n)
    return acc, cnt


def _regularize(P, faces, frozen, projector, rho_v, iters=REGULARIZE_ITERS,
                w_square=W_SQUARE, w_even=W_EVEN, w_lap=W_LAPLACE,
                step=REGULARIZE_STEP, max_drift=MAX_DRIFT,
                size_lock=SIZE_LOCK, rest_smooth=REST_SMOOTH,
                rest_rho=REST_RHO, project_every=PROJECT_EVERY,
                tol=REGULARIZE_TOL, uncapped=None, chain_nb=None,
                chain_move=None, feat_proj=None, chain_step=FEATURE_STEP,
                chain_drift=FEATURE_MAX_DRIFT, sym_axes=None):
    """Quad fairing: even edge lengths + square corners, on the surface.

    Each iteration blends three tangential pulls - a best-fit-square target
    per quad, a spring that drives every incident edge toward the vertex's own
    mean edge length (scale free, so it never fights the density field), and a
    light umbrella term - then reprojects onto the input surface.  Creases,
    boundaries and pins are frozen and total drift is capped relative to the
    local ``rho`` so sculpted detail cannot wash out.
    """
    if iters <= 0 or len(faces) == 0:
        return P
    n = len(P)
    tris, quads = _tri_quad_arrays(faces)
    pairs = []
    for f in faces:
        k = len(f)
        for a in range(k):
            pairs.append((f[a], f[(a + 1) % k]))
    e = np.asarray(pairs, dtype=np.int64)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.maximum(np.bincount(src, minlength=n).astype(np.float64), 1.0)

    Nv = None
    P0 = P.copy()
    P = P.copy()
    move = ~frozen
    if not move.any():
        return P
    midx = np.nonzero(move)[0]
    cidx = (np.nonzero(chain_move)[0] if chain_move is not None
            else np.zeros(0, dtype=np.int64))
    drift_cap = max_drift * rho_v
    step_cap = step * 1.5 * rho_v

    for _it in range(iters):
        do_proj = (projector is not None
                   and (_it % project_every == 0 or _it == iters - 1))
        Pd = P[dst]
        d = Pd - P[src]
        L = np.sqrt(np.einsum("ij,ij->i", d, d))
        Ls = np.maximum(L, 1e-14)
        # The rest length must be a *smooth* field.  Using each vertex's own
        # mean incident length makes a locally-uniform-but-oversized patch a
        # fixed point, which is precisely the 2x quad-size jitter the fairing
        # is supposed to remove; diffusing it (and anchoring to the density
        # field rho) is what actually equalises quad areas.
        rest = np.bincount(src, weights=L, minlength=n) / deg
        for _ in range(rest_smooth):
            acc_r = np.bincount(src, weights=rest[dst], minlength=n) / deg
            rest = 0.5 * (rest + acc_r)
        if rest_rho > 0.0:
            mr = float(np.mean(rho_v))
            anchor = rho_v * (float(np.mean(L)) / mr) if mr > 1e-14 else rest
            rest = (1.0 - rest_rho) * rest + rest_rho * anchor

        delta = np.zeros((n, 3))
        wsum = float(w_even + w_lap + w_square) or 1.0
        if w_even:
            f = d * (1.0 - rest[src] / Ls)[:, None]
            ev = np.empty((n, 3))
            for c in range(3):
                ev[:, c] = np.bincount(src, weights=f[:, c], minlength=n)
            delta += w_even * (ev / deg[:, None])
        if w_lap:
            lap = np.empty((n, 3))
            for c in range(3):
                lap[:, c] = np.bincount(src, weights=Pd[:, c], minlength=n)
            delta += w_lap * (lap / deg[:, None] - P)
        if w_square and len(quads):
            acc, cnt = _square_targets(P, quads, n, rest=rest,
                                       size_lock=size_lock)
            has = cnt > 0
            sq = np.zeros((n, 3))
            sq[has] = acc[has] / cnt[has][:, None] - P[has]
            delta += w_square * sq
        # convex blend: summing the terms would scale the effective step with
        # the weights and diverge
        delta /= wsum

        # tangential only - the normal component is the projector's job.
        # Normals drift slowly, so they are refreshed on the projection beat.
        if Nv is None or do_proj:
            Nv = _vertex_normals_poly(P, tris, quads, n)
        delta -= Nv * _dot(delta, Nv)[:, None]

        dl = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        scl = np.minimum(1.0, step_cap / np.maximum(dl, 1e-18))
        cand = P[midx] + step * delta[midx] * scl[midx][:, None]
        if do_proj:
            cand = projector.project(cand)
        # keep every vertex within max_drift * rho of where extraction put it
        dr = cand - P0[midx]
        drl = np.sqrt(np.einsum("ij,ij->i", dr, dr))
        cap = drift_cap[midx]
        if uncapped is not None:
            cap = np.where(uncapped[midx], np.inf, cap)
        over = drl > cap
        if over.any():
            fac = (cap[over] / np.maximum(drl[over], 1e-18))[:, None]
            cand[over] = P0[midx[over]] + dr[over] * fac
            if do_proj:
                cand[over] = projector.project(cand[over])
        Pn = P.copy()
        Pn[midx] = cand
        if len(cidx):
            # 1D umbrella ALONG the feature polyline, then snap back onto the
            # true input curve so the feature can never drift off it.  Without
            # this the chains keep the raw extraction jitter and a thin rim
            # (an ear border) reads as saw teeth on the silhouette.
            tgt = 0.5 * (P[chain_nb[cidx, 0]] + P[chain_nb[cidx, 1]])
            cc = P[cidx] + chain_step * (tgt - P[cidx])
            if feat_proj is not None:
                cc = feat_proj.project(cc)
            dr = cc - P0[cidx]
            drl = np.sqrt(np.einsum("ij,ij->i", dr, dr))
            capc = chain_drift * rho_v[cidx]
            over = drl > capc
            if over.any():
                fac = (capc[over] / np.maximum(drl[over], 1e-18))[:, None]
                cc[over] = P0[cidx[over]] + dr[over] * fac
                if feat_proj is not None:
                    cc[over] = feat_proj.project(cc[over])
            if sym_axes:
                # a vertex the bisect put on a mirror plane must stay on it
                for ax, tolp in sym_axes:
                    on = np.abs(P0[cidx][:, ax]) <= tolp
                    if on.any():
                        cc[on, ax] = 0.0
            Pn[cidx] = cc
        shift = np.abs(cand - P[midx]).max(axis=1) / np.maximum(rho_v[midx], EPS)
        P = Pn
        if do_proj and float(shift.max()) < tol:
            break
    return P


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------

def _extract_core(O, Q, N, rho, edges, bnd_verts=None, projector=None,
                  pin_verts=None, reg=None, quad_floor=QUAD_FLOOR,
                  feat_proj=None, corner_pts=None, sym_axes=None,
                  feat_capture=FEATURE_CAPTURE, polish=True, state=None):
    """Position field -> repaired quad-dominant mesh.

    ``polish=False`` returns the same faces with unfaired, unprojected point
    positions - all that the face-count search needs, at roughly half the
    cost.  Pass a dict as ``state`` to receive the pre-polish intermediate,
    which :func:`_polish_positions` can finish later.
    """
    n = O.shape[0]
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return np.zeros((0, 3)), []

    pin_in = None
    if pin_verts is not None or bnd_verts is not None:
        pin_in = np.zeros(O.shape[0], dtype=bool)
        if pin_verts is not None:
            pin_in |= np.asarray(pin_verts, dtype=bool)
        if bnd_verts is not None:
            pin_in |= np.asarray(bnd_verts, dtype=bool)
    cluster, CP, CN, CQ, ce = _collapse(O, Q, N, rho, edges, pin_in)
    nc = CP.shape[0]
    if len(ce) == 0:
        return np.zeros((0, 3)), []

    crho = np.bincount(cluster, weights=rho, minlength=nc) / np.maximum(
        np.bincount(cluster, minlength=nc), 1)

    if bnd_verts is not None and np.any(bnd_verts):
        cbnd = np.zeros(nc, dtype=bool)
        cbnd[cluster[np.asarray(bnd_verts, dtype=bool)]] = True
    else:
        cbnd = np.zeros(nc, dtype=bool)
    # creases pin the lattice but are *not* mesh boundaries: they must not be
    # filled around, yet they must not be smoothed across either
    cpin = np.zeros(nc, dtype=bool)
    if pin_verts is not None and np.any(pin_verts):
        cpin[cluster[np.asarray(pin_verts, dtype=bool)]] = True

    cycles = _rotation_faces(CP, CN, CQ, ce, nc, cbnd)
    if not cycles:
        return np.zeros((0, 3)), []

    faces = []
    for f in cycles:
        if len(f) <= 4:
            faces.append(tuple(f))
        else:
            parts = _split_ngon(CP, tuple(f))
            if parts:
                faces.extend(parts)

    faces = _dedupe(CP, faces)
    if not faces:
        return np.zeros((0, 3)), []
    faces = _enforce_edge_manifold(CP, faces)
    faces = _orient(CP, CN, faces)
    faces = _make_injective(faces)

    # ---- hole filling / doublets / tri fusion, to a fixed point ----------
    P = CP
    n_new = 0

    def _round_trip(P, faces, n_new, do_merge):
        faces, new_pts, welds, changed = _fill_holes(P, faces, cbnd)
        for keep, drop in welds:
            faces = _collapse_vertex(faces, keep, drop)
        if new_pts:
            pts = np.asarray(new_pts, dtype=np.float64).reshape(-1, 3)
            if projector is not None:
                pts = projector.project(pts)
            P = np.concatenate([P, pts], axis=0)
            n_new += len(pts)
        passes = [("dedupe", lambda fl: _dedupe(P, fl)),
                  ("isolated", _drop_isolated),
                  ("manifold", lambda fl: _enforce_edge_manifold(P, fl)),
                  ("injective", _make_injective),
                  ("doublets", lambda fl: _remove_doublets(P, fl))]
        if do_merge:
            # Short, well-shaped cancellations first.  Every step of an
            # annihilation walk distorts the quads it passes through and the
            # final fusion leaves a valence-3 pair, so a depth-16 walk costs
            # real mesh evenness - escalate the walk length only while the
            # quad ratio is still short of the gate.
            def _stage(fl):
                for depth, mq in ((2, 0.10), (2, -1.0), (4, -1.0)):
                    fl = _annihilate_triangles(P, fl, max_depth=depth,
                                               min_quality=mq)
                for depth in (8, 16):
                    nq = sum(1 for f in fl if len(f) == 4)
                    if nq >= quad_floor * len(fl):
                        break
                    fl = _annihilate_triangles(P, fl, max_depth=depth)
                return fl
            passes += [
                ("tripairs", lambda fl: _merge_tri_pairs(P, fl)),
                ("annihil", _stage),
            ]
        passes += [("dedupe2", lambda fl: _dedupe(P, fl)),
                   ("manifold2", lambda fl: _enforce_edge_manifold(P, fl)),
                   ("injective2", _make_injective)]
        for tag, fn in passes:
            faces = fn(faces)
            if _DEBUG:
                nb = sum(1 for l in _edge_use(faces).values() if len(l) == 1)
                nt = sum(1 for f in faces if len(f) == 3)
                print("   [post] %-10s faces=%d bnd=%d tri=%d"
                      % (tag, len(faces), nb, nt))
        return P, faces, n_new, changed

    for _round in range(7):
        do_merge = (_round in (1, 3))
        P, faces, n_new, changed = _round_trip(P, faces, n_new, do_merge)
        if not changed and not do_merge and _round >= 4:
            break

    if not faces:
        return np.zeros((0, 3)), []

    # ---- relax + reproject ----------------------------------------------
    npt = len(P)
    if npt > nc:
        pad = npt - nc
        crho_full = np.concatenate([crho, np.full(pad, float(np.mean(crho)))])
        cbnd_full = np.concatenate([cbnd, np.zeros(pad, dtype=bool)])
    else:
        crho_full = crho
        cbnd_full = cbnd
    is_new = np.zeros(npt, dtype=bool)
    if npt > nc:
        is_new[nc:] = True
    if npt > nc:
        cpin_full = np.concatenate([cpin, np.zeros(npt - nc, dtype=bool)])
    else:
        cpin_full = cpin

    # The polish stage below moves points and never touches `faces`, so the
    # face count is already final here.  `state` hands that intermediate out
    # so the count-adherence search can run attempt after attempt without
    # paying for fairing and reprojection, and polish only the winner.
    if state is not None:
        state.update(P=P, faces=faces, crho_full=crho_full,
                     cbnd_full=cbnd_full, cpin_full=cpin_full, is_new=is_new)
    if polish and projector is not None:
        P = _polish_positions(
            P, faces, crho_full, cbnd_full, cpin_full, is_new, projector,
            reg=reg, feat_proj=feat_proj, corner_pts=corner_pts,
            sym_axes=sym_axes, feat_capture=feat_capture)
    return _compact(P, faces)


def _compact(P, faces):
    """Drop the points no face uses and reindex."""
    used = np.zeros(len(P), dtype=bool)
    for f in faces:
        used[list(f)] = True
    remap = np.full(len(P), -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    return P[used], [tuple(int(remap[v]) for v in f) for f in faces]


def _polish_positions(P, faces, crho_full, cbnd_full, cpin_full, is_new,
                      projector, reg=None, feat_proj=None, corner_pts=None,
                      sym_axes=None, feat_capture=FEATURE_CAPTURE):
    """Fairing + surface/feature reprojection of an extracted mesh.

    Point positions only: ``faces`` is read, never rewritten.
    """
    if projector is not None:
        frozen = cbnd_full | cpin_full
        if feat_proj is not None and feat_capture > 0.0:
            # A vertex can land on the rim without its cluster having captured
            # any pinned input sample; it then gets surface-faired, wanders off
            # the feature and kinks the chain.  Anything already sitting within
            # a narrow band of the feature curve joins the chain instead.
            # Band test only: `project_within` stops as soon as it has proved
            # a point is outside the band, which is the difference between a
            # bounded grid query and brute-forcing every vertex against every
            # feature segment (9.6 s of a 59 s solve on a 230k-tri shell).
            # Inside the band it returns exactly what `project` would, so the
            # distance - and therefore the membership - is bit for bit what
            # the full projection gave.
            band = feat_capture * crho_full
            fcp, fhit = feat_proj.project_within(P, band)
            dfe = np.where(fhit, np.linalg.norm(fcp - P, axis=1), np.inf)
            near = dfe < band
            if _DEBUG:
                print("   [capture] pinned=%d + geometric=%d"
                      % (int(frozen.sum()), int((near & ~frozen).sum())))
            frozen = frozen | near
        r = dict(reg or {})
        chain_nb, chain_move, snap_i, snap_p = _build_chains(
            faces, frozen, P, corner_pts, crho_full)
        if len(snap_i):
            P = P.copy()
            P[snap_i] = snap_p
        if _DEBUG:
            print("   [chains] pinned=%d faired=%d corner_anchors=%d"
                  % (int(frozen.sum()), int(chain_move.sum()), len(snap_i)))
            _turn_report("chain BEFORE", P, chain_nb, chain_move)
        P = _regularize(P, faces, frozen, projector, crho_full,
                        iters=int(r.get("regularize_iters", REGULARIZE_ITERS)),
                        w_square=float(r.get("w_square", W_SQUARE)),
                        w_even=float(r.get("w_even", W_EVEN)),
                        w_lap=float(r.get("w_laplace", W_LAPLACE)),
                        step=float(r.get("regularize_step", REGULARIZE_STEP)),
                        max_drift=float(r.get("max_drift", MAX_DRIFT)),
                        size_lock=float(r.get("size_lock", SIZE_LOCK)),
                        rest_smooth=int(r.get("rest_smooth", REST_SMOOTH)),
                        rest_rho=float(r.get("rest_rho", REST_RHO)),
                        project_every=int(r.get("project_every",
                                                PROJECT_EVERY)),
                        tol=float(r.get("regularize_tol", REGULARIZE_TOL)),
                        uncapped=is_new, chain_nb=chain_nb,
                        chain_move=chain_move, feat_proj=feat_proj,
                        chain_step=float(r.get("feature_step", FEATURE_STEP)),
                        chain_drift=float(r.get("feature_max_drift",
                                                FEATURE_MAX_DRIFT)),
                        sym_axes=sym_axes)
        if _DEBUG:
            _turn_report("chain AFTER ", P, chain_nb, chain_move)
        # Everything gets a final snap to the surface - including the frozen
        # crease/boundary vertices, which never move during fairing but still
        # start life on a tangent plane rather than on the mesh itself.
        # chains are snapped to the feature curve, never to the surface
        moved = projector.project(P)
        d = moved - P
        dl = np.sqrt(np.einsum("ij,ij->i", d, d))
        ok = (dl <= 0.75 * crho_full) | is_new
        ok &= ~frozen if feat_proj is not None else np.ones(len(P), bool)
        P = np.where(ok[:, None], moved, P)
        if feat_proj is not None and np.any(frozen):
            fi = np.nonzero(frozen)[0]
            P[fi] = feat_proj.project(P[fi])
            if len(snap_i):
                P[snap_i] = snap_p
            if _DEBUG:
                dd = np.linalg.norm(feat_proj.project(P[fi]) - P[fi], axis=1)
                cm2 = np.nonzero(chain_move)[0]
                dc = (np.linalg.norm(feat_proj.project(P[cm2]) - P[cm2], axis=1)
                      / np.maximum(crho_full[cm2], EPS)) if len(cm2) else np.zeros(1)
                print("   [featfid] pinned max d/rho=%.2e  faired-chain max d/rho=%.2e"
                      % ((dd / np.maximum(crho_full[fi], EPS)).max(), dc.max()))
            if sym_axes:
                for ax, tolp in sym_axes:
                    on = np.abs(P[fi][:, ax]) <= tolp
                    if on.any():
                        P[fi[on], ax] = 0.0
    return P


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def extract_quads(O, Q, N, rho, edges, min_cycle=3, max_cycle=6,
                  bnd_verts=None):
    """v1 entry point: the caller supplies an already-solved position field.

    ``min_cycle`` / ``max_cycle`` are accepted for signature compatibility and
    ignored - the orbit length limits are now fixed by ``MAX_ORBIT`` and the
    repair stage.  Surface reprojection is skipped here because this entry
    point does not receive the input triangles.
    """
    return _extract_core(O, Q, N, rho, np.asarray(edges), bnd_verts=bnd_verts,
                         projector=None)


def extract(V, F, sol, params=None):
    """NATIVE_V2 entry point.

    Solves the position field for the orientation field in ``sol`` and returns
    a repaired quad-dominant mesh ``(VQ (k,3) float64, FQ list[3|4-tuples])``.

    Recognised ``params`` keys: ``target_faces``, ``seed``, ``sharp_edges``,
    ``preserve_boundaries``, ``pos_iters``, ``attempts``, ``project``, and the
    fairing knobs ``regularize_iters`` / ``regularize_step`` / ``w_square`` /
    ``w_even`` / ``w_laplace`` / ``size_lock`` / ``rest_smooth`` / ``rest_rho``
    / ``project_every`` / ``max_drift`` / ``quad_floor``.
    """
    p = dict(params or {})
    V = np.ascontiguousarray(np.asarray(V, dtype=np.float64).reshape(-1, 3))
    F = np.ascontiguousarray(np.asarray(F, dtype=np.int64).reshape(-1, 3))
    n = V.shape[0]
    if n < 4 or len(F) < 2 or F.size == 0:
        return np.zeros((0, 3)), []
    if int(F.max()) >= n or int(F.min()) < 0:
        raise ValueError("triangle indices out of range")

    N = np.asarray(_sol_get(sol, "N"), dtype=np.float64).reshape(n, 3)
    Q = np.asarray(_sol_get(sol, "Q"), dtype=np.float64).reshape(n, 3)
    rho = np.asarray(_sol_get(sol, "rho"), dtype=np.float64).reshape(n)
    N = normalize(N)
    Q = Q - N * _dot(Q, N)[:, None]
    ql = np.sqrt(np.einsum("ij,ij->i", Q, Q))
    bad = ql < 1e-9
    if bad.any():
        Q[bad] = _any_tangent(N[bad])
    Q = normalize(Q)

    # ---- make the input resolvable by the lattice -------------------------
    # Built on the *original* triangles: midpoint refinement does not move the
    # surface, so the projector stays valid (and cheap).
    V0, F0 = V, F
    projector = _Projector(V0, F0) if p.get("project", True) else None

    sharp = p.get("sharp_edges")
    if sharp is not None and len(sharp):
        sharp = np.asarray(sharp, dtype=np.int64).reshape(-1, 2)
        sharp = sharp[(sharp[:, 0] != sharp[:, 1]) & (sharp.min(axis=1) >= 0)
                      & (sharp.max(axis=1) < n)]
        sharp = np.sort(sharp, axis=1)
    else:
        sharp = None

    _t = _time.time() if _DEBUG else 0.0

    def _lap(tag):
        nonlocal _t
        if _DEBUG:
            now = _time.time()
            print("   [time] %-12s %.2fs" % (tag, now - _t))
            _t = now

    _lap("projector")
    if p.get("shell_clamp", True):
        rho = _clamp_rho_per_shell(
            V, F, rho, _build_edges(F),
            min_quads=float(p.get("min_shell_quads", MIN_SHELL_QUADS)),
            target=int(p.get("target_faces", 0) or 0),
            floor_share=float(p.get("shell_floor_share", SHELL_FLOOR_SHARE)))
    # ---- opening rings: pin a loop on the first offset contour -----------
    # The orientation field can only *ask* for concentric loops around an eye
    # socket; whether the lattice actually closes one is decided here.  When
    # fields.solve_fields ran the ring machinery it leaves the geodesic
    # distance to the rims on the solution, and its iso-contour a quad and a
    # half out is where the first lattice line after the rim wants to be - so
    # it is handed over as a feature curve and the loop exists by
    # construction rather than by hoping the field alignment survives.  ``ring_dist`` is None on every solve that did not ask for
    # rings, so this costs nothing (and imports nothing) by default.
    ring_dist = _sol_get(sol, "ring_dist", None)
    if ring_dist is not None and p.get("ring_pin", True):
        try:
            from . import rings as _rings
            rseg = _rings.ring_pin_segments(V, F, ring_dist, rho, params=p)
        except Exception:
            rseg = None
        if rseg is not None and len(rseg):
            sharp = (rseg if sharp is None or not len(sharp)
                     else np.unique(np.sort(np.concatenate([sharp, rseg],
                                                           axis=0), axis=1),
                                    axis=0))

    if p.get("refine", True):
        # allow one retry step of rho headroom before the lattice gets finer
        V, F, N, Q, rho, sharp = _refine_for_lattice(
            V, F, N, Q, rho * 0.8, sharp,
            max_verts=int(p.get("max_refine_verts", MAX_REFINE_VERTS)))
        rho = rho / 0.8
        n = V.shape[0]
    _lap("refine")

    edges = _build_edges(F)
    if len(edges) == 0:
        return np.zeros((0, 3)), []
    indptr, indices, _src = _build_csr(edges, n)

    # ---- constraints -----------------------------------------------------
    be = _boundary_edges(F)
    bnd_verts = np.zeros(n, dtype=bool)
    if len(be):
        bnd_verts[be.ravel()] = True
    if sharp is not None and len(sharp):
        # Dihedral detection on a sculpt is noisy: the Dinasty head yields 60
        # feature components, half of them 1-2 segment stubs.  Pinning a stub
        # anchors an output vertex to a meaningless "corner" and blocks the
        # chain around it from fairing, so short fragments are dropped.  Real
        # mesh boundaries are never filtered.
        sharp = _filter_feature_fragments(
            V, sharp, float(np.mean(rho)),
            min_segs=int(p.get("feature_min_segs", FEATURE_MIN_SEGS)),
            min_len=float(p.get("feature_min_len", FEATURE_MIN_LEN)))
        # bridge only AFTER the noise stubs are gone - joining stubs just
        # manufactures more spurious chains and corners
        if len(sharp) and p.get("feature_bridge", FEATURE_BRIDGE):
            sharp = _bridge_feature_gaps(
                V, sharp, indptr, indices, rho,
                gap=float(p.get("feature_bridge_gap", FEATURE_BRIDGE_GAP)))
    pin_list = []
    if sharp is not None and len(sharp):
        pin_list.append(sharp)
    if len(be) and p.get("preserve_boundaries", True):
        pin_list.append(be)
    pin_mask = np.zeros(n, dtype=bool)
    pin_dir = np.zeros((n, 3))
    pin_corner = np.zeros(n, dtype=bool)
    feat_segs = np.zeros((0, 2), dtype=np.int64)
    if pin_list:
        se = np.concatenate(pin_list, axis=0)
        se = se[(se[:, 0] != se[:, 1]) & (se[:, 0] >= 0) & (se[:, 1] >= 0)
                & (se[:, 0] < n) & (se[:, 1] < n)]
        se = np.unique(np.sort(se, axis=1), axis=0)
        if len(se):
            feat_segs = se
            # Slide direction = the true feature TANGENT.  It must not be the
            # 4-RoSy class representative used for the orientation field: that
            # representative is just as happy pointing across the crease, and
            # sliding a pinned sample along the perpendicular walks it clean
            # off the feature (measured: up to 1.15 rho on a cube edge).
            d = normalize(V[se[:, 1]] - V[se[:, 0]])
            vi = np.concatenate([se[:, 0], se[:, 1]])
            vd = np.concatenate([d, -d])
            fdeg = np.bincount(vi, minlength=n)
            pin_mask = fdeg > 0
            # sign-match every incident segment against a per-vertex reference
            ref = np.zeros((n, 3))
            ref[vi] = vd
            sgn = np.where(_dot(vd, ref[vi]) < 0.0, -1.0, 1.0)
            acc = np.zeros((n, 3))
            for c in range(3):
                acc[:, c] = np.bincount(vi, weights=vd[:, c] * sgn,
                                        minlength=n)
            al = np.sqrt(np.einsum("ij,ij->i", acc, acc))
            good = pin_mask & (al > 1e-7)
            pin_dir[good] = acc[good] / al[good][:, None]
            # A junction (feature valence != 2) or a genuine corner cannot
            # slide at all: zero direction means "hold exactly at P".
            pin_corner = _feature_corners(
                V, se, n, fdeg,
                corner_deg=float(p.get("feature_corner_deg",
                                       FEATURE_CORNER_DEG)),
                span=int(p.get("feature_corner_span", FEATURE_CORNER_SPAN)))
            pin_corner &= pin_mask
            pin_dir[pin_corner] = 0.0


    feat_proj = None
    corner_pts = None
    if len(feat_segs):
        # snap target = the FAIRED feature curve, not the raw polyline
        FV, FS, _fi = (_fair_feature_curves(
            V, feat_segs, rho, pin_corner,
            iters=int(p.get("feature_fair_iters", FEATURE_FAIR_ITERS)),
            lam=float(p.get("feature_fair_lambda", FEATURE_FAIR_LAMBDA)),
            drift=float(p.get("feature_fair_drift", FEATURE_FAIR_DRIFT)),
            resample=float(p.get("feature_resample", FEATURE_RESAMPLE)))
            if p.get("feature_fair", True) else (None, None, {}))
        if FV is not None:
            feat_proj = _Projector(FV, FS)
        else:
            feat_proj = _Projector(V, feat_segs)
        if np.any(pin_corner):
            corner_pts = V[pin_corner]
    sym = p.get("symmetry") or ()
    ext = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    sym_axes = [(ax, 1e-5 * max(ext, 1e-12))
                for ax in range(min(3, len(sym))) if sym[ax]] or None

    # ---- hierarchy + position field -------------------------------------
    rng = np.random.default_rng(int(p.get("seed", 0) or 0) & 0x7FFFFFFF)
    levels = _build_pos_hierarchy(V, N, Q, rho, indptr, indices,
                                  pin_mask, pin_dir, rng)
    _lap("hierarchy")
    pos_iters = int(p.get("pos_iters", 20) or 20)
    target = int(p.get("target_faces", 0) or 0)
    in_area = float(_tri_areas(V0, F0).sum())

    tol = float(p.get("count_tol", COUNT_TOL))
    attempts = max(1, int(p.get("attempts", COUNT_ATTEMPTS) or COUNT_ATTEMPTS))
    tries = []          # (scale, faces, quad fraction, coverage, state)
    samples = []        # (log scale, log faces), for the secant
    tried = []          # scales already spent
    scale = 1.0
    for _a in range(attempts):
        O = _solve_positions(levels, scale, pos_iters)
        _lap("positions")
        st = {}
        VQ, FQ = _extract_core(O, Q, N, rho * scale, edges,
                               bnd_verts=bnd_verts, projector=projector,
                               pin_verts=pin_mask, reg=p,
                               quad_floor=float(p.get("quad_floor",
                                                      QUAD_FLOOR)),
                               feat_proj=feat_proj, corner_pts=corner_pts,
                               sym_axes=sym_axes,
                               feat_capture=float(p.get("feature_capture",
                                                        FEATURE_CAPTURE)),
                               polish=False, state=st)
        _lap("extract_core")
        tried.append(scale)
        nf = len(FQ)
        nq = sum(1 for f in FQ if len(f) == 4)
        cover = (_poly_area(VQ, FQ) / in_area) if (nf and in_area > 0) else 0.0
        if nf and st.get("faces"):
            tries.append((scale, nf, nq / float(nf), cover, st))
        if _DEBUG:
            print("   [attempt] scale=%.4f faces=%d quad%%=%.1f cover=%.1f%% "
                  "err=%+.1f%%" % (scale, nf, 100.0 * nq / max(nf, 1),
                                   100.0 * cover,
                                   100.0 * (nf / float(target) - 1.0)
                                   if target > 0 else 0.0))
        if target <= 0:
            break
        if nf == 0:
            scale *= 0.6
            continue
        err = float(np.log(nf / float(target)))
        samples.append((float(np.log(scale)), float(np.log(nf))))
        if abs(err) <= np.log1p(tol):
            break
        # Secant on log(count) vs log(scale): the previous two samples measure
        # the local exponent, so the step self-corrects instead of trusting
        # the ideal-lattice value.  Before there are two of them, start from
        # the exponent real sculpts show.
        e = float(np.clip(
            COUNT_E0 + COUNT_E_YIELD * max(0.0, 1.0 - nf / float(target)),
            COUNT_E_MIN, COUNT_E_MAX))
        if len(samples) >= 2:
            (s1, f1), (s2, f2) = samples[-2], samples[-1]
            if abs(s2 - s1) > 1e-6:
                ee = -(f2 - f1) / (s2 - s1)
                if not np.isfinite(ee):
                    ee = e
                if ee < COUNT_E_DEAD:
                    # the count barely answered a real change of scale: this
                    # input is saturated (a shell too small to carry another
                    # quad ring at any budget), and further attempts are pure
                    # latency for a count that cannot move
                    break
                e = float(np.clip(ee, COUNT_E_MIN, COUNT_E_MAX))
        nxt = scale * float(np.clip(np.exp(err / e), 0.55, 1.8))
        # a step that lands on a scale already spent buys nothing
        if any(abs(nxt / s - 1.0) < 5e-3 for s in tried):
            break
        scale = nxt

    if not tries:
        return np.zeros((0, 3)), []

    # Coverage is a veto, not a currency: an attempt that dropped surface is
    # ineligible however well its face count reads, but among attempts that
    # all cover the input, the count decides.
    cov_ref = min(1.0, max(t[3] for t in tries))
    gate = min(cov_ref, max(0.95, cov_ref - COVER_SLACK)) - 1e-9
    feasible = [t for t in tries if min(1.0, t[3]) >= gate] or tries
    if target > 0:
        pick = min(feasible,
                   key=lambda t: (abs(np.log(t[1] / float(target))), -t[2]))
    else:
        pick = max(feasible, key=lambda t: (min(1.0, t[3]), t[2]))
    if _DEBUG:
        print("   [pick] scale=%.4f faces=%d of %d attempt(s)"
              % (pick[0], pick[1], len(tries)))
    st = pick[4]
    P = _polish_positions(
        st["P"], st["faces"], st["crho_full"], st["cbnd_full"],
        st["cpin_full"], st["is_new"], projector, reg=p, feat_proj=feat_proj,
        corner_pts=corner_pts, sym_axes=sym_axes,
        feat_capture=float(p.get("feature_capture", FEATURE_CAPTURE)))
    _lap("polish")
    return _compact(P, st["faces"])
