"""Orientation (4-RoSy) and position field computation - pure numpy.

Implements the field-smoothing half of the Instant Meshes pipeline
(Jakob et al., "Instant Field-Aligned Meshes", SIGGRAPH Asia 2015):

* an extrinsically-smoothed 4-RoSy orientation field,
* a lattice-compatible position field living on the tangent planes,
* a vertex-clustering multiresolution hierarchy so that both fields
  converge in a handful of iterations per level.

v2 adds the *curvature-aligned* half of the story (see NATIVE_V2.md):

* a per-vertex shape-operator fit over the 1-ring (2-ring where the valence
  is low) giving principal curvature directions and an anisotropy measure,
* a soft alignment term in the 4-RoSy energy weighted by
  ``curvature_align * smoothstep(anisotropy)``, so quad loops follow the
  natural flow of the surface (around eyes, along muscles, around a torus
  tube) instead of an arbitrary smooth field,
* per-vertex target edge lengths (``rho``) from the face budget, the painted
  density attribute and curvature adaptivity,
* the ``solve_fields(V, F, params) -> FieldSolution`` entry point.

Everything is vectorised over *directed edges* and accumulated with
``np.bincount``; the only Python-level loops are the greedy matching used
to build the hierarchy (O(|E|) with tiny constant) and the loop over
hierarchy levels.

No scipy, no bpy - this module is importable from plain CPython which is
what the standalone self-tests use.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field as _dc_field

import numpy as np

EPS = 1e-12

__all__ = [
    # --- v1 (still used by solver.py / extract.py) ---
    "normalize",
    "face_normals_areas",
    "vertex_normals",
    "build_edges",
    "build_csr",
    "boundary_edges",
    "random_tangents",
    "build_constraints",
    "build_hierarchy",
    "smooth_orientations",
    "smooth_positions",
    "prolong_orientations",
    "prolong_positions",
    "round_to_cell",
    # --- v2 ---
    "FieldSolution",
    "CurvatureField",
    "solve_fields",
    "principal_curvatures",
    "tangent_basis",
    "ring_pairs",
    "smooth_cross_field",
    "smooth_curvature",
    "alignment_weight",
    "target_edge_lengths",
    "smoothstep",
    "rosy4_representative",
    "rosy4_angle",
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def normalize(A, eps=EPS):
    """Row-wise normalisation of an (n, 3) array (safe against zeros)."""
    n = np.sqrt(np.einsum("ij,ij->i", A, A))[:, None]
    return A / np.maximum(n, eps)


def _dot(A, B):
    return np.einsum("ij,ij->i", A, B)


def face_normals_areas(V, F):
    """Un-normalised face normals (=2*area*n) and face areas."""
    a = V[F[:, 1]] - V[F[:, 0]]
    b = V[F[:, 2]] - V[F[:, 0]]
    cr = np.cross(a, b)
    ar = 0.5 * np.sqrt(np.einsum("ij,ij->i", cr, cr))
    return cr, ar


def vertex_areas(V, F, n=None, areas=None):
    """Lumped (barycentric) vertex areas - a third of every incident triangle.

    ``sum(vertex_areas(...))`` is the surface area, so ``sum(A_v / rho_v**2)``
    is the number of ``rho``-sized quads the surface can carry.  That sum is
    the only honest way to read a face budget off a *non-uniform* ``rho``.
    """
    if n is None:
        n = V.shape[0]
    if areas is None:
        _, areas = face_normals_areas(V, F)
    w = np.zeros(int(n), dtype=np.float64)
    third = np.asarray(areas, dtype=np.float64) / 3.0
    for k in range(3):
        w += np.bincount(F[:, k], weights=third, minlength=int(n))
    return w


def budget_scale(rho, w, target):
    """Uniform factor that puts ``rho``'s predicted quad count on ``target``.

    A quad of side ``rho_v`` covers ``rho_v**2`` of surface, so the field
    carries ``pred = sum(w_v / rho_v**2)`` cells and scaling the whole field
    by ``s`` scales that by ``1/s**2``; ``s = sqrt(pred / target)`` therefore
    lands it on ``target`` while leaving every *ratio* inside ``rho``
    untouched - the distribution of detail is preserved, only the total moves.

    Returns 1.0 for a degenerate input (the caller then keeps ``rho`` as is).
    """
    rho = np.asarray(rho, dtype=np.float64)
    pred = float(np.sum(np.asarray(w, dtype=np.float64)
                        / np.maximum(rho, EPS) ** 2))
    tgt = float(target)
    if not np.isfinite(pred) or pred <= 0.0 or not np.isfinite(tgt) or tgt <= 0.0:
        return 1.0
    return float(np.sqrt(pred / tgt))


def vertex_normals(V, F):
    """Area-weighted vertex normals."""
    n = V.shape[0]
    cr, _ = face_normals_areas(V, F)
    N = np.zeros((n, 3), dtype=np.float64)
    for k in range(3):
        idx = F[:, k]
        for d in range(3):
            N[:, d] += np.bincount(idx, weights=cr[:, d], minlength=n)
    ln = np.sqrt(np.einsum("ij,ij->i", N, N))
    bad = ln < 1e-14
    if bad.any():
        N[bad] = (0.0, 0.0, 1.0)
    return normalize(N)


def build_edges(F):
    """Unique undirected edges of a triangle soup, as (E, 2) int64, i<j."""
    e = np.concatenate(
        [F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0
    ).astype(np.int64)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    e = e[e[:, 0] != e[:, 1]]
    return e


def build_csr(edges, n):
    """CSR adjacency (both directions) from undirected edge list.

    Returns ``(indptr, indices, src)`` where ``src`` is the expanded source
    index array (``np.repeat(arange(n), degree)``) that pairs with
    ``indices``.
    """
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    src = src[order]
    dst = dst[order]
    deg = np.bincount(src, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    return indptr, dst, src


def boundary_edges(F):
    """Edges incident to exactly one triangle."""
    e = np.concatenate(
        [F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0
    ).astype(np.int64)
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1]


def random_tangents(N, rng):
    """A random unit vector inside each tangent plane."""
    n = N.shape[0]
    R = rng.normal(size=(n, 3))
    Q = R - N * _dot(R, N)[:, None]
    ln = np.sqrt(np.einsum("ij,ij->i", Q, Q))
    bad = ln < 1e-6
    if bad.any():
        # fall back to a deterministic perpendicular
        alt = np.zeros((int(bad.sum()), 3))
        nb = N[bad]
        pick = np.abs(nb[:, 0]) < 0.9
        alt[pick] = (1.0, 0.0, 0.0)
        alt[~pick] = (0.0, 1.0, 0.0)
        Q[bad] = alt - nb * _dot(alt, nb)[:, None]
    return normalize(Q)


# --------------------------------------------------------------------------
# constraints (sharp edges + guide directions)
# --------------------------------------------------------------------------

def build_constraints(V, N, n, sharp_edges=None, guide_vert_dirs=None):
    """Build per-vertex hard orientation constraints.

    ``sharp_edges``      (k, 2) int  - crease / boundary / guide-path edges.
    ``guide_vert_dirs``  (n, 3) float - per-vertex guide direction, 0 = none.

    Directions are projected into the tangent plane.  A vertex touched by
    several sharp edges gets the 4-RoSy average of them (perpendicular
    creases are compatible in a 4-RoSy field, so a box corner resolves
    cleanly).  Sharp edges win over guides.

    Returns ``(mask (n,) bool, dirs (n, 3) float)``.
    """
    mask = np.zeros(n, dtype=bool)
    dirs = np.zeros((n, 3), dtype=np.float64)

    if guide_vert_dirs is not None:
        g = np.asarray(guide_vert_dirs, dtype=np.float64).reshape(n, 3)
        gl = np.sqrt(np.einsum("ij,ij->i", g, g))
        gm = gl > 1e-9
        if gm.any():
            gp = g[gm] - N[gm] * _dot(g[gm], N[gm])[:, None]
            gpl = np.sqrt(np.einsum("ij,ij->i", gp, gp))
            ok = gpl > 1e-7
            sel = np.nonzero(gm)[0][ok]
            dirs[sel] = gp[ok] / gpl[ok][:, None]
            mask[sel] = True

    if sharp_edges is not None and len(sharp_edges):
        se = np.asarray(sharp_edges, dtype=np.int64).reshape(-1, 2)
        se = se[(se[:, 0] != se[:, 1])]
        se = se[(se[:, 0] >= 0) & (se[:, 1] >= 0) & (se[:, 0] < n) & (se[:, 1] < n)]
        if len(se):
            d = V[se[:, 1]] - V[se[:, 0]]
            d = normalize(d)
            # both endpoints, both orientations
            vi = np.concatenate([se[:, 0], se[:, 1]])
            vd = np.concatenate([d, d])
            # project into the tangent plane of the receiving vertex
            nv = N[vi]
            vd = vd - nv * _dot(vd, nv)[:, None]
            vl = np.sqrt(np.einsum("ij,ij->i", vd, vd))
            keep = vl > 1e-7
            vi = vi[keep]
            vd = vd[keep] / vl[keep][:, None]

            # seed: last write wins -> deterministic reference per vertex
            ref = np.zeros((n, 3))
            refm = np.zeros(n, dtype=bool)
            ref[vi] = vd
            refm[vi] = True

            # one 4-RoSy accumulation pass against that reference
            nv = N[vi]
            perp = np.cross(nv, vd)
            r = ref[vi]
            d0 = _dot(vd, r)
            d1 = _dot(perp, r)
            use1 = np.abs(d1) > np.abs(d0)
            rep = np.where(use1[:, None], perp, vd)
            sg = np.where(use1, d1, d0)
            rep = rep * np.where(sg < 0.0, -1.0, 1.0)[:, None]
            acc = np.zeros((n, 3))
            for c in range(3):
                acc[:, c] = np.bincount(vi, weights=rep[:, c], minlength=n)
            acc = acc - N * _dot(acc, N)[:, None]
            al = np.sqrt(np.einsum("ij,ij->i", acc, acc))
            good = refm & (al > 1e-7)
            dirs[good] = acc[good] / al[good][:, None]
            # degenerate accumulation -> keep the raw reference
            fallback = refm & ~good
            dirs[fallback] = ref[fallback]
            mask[refm] = True

    return mask, dirs


# --------------------------------------------------------------------------
# multiresolution hierarchy (randomised greedy vertex clustering)
# --------------------------------------------------------------------------

def _greedy_match(n, indptr, indices, rng):
    """Randomised greedy matching -> parent[i] in [0, n_coarse)."""
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


def _coarsen_cross(parent, nc, Nf, Nc, D, W):
    """Restrict a weighted 4-RoSy target field onto the next coarser level.

    The reference direction of a cluster is the child with the largest weight
    (ties broken by the lower index, so it is deterministic); the children are
    then accumulated onto it with their weights.  The coarse weight is the
    mean of the children's weights.
    """
    cnt = np.bincount(parent, minlength=nc).astype(np.float64)
    Wc = np.bincount(parent, weights=W, minlength=nc) / np.maximum(cnt, 1.0)

    # deterministic reference: highest-weight child wins (last write)
    order = np.lexsort((np.arange(len(parent))[::-1], W))
    ref = np.zeros((nc, 3))
    ref[parent[order]] = D[order]
    ref = ref - Nc * _dot(ref, Nc)[:, None]
    rl = np.sqrt(np.einsum("ij,ij->i", ref, ref))
    ref = ref / np.maximum(rl, EPS)[:, None]

    rep = rosy4_representative(D, Nf, ref[parent]) * W[:, None]
    acc = np.zeros((nc, 3))
    for c in range(3):
        acc[:, c] = np.bincount(parent, weights=rep[:, c], minlength=nc)
    acc -= Nc * _dot(acc, Nc)[:, None]
    al = np.sqrt(np.einsum("ij,ij->i", acc, acc))
    Dc = np.zeros((nc, 3))
    good = al > 1e-9
    Dc[good] = acc[good] / al[good][:, None]
    Wc = np.where(good, Wc, 0.0)
    return Dc, Wc


def build_hierarchy(P, N, indptr, indices, rho, con_mask, con_dir, rng,
                    min_verts=48, max_levels=12, pin_mask=None,
                    align_dir=None, align_w=None):
    """Vertex-clustering hierarchy.

    Each level is a dict with keys ``P N indptr indices src rho con_mask
    con_dir pin_mask parent n``.  ``parent`` maps *this* level's vertices to
    the next (coarser) level; it is ``None`` on the coarsest level.

    ``pin_mask`` is the subset of ``con_mask`` whose *position* is also
    constrained (creases and boundaries - guides only steer orientation).

    When ``align_dir`` / ``align_w`` are given (the v2 curvature alignment
    target) each level additionally carries restricted ``align_dir`` and
    ``align_w`` entries, so the soft alignment acts on every level of the
    multiresolution solve rather than only on the finest one.
    """
    if pin_mask is None:
        pin_mask = np.zeros(len(P), dtype=bool)
    has_align = align_dir is not None and align_w is not None
    levels = []
    cur = dict(P=P, N=N, indptr=indptr, indices=indices,
               src=np.repeat(np.arange(len(P), dtype=np.int64), np.diff(indptr)),
               rho=rho, con_mask=con_mask, con_dir=con_dir,
               pin_mask=pin_mask, parent=None, n=len(P),
               align_dir=align_dir if has_align else None,
               align_w=align_w if has_align else None)
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

        # coarse graph
        pi = parent[cur["src"]]
        pj = parent[cur["indices"]]
        keep = pi != pj
        pi, pj = pi[keep], pj[keep]
        if len(pi) == 0:
            break
        ce = np.sort(np.stack([pi, pj], axis=1), axis=1)
        ce = np.unique(ce, axis=0)
        cip, cidx, csrc = build_csr(ce, nc)

        # constraints propagate upward: lowest-index constrained child wins
        cmask = np.zeros(nc, dtype=bool)
        cpin = np.zeros(nc, dtype=bool)
        cdir = np.zeros((nc, 3))
        ci = np.nonzero(cur["con_mask"])[0]
        if len(ci):
            rev = ci[::-1]
            cdir[parent[rev]] = cur["con_dir"][rev]
            cpin[parent[rev]] = cur["pin_mask"][rev]
            cmask[parent[ci]] = True
            cd = cdir - cN * _dot(cdir, cN)[:, None]
            cl = np.sqrt(np.einsum("ij,ij->i", cd, cd))
            ok = cmask & (cl > 1e-7)
            cdir[ok] = cd[ok] / cl[ok][:, None]
            cmask = ok
            cpin &= cmask

        cadir = cawgt = None
        if has_align:
            cadir, cawgt = _coarsen_cross(parent, nc, cur["N"], cN,
                                          cur["align_dir"], cur["align_w"])

        cur = dict(P=cP, N=cN, indptr=cip, indices=cidx, src=csrc,
                   rho=crho, con_mask=cmask, con_dir=cdir, pin_mask=cpin,
                   parent=None, n=nc, align_dir=cadir, align_w=cawgt)
        levels[-1]["parent"] = parent
        levels.append(cur)

    return levels


# --------------------------------------------------------------------------
# 4-RoSy orientation smoothing
# --------------------------------------------------------------------------

def rosy4_representative(D, ND, R):
    """Representative of the 4-RoSy class of ``D`` (normal ``ND``) closest to
    the reference direction ``R``.

    Row-wise; all inputs are ``(k, 3)``.  The class of ``D`` is
    ``{+-D, +-(ND x D)}``; the returned vector is the member with the largest
    positive dot product against ``R``.  A zero ``D`` maps to zero (so it can
    be used with a zero weight without producing NaNs).
    """
    perp = np.cross(ND, D)
    d0 = _dot(D, R)
    d1 = _dot(perp, R)
    use1 = np.abs(d1) > np.abs(d0)
    rep = np.where(use1[:, None], perp, D)
    sg = np.where(use1, d1, d0)
    return rep * np.where(sg < 0.0, -1.0, 1.0)[:, None]


def rosy4_angle(A, B):
    """Angle (radians, in ``[0, pi/4]``) between the 4-RoSy classes of two
    row-wise unit direction arrays that share a tangent plane.

    A 4-RoSy class is invariant under 90-degree rotations, so the metric
    folds twice: once for the sign, once at 45 degrees.
    """
    c = np.clip(np.abs(_dot(A, B)), 0.0, 1.0)   # fold the sign
    ang = np.arccos(c)                          # 0 .. pi/2
    return np.minimum(ang, 0.5 * np.pi - ang)   # fold at 45 deg -> 0 .. pi/4


def smooth_orientations(Q, N, src, dst, con_mask, con_dir, iters,
                        self_weight=1.0, align_dir=None, align_w=None):
    """Vectorised (Jacobi) 4-RoSy smoothing with extrinsic matching.

    For every directed edge (i -> j) we pick the representative of the
    4-RoSy class of ``Q[j]`` - one of ``+-Q[j]``, ``+-(N[j] x Q[j])`` -
    that is closest to the current ``Q[i]``, then average.

    ``align_dir`` / ``align_w`` add the v2 *soft* curvature alignment term:
    after the neighbour average the result is blended towards the 4-RoSy
    representative of ``align_dir`` with per-vertex weight ``align_w`` in
    ``[0, 1]`` (0 = pure smoothing, 1 = snap to the principal direction).
    Hard constraints still win over both.
    """
    n = Q.shape[0]
    Q = np.array(Q, dtype=np.float64, copy=True)
    has_con = bool(con_mask.any())
    if has_con:
        Q[con_mask] = con_dir[con_mask]
    deg = np.bincount(src, minlength=n).astype(np.float64)
    denom = deg + self_weight

    has_align = (align_dir is not None and align_w is not None
                 and float(np.max(align_w)) > 1e-9)
    if has_align:
        aw = np.asarray(align_w, dtype=np.float64).reshape(n)[:, None]
        ad = np.asarray(align_dir, dtype=np.float64).reshape(n, 3)

    for _ in range(iters):
        qi = Q[src]
        qj = Q[dst]
        perp = np.cross(N[dst], qj)
        d0 = _dot(qj, qi)
        d1 = _dot(perp, qi)
        use1 = np.abs(d1) > np.abs(d0)
        rep = np.where(use1[:, None], perp, qj)
        sg = np.where(use1, d1, d0)
        rep = rep * np.where(sg < 0.0, -1.0, 1.0)[:, None]

        acc = np.empty((n, 3))
        for d in range(3):
            acc[:, d] = np.bincount(src, weights=rep[:, d], minlength=n)
        acc += Q * self_weight
        acc /= denom[:, None]
        acc -= N * _dot(acc, N)[:, None]
        ln = np.sqrt(np.einsum("ij,ij->i", acc, acc))
        good = ln > 1e-9
        Qn = Q.copy()
        Qn[good] = acc[good] / ln[good][:, None]

        if has_align:
            # blend towards the principal direction inside the 4-RoSy class
            arep = rosy4_representative(ad, N, Qn)
            mix = (1.0 - aw) * Qn + aw * arep
            mix -= N * _dot(mix, N)[:, None]
            ml = np.sqrt(np.einsum("ij,ij->i", mix, mix))
            ok = ml > 1e-9
            Qn[ok] = mix[ok] / ml[ok][:, None]

        if has_con:
            Qn[con_mask] = con_dir[con_mask]
        Q = Qn
    return Q


def prolong_orientations(Q_coarse, parent, N_fine):
    """Copy the coarse field down and re-project into the fine tangent planes."""
    Q = Q_coarse[parent]
    Q = Q - N_fine * _dot(Q, N_fine)[:, None]
    ln = np.sqrt(np.einsum("ij,ij->i", Q, Q))
    bad = ln < 1e-7
    if bad.any():
        # degenerate: pick any tangent
        alt = np.zeros((int(bad.sum()), 3))
        nb = N_fine[bad]
        pick = np.abs(nb[:, 0]) < 0.9
        alt[pick] = (1.0, 0.0, 0.0)
        alt[~pick] = (0.0, 1.0, 0.0)
        Q[bad] = alt - nb * _dot(alt, nb)[:, None]
    return normalize(Q)


# --------------------------------------------------------------------------
# position field
# --------------------------------------------------------------------------

def round_to_cell(O, Q, N, P, rho):
    """Move ``O`` by integer multiples of ``rho`` along (Q, N x Q) so that it
    lands in the lattice cell containing ``P`` (Instant Meshes
    ``position_round_4``)."""
    T = np.cross(N, Q)
    d = P - O
    a = np.round(_dot(Q, d) / rho)
    b = np.round(_dot(T, d) / rho)
    return O + Q * (a * rho)[:, None] + T * (b * rho)[:, None]


def smooth_positions(O, P, Q, N, rho, src, dst, iters, self_weight=1.0,
                     con_mask=None, con_dir=None):
    """Vectorised lattice-compatible position smoothing.

    For a directed edge (i -> j) the neighbour position ``O[j]`` is
    translated by integer multiples of ``rho`` along ``Q[j]`` and
    ``N[j] x Q[j]`` so that it becomes the representative closest to
    ``O[i]``.  The result is averaged, re-projected onto the tangent plane
    of ``i`` and snapped back into the cell containing ``P[i]``.

    When ``con_mask`` is given, constrained vertices (creases, boundaries,
    guides) may only slide *along* their constraint direction, which pins a
    lattice line onto every crease so the extracted quad mesh reproduces it.
    """
    n = O.shape[0]
    has_con = con_mask is not None and bool(np.any(con_mask))
    O = np.array(O, dtype=np.float64, copy=True)
    deg = np.bincount(src, minlength=n).astype(np.float64)
    denom = deg + self_weight
    rho_e = 0.5 * (rho[src] + rho[dst])
    inv_e = 1.0 / rho_e
    Tj = np.cross(N[dst], Q[dst])
    Qj = Q[dst]

    for _ in range(iters):
        d = O[src] - O[dst]
        a = np.round(_dot(Qj, d) * inv_e)
        b = np.round(_dot(Tj, d) * inv_e)
        rep = O[dst] + Qj * (a * rho_e)[:, None] + Tj * (b * rho_e)[:, None]

        acc = np.empty((n, 3))
        for c in range(3):
            acc[:, c] = np.bincount(src, weights=rep[:, c], minlength=n)
        acc += O * self_weight
        acc /= denom[:, None]
        # keep it on the tangent plane through P
        acc -= N * _dot(acc - P, N)[:, None]
        # and inside the cell of P
        O = round_to_cell(acc, Q, N, P, rho)
        if has_con:
            cm = con_mask
            cd = con_dir[cm]
            rel = O[cm] - P[cm]
            O[cm] = P[cm] + cd * _dot(rel, cd)[:, None]
    return O


def prolong_positions(O_coarse, parent, P_fine, Q_fine, N_fine, rho_fine):
    O = O_coarse[parent]
    O = O - N_fine * _dot(O - P_fine, N_fine)[:, None]
    return round_to_cell(O, Q_fine, N_fine, P_fine, rho_fine)


# ==========================================================================
#  v2: principal-curvature-aligned 4-RoSy field
# ==========================================================================

def smoothstep(x, e0=0.0, e1=1.0):
    """Hermite smoothstep clamped to ``[0, 1]``."""
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / max(e1 - e0, EPS),
                0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def tangent_basis(N):
    """Deterministic orthonormal tangent frame ``(e1, e2)`` for unit normals.

    Branchless construction (Duff et al. 2017, "Building an Orthonormal Basis,
    Revisited") - no random numbers, no degenerate case, and continuous away
    from the -Z pole, which keeps the curvature fit reproducible.
    """
    nx, ny, nz = N[:, 0], N[:, 1], N[:, 2]
    sg = np.where(nz >= 0.0, 1.0, -1.0)
    a = -1.0 / (sg + nz)            # |sg + nz| >= 1 by construction
    b = nx * ny * a
    e1 = np.stack([1.0 + sg * nx * nx * a, sg * b, -sg * nx], axis=1)
    e2 = np.stack([b, sg + ny * ny * a, -ny], axis=1)
    return normalize(e1), normalize(e2)


def _csr_neighbors(indptr, indices, verts):
    """Flattened 1-ring of ``verts``: ``(src, dst)`` index arrays."""
    deg = (indptr[1:] - indptr[:-1])[verts]
    tot = int(deg.sum())
    if tot == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z
    starts = indptr[verts]
    csum = np.cumsum(deg)
    offs = np.arange(tot, dtype=np.int64) - np.repeat(csum - deg, deg)
    return np.repeat(verts, deg), indices[np.repeat(starts, deg) + offs]


def ring_pairs(indptr, indices, n, min_valence=5):
    """Sample pairs ``(i, j)`` for the per-vertex curvature fit.

    Every vertex contributes its 1-ring; vertices with a valence below
    ``min_valence`` also contribute their 2-ring, because three or four
    samples cannot pin down the three unknowns of the shape operator
    robustly (a valence-3 corner has exactly as many equations as unknowns
    in the position-only formulation).
    """
    allv = np.arange(n, dtype=np.int64)
    s, d = _csr_neighbors(indptr, indices, allv)
    deg = indptr[1:] - indptr[:-1]
    low = np.nonzero(deg < int(min_valence))[0]
    if len(low):
        s1, d1 = _csr_neighbors(indptr, indices, low)
        if len(d1):
            cnt = deg[d1]
            _, d2 = _csr_neighbors(indptr, indices, d1)
            s2 = np.repeat(s1, cnt)
            s = np.concatenate([s, s2])
            d = np.concatenate([d, d2])
    keep = s != d
    s, d = s[keep], d[keep]
    key = s * np.int64(n) + d
    _, uniq = np.unique(key, return_index=True)
    uniq.sort()
    return s[uniq], d[uniq]


@dataclass
class CurvatureField:
    """Per-vertex principal curvature estimate.

    The shape operator is stored twice: as the principal pair
    ``(k1, D1) / (k2, D2)`` and as the equivalent *isotropic + deviatoric*
    split ``km = (k1+k2)/2`` (mean curvature, a scalar) and ``kd =
    |k1-k2|/2`` with direction ``D1`` (a spin-2 / line quantity).  The split
    is what :func:`smooth_curvature` averages, because only there does noise
    actually cancel.
    """
    k1: np.ndarray          # (n,) larger principal curvature
    k2: np.ndarray          # (n,) smaller principal curvature
    D1: np.ndarray          # (n,3) unit direction of k1 (tangent)
    D2: np.ndarray          # (n,3) unit direction of k2 = N x D1
    aniso: np.ndarray       # (n,) |k1-k2| / (|k1|+|k2|+eps)  in [0,1]
    conf: np.ndarray        # (n,) magnitude confidence in [0,1]
    km: np.ndarray = None   # (n,) mean curvature (k1+k2)/2
    kd: np.ndarray = None   # (n,) deviatoric magnitude |k1-k2|/2
    e1: np.ndarray = None   # (n,3) tangent frame used for the fit
    e2: np.ndarray = None   # (n,3)
    edge_len: np.ndarray = None   # (n,) mean incident edge length


def principal_curvatures(V, F=None, N=None, indptr=None, indices=None,
                         min_valence=5, w_pos=1.0, w_normal=1.0,
                         edges=None):
    """Least-squares shape-operator fit per vertex.

    In the tangent frame ``(e1, e2)`` of vertex *i* the shape operator is the
    symmetric matrix ``S = [[a, b], [b, c]]``.  For every ring sample *j* with
    tangent offset direction ``t = (u, v)`` (unit) and 3D distance ``L`` we
    stack two kinds of equation:

    * *position* (one row): the normal curvature of the normal section,
      ``t^T S t = -2 (p_j - p_i) . n_i / L^2`` - purely geometric, immune to
      normal-estimation error;
    * *normal variation* (two rows): ``S t = (n_j - n_i)_tangent / L`` -
      the definition of the Weingarten map, better conditioned on smooth
      meshes because it constrains the full matrix, not just a quadratic form.

    Both are accumulated into the 3x3 normal equations with ``np.bincount``
    and solved in one batched ``np.linalg.solve``.  Eigen-decomposition of the
    2x2 result is closed form.

    Returns a :class:`CurvatureField`.
    """
    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    n = V.shape[0]
    if N is None:
        N = vertex_normals(V, F)
    if indptr is None or indices is None:
        if edges is None:
            edges = build_edges(F)
        indptr, indices, _ = build_csr(edges, n)

    e1, e2 = tangent_basis(N)
    si, di = ring_pairs(indptr, indices, n, min_valence=min_valence)
    if len(si) == 0:
        z = np.zeros(n)
        return CurvatureField(z, z.copy(), e1.copy(), e2.copy(),
                              z.copy(), z.copy(), km=z.copy(), kd=z.copy(),
                              e1=e1, e2=e2, edge_len=np.ones(n))

    d = V[di] - V[si]
    L = np.sqrt(np.einsum("ij,ij->i", d, d))
    E1, E2, NI = e1[si], e2[si], N[si]
    u = _dot(d, E1)
    v = _dot(d, E2)
    tl = np.sqrt(u * u + v * v)
    ok = (L > 1e-12) & (tl > 1e-9 * np.maximum(L, 1e-12))
    if not ok.any():
        z = np.zeros(n)
        return CurvatureField(z, z.copy(), e1.copy(), e2.copy(),
                              z.copy(), z.copy(), km=z.copy(), kd=z.copy(),
                              e1=e1, e2=e2, edge_len=np.ones(n))
    si, di = si[ok], di[ok]
    d, L, tl = d[ok], L[ok], tl[ok]
    NI = NI[ok]
    cu = u[ok] / tl
    cv = v[ok] / tl

    inv_l = 1.0 / L
    # --- position rows: [cu^2, 2 cu cv, cv^2] . (a,b,c) = kn ---------------
    g0 = cu * cu
    g1 = 2.0 * cu * cv
    g2 = cv * cv
    kn = -2.0 * _dot(d, NI) * inv_l * inv_l   # sign: matches dN/dx convention

    # --- normal-variation rows: S t = dn_t / L ----------------------------
    dn = N[di] - N[si]
    y1 = _dot(dn, e1[si]) * inv_l
    y2 = _dot(dn, e2[si]) * inv_l

    wp, wn = float(w_pos), float(w_normal)
    cucv = cu * cv
    ent = [
        wp * g0 * g0 + wn * (cu * cu),                      # M00
        wp * g0 * g1 + wn * cucv,                           # M01
        wp * g0 * g2,                                       # M02
        wp * g1 * g1 + wn * 1.0,                            # M11 (cu^2+cv^2=1)
        wp * g1 * g2 + wn * cucv,                           # M12
        wp * g2 * g2 + wn * (cv * cv),                      # M22
        wp * g0 * kn + wn * (cu * y1),                       # r0
        wp * g1 * kn + wn * (cv * y1 + cu * y2),             # r1
        wp * g2 * kn + wn * (cv * y2),                       # r2
    ]
    acc = np.empty((9, n))
    for k in range(9):
        acc[k] = np.bincount(si, weights=ent[k], minlength=n)

    M = np.empty((n, 3, 3))
    M[:, 0, 0] = acc[0]
    M[:, 0, 1] = M[:, 1, 0] = acc[1]
    M[:, 0, 2] = M[:, 2, 0] = acc[2]
    M[:, 1, 1] = acc[3]
    M[:, 1, 2] = M[:, 2, 1] = acc[4]
    M[:, 2, 2] = acc[5]
    rhs = np.stack([acc[6], acc[7], acc[8]], axis=1)

    # Tikhonov ridge: keeps the batched solve non-singular on isolated or
    # collinear-ring vertices without biasing well-conditioned fits.
    tr = np.maximum(M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2], EPS)
    ridge = (1e-9 * tr + 1e-30)[:, None, None] * np.eye(3)[None]
    # numpy >= 2 requires an explicit column for batched right-hand sides
    X = np.linalg.solve(M + ridge, rhs[:, :, None])[:, :, 0]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    a, b, c = X[:, 0], X[:, 1], X[:, 2]
    km = 0.5 * (a + c)                                    # isotropic part
    kd = np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0.0))   # deviatoric
    theta = 0.5 * np.arctan2(2.0 * b, a - c)
    D1 = e1 * np.cos(theta)[:, None] + e2 * np.sin(theta)[:, None]

    el = np.bincount(si, weights=L, minlength=n) / np.maximum(
        np.bincount(si, minlength=n), 1).astype(np.float64)
    el[el <= 0.0] = float(np.mean(L))
    return _finish_curvature(km, kd, D1, N, e1, e2, el)


def _finish_curvature(km, kd, D1, N, e1, e2, edge_len):
    """(mean, deviatoric, direction) -> a complete :class:`CurvatureField`."""
    D1 = D1 - N * _dot(D1, N)[:, None]
    dl = np.sqrt(np.einsum("ij,ij->i", D1, D1))
    bad = dl < 1e-9
    if bad.any():
        D1 = D1.copy()
        D1[bad] = e1[bad]
    D1 = normalize(D1)
    D2 = normalize(np.cross(N, D1))
    k1 = km + kd
    k2 = km - kd

    # |k1-k2| / (|k1|+|k2|) reduces exactly to kd / max(|km|, kd)
    denom = np.maximum(np.abs(km), kd)
    ref = float(np.percentile(denom, 75.0)) if len(denom) else 0.0
    aniso = np.clip(kd / (denom + max(ref * 1e-3, EPS)), 0.0, 1.0)
    aniso = np.nan_to_num(aniso)

    # magnitude confidence: curvature * local edge length is the turning angle
    # per edge; below ~0.5 degrees per edge the direction is numerical noise.
    turn = np.maximum(np.abs(k1), np.abs(k2)) * edge_len
    conf = smoothstep(turn, 0.008, 0.04)
    return CurvatureField(k1, k2, D1, D2, aniso, conf,
                          km=km, kd=kd, e1=e1, e2=e2, edge_len=edge_len)


def smooth_curvature(cur, N, src, dst, iters=4, self_weight=1.0):
    """Diffuse the *shape operator* (not just its direction) over the 1-ring.

    A symmetric 2x2 tensor splits into a scalar mean curvature ``km`` and a
    spin-2 deviatoric part of magnitude ``kd`` pointing along ``D1``.  Under a
    rotation by ``phi`` the deviatoric part rotates by ``2*phi``, so averaging
    it over the neighbourhood is exactly averaging the tensors - and that is
    the point: on a *noisy* surface the deviatoric parts of neighbouring
    vertices point in unrelated directions and **cancel**, which shrinks
    ``kd`` and therefore the anisotropy, so the alignment weight backs off on
    its own.  Smoothing unit directions instead (the obvious thing to do)
    renormalises noise back up to full strength and actively harms the field.

    ``iters`` Jacobi sweeps diffuse over a radius of roughly
    ``sqrt(iters) * edge_length``; :func:`solve_fields` picks it from the
    target edge length so the field only chases detail the quads can carry.
    """
    if iters <= 0:
        return cur
    n = len(N)
    km = np.array(cur.km, dtype=np.float64, copy=True)
    kd = np.array(cur.kd, dtype=np.float64, copy=True)
    D1 = np.array(cur.D1, dtype=np.float64, copy=True)
    e1, e2 = cur.e1, cur.e2
    deg = np.bincount(src, minlength=n).astype(np.float64)
    denom = deg + float(self_weight)

    for _ in range(int(iters)):
        # express the neighbour's principal direction in the frame of `src`
        p = D1[dst] - N[src] * _dot(D1[dst], N[src])[:, None]
        cc = _dot(p, e1[src])
        ss = _dot(p, e2[src])
        nn = cc * cc + ss * ss
        w = kd[dst] / np.maximum(nn, EPS)
        zx = w * (cc * cc - ss * ss)          # kd * cos(2 theta)
        zy = w * (2.0 * cc * ss)              # kd * sin(2 theta)

        Zx = np.bincount(src, weights=zx, minlength=n)
        Zy = np.bincount(src, weights=zy, minlength=n)
        Hs = np.bincount(src, weights=km[dst], minlength=n)

        # own contribution, in its own frame
        c0 = _dot(D1, e1)
        s0 = _dot(D1, e2)
        n0 = np.maximum(c0 * c0 + s0 * s0, EPS)
        Zx += self_weight * kd * (c0 * c0 - s0 * s0) / n0
        Zy += self_weight * kd * (2.0 * c0 * s0) / n0
        Hs += self_weight * km

        Zx /= denom
        Zy /= denom
        km = Hs / denom
        kd = np.sqrt(Zx * Zx + Zy * Zy)
        th = 0.5 * np.arctan2(Zy, Zx)
        D1 = e1 * np.cos(th)[:, None] + e2 * np.sin(th)[:, None]

    return _finish_curvature(km, kd, D1, N, e1, e2, cur.edge_len)


def smooth_cross_field(D, W, N, src, dst, iters=3):
    """Weighted 4-RoSy smoothing of a *target* cross field (directions only).

    Unlike :func:`smooth_orientations` this keeps the per-vertex weights fixed
    and only cleans up the directions.

    Note that :func:`solve_fields` deliberately does **not** use this on the
    curvature target: smoothing unit directions renormalises noise back to
    full strength.  :func:`smooth_curvature` diffuses the shape operator
    itself, where noise cancels.  This helper stays for callers that already
    have a direction field with no magnitude attached (e.g. guide fields).
    """
    n = D.shape[0]
    D = np.array(D, dtype=np.float64, copy=True)
    W = np.asarray(W, dtype=np.float64).reshape(n)
    if iters <= 0 or not np.any(W > 0.0):
        return D
    we = W[dst]
    for _ in range(iters):
        rep = rosy4_representative(D[dst], N[dst], D[src]) * we[:, None]
        acc = np.empty((n, 3))
        for c in range(3):
            acc[:, c] = np.bincount(src, weights=rep[:, c], minlength=n)
        acc += D * (W[:, None] + 0.5)      # keep a share of the own estimate
        acc -= N * _dot(acc, N)[:, None]
        ln = np.sqrt(np.einsum("ij,ij->i", acc, acc))
        good = ln > 1e-9
        Dn = D.copy()
        Dn[good] = acc[good] / ln[good][:, None]
        D = Dn
    return D


def alignment_weight(cur, curvature_align=0.7, lo=0.15, hi=0.45):
    """Per-vertex soft-alignment weight ``curvature_align * smoothstep(a)``.

    The magnitude confidence of the fit is folded in as well, which is what
    keeps a sphere (``a ~ 0`` but also no reliable direction) and flat panels
    from picking up noise directions.
    """
    w = float(np.clip(curvature_align, 0.0, 1.0)) * smoothstep(cur.aniso, lo, hi)
    return np.clip(w * cur.conf, 0.0, 1.0)


def target_edge_lengths(V, F, n, target_faces, density=None, adaptive=0.0,
                        cur=None, areas=None):
    """Per-vertex target edge length ``rho``.

    * base: ``sqrt(area / target_faces)`` - a quad of side ``rho`` covers
      ``rho^2`` of surface (v1 behaviour, unchanged),
    * ``density`` (1 = neutral, >1 = denser) divides it, exactly as v1,
    * ``adaptive`` in ``[0, 1]`` additionally shrinks ``rho`` where the
      surface curves - ``rho ~ kappa^(-adaptive/2)`` clamped to a 3x band,
    * the field is renormalised so that its *predicted cell count*
      ``sum(A_v / rho_v**2)`` is the requested one, so density and adaptivity
      only *redistribute* the face budget.

    Renormalising the *mean* of ``rho`` (what this did before) is not the same
    thing: the count is driven by ``1/rho**2``, which is convex, so spreading
    ``rho`` at a fixed mean silently buys extra faces (Jensen).  A uniform
    ``rho`` is a fixed point of both rules, so nothing changes for inputs
    without a density attribute or adaptivity.
    """
    if areas is None:
        _, areas = face_normals_areas(V, F)
    area = float(np.sum(areas))
    target = max(12.0, float(target_faces))
    rho0 = np.sqrt(max(area, EPS) / target)
    rho = np.full(n, rho0, dtype=np.float64)

    if density is not None:
        d = np.asarray(density, dtype=np.float64).ravel()
        if d.size == n:
            d = np.clip(np.nan_to_num(d, nan=1.0), 0.25, 4.0)
            rho = rho0 / d

    a = float(np.clip(adaptive, 0.0, 1.0))
    if a > 0.0 and cur is not None:
        kk = np.maximum(np.abs(cur.k1), np.abs(cur.k2))
        kk = np.nan_to_num(kk, nan=0.0, posinf=0.0, neginf=0.0)
        ref = float(np.percentile(kk, 60.0))
        if ref > EPS:
            fac = np.clip(kk / ref, 1.0 / 3.0, 3.0) ** (0.5 * a)
            rho = rho / fac

    return rho * budget_scale(rho, vertex_areas(V, F, n, areas=areas), target)


@dataclass
class FieldSolution:
    """Result of :func:`solve_fields` (NATIVE_V2.md module contract).

    ``N``, ``Q`` and ``rho`` are the contract; the remaining fields are
    diagnostics the extractor / benchmark may use and are always present.
    """
    N: np.ndarray            # (n,3) f64 unit vertex normals
    Q: np.ndarray            # (n,3) f64 unit tangent 4-RoSy representative
    rho: np.ndarray          # (n,)  f64 target edge length
    # --- extras (not part of the minimal contract) ---
    aniso: np.ndarray = None         # (n,)  anisotropy |k1-k2|/(|k1|+|k2|)
    align_w: np.ndarray = None       # (n,)  soft alignment weight actually used
    align_dir: np.ndarray = None     # (n,3) smoothed principal direction target
    k1: np.ndarray = None            # (n,)
    k2: np.ndarray = None            # (n,)
    con_mask: np.ndarray = None      # (n,)  hard-constrained vertices
    con_dir: np.ndarray = None       # (n,3)
    pin_mask: np.ndarray = None      # (n,)  position-pinned subset (creases)
    bnd_verts: np.ndarray = None     # (n,)  boundary vertices
    edges: np.ndarray = None         # (e,2) undirected edge list
    levels: list = _dc_field(default=None, repr=False)
    stats: dict = _dc_field(default_factory=dict)


FIELD_DEFAULTS = {
    "target_faces": 5000,
    "density": None,
    "adaptive": 0.0,
    "sharp_edges": None,
    "guide_dirs": None,
    "curvature_align": 0.7,
    "seed": 0,
    "orient_iters": 20,
    "curvature_smooth": "auto",   # int, or "auto" = derive from rho
    "curvature_scale": 0.6,       # smoothing radius as a fraction of rho
    "min_valence": 5,
    "preserve_boundaries": True,
    "verbose": False,
}


def _guide_dirs_to_vertices(guide_dirs, n, nf, F, N):
    """Accept every documented guide-direction shape -> per-vertex (n,3).

    * ``{vidx: (3,)}``           (NATIVE_V2.md contract)
    * ``(n, 3)`` array           per-vertex
    * ``(m, 3)`` array, m = |F|  per-face (v1 / CONTRACTS.md); a face vector
      is spread onto its three corners.
    """
    if guide_dirs is None:
        return None
    if isinstance(guide_dirs, dict):
        if not guide_dirs:
            return None
        gd = np.zeros((n, 3), dtype=np.float64)
        for vi, vec in guide_dirs.items():
            i = int(vi)
            if 0 <= i < n:
                gd[i] = np.asarray(vec, dtype=np.float64).reshape(3)
        return gd
    g = np.asarray(guide_dirs, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 3 or len(g) == 0:
        return None
    if len(g) == nf:                       # per-face (v1 semantics win a tie)
        ln = np.sqrt(np.einsum("ij,ij->i", g, g))
        live = ln > 1e-9
        if not live.any():
            return None
        gu = g[live] / ln[live][:, None]
        fi = np.nonzero(live)[0]
        gd = np.zeros((n, 3))
        for k in range(3):
            vi = F[fi, k]
            for c in range(3):
                gd[:, c] += np.bincount(vi, weights=gu[:, c], minlength=n)
        return gd
    if len(g) == n:
        return g
    return None


def solve_fields(V, F, params=None):
    """Curvature-aligned 4-RoSy orientation field + target edge lengths.

    ``params`` (all optional, see :data:`FIELD_DEFAULTS`)::

        target_faces    int        face budget, drives the base rho
        density         (n,)|None  1 = neutral, >1 = denser
        adaptive        float      0..1 curvature adaptivity of rho
                                   (0..100 is accepted and read as percent)
        sharp_edges     (e,2)i32   hard crease constraints
        guide_dirs      dict|array per-vertex or per-face guide directions
        curvature_align float      0..1 weight of the principal-dir alignment
        seed            int        deterministic
        orient_iters    int        smoothing iterations on the finest level

    Returns a :class:`FieldSolution`.  Deterministic for a given seed.
    """
    t0 = _time.time()
    p = dict(FIELD_DEFAULTS)
    for k, v in (params or {}).items():
        if v is not None or k not in p:
            p[k] = v
    for k in ("target_faces", "orient_iters", "seed", "adaptive",
              "curvature_align", "curvature_smooth", "curvature_scale",
              "min_valence"):
        if p.get(k) is None:
            p[k] = FIELD_DEFAULTS[k]

    V = np.ascontiguousarray(np.asarray(V, dtype=np.float64).reshape(-1, 3))
    F = np.ascontiguousarray(np.asarray(F, dtype=np.int64).reshape(-1, 3))
    n = V.shape[0]
    if n < 3 or len(F) < 1:
        raise ValueError("solve_fields: mesh too small")
    if F.max() >= n or F.min() < 0:
        raise ValueError("solve_fields: triangle indices out of range")

    rng = np.random.default_rng(int(p["seed"]) & 0x7FFFFFFF)

    # ---- topology ------------------------------------------------------
    N = vertex_normals(V, F)
    edges = build_edges(F)
    if len(edges) == 0:
        raise ValueError("solve_fields: mesh has no edges")
    indptr, indices, _src = build_csr(edges, n)
    _, areas = face_normals_areas(V, F)
    t_topo = _time.time()

    # ---- principal curvature -------------------------------------------
    ca = float(p["curvature_align"])
    if ca > 1.0:                       # tolerate a 0..100 percentage
        ca /= 100.0
    ca = float(np.clip(ca, 0.0, 1.0))
    cur = principal_curvatures(V, F, N=N, indptr=indptr, indices=indices,
                               min_valence=int(p["min_valence"]))

    # Curvature is only useful at the scale the output quads can represent, so
    # diffuse the shape operator over a radius of ~curvature_scale * rho.  A
    # Jacobi sweep spreads about one edge length, radius grows as sqrt(iters).
    rho0 = np.sqrt(max(float(np.sum(areas)), EPS)
                   / max(12.0, float(p["target_faces"])))
    mean_edge = float(np.mean(cur.edge_len))
    csm = p.get("curvature_smooth")
    if csm is None or csm == "auto":
        radius = float(p.get("curvature_scale", 0.6)) * rho0
        csm = int(np.clip(round((radius / max(mean_edge, EPS)) ** 2), 6, 24))
    csm = int(csm)
    src_all = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    if ca > 0.0 and csm > 0:
        cur = smooth_curvature(cur, N, src_all, indices, iters=csm)
    align_w = alignment_weight(cur, ca)
    align_dir = cur.D1
    t_curv = _time.time()

    # ---- constraints (sharp edges + boundary + guides) -------------------
    sharp_list = []
    se = p.get("sharp_edges")
    if se is not None and len(se):
        sharp_list.append(np.asarray(se, dtype=np.int64).reshape(-1, 2))
    be = boundary_edges(F)
    bnd_verts = np.zeros(n, dtype=bool)
    if len(be):
        bnd_verts[be.ravel()] = True
        if p.get("preserve_boundaries", True):
            sharp_list.append(be)
    sharp_all = np.concatenate(sharp_list, axis=0) if sharp_list else None

    gverts = _guide_dirs_to_vertices(p.get("guide_dirs"), n, len(F), F, N)
    con_mask, con_dir = build_constraints(V, N, n, sharp_all, gverts)
    if sharp_all is not None and len(sharp_all):
        pin_mask, _ = build_constraints(V, N, n, sharp_all, None)
        pin_mask &= con_mask
    else:
        pin_mask = np.zeros(n, dtype=bool)
    # a hard constraint overrides the soft curvature pull entirely
    align_w = np.where(con_mask, 0.0, align_w)

    # ---- target edge length ---------------------------------------------
    adaptive = float(p["adaptive"])
    if adaptive > 1.0:                 # Blender hands over 0..100
        adaptive /= 100.0
    rho = target_edge_lengths(V, F, n, p["target_faces"],
                              density=p.get("density"), adaptive=adaptive,
                              cur=cur, areas=areas)

    # ---- multiresolution 4-RoSy solve ------------------------------------
    levels = build_hierarchy(V, N, indptr, indices, rho, con_mask, con_dir,
                             rng, pin_mask=pin_mask,
                             align_dir=align_dir, align_w=align_w)
    t_hier = _time.time()

    top = levels[-1]
    Q = random_tangents(top["N"], rng)
    # seed the coarsest level from the (restricted) curvature field where it
    # is trustworthy: fewer iterations to converge and no dependence on which
    # random tangent happened to land near a feature
    tw, td = top.get("align_w"), top.get("align_dir")
    if tw is not None:
        seedm = tw > 0.25 * max(ca, EPS)
        if seedm.any():
            Q[seedm] = td[seedm]
    if top["con_mask"].any():
        Q[top["con_mask"]] = top["con_dir"][top["con_mask"]]

    oit = int(p["orient_iters"])
    for li in range(len(levels) - 1, -1, -1):
        lv = levels[li]
        it = oit if li == 0 else max(6, oit // 2)
        Q = smooth_orientations(Q, lv["N"], lv["src"], lv["indices"],
                                lv["con_mask"], lv["con_dir"], it,
                                align_dir=lv.get("align_dir"),
                                align_w=lv.get("align_w"))
        if li > 0:
            Q = prolong_orientations(Q, levels[li - 1]["parent"],
                                     levels[li - 1]["N"])
    t_orient = _time.time()

    Q = normalize(Q - N * _dot(Q, N)[:, None])
    if not np.isfinite(Q).all():
        bad = ~np.isfinite(Q).all(axis=1)
        Q[bad] = random_tangents(N[bad], np.random.default_rng(0))

    stats = {
        "n": n, "tris": int(len(F)), "levels": len(levels),
        "curvature_align": ca, "adaptive": adaptive, "curvature_smooth": csm,
        "aligned_verts": int(np.count_nonzero(align_w > 0.05)),
        "t_topology": t_topo - t0, "t_curvature": t_curv - t_topo,
        "t_hierarchy": t_hier - t_curv, "t_orient": t_orient - t_hier,
        "t_total": t_orient - t0,
    }
    if p.get("verbose"):
        print("[quadforge.native.fields] n=%d tris=%d levels=%d aligned=%d "
              "curv=%.2fs hier=%.2fs orient=%.2fs total=%.2fs"
              % (n, len(F), len(levels), stats["aligned_verts"],
                 stats["t_curvature"], stats["t_hierarchy"],
                 stats["t_orient"], stats["t_total"]))

    return FieldSolution(
        N=N, Q=Q, rho=rho,
        aniso=cur.aniso, align_w=align_w, align_dir=align_dir,
        k1=cur.k1, k2=cur.k2,
        con_mask=con_mask, con_dir=con_dir, pin_mask=pin_mask,
        bnd_verts=bnd_verts, edges=edges, levels=levels, stats=stats,
    )
