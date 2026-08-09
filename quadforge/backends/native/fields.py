"""Orientation (4-RoSy) and position field computation - pure numpy.

Implements the field-smoothing half of the Instant Meshes pipeline
(Jakob et al., "Instant Field-Aligned Meshes", SIGGRAPH Asia 2015):

* an extrinsically-smoothed 4-RoSy orientation field,
* a lattice-compatible position field living on the tangent planes,
* a vertex-clustering multiresolution hierarchy so that both fields
  converge in a handful of iterations per level.

Everything is vectorised over *directed edges* and accumulated with
``np.bincount``; the only Python-level loops are the greedy matching used
to build the hierarchy (O(|E|) with tiny constant) and the loop over
hierarchy levels.

No scipy, no bpy - this module is importable from plain CPython which is
what the standalone self-tests use.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12

__all__ = [
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


def build_hierarchy(P, N, indptr, indices, rho, con_mask, con_dir, rng,
                    min_verts=48, max_levels=12, pin_mask=None):
    """Vertex-clustering hierarchy.

    Each level is a dict with keys ``P N indptr indices src rho con_mask
    con_dir pin_mask parent n``.  ``parent`` maps *this* level's vertices to
    the next (coarser) level; it is ``None`` on the coarsest level.

    ``pin_mask`` is the subset of ``con_mask`` whose *position* is also
    constrained (creases and boundaries - guides only steer orientation).
    """
    if pin_mask is None:
        pin_mask = np.zeros(len(P), dtype=bool)
    levels = []
    cur = dict(P=P, N=N, indptr=indptr, indices=indices,
               src=np.repeat(np.arange(len(P), dtype=np.int64), np.diff(indptr)),
               rho=rho, con_mask=con_mask, con_dir=con_dir,
               pin_mask=pin_mask, parent=None, n=len(P))
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

        cur = dict(P=cP, N=cN, indptr=cip, indices=cidx, src=csrc,
                   rho=crho, con_mask=cmask, con_dir=cdir, pin_mask=cpin,
                   parent=None, n=nc)
        levels[-1]["parent"] = parent
        levels.append(cur)

    return levels


# --------------------------------------------------------------------------
# 4-RoSy orientation smoothing
# --------------------------------------------------------------------------

def smooth_orientations(Q, N, src, dst, con_mask, con_dir, iters,
                        self_weight=1.0):
    """Vectorised (Jacobi) 4-RoSy smoothing with extrinsic matching.

    For every directed edge (i -> j) we pick the representative of the
    4-RoSy class of ``Q[j]`` - one of ``+-Q[j]``, ``+-(N[j] x Q[j])`` -
    that is closest to the current ``Q[i]``, then average.
    """
    n = Q.shape[0]
    Q = np.array(Q, dtype=np.float64, copy=True)
    has_con = bool(con_mask.any())
    if has_con:
        Q[con_mask] = con_dir[con_mask]
    deg = np.bincount(src, minlength=n).astype(np.float64)
    denom = deg + self_weight

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
