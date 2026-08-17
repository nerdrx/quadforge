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
   * remove 2-valence doublets (two quads sharing two edges -> one quad),
   * fuse leftover adjacent triangle pairs into quads,
   * repeat until no hole and no over-used edge remains.

5. **Relax + reproject.**  3-5 tangent-space Laplacian iterations, each
   followed by a projection back onto the nearest input triangle through a
   uniform spatial hash, so repair patches blend in and no output vertex
   drifts off the input surface.  Creases, boundaries and pinned vertices are
   frozen.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12

# collapse an input edge whose two position-field samples are closer than
# this fraction of the local target edge length, even when the two tangent
# frames disagree about the integer offset
WELD_EPS = 0.30

# orbits longer than this are treated as extraction failures and left to the
# hole filler
MAX_ORBIT = 8

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


def _build_edges(F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]],
                       axis=0).astype(np.int64)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
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
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]],
                       axis=0).astype(np.int64)
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1]


def _any_tangent(N):
    alt = np.zeros((len(N), 3))
    pick = np.abs(N[:, 0]) < 0.9
    alt[pick] = (1.0, 0.0, 0.0)
    alt[~pick] = (0.0, 1.0, 0.0)
    return normalize(alt - N * _dot(alt, N)[:, None])


def make_solution(N, Q, rho):
    """Tiny stand-in for ``fields.FieldSolution`` (handy in tests)."""
    from types import SimpleNamespace
    return SimpleNamespace(N=np.asarray(N, dtype=np.float64),
                           Q=np.asarray(Q, dtype=np.float64),
                           rho=np.asarray(rho, dtype=np.float64))


def _sol_get(sol, name):
    if isinstance(sol, dict):
        return sol[name]
    return getattr(sol, name)


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
    n = O.shape[0]
    has_con = con_mask is not None and bool(np.any(con_mask))
    O = np.array(O, dtype=np.float64, copy=True)
    deg = np.bincount(src, minlength=n).astype(np.float64)
    denom = deg + self_weight
    rho_e = 0.5 * (rho[src] + rho[dst])
    inv_e = 1.0 / np.maximum(rho_e, EPS)
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
        acc -= N * _dot(acc - P, N)[:, None]
        O = _round_to_cell(acc, Q, N, P, rho)
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
        ce = np.unique(np.sort(np.stack([pi, pj], axis=1), axis=1), axis=0)
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
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    return np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)


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


def _collapse(O, Q, N, rho, edges):
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
    ce = np.unique(np.sort(np.stack([ei, ej], axis=1), axis=1), axis=0)
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
    """0 (degenerate / reflex) .. 1 (square)."""
    p = P[list(quad)]
    nrm = np.zeros(3)
    for k in range(4):
        nrm += np.cross(p[(k + 1) % 4] - p[k], p[(k + 2) % 4] - p[(k + 1) % 4])
    ln = float(np.dot(nrm, nrm))
    if ln < 1e-24:
        return 0.0
    nrm /= np.sqrt(ln)
    worst = 1.0
    for k in range(4):
        e0 = p[k] - p[(k - 1) % 4]
        e1 = p[(k + 1) % 4] - p[k]
        c = np.cross(e0, e1)
        if float(np.dot(c, nrm)) <= 0.0:
            return 0.0
        l0 = np.linalg.norm(e0)
        l1 = np.linalg.norm(e1)
        if l0 < 1e-12 or l1 < 1e-12:
            return 0.0
        worst = min(worst, 1.0 - abs(float(np.dot(e0, e1)) / (l0 * l1)))
    return worst


def _face_area(P, f):
    p = P[list(f)]
    a = 0.0
    for k in range(1, len(f) - 1):
        a += 0.5 * float(np.linalg.norm(np.cross(p[k] - p[0], p[k + 1] - p[0])))
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


def _split_ngon(P, f, forbidden=None):
    """Split an n-gon (n >= 5) into quads plus at most one triangle.

    ``forbidden`` is a set of undirected edges that must not be created (they
    already exist elsewhere in the mesh and would become non-manifold).
    Returns ``None`` when every rotation would need a forbidden chord.
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
                    if e in forbidden:
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
    use = {}
    for i, f in enumerate(faces):
        k = len(f)
        for a in range(k):
            u, v = f[a], f[(a + 1) % k]
            e = (u, v) if u < v else (v, u)
            use.setdefault(e, []).append(i)
    return use


def _dedupe(P, faces):
    seen = set()
    out = []
    for f in faces:
        if len(f) < 3 or len(set(f)) != len(f):
            continue
        key = tuple(sorted(f))
        if key in seen:
            continue
        pts = P[list(f)]
        a = pts[1] - pts[0]
        b = pts[2] - pts[0]
        c = np.cross(a, b)
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
    for i, f in enumerate(faces):
        k = len(f)
        for a in range(k):
            u, v = f[a], f[(a + 1) % k]
            e = (u, v) if u < v else (v, u)
            dmap.setdefault(e, []).append((i, u < v))

    flip = np.zeros(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)
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
            for a in range(k):
                u, v = f[a], f[(a + 1) % k]
                e = (u, v) if u < v else (v, u)
                for (j, fwd) in dmap[e]:
                    if j == i or seen[j]:
                        continue
                    # i traverses e as (u<v) == (u < v); j must be opposite
                    fwd_i = (u < v) != bool(flip[i])
                    flip[j] = (fwd == fwd_i)
                    seen[j] = True
                    comp.append(j)
                    stack.append(j)
        # component-wide sign vote against the reference normals
        votes = 0
        for i in comp:
            f = faces[i] if not flip[i] else tuple(reversed(faces[i]))
            p0, p1, p2 = P[f[0]], P[f[1]], P[f[2]]
            fn = np.cross(p1 - p0, p2 - p0)
            dp = float(fn @ NRM[list(f)].sum(axis=0))
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
        for _ in range(200000):
            if h in visited:
                ok = h is h0
                break
            visited.add(h)
            seq.append(h[0])
            u, v = h
            e = (v, u)
            nxt = None
            for _r in range(64):
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


def _fill_holes(P, faces, cbnd, extra_pts, project=None):
    """Close every hole that is not a genuine input boundary.

    ``extra_pts`` is appended to in place with the centroid vertices; their
    indices continue after ``len(P) + len(extra_pts)``.
    """
    loops = _hole_loops(faces)
    if not loops:
        return faces, False
    existing = set(_edge_use(faces))
    added = []
    changed = False
    for loop in loops:
        k = len(loop)
        if k < 3:
            continue
        if cbnd is not None and all(
                v < len(cbnd) and cbnd[v] for v in loop):
            continue                     # real opening of an open input mesh
        parts = None
        if k == 3 or k == 4:
            parts = [tuple(loop)]
        elif k <= 6:
            parts = _split_ngon(P, tuple(loop), forbidden=existing)
        if parts is None:
            # centroid fan: touches only the loop's own (single-use) edges
            pts = P[[v for v in loop if v < len(P)]] if k else None
            cen = pts.mean(axis=0) if pts is not None and len(pts) else None
            if cen is None:
                continue
            if project is not None:
                cen = project(cen[None, :])[0]
            ci = len(P) + len(extra_pts)
            extra_pts.append(cen)
            parts = []
            a = 0
            while k - a >= 3:
                parts.append((loop[a], loop[a + 1], loop[a + 2], ci))
                a += 2
            if a == k - 2:
                parts.append((loop[a], loop[a + 1], ci))
            elif a == k - 1:
                parts.append((loop[a], loop[0], ci))
            # k - a == 2 -> the wrap-around quad already closed the fan
            if a == k - 2:
                pass
        for part in parts:
            m = len(part)
            for b in range(m):
                u, v = part[b], part[(b + 1) % m]
                existing.add((u, v) if u < v else (v, u))
        added.extend(parts)
        changed = True
    return (faces + added if added else faces), changed


def _remove_doublets(P, faces):
    """Two faces meeting at a valence-2 interior vertex -> one face."""
    for _ in range(4):
        use = _edge_use(faces)
        vert_edges = {}
        for (u, v), fl in use.items():
            vert_edges.setdefault(u, []).append(((u, v), len(fl)))
            vert_edges.setdefault(v, []).append(((u, v), len(fl)))
        vert_faces = {}
        for i, f in enumerate(faces):
            for v in f:
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


def _merge_tri_pairs(P, faces, min_quality=0.05):
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


# --------------------------------------------------------------------------
# surface projection (uniform spatial hash over the input triangles)
# --------------------------------------------------------------------------

def _closest_on_tri(P, A, B, C):
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
    inv = 1.0 / np.where(np.abs(denom) < 1e-30, 1.0, denom)
    v = vb * inv
    w = vc * inv
    res = A + AB * v[:, None] + AC * w[:, None]
    res = np.where((np.abs(denom) < 1e-30)[:, None], A, res)

    def _set(mask, val):
        if mask.any():
            res[mask] = val[mask]

    den = d1 - d3
    t = d1 / np.where(np.abs(den) < 1e-30, 1.0, den)
    _set((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + AB * t[:, None])
    den = d2 - d6
    t = d2 / np.where(np.abs(den) < 1e-30, 1.0, den)
    _set((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + AC * t[:, None])
    num = d4 - d3
    den = num + (d5 - d6)
    t = num / np.where(np.abs(den) < 1e-30, 1.0, den)
    _set((va <= 0) & (num >= 0) & ((d5 - d6) >= 0), B + (C - B) * t[:, None])

    _set((d1 <= 0) & (d2 <= 0), A)
    _set((d3 >= 0) & (d4 <= d3), B)
    _set((d6 >= 0) & (d5 <= d6), C)
    return res


class _Projector:
    """Nearest point on a triangle mesh via a uniform grid."""

    def __init__(self, V, F, res=64):
        self.V = V
        self.F = F
        self.lo = V.min(axis=0)
        hi = V.max(axis=0)
        ext = np.maximum(hi - self.lo, 1e-12)
        self.diag = float(np.linalg.norm(ext))
        nf = max(len(F), 1)
        cell = max(self.diag / float(res), self.diag / max(nf ** (1.0 / 3.0), 1.0))
        self.dims = np.maximum(np.ceil(ext / max(cell, 1e-12)).astype(np.int64), 1)
        self.dims = np.minimum(self.dims, 128)
        self.cell = ext / self.dims

        # every triangle goes into the cells of its three corners + centroid
        cen = V[F].mean(axis=1)
        pts = np.concatenate([V[F[:, 0]], V[F[:, 1]], V[F[:, 2]], cen], axis=0)
        tid = np.tile(np.arange(len(F), dtype=np.int64), 4)
        cid = self._cell_of(pts)
        key = self._key(cid)
        order = np.argsort(key, kind="stable")
        self.tri_sorted = tid[order]
        self.key_sorted = key[order]
        self.uniq_key, self.uniq_start = np.unique(self.key_sorted,
                                                   return_index=True)
        self.uniq_end = np.append(self.uniq_start[1:], len(self.key_sorted))

    def _cell_of(self, P):
        c = np.floor((P - self.lo) / self.cell).astype(np.int64)
        return np.clip(c, 0, self.dims - 1)

    def _key(self, c):
        return (c[:, 0] * self.dims[1] + c[:, 1]) * self.dims[2] + c[:, 2]

    def _tris_in(self, keys):
        out = []
        pos = np.searchsorted(self.uniq_key, keys)
        pos = np.clip(pos, 0, len(self.uniq_key) - 1)
        hit = self.uniq_key[pos] == keys
        for p in pos[hit]:
            out.append(self.tri_sorted[self.uniq_start[p]:self.uniq_end[p]])
        if not out:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(out))

    def project(self, P, max_move=None):
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        out = P.copy()
        if len(self.F) == 0 or len(P) == 0:
            return out
        cid = self._cell_of(P)
        key = self._key(cid)
        order = np.argsort(key, kind="stable")
        skey = key[order]
        bounds = np.append(np.append(0, np.nonzero(np.diff(skey))[0] + 1),
                           len(skey))
        offs = np.arange(-1, 2)
        neigh = np.stack(np.meshgrid(offs, offs, offs, indexing="ij"),
                         axis=-1).reshape(-1, 3)
        for b in range(len(bounds) - 1):
            sel = order[bounds[b]:bounds[b + 1]]
            base = cid[sel[0]]
            tris = np.zeros(0, dtype=np.int64)
            for r in (1, 2, 4):
                if r == 1:
                    cells = base[None, :] + neigh
                else:
                    o = np.arange(-r, r + 1)
                    cells = np.stack(
                        np.meshgrid(o, o, o, indexing="ij"),
                        axis=-1).reshape(-1, 3) + base[None, :]
                ok = np.all((cells >= 0) & (cells < self.dims[None, :]), axis=1)
                cells = cells[ok]
                if len(cells) == 0:
                    continue
                tris = self._tris_in(self._key(cells))
                if len(tris):
                    break
            if len(tris) == 0:
                tris = np.arange(len(self.F), dtype=np.int64)
            pts = P[sel]
            step = max(1, int(4e6 // max(len(tris), 1)))
            for c0 in range(0, len(pts), step):
                chunk = pts[c0:c0 + step]
                k = len(chunk)
                t = len(tris)
                pp = np.repeat(chunk, t, axis=0)
                tt = np.tile(tris, k)
                tri = self.F[tt]
                cp = _closest_on_tri(pp, self.V[tri[:, 0]], self.V[tri[:, 1]],
                                     self.V[tri[:, 2]])
                d = ((cp - pp) ** 2).sum(axis=1).reshape(k, t)
                best = np.argmin(d, axis=1)
                res = cp.reshape(k, t, 3)[np.arange(k), best]
                out[sel[c0:c0 + step]] = res
        if max_move is not None:
            dv = out - P
            dl = np.sqrt(np.einsum("ij,ij->i", dv, dv))
            too_far = dl > max_move
            if np.isscalar(max_move):
                out[too_far] = P[too_far]
            else:
                out[too_far] = P[too_far]
        return out


# --------------------------------------------------------------------------
# relaxation
# --------------------------------------------------------------------------

def _relax(P, faces, frozen, projector, rho_v, iters=4, step=0.5):
    if iters <= 0 or len(faces) == 0:
        return P
    pairs = []
    for f in faces:
        k = len(f)
        for a in range(k):
            pairs.append((f[a], f[(a + 1) % k]))
    e = np.asarray(pairs, dtype=np.int64)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    n = len(P)
    deg = np.bincount(src, minlength=n).astype(np.float64)
    deg = np.maximum(deg, 1.0)
    P = P.copy()
    move = ~frozen
    if not move.any():
        return P
    for _ in range(iters):
        acc = np.empty((n, 3))
        for c in range(3):
            acc[:, c] = np.bincount(src, weights=P[dst][:, c], minlength=n)
        acc /= deg[:, None]
        delta = acc - P
        # tangential component only (uses the current vertex normal estimate)
        Nv = _vertex_normals_poly(P, faces, n)
        delta -= Nv * _dot(delta, Nv)[:, None]
        Pn = P.copy()
        Pn[move] = P[move] + step * delta[move]
        if projector is not None:
            Pn[move] = projector.project(Pn[move])
            d = Pn[move] - P[move]
            dl = np.sqrt(np.einsum("ij,ij->i", d, d))
            cap = 0.75 * rho_v[move]
            bad = dl > cap
            if bad.any():
                idx = np.nonzero(move)[0][bad]
                Pn[idx] = P[idx]
        P = Pn
    return P


def _vertex_normals_poly(P, faces, n):
    N = np.zeros((n, 3))
    for f in faces:
        p = P[list(f)]
        k = len(f)
        nrm = np.zeros(3)
        for a in range(k):
            nrm += np.cross(p[a], p[(a + 1) % k])
        for v in f:
            N[v] += nrm
    ln = np.sqrt(np.einsum("ij,ij->i", N, N))
    bad = ln < 1e-14
    if bad.any():
        N[bad] = (0.0, 0.0, 1.0)
    return normalize(N)


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------

def _extract_core(O, Q, N, rho, edges, bnd_verts=None, V=None, F=None,
                  relax_iters=4, projector=None):
    n = O.shape[0]
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return np.zeros((0, 3)), []

    cluster, CP, CN, CQ, ce = _collapse(O, Q, N, rho, edges)
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
    extra = []

    def _proj_pt(pt):
        if projector is None:
            return pt
        return projector.project(pt)

    for _round in range(4):
        faces, changed = _fill_holes(P, faces, cbnd, extra,
                                     project=_proj_pt if projector else None)
        if extra:
            P = np.concatenate([CP, np.asarray(extra).reshape(-1, 3)], axis=0)
        faces = _dedupe(P, faces)
        faces = _enforce_edge_manifold(P, faces)
        faces = _make_injective(faces)
        faces = _remove_doublets(P, faces)
        faces = _dedupe(P, faces)
        faces = _enforce_edge_manifold(P, faces)
        if not changed:
            break

    faces = _merge_tri_pairs(P, faces)
    faces = _dedupe(P, faces)
    faces = _enforce_edge_manifold(P, faces)
    # a fusion can (very rarely) open a hole again
    faces, changed = _fill_holes(P, faces, cbnd, extra,
                                 project=_proj_pt if projector else None)
    if changed:
        if extra:
            P = np.concatenate([CP, np.asarray(extra).reshape(-1, 3)], axis=0)
        faces = _dedupe(P, faces)
        faces = _enforce_edge_manifold(P, faces)
    faces = _orient(P, np.concatenate(
        [CN, np.tile((0.0, 0.0, 1.0), (len(P) - nc, 1))], axis=0)
        if len(P) > nc else CN, faces)

    if not faces:
        return np.zeros((0, 3)), []

    # ---- relax + reproject ----------------------------------------------
    npt = len(P)
    if npt > nc:
        crho_full = np.concatenate([crho, np.full(npt - nc, float(np.mean(crho)))])
        cbnd_full = np.concatenate([cbnd, np.zeros(npt - nc, dtype=bool)])
    else:
        crho_full = crho
        cbnd_full = cbnd
    if projector is not None:
        frozen = cbnd_full.copy()
        P = _relax(P, faces, frozen, projector, crho_full, iters=relax_iters)
        moved = P.copy()
        moved[~frozen] = projector.project(P[~frozen])
        d = moved - P
        dl = np.sqrt(np.einsum("ij,ij->i", d, d))
        ok = dl <= 0.75 * crho_full
        P = np.where(ok[:, None], moved, P)

    # ---- compact ---------------------------------------------------------
    used = np.zeros(len(P), dtype=bool)
    for f in faces:
        used[list(f)] = True
    remap = np.full(len(P), -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    VQ = P[used]
    FQ = [tuple(int(remap[v]) for v in f) for f in faces]
    return VQ, FQ


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def extract_quads(O, Q, N, rho, edges, min_cycle=3, max_cycle=6,
                  bnd_verts=None):
    """v1 entry point: the caller supplies an already-solved position field."""
    return _extract_core(O, Q, N, rho, np.asarray(edges), bnd_verts=bnd_verts,
                         projector=None, relax_iters=0)


def extract(V, F, sol, params=None):
    """NATIVE_V2 entry point.

    Solves the position field for the orientation field in ``sol`` and returns
    a repaired quad-dominant mesh ``(VQ (k,3) float64, FQ list[3|4-tuples])``.
    """
    p = dict(params or {})
    V = np.ascontiguousarray(np.asarray(V, dtype=np.float64).reshape(-1, 3))
    F = np.ascontiguousarray(np.asarray(F, dtype=np.int64).reshape(-1, 3))
    n = V.shape[0]

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

    edges = _build_edges(F)
    if len(edges) == 0:
        return np.zeros((0, 3)), []
    indptr, indices, _src = _build_csr(edges, n)

    # ---- constraints -----------------------------------------------------
    be = _boundary_edges(F)
    bnd_verts = np.zeros(n, dtype=bool)
    if len(be):
        bnd_verts[be.ravel()] = True
    pin_list = []
    sharp = p.get("sharp_edges")
    if sharp is not None and len(sharp):
        pin_list.append(np.asarray(sharp, dtype=np.int64).reshape(-1, 2))
    if len(be) and p.get("preserve_boundaries", True):
        pin_list.append(be)
    pin_mask = np.zeros(n, dtype=bool)
    pin_dir = np.zeros((n, 3))
    if pin_list:
        se = np.concatenate(pin_list, axis=0)
        se = se[(se[:, 0] != se[:, 1]) & (se[:, 0] >= 0) & (se[:, 1] >= 0)
                & (se[:, 0] < n) & (se[:, 1] < n)]
        if len(se):
            d = normalize(V[se[:, 1]] - V[se[:, 0]])
            vi = np.concatenate([se[:, 0], se[:, 1]])
            vd = np.concatenate([d, d])
            nv = N[vi]
            vd = vd - nv * _dot(vd, nv)[:, None]
            vl = np.sqrt(np.einsum("ij,ij->i", vd, vd))
            keep = vl > 1e-7
            vi = vi[keep]
            vd = vd[keep] / vl[keep][:, None]
            ref = np.zeros((n, 3))
            ref[vi] = vd
            rep = _match_4rosy(vd, N[vi], ref[vi])
            acc = np.zeros((n, 3))
            for c in range(3):
                acc[:, c] = np.bincount(vi, weights=rep[:, c], minlength=n)
            acc -= N * _dot(acc, N)[:, None]
            al = np.sqrt(np.einsum("ij,ij->i", acc, acc))
            pin_mask[vi] = True
            good = pin_mask & (al > 1e-7)
            pin_dir[good] = acc[good] / al[good][:, None]
            fb = pin_mask & ~good
            pin_dir[fb] = ref[fb]
            pin_mask = good | fb

    # ---- hierarchy + position field -------------------------------------
    rng = np.random.default_rng(int(p.get("seed", 0) or 0) & 0x7FFFFFFF)
    levels = _build_pos_hierarchy(V, N, Q, rho, indptr, indices,
                                  pin_mask, pin_dir, rng)
    pos_iters = int(p.get("pos_iters", 20) or 20)

    projector = _Projector(V, F) if p.get("project", True) else None
    target = int(p.get("target_faces", 0) or 0)

    best = None
    scale = 1.0
    attempts = int(p.get("attempts", 3) or 3)
    for _a in range(max(1, attempts)):
        O = _solve_positions(levels, scale, pos_iters)
        VQ, FQ = _extract_core(O, Q, N, rho * scale, edges,
                               bnd_verts=bnd_verts, V=V, F=F,
                               projector=projector,
                               relax_iters=int(p.get("relax_iters", 4)))
        nf = len(FQ)
        nq = sum(1 for f in FQ if len(f) == 4)
        if target > 0:
            score = abs(np.log(max(nf, 1) / float(target))) - 0.25 * (
                nq / float(max(nf, 1)))
        else:
            score = -(nq / float(max(nf, 1)))
        if best is None or score < best[0]:
            best = (score, VQ, FQ)
        if target <= 0:
            break
        if nf == 0:
            scale *= 0.6
            continue
        ratio = nf / float(target)
        if 0.82 <= ratio <= 1.22:
            break
        scale *= float(np.clip(np.sqrt(ratio), 0.55, 1.8))

    return best[1], best[2]
