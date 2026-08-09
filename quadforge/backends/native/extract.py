"""Quad extraction from a 4-RoSy orientation field + position field.

Strategy (a numpy re-implementation of the Instant Meshes graph-extraction
step, simplified):

1. Every input vertex carries a position-field sample ``O[i]`` that already
   sits (approximately) on a locally-consistent square lattice of spacing
   ``rho``.  For every input edge (i, j) we express ``O[j] - O[i]`` in the
   tangent frame ``(Q[i], N[i] x Q[i])`` and round to integers.

   * offset (0, 0)      -> the two vertices occupy the same lattice cell:
                           union-find **collapse** them,
   * |da| + |db| == 1   -> unit lattice step: keep as an **edge** of the
                           extracted graph,
   * anything else      -> discard (diagonal / long jump).

2. Clusters become output vertices (mean of their ``O``), the surviving
   unit-step edges become output edges.  Degree<=1 vertices are pruned.

3. The extracted graph is embedded in the surface, so its faces can be
   recovered combinatorially: sort each vertex's neighbours by angle in its
   tangent frame (a *rotation system*) and trace the orbits of
   ``next(u->v) = v->prev_ccw(u)``.  Every directed edge belongs to exactly
   one orbit, so faces are produced without duplicates and each edge is used
   by at most two faces.  On a clean lattice region the orbits are 4-cycles.

4. Orbits of length 3 stay triangles, 4 stay quads, 5 and 6 are fan-split into
   quads (+ at most one triangle), everything longer (holes, the outer face of
   an open mesh) is discarded.

5. Clean-up: dedupe, drop degenerate faces, drop the least valuable face on
   any edge shared by more than two faces, close small holes that are not on a
   genuine input boundary, fuse leftover adjacent triangle pairs into quads,
   and compact away unused vertices.
"""

from __future__ import annotations

import numpy as np

from .fields import normalize

EPS = 1e-12

__all__ = ["extract_quads"]


def _dot(A, B):
    return np.einsum("ij,ij->i", A, B)


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
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    return roots


def _prune_low_degree(nv, e0, e1, min_deg=2):
    """Iteratively drop vertices with degree < min_deg."""
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


# --------------------------------------------------------------------------

def _quad_quality(P, quad):
    """0 (degenerate / reflex) .. 1 (square).  Used to rank triangle merges."""
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
            return 0.0                     # reflex corner -> not a valid quad
        l0 = np.linalg.norm(e0)
        l1 = np.linalg.norm(e1)
        if l0 < 1e-12 or l1 < 1e-12:
            return 0.0
        cosang = abs(float(np.dot(e0, e1)) / (l0 * l1))
        worst = min(worst, 1.0 - cosang)   # 1 at 90 degrees, 0 when collinear
    return worst


def _face_area(P, f):
    p = P[list(f)]
    a = 0.0
    for k in range(1, len(f) - 1):
        a += 0.5 * float(np.linalg.norm(np.cross(p[k] - p[0], p[k + 1] - p[0])))
    return a


def _split_ngon(P, f):
    """Fan-split an n-gon (n >= 5) into quads plus at most one triangle,
    choosing the fan apex that gives the best-shaped quads."""
    k = len(f)
    best = None
    for r in range(k):
        g = list(f[r:]) + list(f[:r])
        parts = []
        worst = 1.0
        i = 1
        while k - i >= 3:
            quad = (g[0], g[i], g[i + 1], g[i + 2])
            parts.append(quad)
            worst = min(worst, _quad_quality(P, quad))
            i += 2
        if i < k - 1:
            parts.append((g[0], g[i], g[i + 1]))
        if best is None or worst > best[0]:
            best = (worst, parts)
    return best[1]


def _repair_nonmanifold(P, faces):
    """Drop the least valuable face on any edge used by more than two faces."""
    use = {}
    for i, f in enumerate(faces):
        k = len(f)
        for a in range(k):
            e = (min(f[a], f[(a + 1) % k]), max(f[a], f[(a + 1) % k]))
            use.setdefault(e, []).append(i)
    bad = [e for e, l in use.items() if len(l) > 2]
    if not bad:
        return faces
    drop = set()
    for e in bad:
        fl = [i for i in use[e] if i not in drop]
        if len(fl) <= 2:
            continue
        fl.sort(key=lambda i: (len(faces[i]) == 4, _face_area(P, faces[i])),
                reverse=True)
        drop.update(fl[2:])
    return [f for i, f in enumerate(faces) if i not in drop]


def _fill_small_holes(P, NRM, faces, blocked, max_len=6):
    """Close small holes left by incomplete cycles.

    ``blocked`` flags output vertices that sit on a *genuine* boundary of the
    input mesh - loops touching those are real openings and are left alone.
    """
    use = {}
    for i, f in enumerate(faces):
        k = len(f)
        for a in range(k):
            e = (min(f[a], f[(a + 1) % k]), max(f[a], f[(a + 1) % k]))
            use.setdefault(e, []).append(i)
    bnd = [e for e, l in use.items() if len(l) == 1]
    if not bnd:
        return faces

    adj = {}
    for a, b in bnd:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    seen = set()
    added = []
    for v0 in list(adj):
        if v0 in seen:
            continue
        loop = []
        prev = None
        cur = v0
        ok = True
        while True:
            if len(adj[cur]) != 2 or cur in loop:
                ok = False
                break
            loop.append(cur)
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt:
                ok = False
                break
            prev, cur = cur, nxt[0]
            if cur == v0:
                break
            if len(loop) > max_len:
                ok = False
                break
        seen.update(loop)
        if not ok or not (3 <= len(loop) <= max_len):
            continue
        if any(blocked[v] for v in loop):
            continue
        # orient the patch consistently with the surrounding faces
        p = P[loop]
        nrm = np.zeros(3)
        for k in range(len(loop)):
            nrm += np.cross(p[k], p[(k + 1) % len(loop)])
        if float(nrm @ NRM[loop].sum(axis=0)) < 0.0:
            loop = loop[::-1]
        if len(loop) <= 4:
            added.append(tuple(loop))
        else:
            added.extend(_split_ngon(P, tuple(loop)))
    return faces + added if added else faces


def _merge_tri_pairs(P, faces, min_quality=0.15):
    """Greedily merge adjacent triangle pairs into quads (quad-dominance pass)."""
    tri_idx = [i for i, f in enumerate(faces) if len(f) == 3]
    if len(tri_idx) < 2:
        return faces

    use = {}
    for i, f in enumerate(faces):
        k = len(f)
        for a in range(k):
            e = (min(f[a], f[(a + 1) % k]), max(f[a], f[(a + 1) % k]))
            use.setdefault(e, []).append(i)

    is_tri = [len(f) == 3 for f in faces]
    cand = []
    for e, fl in use.items():
        if len(fl) != 2:
            continue
        i, j = fl
        if not (is_tri[i] and is_tri[j]):
            continue
        fi, fj = faces[i], faces[j]
        a, b = e
        # windings must be opposite along the shared edge
        ai = fi.index(a)
        forward_i = fi[(ai + 1) % 3] == b
        aj = fj.index(a)
        forward_j = fj[(aj + 1) % 3] == b
        if forward_i == forward_j:
            continue
        if not forward_i:
            i, j = j, i
            fi, fj = fj, fi
            ai = fi.index(a)
        c = [v for v in fi if v not in (a, b)][0]
        d = [v for v in fj if v not in (a, b)][0]
        quad = (b, c, a, d)
        if len(set(quad)) != 4:
            continue
        q = _quad_quality(P, quad)
        if q < min_quality:
            continue
        cand.append((-q, i, j, quad))

    if not cand:
        return faces
    cand.sort()
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


def extract_quads(O, Q, N, rho, edges, min_cycle=3, max_cycle=6,
                  bnd_verts=None):
    """Return ``(VQ (k,3) float64, faces list[tuple[int, ...]])``.

    ``bnd_verts`` is an optional per-input-vertex bool mask marking genuine
    mesh boundaries, so that small holes elsewhere can be closed while real
    openings are preserved.
    """
    n = O.shape[0]
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return np.zeros((0, 3)), []

    i = edges[:, 0]
    j = edges[:, 1]
    rho_e = 0.5 * (rho[i] + rho[j])
    inv = 1.0 / np.maximum(rho_e, EPS)

    d = O[j] - O[i]
    Ti = np.cross(N[i], Q[i])
    ai = np.round(_dot(Q[i], d) * inv)
    bi = np.round(_dot(Ti, d) * inv)

    Tj = np.cross(N[j], Q[j])
    aj = np.round(_dot(Q[j], -d) * inv)
    bj = np.round(_dot(Tj, -d) * inv)

    zi = (ai == 0) & (bi == 0)
    zj = (aj == 0) & (bj == 0)
    collapse = zi & zj

    si = np.abs(ai) + np.abs(bi)
    sj = np.abs(aj) + np.abs(bj)
    unit = (si == 1) & (sj == 1)

    # ---- 1. collapse -------------------------------------------------
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

    # representative tangent per cluster (lowest member index wins),
    # re-projected onto the cluster normal
    rep_order = np.arange(n)[::-1]
    CQ = np.zeros((nc, 3))
    CQ[cluster[rep_order]] = Q[rep_order]
    CQ = CQ - CN * _dot(CQ, CN)[:, None]
    ql = np.sqrt(np.einsum("ij,ij->i", CQ, CQ))
    qbad = ql < 1e-9
    if qbad.any():
        alt = np.zeros((int(qbad.sum()), 3))
        nb = CN[qbad]
        pick = np.abs(nb[:, 0]) < 0.9
        alt[pick] = (1.0, 0.0, 0.0)
        alt[~pick] = (0.0, 1.0, 0.0)
        CQ[qbad] = alt - nb * _dot(alt, nb)[:, None]
    CQ = normalize(CQ)

    # ---- 2. extracted graph -----------------------------------------
    ei = cluster[i[unit]]
    ej = cluster[j[unit]]
    m = ei != ej
    ei, ej = ei[m], ej[m]
    if len(ei) == 0:
        return np.zeros((0, 3)), []
    ce = np.sort(np.stack([ei, ej], axis=1), axis=1)
    ce = np.unique(ce, axis=0)

    alive, keep = _prune_low_degree(nc, ce[:, 0], ce[:, 1], min_deg=2)
    ce = ce[keep]
    if len(ce) == 0:
        return np.zeros((0, 3)), []

    # ---- 3. rotation system ------------------------------------------
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
    pos = np.searchsorted(skeys, rkeys)
    pos = np.clip(pos, 0, ne - 1)
    rev = korder[pos]
    if not np.array_equal(keys[rev], rkeys):
        # should not happen (graph is symmetric & deduped) - bail out safely
        good = keys[rev] == rkeys
        rev = np.where(good, rev, np.arange(ne))

    dv_local = rev - start[dst]
    nxt = start[dst] + (dv_local - 1) % np.maximum(deg[dst], 1)

    # ---- 4. trace orbits ---------------------------------------------
    visited = np.zeros(ne, dtype=bool)
    faces = []
    nxt_l = nxt.tolist()
    src_l = src.tolist()
    for s0 in range(ne):
        if visited[s0]:
            continue
        cyc = []
        s = s0
        while not visited[s]:
            visited[s] = True
            cyc.append(src_l[s])
            s = nxt_l[s]
            if len(cyc) > max_cycle + 1:
                break
        if min_cycle <= len(cyc) <= max_cycle and len(set(cyc)) == len(cyc):
            faces.append(cyc)

    if not faces:
        return np.zeros((0, 3)), []

    # ---- 5. split 5/6-gons, orient, clean ----------------------------
    out = []
    for f in faces:
        k = len(f)
        if k <= 4:
            out.append(tuple(f))
        else:
            out.extend(_split_ngon(CP, f))

    # consistent winding: compare against the cluster normals
    flip_votes = 0
    total_votes = 0
    for f in out:
        if len(f) < 3:
            continue
        p0, p1, p2 = CP[f[0]], CP[f[1]], CP[f[2]]
        fn = np.cross(p1 - p0, p2 - p0)
        nn = CN[list(f)].sum(axis=0)
        dp = float(fn @ nn)
        if dp != 0.0:
            total_votes += 1
            if dp < 0.0:
                flip_votes += 1
    if total_votes and flip_votes * 2 > total_votes:
        out = [tuple(reversed(f)) for f in out]

    # dedupe by vertex set + drop degenerate
    seen = set()
    clean = []
    for f in out:
        if len(set(f)) != len(f):
            continue
        key = tuple(sorted(f))
        if key in seen:
            continue
        seen.add(key)
        pts = CP[list(f)]
        a = pts[1] - pts[0]
        b = pts[2] - pts[0]
        if float(np.cross(a, b) @ np.cross(a, b)) < 1e-24:
            continue
        clean.append(f)

    if not clean:
        return np.zeros((0, 3)), []

    # manifold repair, small-hole closing, then a quad-dominance pass that
    # fuses leftover triangle pairs
    clean = _repair_nonmanifold(CP, clean)
    if bnd_verts is not None:
        cbnd = np.zeros(nc, dtype=bool)
        cbnd[cluster[np.asarray(bnd_verts, dtype=bool)]] = True
    else:
        cbnd = np.ones(nc, dtype=bool)
    clean = _fill_small_holes(CP, CN, clean, cbnd)
    clean = _merge_tri_pairs(CP, clean)

    used = np.zeros(nc, dtype=bool)
    for f in clean:
        used[list(f)] = True
    remap = np.full(nc, -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    VQ = CP[used]
    FQ = [tuple(int(remap[v]) for v in f) for f in clean]
    return VQ, FQ
