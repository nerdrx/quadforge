"""Data-preservation *quality*, measured rather than assumed.

test_05 asks whether UVs / weights / shape keys / materials survived at all.
This module asks how good the survivors are, with metrics that a texture or a
rig would actually notice:

  seam bleed      output faces whose corners trace back to two different UV
                  islands of the source — a face textured out of two unrelated
                  parts of the layout, i.e. a smear across the whole UV space
  texel stretch   output face edges whose UV length blows past the source's own
                  texel density
  back-projection how far the texel that now lands on a point is from where it
                  used to live on the surface
  deformation     surface-to-surface deviation between original and remesh with
                  the rig posed / a shape key at 1.0, minus the rest-state
                  deviation, which is remeshing error and not a transfer fault
  crevice bleed   weights grabbed across a gap of 8% of a tube's radius

Everything runs on the NATIVE backend with a fixed seed so the rows are
deterministic; QuadriFlow's are not.
"""

import math

import numpy as np

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

SEED = 7


# ---------------------------------------------------------------- harness

def _fget(coll, attr, size, dtype='f8'):
    a = np.empty(size, dtype)
    coll.foreach_get(attr, a)
    return a


def _verts(me):
    return _fget(me.vertices, "co", len(me.vertices) * 3).reshape(-1, 3)


def _polys(me):
    npo, nl = len(me.polygons), len(me.loops)
    return (_fget(me.polygons, "loop_start", npo, 'i4'),
            _fget(me.polygons, "loop_total", npo, 'i4'),
            _fget(me.loops, "vertex_index", nl, 'i4'))


def _fan(starts, totals, lv):
    per = np.maximum(totals - 2, 0).astype('i8')
    total = int(per.sum())
    if total == 0:
        z = np.zeros((0, 3), 'i4')
        return z, z, np.zeros(0, 'i4')
    pidx = np.repeat(np.arange(starts.size, dtype='i4'), per)
    ends = np.cumsum(per)
    k = np.arange(total, dtype='i8') - np.repeat(ends - per, per) + 1
    s = starts[pidx].astype('i8')
    tl = np.stack([s, s + k, s + k + 1], axis=1).astype('i4')
    return tl, lv[tl], pidx


def _diag(V):
    return float(np.linalg.norm(V.max(0) - V.min(0))) if V.size else 1.0


def _bvh(V, tris):
    if tris.shape[0] == 0:
        return None
    return BVHTree.FromPolygons(V.tolist(), [tuple(t) for t in tris.tolist()],
                                all_triangles=True)


def _dists(points, bvh):
    out = np.zeros(points.shape[0], 'f8')
    fn = bvh.find_nearest
    for i, p in enumerate(points.tolist()):
        r = fn(p)
        out[i] = r[3] if (r is not None and r[3] is not None) else np.nan
    return out


def _stat(a, scale=1.0):
    a = np.asarray(a, 'f8')
    a = a[np.isfinite(a)] / scale
    if a.size == 0:
        return {'n': 0, 'mean': 0.0, 'p99': 0.0, 'max': 0.0}
    return {'n': int(a.size), 'mean': float(a.mean()),
            'p99': float(np.percentile(a, 99)), 'max': float(a.max())}


def _eval_surface(obj):
    """(verts, tris) of the fully evaluated mesh, object-local."""
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), depsgraph=dg)
    try:
        V = _verts(me)
        st, to, lv = _polys(me)
        _tl, tris, _tp = _fan(st, to, lv)
    finally:
        bpy.data.meshes.remove(me)
    return V, tris


def _deviation(src_obj, out_obj, scale):
    sV, stris = _eval_surface(src_obj)
    oV, _ = _eval_surface(out_obj)
    b = _bvh(sV, stris)
    if b is None:
        return {'n': 0, 'mean': 0.0, 'p99': 0.0, 'max': 0.0}, sV
    return _stat(_dists(oV, b), scale), sV


class _UVRef:
    """Source UV layout: island ids + a UV -> surface trace."""

    def __init__(self, me):
        self.V = _verts(me)
        self.starts, self.totals, self.lv = _polys(me)
        self.tri_loops, self.tris, self.tri_poly = _fan(
            self.starts, self.totals, self.lv)
        uvl = me.uv_layers.active or (me.uv_layers[0] if me.uv_layers else None)
        self.name = uvl.name if uvl else ""
        nl = len(me.loops)
        a = np.zeros(nl * 2, 'f4')
        if uvl is not None:
            uvl.data.foreach_get("uv", a)
        self.uv = a.reshape(-1, 2).astype('f8')
        self.island = self._islands()
        self._bvh = None

    def _islands(self):
        npo = self.starts.size
        nl = self.lv.size
        if npo == 0:
            return np.zeros(0, 'i4')
        lp = np.repeat(np.arange(npo, dtype='i4'), self.totals)
        idx = np.arange(nl, dtype='i8')
        local = idx - np.repeat(self.starts.astype('i8'), self.totals)
        nxt = self.starts[lp].astype('i8') + (local + 1) % np.maximum(self.totals[lp], 1)
        ekey = np.sort(np.stack([self.lv, self.lv[nxt]], 1), axis=1)
        order = np.lexsort((ekey[:, 1], ekey[:, 0]))
        e = ekey[order]
        if e.shape[0] < 2:
            return np.arange(npo, dtype='i4')
        same = np.all(e[1:] == e[:-1], axis=1)
        la, lb = order[:-1][same], order[1:][same]
        uv = self.uv
        a0, a1 = uv[la], uv[nxt[la]]
        b0, b1 = uv[lb], uv[nxt[lb]]
        eps = 1e-6
        m1 = (np.abs(a0 - b1).max(1) < eps) & (np.abs(a1 - b0).max(1) < eps)
        m2 = (np.abs(a0 - b0).max(1) < eps) & (np.abs(a1 - b1).max(1) < eps)
        join = m1 | m2
        parent = list(range(npo))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for x, y in zip(lp[la[join]].tolist(), lp[lb[join]].tolist()):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
        roots = np.fromiter((find(i) for i in range(npo)), 'i4', npo)
        _u, inv = np.unique(roots, return_inverse=True)
        return inv.astype('i4')

    @property
    def n_islands(self):
        return int(self.island.max()) + 1 if self.island.size else 0

    def uvbvh(self):
        if self._bvh is None:
            U = self.uv[self.tri_loops].reshape(-1, 2)
            P = np.zeros((U.shape[0], 3), 'f8')
            P[:, :2] = U
            t = np.arange(U.shape[0], dtype='i4').reshape(-1, 3)
            self._bvh = BVHTree.FromPolygons(
                P.tolist(), [tuple(x) for x in t.tolist()], all_triangles=True)
        return self._bvh

    def trace_affine(self, UV, P, radius=1e-4):
        """Like ``trace``, but the source triangle's map is *not* clipped.

        A UV outside every island still means something: it is the source
        parameterisation continued past the island's edge, which is exactly what
        bounded extrapolation writes and what a tiling or coordinate-driven
        texture reads.  Clipping (as ``trace`` does) measures the same UV as if
        the sampler snapped it back to the island outline.
        """
        bvh = self.uvbvh()
        rng, near = bvh.find_nearest_range, bvh.find_nearest
        own, tri = [], []
        for i, uv in enumerate(UV.tolist()):
            q = (uv[0], uv[1], 0.0)
            cs = rng(q, radius)
            if not cs:
                r = near(q)
                if r is None or r[2] is None:
                    continue
                cs = [r]
            for c in cs:
                own.append(i)
                tri.append(c[2])
        own = np.asarray(own, 'i8')
        tri = np.asarray(tri, 'i8')
        if own.size == 0:
            return own, np.zeros(0)
        L = self.tri_loops[tri]
        a, b, c = self.uv[L[:, 0]], self.uv[L[:, 1]], self.uv[L[:, 2]]
        v0, v1, v2 = b - a, c - a, UV[own] - a
        d = lambda x, y: np.einsum('ij,ij->i', x, y)      # noqa: E731
        d00, d01, d11 = d(v0, v0), d(v0, v1), d(v1, v1)
        d20, d21 = d(v2, v0), d(v2, v1)
        den = d00 * d11 - d01 * d01
        den = np.where(np.abs(den) < 1e-18, 1.0, den)
        vv = (d11 * d20 - d01 * d21) / den
        ww = (d00 * d21 - d01 * d20) / den
        B = np.stack([1.0 - vv - ww, vv, ww], 1)
        pos = np.einsum('ijk,ij->ik', self.V[self.tris[tri]], B)
        return own, np.linalg.norm(pos - P[own], axis=1)

    def trace(self, UV, P, radius=1e-4):
        """All UV-coincident candidates: (own, err, island) flat arrays."""
        bvh = self.uvbvh()
        rng, near = bvh.find_nearest_range, bvh.find_nearest
        own, tri = [], []
        for i, uv in enumerate(UV.tolist()):
            q = (uv[0], uv[1], 0.0)
            cs = rng(q, radius)
            if not cs:
                r = near(q)
                if r is None or r[2] is None:
                    continue
                cs = [r]
            for c in cs:
                own.append(i)
                tri.append(c[2])
        own = np.asarray(own, 'i8')
        tri = np.asarray(tri, 'i8')
        if own.size == 0:
            return own, np.zeros(0), np.zeros(0, 'i4')
        L = self.tri_loops[tri]
        a, b, c = self.uv[L[:, 0]], self.uv[L[:, 1]], self.uv[L[:, 2]]
        v0, v1, v2 = b - a, c - a, UV[own] - a
        d = lambda x, y: np.einsum('ij,ij->i', x, y)      # noqa: E731
        d00, d01, d11 = d(v0, v0), d(v0, v1), d(v1, v1)
        d20, d21 = d(v2, v0), d(v2, v1)
        den = d00 * d11 - d01 * d01
        den = np.where(np.abs(den) < 1e-18, 1.0, den)
        vv = (d11 * d20 - d01 * d21) / den
        ww = (d00 * d21 - d01 * d20) / den
        B = np.stack([1.0 - vv - ww, vv, ww], 1)
        np.clip(B, 0.0, 1.0, out=B)
        B /= np.maximum(B.sum(1), 1e-12)[:, None]
        pos = np.einsum('ijk,ij->ik', self.V[self.tris[tri]], B)
        return own, np.linalg.norm(pos - P[own], axis=1), \
            self.island[self.tri_poly[tri]]


def _texel_density(starts, totals, V, lv, uv):
    npo, nl = starts.size, lv.size
    lp = np.repeat(np.arange(npo, dtype='i4'), totals)
    idx = np.arange(nl, dtype='i8')
    local = idx - np.repeat(starts.astype('i8'), totals)
    nxt = starts[lp].astype('i8') + (local + 1) % np.maximum(totals[lp], 1)
    dp = np.linalg.norm(V[lv[nxt]] - V[lv], axis=1)
    du = np.linalg.norm(uv[nxt] - uv, axis=1)
    return np.where(dp > 1e-12, du / np.maximum(dp, 1e-12), 0.0), lp


def uv_quality(src_me, out_me):
    """Seam bleed, texel stretch and back-projection error of a remesh."""
    ref = _UVRef(src_me)
    V = _verts(out_me)
    starts, totals, lv = _polys(out_me)
    npo, nl = starts.size, lv.size
    uvl = out_me.uv_layers.get(ref.name) or out_me.uv_layers.active
    if uvl is None or nl == 0:
        return {'ok': False, 'islands': ref.n_islands}
    a = np.empty(nl * 2, 'f4')
    uvl.data.foreach_get("uv", a)
    OUV = a.reshape(-1, 2).astype('f8')
    P = V[lv]
    diag = _diag(ref.V)

    own, err, isl = ref.trace(OUV, P)
    best = np.full(nl, np.inf)
    if own.size:
        np.minimum.at(best, own, err)
    # an island counts as a plausible source when it lands essentially as close
    # as the best candidate; that is what keeps mirrored layouts from reading as
    # bleed
    ok = err <= best[own] + 1e-3 * diag
    sets = [set() for _ in range(nl)]
    for o, i in zip(own[ok].tolist(), isl[ok].tolist()):
        sets[o].add(i)

    bleed = 0
    for p in range(npo):
        common = None
        for k in range(int(starts[p]), int(starts[p]) + int(totals[p])):
            if sets[k]:
                common = sets[k] if common is None else (common & sets[k])
        if common is not None and not common:
            bleed += 1

    src_r, _ = _texel_density(ref.starts, ref.totals, ref.V, ref.lv, ref.uv)
    src_r = src_r[src_r > 0]
    med = float(np.median(src_r)) if src_r.size else 0.0
    lim = max(4.0 * med, 1.5 * float(np.percentile(src_r, 99.9))) if src_r.size else 0.0
    r, lp = _texel_density(starts, totals, V, lv, OUV)
    fbad = np.zeros(npo, bool)
    if lim > 0:
        fbad[lp[r > lim]] = True

    out = _stat(np.where(np.isfinite(best), best, np.nan), diag)
    out.update({
        'ok': True,
        'islands': ref.n_islands,
        'faces': npo,
        'bleed_faces': bleed,
        'bleed_pct': 100.0 * bleed / max(npo, 1),
        'stretch_faces': int(fbad.sum()),
        'stretch_pct': 100.0 * float(fbad.sum()) / max(npo, 1),
    })
    return out


def group_weights(obj, name):
    """Per-vertex weight of one vertex group as a dense array."""
    vg = obj.vertex_groups.get(name)
    n = len(obj.data.vertices)
    w = np.zeros(n, 'f8')
    if vg is None:
        return w, False
    gi = vg.index
    for v in obj.data.vertices:
        for ge in v.groups:
            if ge.group == gi:
                w[v.index] = ge.weight
                break
    return w, True


# --------------------------------------------------------------- fixtures

def two_island_sphere(ctx, segments=48, rings=24, name="TwoIsland"):
    """Sphere whose UV layout is two far-apart planar islands split at the
    equator: every equator face is a seam and grabbing the wrong island puts
    the sample half a UV layout away.  Also carries two materials on the same
    split, an analytic Z-gradient weight and two shape keys."""
    obj = ctx.uv_sphere(segments=segments, rings=rings, name=name)
    me = obj.data
    uvl = me.uv_layers.active or me.uv_layers.new(name="UVMap")
    for p in me.polygons:
        up = p.center.z > 0.0
        for li in p.loop_indices:
            v = me.vertices[me.loops[li].vertex_index].co
            u, w = (v.x + 1.0) * 0.5, (v.y + 1.0) * 0.5
            uvl.data[li].uv = ((0.02 + 0.44 * u, 0.02 + 0.44 * w) if up
                               else (0.54 + 0.44 * u, 0.54 + 0.44 * w))
    m0 = bpy.data.materials.new("qf_isl_low")
    m1 = bpy.data.materials.new("qf_isl_high")
    me.materials.append(m0)
    me.materials.append(m1)
    for p in me.polygons:
        p.material_index = 1 if p.center.z > 0.0 else 0
    vg = obj.vertex_groups.new(name="qf_grad")
    for v in me.vertices:
        vg.add([v.index], (v.co.z + 1.0) * 0.5, 'REPLACE')
    obj.shape_key_add(name="Basis", from_mix=False)
    k = obj.shape_key_add(name="qf_bulge", from_mix=False)
    for i, v in enumerate(me.vertices):
        k.data[i].co = v.co * (1.0 + 0.3 * math.exp(-((v.co.z - 1.0) ** 2) / 0.2))
    k2 = obj.shape_key_add(name="qf_shift", from_mix=False)
    for i, v in enumerate(me.vertices):
        k2.data[i].co = v.co + Vector((0.15, 0.0, 0.0))
    for kb in me.shape_keys.key_blocks:
        kb.value = 0.0
    return obj


def crease_rim_cube(ctx, subdiv=3, name="CreaseRim"):
    """Cube whose top rim is creased 1.0 and bottom rim bevel-weighted 1.0.

    Both are single closed loops of exactly known length (8.0, the cube's
    perimeter at 2 units a side), which is what makes a per-edge nearest match
    obvious: one source ring comes back as a bush of stubs and junctions rather
    than one ring of comparable length.
    """
    obj = ctx.cube(size=2.0, subdiv=subdiv, name=name)
    me = obj.data
    ne = len(me.edges)
    EV = _fget(me.edges, "vertices", ne * 2, 'i4').reshape(-1, 2)
    V = _verts(me)
    cr = np.zeros(ne, 'f4')
    bw = np.zeros(ne, 'f4')
    rim = np.max(np.abs(V[:, :2]), axis=1) > 1.0 - 1e-6
    for i, (a, b) in enumerate(EV.tolist()):
        if not (rim[a] and rim[b]):
            continue
        if abs(V[a, 2] - 1.0) < 1e-6 and abs(V[b, 2] - 1.0) < 1e-6:
            cr[i] = 1.0
        elif abs(V[a, 2] + 1.0) < 1e-6 and abs(V[b, 2] + 1.0) < 1e-6:
            bw[i] = 1.0
    for attr, vals in (("crease_edge", cr), ("bevel_weight_edge", bw)):
        at = me.attributes.get(attr) or me.attributes.new(attr, 'FLOAT', 'EDGE')
        at.data.foreach_set("value", vals)
    return obj


def material_band_sphere(ctx, segments=64, rings=32, name="MatBand"):
    """Sphere carrying three materials split along a wandering band.

    The boundary wiggles at (and below) the size of an output face, which is
    where a single sample at the face centre stops being a good guess for the
    material that owns most of that face — real assets get this from material
    borders that follow the source mesh's own zig-zagging edges.
    """
    obj = ctx.uv_sphere(segments=segments, rings=rings, name=name)
    me = obj.data
    for nm in ("qf_band_a", "qf_band_b", "qf_band_c"):
        me.materials.append(bpy.data.materials.new(nm))
    for p in me.polygons:
        c = p.center
        th = math.atan2(c.y, c.x)
        w = (0.45 * math.sin(3.0 * th) + 0.15 * math.sin(7.0 * th)
             + 0.10 * math.sin(29.0 * th) + 0.06 * math.sin(53.0 * th))
        p.material_index = 1 if c.z > w else (2 if c.z < w - 0.8 else 0)
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    return obj


def edge_chain_stats(me, attr):
    """Topology of the subgraph an edge scalar marks: the shape of the chain.

    A transferred crease that double-assigns shows up here as many components
    with degree-3+ junctions and loose ends; one clean ring is comps=1,
    ends=0, junctions=0.
    """
    ne = len(me.edges)
    at = me.attributes.get(attr)
    out = {'n': 0, 'sum': 0.0, 'len': 0.0, 'comps': 0, 'ends': 0,
           'junctions': 0, 'maxdeg': 0}
    if at is None or ne == 0:
        return out
    v = np.empty(ne, 'f4')
    at.data.foreach_get("value", v)
    idx = np.nonzero(v > 0.0)[0]
    if idx.size == 0:
        return out
    EV = _fget(me.edges, "vertices", ne * 2, 'i4').reshape(-1, 2)
    V = _verts(me)
    e = EV[idx]
    vs, cnt = np.unique(e.ravel(), return_counts=True)
    parent = {int(x): int(x) for x in vs}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in e.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    out.update({
        'n': int(idx.size), 'sum': float(v[idx].sum()),
        'len': float(np.linalg.norm(V[e[:, 1]] - V[e[:, 0]], axis=1).sum()),
        'comps': len({find(int(x)) for x in vs}),
        'ends': int((cnt == 1).sum()), 'junctions': int((cnt > 2).sum()),
        'maxdeg': int(cnt.max()),
    })
    return out


def material_mismatch(src_me, out_me, seed=11, per_tri=4):
    """How much of the output surface carries the wrong material.

    Deterministic stratified samples over every output face are mapped to the
    nearest point of the source and compared with the material the face was
    assigned; area-weighted, so it reads as "how much of the model is wrong",
    not "how many faces".

    Returns (wrong_pct, majority_pct).  The second number is the part that was
    actually decidable: area sitting on a face whose assigned material is not
    even the one that owns most of that face.  A face split by a material
    boundary always costs the minority side no matter what is assigned, so
    ``wrong_pct`` can never reach zero, while ``majority_pct`` can.
    """
    sV = _verts(src_me)
    s_st, s_to, s_lv = _polys(src_me)
    _tl, s_tris, s_tp = _fan(s_st, s_to, s_lv)
    s_mat = _fget(src_me.polygons, "material_index", len(src_me.polygons), 'i4')
    bvh = _bvh(sV, s_tris)
    if bvh is None:
        return None

    oV = _verts(out_me)
    o_st, o_to, o_lv = _polys(out_me)
    o_mat = _fget(out_me.polygons, "material_index", len(out_me.polygons), 'i4')
    _otl, o_tris, o_tp = _fan(o_st, o_to, o_lv)
    A = oV[o_tris]
    area = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]),
                                axis=1)
    rng = np.random.default_rng(seed)
    r1 = np.sqrt(rng.random((A.shape[0], per_tri)))
    r2 = rng.random((A.shape[0], per_tri))
    P = (A[:, 0][:, None, :] * (1.0 - r1)[:, :, None]
         + A[:, 1][:, None, :] * (r1 * (1.0 - r2))[:, :, None]
         + A[:, 2][:, None, :] * (r1 * r2)[:, :, None]).reshape(-1, 3)
    w = np.repeat(area / per_tri, per_tri)
    face = np.repeat(o_tp, per_tri)
    fn = bvh.find_nearest
    hit = np.full(P.shape[0], -1, 'i8')
    for i, p in enumerate(P.tolist()):
        r = fn(p)
        if r is not None and r[2] is not None:
            hit[i] = r[2]
    want = s_mat[s_tp[np.where(hit >= 0, hit, 0)]]
    ok = hit >= 0
    bad = (want != o_mat[face]) & ok
    total = max(float(w.sum()), 1e-12)
    npo = len(out_me.polygons)
    nslots = int(max(want.max(initial=0), o_mat.max(initial=0))) + 1
    acc = np.zeros((npo, nslots), 'f8')
    np.add.at(acc, (face[ok].astype('i8'), want[ok].astype('i8')), w[ok])
    major = np.argmax(acc, axis=1)
    seen = acc.max(axis=1) > 0.0
    off = seen & (major != o_mat[:npo])
    return (100.0 * float(w[bad].sum()) / total,
            100.0 * float(w[off[face]].sum()) / total)


def uv_affine_backprojection(src_me, out_me):
    """Back-projection error measured against the *unclipped* source map."""
    ref = _UVRef(src_me)
    nl = len(out_me.loops)
    uvl = out_me.uv_layers.get(ref.name) or out_me.uv_layers.active
    if uvl is None or nl == 0:
        return {'n': 0, 'mean': 0.0, 'p99': 0.0, 'max': 0.0}
    a = np.empty(nl * 2, 'f4')
    uvl.data.foreach_get("uv", a)
    OUV = a.reshape(-1, 2).astype('f8')
    V = _verts(out_me)
    _st, _to, lv = _polys(out_me)
    own, err = ref.trace_affine(OUV, V[lv])
    best = np.full(nl, np.inf)
    if own.size:
        np.minimum.at(best, own, err)
    return _stat(np.where(np.isfinite(best), best, np.nan), _diag(ref.V))


def uv_island_incursion(src_me, out_me, cap=0.05):
    """How close transferred UVs get to an island they do not belong to.

    Returns (n_outside, worst_uv_shift, closest_foreign_island).  Bounded
    extrapolation is only honest if the corners it pushes past an island's
    outline stay clear of every other island's texels.
    """
    ref = _UVRef(src_me)
    nl = len(out_me.loops)
    uvl = out_me.uv_layers.get(ref.name) or out_me.uv_layers.active
    if uvl is None or nl == 0:
        return 0, 0.0, cap
    a = np.empty(nl * 2, 'f4')
    uvl.data.foreach_get("uv", a)
    OUV = a.reshape(-1, 2).astype('f8')
    isl_tri = ref.island[ref.tri_poly]
    bvh = ref.uvbvh()
    worst_out = 0.0
    closest = cap
    n_out = 0
    for i, uvp in enumerate(OUV.tolist()):
        q = (uvp[0], uvp[1], 0.0)
        r = bvh.find_nearest(q)
        if r is None or r[2] is None or r[3] is None or r[3] <= 1e-9:
            continue
        n_out += 1
        worst_out = max(worst_out, float(r[3]))
        own = isl_tri[int(r[2])]
        for hit in bvh.find_nearest_range(q, cap):
            if hit is None or hit[2] is None or hit[3] is None:
                continue
            if isl_tri[int(hit[2])] != own:
                closest = min(closest, float(hit[3]))
    return n_out, worst_out, closest


GAP = 0.02
RAD = 0.25


def crevice_pair(ctx, nseg=28, nring=20, name="Crevice"):
    """Two capped tubes along X, a gap of 8% of their radius apart.

    A plain nearest-point lookup in the crevice happily grabs the *other* tube,
    which is what makes an armpit fly apart when the arm comes down."""
    verts, faces = [], []
    groups = {0: [], 1: []}
    for gid, zoff in ((0, 0.0), (1, 2 * RAD + GAP)):
        base = len(verts)
        for i in range(nseg + 1):
            x = -1.0 + 2.0 * i / nseg
            for j in range(nring):
                a = 2 * math.pi * j / nring
                verts.append((x, RAD * math.cos(a), zoff + RAD * math.sin(a)))
                groups[gid].append(len(verts) - 1)
        for i in range(nseg):
            for j in range(nring):
                a = base + i * nring + j
                b = base + i * nring + (j + 1) % nring
                c = base + (i + 1) * nring + (j + 1) % nring
                d = base + (i + 1) * nring + j
                faces.append((a, b, c, d))
        for end, rev in ((0, False), (nseg, True)):
            ring = [base + end * nring + j for j in range(nring)]
            faces.append(tuple(ring[::-1] if rev else ring))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    uvl = me.uv_layers.new(name="UVMap")
    for p in me.polygons:
        hi = p.center.z > RAD + GAP * 0.5
        for li in p.loop_indices:
            v = me.vertices[me.loops[li].vertex_index].co
            ang = math.atan2(v.y, v.z - ((2 * RAD + GAP) if hi else 0.0))
            uvl.data[li].uv = ((v.x + 1.0) * 0.22 + 0.03,
                               (ang / (2 * math.pi) + 0.5) * 0.4
                               + (0.55 if hi else 0.03))
    for gid, gname in ((0, "tube_low"), (1, "tube_up")):
        vg = obj.vertex_groups.new(name=gname)
        for i in groups[gid]:
            vg.add([i], 1.0, 'REPLACE')
    return obj


def crevice_rig(ctx, mesh_obj):
    """Two-bone armature, one bone per tube, bound to `mesh_obj`."""
    arm_data = bpy.data.armatures.new("qf_arm")
    arm = bpy.data.objects.new("qf_arm", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    ctx.activate(arm)
    bpy.ops.object.mode_set(mode='EDIT')
    b0 = arm_data.edit_bones.new("tube_low")
    b0.head = (-1.0, 0.0, 0.0)
    b0.tail = (1.0, 0.0, 0.0)
    b1 = arm_data.edit_bones.new("tube_up")
    b1.head = (-1.0, 0.0, 2 * RAD + GAP)
    b1.tail = (1.0, 0.0, 2 * RAD + GAP)
    b1.parent = b0
    b1.use_connect = False
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh_obj.parent = arm
    m = mesh_obj.modifiers.new("Armature", 'ARMATURE')
    m.object = arm
    ctx.activate(mesh_obj)
    return arm


def _visible(obj):
    """Put `obj` in the scene collection, unhidden.

    keep_original stows the source in a hidden 'QuadForge Originals'
    collection, and objects the depsgraph cannot see do not evaluate their
    modifiers — every deformation number would silently read zero.
    """
    sc = bpy.context.scene.collection
    for coll in list(obj.users_collection):
        if coll is not sc:
            try:
                coll.objects.unlink(obj)
            except Exception:
                pass
    if obj.name not in sc.objects:
        sc.objects.link(obj)
    obj.hide_viewport = False
    obj.hide_render = False
    try:
        obj.hide_set(False)
    except Exception:
        pass
    return obj


def _bind(out_obj, arm):
    if out_obj.parent is not arm:
        out_obj.parent = arm
    for m in list(out_obj.modifiers):
        if m.type == 'ARMATURE':
            m.object = arm
            return
    m = out_obj.modifiers.new("Armature", 'ARMATURE')
    m.object = arm


def _remesh(ctx, obj, target, **over):
    s = ctx.settings(obj, backend='NATIVE', seed=SEED, mode='FACES',
                     target_count=target, keep_original=True, **over)
    ctx.activate(obj)
    return ctx.pipeline().run_remesh(bpy.context, obj, s)


# ------------------------------------------------------------------- tests

def run(ctx):
    r = ctx.results()
    st = {}

    # ------------------------------------------------ UV island fixture
    with r.case("island_remesh_ok") as c:
        ctx.fresh_scene()
        src = two_island_sphere(ctx)
        res = _remesh(ctx, src, 1500)
        c.require(res.get("ok") is True, "run failed: %r" % (res.get("error"),))
        out = res.get("object")
        c.require(ctx.is_mesh_valid(out), "no result mesh")
        st['src'] = src
        st['out'] = out
        st['res'] = res
        _visible(src)
        _visible(out)
        c.note("faces=%d" % len(out.data.polygons))

    with r.case("uv_islands_detected") as c:
        transfer = ctx.imp("quadforge.core.transfer")
        c.require(hasattr(transfer, "_uv_regions"),
                  "transfer._uv_regions missing (island-constrained UVs gone?)")
        snap = transfer.capture(st['src'])
        reg = transfer._uv_regions(snap)
        n = int(reg.max()) + 1 if reg.size else 0
        c.require(n == 2, "expected 2 UV islands on the fixture, got %d" % n)
        c.note("islands=%d over %d polys" % (n, reg.size))

    with r.case("uv_seam_bleed") as c:
        out = st.get('out')
        c.require(out is not None, "no result mesh (island_remesh_ok failed)")
        q = uv_quality(st['src'].data, out.data)
        st['uv'] = q
        c.require(q.get('ok'), "no UV layer on the result")
        c.require(q['islands'] == 2, "source islands=%d, want 2" % q['islands'])
        # this fixture before island-constrained UVs: 85/1516 faces, 5.61%
        c.require(q['bleed_pct'] <= 0.6,
                  "%d/%d faces (%.2f%%) are textured out of two UV islands"
                  % (q['bleed_faces'], q['faces'], q['bleed_pct']))
        c.note("bleed=%d/%d (%.2f%%)" % (q['bleed_faces'], q['faces'], q['bleed_pct']))

    with r.case("uv_texel_stretch") as c:
        q = st.get('uv')
        c.require(q is not None and q.get('ok'), "no UV measurement")
        # this fixture before island-constrained UVs: 85/1516 faces, 5.61%
        c.require(q['stretch_pct'] <= 0.6,
                  "%d/%d faces (%.2f%%) stretch past the source texel density"
                  % (q['stretch_faces'], q['faces'], q['stretch_pct']))
        c.note("stretch=%d (%.2f%%)" % (q['stretch_faces'], q['stretch_pct']))

    with r.case("uv_back_projection") as c:
        q = st.get('uv')
        c.require(q is not None and q.get('ok'), "no UV measurement")
        c.require(q['mean'] <= 2.0e-3,
                  "mean UV back-projection error %.5f of the bbox diagonal" % q['mean'])
        c.require(q['max'] <= 4.0e-2,
                  "worst UV back-projection error %.5f of the bbox diagonal" % q['max'])
        c.note("mean=%.6f p99=%.6f max=%.6f (of bbox diag)"
               % (q['mean'], q['p99'], q['max']))

    with r.case("transfer_reports_seams") as c:
        rep = (st.get('res') or {}).get('report', {}).get('transfer', {})
        c.require(rep, "pipeline report carries no transfer block")
        for k in ("uv_seam_faces", "uv_seam_faces_fixed", "uv_seam_corners_fixed"):
            c.require(k in rep, "transfer report missing %r" % k)
        c.require(rep['uv_seam_faces'] > 0,
                  "no face straddled a UV seam on a fixture built out of two "
                  "islands — the seam detector is not seeing them")
        c.require(rep['uv_seam_faces_fixed'] >= 0.9 * rep['uv_seam_faces'],
                  "only %d of %d straddling faces were pulled onto one island"
                  % (rep['uv_seam_faces_fixed'], rep['uv_seam_faces']))
        c.note("straddling=%d fixed=%d corners=%d"
               % (rep['uv_seam_faces'], rep['uv_seam_faces_fixed'],
                  rep['uv_seam_corners_fixed']))

    with r.case("weights_follow_analytic_gradient") as c:
        out = st.get('out')
        c.require(out is not None, "no result mesh")
        w, have = group_weights(out, "qf_grad")
        c.require(have, "vertex group 'qf_grad' missing")
        V = ctx.verts_np(out)
        err = np.abs(w - (V[:, 2] + 1.0) * 0.5)
        c.require(float(err.mean()) <= 0.02,
                  "mean weight error %.4f against the analytic gradient" % err.mean())
        c.require(float(err.max()) <= 0.08,
                  "worst weight error %.4f against the analytic gradient" % err.max())
        c.note("mean=%.5f max=%.5f over %d verts" % (err.mean(), err.max(), err.size))

    with r.case("shape_keys_deform_alike") as c:
        src, out = st.get('src'), st.get('out')
        c.require(out is not None, "no result mesh")
        scale = _diag(ctx.verts_np(src))
        ks, ko = src.data.shape_keys, out.data.shape_keys
        c.require(ks is not None and ko is not None, "shape keys missing")
        for kb in list(ks.key_blocks) + list(ko.key_blocks):
            kb.value = 0.0
        rest, rest_sV = _deviation(src, out, scale)
        worst = []
        for name in [k.name for k in ks.key_blocks[1:]]:
            c.require(name in ko.key_blocks, "shape key %r missing on the result" % name)
            ks.key_blocks[name].value = 1.0
            ko.key_blocks[name].value = 1.0
            d, sV = _deviation(src, out, scale)
            motion = float(np.linalg.norm(sV - rest_sV, axis=1).max()) / scale
            ks.key_blocks[name].value = 0.0
            ko.key_blocks[name].value = 0.0
            c.require(motion > 1e-4,
                      "shape key %r moves nothing — the comparison would be "
                      "measuring an unposed mesh" % name)
            worst.append((d['mean'] - rest['mean'], d['max'] - rest['max'], name))
        worst.sort(reverse=True)
        exc_mean, exc_max, name = worst[0]
        c.require(exc_mean <= 5.0e-3,
                  "shape key %r adds %.5f of the bbox diagonal to the mean "
                  "surface deviation (rest %.5f)" % (name, exc_mean, rest['mean']))
        c.require(exc_max <= 3.0e-2,
                  "shape key %r adds %.5f of the bbox diagonal to the worst "
                  "surface deviation" % (name, exc_max))
        c.note("rest mean=%.6f; worst key %s +%.6f mean +%.6f max"
               % (rest['mean'], name, exc_mean, exc_max))

    with r.case("materials_area_preserved") as c:
        src, out = st.get('src'), st.get('out')
        c.require(out is not None, "no result mesh")
        ns = len(src.data.materials)
        c.require(len(out.data.materials) == ns,
                  "%d material slots, source had %d" % (len(out.data.materials), ns))
        def frac(me):
            n = len(me.polygons)
            mi = _fget(me.polygons, "material_index", n, 'i4')
            ar = _fget(me.polygons, "area", n)
            return np.bincount(mi, weights=ar, minlength=ns) / max(ar.sum(), 1e-12)
        a, b = frac(src.data), frac(out.data)
        l1 = float(np.abs(a - b).sum())
        c.require(l1 <= 0.03,
                  "material area split drifted by %.4f (L1): %s vs %s"
                  % (l1, np.round(a, 4).tolist(), np.round(b, 4).tolist()))
        c.note("L1=%.5f src=%s out=%s"
               % (l1, np.round(a, 4).tolist(), np.round(b, 4).tolist()))

    # ------------------------------------------- bounded UV extrapolation
    with r.case("uv_extrapolation_is_bounded") as c:
        src, out = st.get('src'), st.get('out')
        c.require(out is not None, "no result mesh")
        transfer = ctx.imp("quadforge.core.transfer")
        c.require(hasattr(transfer, "_uv_clearance"),
                  "transfer._uv_clearance missing (bounded extrapolation gone?)")
        snap = transfer.capture(src)
        s = out.quadforge
        old = transfer._UV_EXTRAPOLATE
        try:
            transfer._UV_EXTRAPOLATE = False
            transfer.apply(snap, out, s)
            flat_aff = uv_affine_backprojection(src.data, out.data)
            flat_clip = uv_quality(src.data, out.data)
            transfer._UV_EXTRAPOLATE = True
            rep = transfer.apply(snap, out, s)
            ext_aff = uv_affine_backprojection(src.data, out.data)
            ext_clip = uv_quality(src.data, out.data)
        finally:
            transfer._UV_EXTRAPOLATE = old
        st['ext_rep'] = rep
        n_out, worst_shift, closest = uv_island_incursion(src.data, out.data)
        c.require(rep.get('uv_corners_extrapolated', 0) > 0,
                  "no corner extrapolated on a fixture with two islands "
                  "0.08 UV apart")
        # the whole point: a corner may leave its island, never reach another
        c.require(closest > 0.0,
                  "an extrapolated UV landed on another island (closest "
                  "approach %.6f UV)" % closest)
        c.require(worst_shift <= 0.5 * closest + 1e-9,
                  "worst UV excursion %.6f is more than half the %.6f clearance "
                  "to the nearest other island" % (worst_shift, closest))
        c.require(worst_shift <= 0.5 * transfer._UV_CLEARANCE_CAP + 1e-9,
                  "UV excursion %.6f exceeds the hard cap" % worst_shift)
        c.require(ext_clip['bleed_faces'] <= flat_clip['bleed_faces'],
                  "extrapolation created %d new two-island faces"
                  % (ext_clip['bleed_faces'] - flat_clip['bleed_faces']))
        # measured gain: the UV now continues the source map past the island
        c.require(ext_aff['p99'] <= 0.5 * flat_aff['p99'],
                  "affine back-projection p99 %.3e is no better than the "
                  "clamped %.3e" % (ext_aff['p99'], flat_aff['p99']))
        c.note("ext=%d worst=%.6f clearance>=%.6f; affine p99 %.3e->%.3e "
               "max %.3e->%.3e; clamped p99 %.3e->%.3e"
               % (rep.get('uv_corners_extrapolated', 0), worst_shift, closest,
                  flat_aff['p99'], ext_aff['p99'], flat_aff['max'],
                  ext_aff['max'], flat_clip['p99'], ext_clip['p99']))

    with r.case("uv_extrapolation_off_without_clearance") as c:
        # islands packed with no room (the reference avatar's layout) must come
        # out exactly as before: no room measured, no excursion taken
        src, out = st.get('src'), st.get('out')
        c.require(out is not None, "no result mesh")
        transfer = ctx.imp("quadforge.core.transfer")
        snap = transfer.capture(src)
        real = transfer._uv_clearance
        try:
            transfer._uv_clearance = lambda sn, li, pts, own: np.zeros(len(pts))
            rep = transfer.apply(snap, out, st['out'].quadforge)
            packed = uv_quality(src.data, out.data)
        finally:
            transfer._uv_clearance = real
        c.require(rep.get('uv_extrapolation_max', 0.0) <= 1e-9,
                  "a layout with zero clearance still moved a corner by %.3e UV"
                  % rep.get('uv_extrapolation_max', 0.0))
        n_out, worst_shift, _closest = uv_island_incursion(src.data, out.data)
        # UVs are stored as float32: a corner sitting exactly on an island's
        # outline quantises to a few ulps outside it, which is not an excursion
        c.require(worst_shift <= 1e-6,
                  "corners left the layout (%.3e UV) with no clearance to spend"
                  % worst_shift)
        c.note("no-clearance: bleed=%d worst_shift=%.3e"
               % (packed['bleed_faces'], worst_shift))
        transfer.apply(snap, out, st['out'].quadforge)

    # ------------------------------------------------ crevice / rig fixture
    with r.case("crevice_remesh_ok") as c:
        ctx.fresh_scene()
        src = crevice_pair(ctx)
        try:
            arm = crevice_rig(ctx, src)
        except Exception as exc:
            c.skip("armature fixture unavailable headless: %s" % exc)
        res = _remesh(ctx, src, 1200)
        c.require(res.get("ok") is True, "run failed: %r" % (res.get("error"),))
        out = res.get("object")
        c.require(ctx.is_mesh_valid(out), "no result mesh")
        _visible(src)
        _visible(out)
        _visible(arm)
        _bind(out, arm)
        st['c_src'], st['c_out'], st['c_arm'] = src, out, arm
        c.note("faces=%d (in %d)" % (len(out.data.polygons), len(src.data.polygons)))

    with r.case("crevice_weights_stay_on_their_shell") as c:
        out = st.get('c_out')
        if out is None:
            c.skip("crevice fixture unavailable")
        lo, _h1 = group_weights(out, "tube_low")
        up, _h2 = group_weights(out, "tube_up")
        V = ctx.verts_np(out)
        want_up = V[:, 2] > RAD + GAP * 0.5
        got_up = up > lo
        bad = int((want_up != got_up).sum())
        # a vertex must also be *committed*: a 50/50 blend across the crevice
        # is just as broken as the wrong bone
        blend = float(np.mean(np.minimum(lo, up) > 0.15))
        c.require(bad <= 0.01 * V.shape[0],
                  "%d/%d vertices take their dominant weight from the other tube"
                  % (bad, V.shape[0]))
        c.require(blend <= 0.02,
                  "%.2f%% of vertices are blended across the crevice between the tubes"
                  % (blend * 100.0))
        c.note("wrong=%d/%d blended=%.3f%%" % (bad, V.shape[0], blend * 100.0))

    with r.case("posed_deformation_matches") as c:
        src, out, arm = st.get('c_src'), st.get('c_out'), st.get('c_arm')
        if out is None:
            c.skip("crevice fixture unavailable")
        scale = _diag(ctx.verts_np(src))
        rest, rest_sV = _deviation(src, out, scale)
        pb = arm.pose.bones.get("tube_up")
        c.require(pb is not None, "rig lost its 'tube_up' bone")
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (0.0, 0.9, 0.0)
        try:
            posed, sV = _deviation(src, out, scale)
        finally:
            pb.rotation_euler = (0.0, 0.0, 0.0)
        motion = float(np.linalg.norm(sV - rest_sV, axis=1).max()) / scale
        c.require(motion > 0.05,
                  "the pose moved the original by only %.4f of the bbox diagonal "
                  "— nothing was actually evaluated (hidden collection?)" % motion)
        c.require(posed['mean'] - rest['mean'] <= 5.0e-3,
                  "posing adds %.5f of the bbox diagonal to the mean deviation "
                  "(rest %.5f, posed %.5f)"
                  % (posed['mean'] - rest['mean'], rest['mean'], posed['mean']))
        c.require(posed['max'] - rest['max'] <= 4.0e-2,
                  "posing adds %.5f of the bbox diagonal to the worst deviation"
                  % (posed['max'] - rest['max']))
        c.note("motion=%.4f rest mean=%.6f posed mean=%.6f (+%.6f) max +%.6f"
               % (motion, rest['mean'], posed['mean'],
                  posed['mean'] - rest['mean'], posed['max'] - rest['max']))

    with r.case("crevice_uv_seam_bleed") as c:
        src, out = st.get('c_src'), st.get('c_out')
        if out is None:
            c.skip("crevice fixture unavailable")
        q = uv_quality(src.data, out.data)
        c.require(q.get('ok'), "no UV layer on the result")
        c.require(q['bleed_pct'] <= 0.6,
                  "%d/%d faces (%.2f%%) textured out of two UV islands"
                  % (q['bleed_faces'], q['faces'], q['bleed_pct']))
        c.note("islands=%d bleed=%d/%d (%.2f%%) mean=%.6f"
               % (q['islands'], q['bleed_faces'], q['faces'], q['bleed_pct'],
                  q['mean']))

    # ------------------------------------------- crease / bevel chains
    with r.case("crease_chains_stay_chains") as c:
        transfer = ctx.imp("quadforge.core.transfer")
        rows = []
        for subdiv, target in ((3, 1500), (3, 400), (4, 300)):
            ctx.fresh_scene()
            src = crease_rim_cube(ctx, subdiv)
            res = _remesh(ctx, src, target)
            c.require(res.get("ok") is True,
                      "subdiv=%d target=%d failed: %r"
                      % (subdiv, target, res.get("error")))
            out = res.get("object")
            c.require(ctx.is_mesh_valid(out), "no result mesh")
            for attr in ("crease_edge", "bevel_weight_edge"):
                a = edge_chain_stats(src.data, attr)
                b = edge_chain_stats(out.data, attr)
                tag = "%s subdiv=%d target=%d" % (attr, subdiv, target)
                c.require(a['comps'] == 1 and a['ends'] == 0,
                          "%s: the fixture's own ring is not a ring" % tag)
                c.require(b['n'] > 0, "%s: nothing transferred" % tag)
                c.require(b['comps'] == 1,
                          "%s: one source ring came out as %d pieces"
                          % (tag, b['comps']))
                c.require(b['junctions'] == 0,
                          "%s: %d vertices carry 3+ marked edges (the source "
                          "ring has none) — edges were double-assigned"
                          % (tag, b['junctions']))
                c.require(b['ends'] == 0,
                          "%s: a closed ring came out with %d loose ends"
                          % (tag, b['ends']))
                ratio = b['len'] / max(a['len'], 1e-12)
                c.require(0.8 <= ratio <= 1.3,
                          "%s: transferred ring is %.2fx the source's length"
                          % (tag, ratio))
                c.require(abs(b['sum'] - b['n']) < 1e-3,
                          "%s: weights drifted (sum %.3f over %d edges, all "
                          "source values are 1.0)" % (tag, b['sum'], b['n']))
                rows.append("%s %d/%d len=%.2fx" % (attr[:6], b['n'], a['n'], ratio))
            st['crease_src'], st['crease_out'] = src, out
        c.note("; ".join(rows))

    with r.case("crease_chains_beat_nearest_edge") as c:
        src, out = st.get('crease_src'), st.get('crease_out')
        c.require(out is not None, "no crease fixture")
        transfer = ctx.imp("quadforge.core.transfer")
        c.require(hasattr(transfer, "_route_edge_scalar"),
                  "transfer._route_edge_scalar missing (chain routing gone?)")
        snap = transfer.capture(src)
        s = out.quadforge
        real = transfer._route_edge_scalar
        try:
            transfer._route_edge_scalar = lambda *a, **k: None   # legacy match
            transfer.apply(snap, out, s)
            legacy = edge_chain_stats(out.data, "crease_edge")
        finally:
            transfer._route_edge_scalar = real
        transfer.apply(snap, out, s)
        chain = edge_chain_stats(out.data, "crease_edge")
        c.require(chain['comps'] <= legacy['comps'],
                  "chain routing (%d pieces) is not tidier than the nearest-edge "
                  "match (%d pieces)" % (chain['comps'], legacy['comps']))
        c.require(chain['junctions'] <= legacy['junctions'],
                  "chain routing left %d junctions, nearest-edge %d"
                  % (chain['junctions'], legacy['junctions']))
        c.require(legacy['comps'] > 1 or legacy['junctions'] > 0,
                  "the nearest-edge match came out clean on this fixture — the "
                  "comparison proves nothing, pick a harder one")
        c.note("nearest-edge: %d edges, %d pieces, %d junctions -> chain: "
               "%d edges, %d pieces, %d junctions"
               % (legacy['n'], legacy['comps'], legacy['junctions'],
                  chain['n'], chain['comps'], chain['junctions']))

    # ------------------------------------------- material point sampling
    with r.case("materials_point_sampling") as c:
        ctx.fresh_scene()
        transfer = ctx.imp("quadforge.core.transfer")
        src = material_band_sphere(ctx)
        res = _remesh(ctx, src, 1200)
        c.require(res.get("ok") is True, "run failed: %r" % (res.get("error"),))
        out = res.get("object")
        c.require(ctx.is_mesh_valid(out), "no result mesh")
        _visible(src)
        _visible(out)
        snap = transfer.capture(src)
        s = out.quadforge
        old = transfer._MATERIAL_SAMPLES
        try:
            transfer._MATERIAL_SAMPLES = 0       # face centre only (pre-0.6.1)
            transfer.apply(snap, out, s)
            c_wrong, c_major = material_mismatch(src.data, out.data)
            transfer._MATERIAL_SAMPLES = old
            rep = transfer.apply(snap, out, s)
            v_wrong, v_major = material_mismatch(src.data, out.data)
        finally:
            transfer._MATERIAL_SAMPLES = old
        # a face split by a boundary always costs its minority side, so the
        # decidable part is "is the face under the material that owns most of
        # it": avatar 46031 -> 35395 faces, 0.627% -> 0.409% of the surface
        # wrong, and 0.501% -> 0.286% of it away from any boundary at all
        c.require(v_major <= 0.75 * c_major,
                  "area-weighted voting puts %.3f%% of the surface under a "
                  "material that does not own its face; face-centre sampling "
                  "managed %.3f%%" % (v_major, c_major))
        c.require(v_wrong <= c_wrong,
                  "area-weighted voting (%.3f%% of the surface wrong) is worse "
                  "than face-centre sampling (%.3f%%)" % (v_wrong, c_wrong))
        c.require(v_major <= 3.0,
                  "%.3f%% of the surface is not under its face's own majority "
                  "material" % v_major)
        c.require(rep.get('material_samples', 0) > len(out.data.polygons),
                  "the material vote used %r samples for %d faces — that is "
                  "still one point per face"
                  % (rep.get('material_samples'), len(out.data.polygons)))
        c.note("centre wrong=%.3f%% major=%.3f%% -> vote wrong=%.3f%% "
               "major=%.3f%% (%d samples, %d faces refined)"
               % (c_wrong, c_major, v_wrong, v_major,
                  rep.get('material_samples', 0),
                  rep.get('material_faces_refined', 0)))

    return r.list()
