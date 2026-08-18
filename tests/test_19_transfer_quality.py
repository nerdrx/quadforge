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

    return r.list()
