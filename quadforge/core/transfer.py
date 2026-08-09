"""QuadForge — data preservation (core/transfer.py).

Captures everything that lives on a mesh *before* remeshing and re-projects it
onto the new topology afterwards using a surface-nearest / barycentric mapping.

Public API (see CONTRACTS.md):

    capture(obj) -> Snapshot
    apply(snapshot, new_obj, s) -> dict

The snapshot is a plain Python object holding numpy arrays only (plus material
datablock references, which survive the mesh swap and are re-resolved by name if
they do not).  No mesh / object / key datablock is kept alive by a Snapshot.

Everything is vectorised with numpy + foreach_get / foreach_set; the only Python
level loops are the BVH nearest-surface queries (~2.5 M queries/s) and the
vertex-group write-back (~1.6 M weights/s), both of which are fast enough for
multi-hundred-thousand vertex meshes.
"""

from __future__ import annotations

import time

import numpy as np

import bpy
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

__all__ = ["Snapshot", "capture", "apply"]


# Generic attribute names Blender 4.1+ / 5.x uses for creases & bevel weights.
CREASE_EDGE = "crease_edge"
CREASE_VERT = "crease_vert"
BWEIGHT_EDGE = "bevel_weight_edge"
BWEIGHT_VERT = "bevel_weight_vert"

_EPS = 1e-12
_WEIGHT_EPS = 1e-6
# How far a loop's sample point is pulled towards its face centre when looking
# up UVs.  Large enough to land on the correct side of a UV seam, small enough
# not to skip across a whole source face.
_LOOP_INSET = 0.08


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _fget(coll, attr, size, dtype):
    a = np.empty(size, dtype)
    coll.foreach_get(attr, a)
    return a


def _attr_array(mesh, name, domain, count):
    """Read a FLOAT attribute as (count,) f4, or None when absent/empty."""
    at = mesh.attributes.get(name)
    if at is None or at.domain != domain or at.data_type != 'FLOAT':
        return None
    if len(at.data) != count or count == 0:
        return None
    a = np.empty(count, 'f4')
    at.data.foreach_get("value", a)
    if not np.any(a):
        return None
    return a


def _ensure_float_attr(mesh, name, domain, count):
    at = mesh.attributes.get(name)
    if at is not None and (at.domain != domain or at.data_type != 'FLOAT'):
        return None
    if at is None:
        at = mesh.attributes.new(name, 'FLOAT', domain)
    if len(at.data) != count:
        return None
    return at


def _poly_arrays(mesh):
    npoly = len(mesh.polygons)
    nloop = len(mesh.loops)
    starts = _fget(mesh.polygons, "loop_start", npoly, 'i4')
    totals = _fget(mesh.polygons, "loop_total", npoly, 'i4')
    lverts = _fget(mesh.loops, "vertex_index", nloop, 'i4')
    return starts, totals, lverts


def _triangulate(starts, totals, loop_verts):
    """Fan-triangulate polygons.  Fully vectorised.

    Returns (tri_loops (nt,3) i4, tris (nt,3) i4, tri_poly (nt,) i4).
    """
    if starts.size == 0:
        z3 = np.zeros((0, 3), 'i4')
        return z3, z3, np.zeros(0, 'i4')
    per = np.maximum(totals - 2, 0).astype('i8')
    total = int(per.sum())
    if total == 0:
        z3 = np.zeros((0, 3), 'i4')
        return z3, z3, np.zeros(0, 'i4')
    poly_idx = np.repeat(np.arange(starts.size, dtype='i4'), per)
    ends = np.cumsum(per)
    k = np.arange(total, dtype='i8') - np.repeat(ends - per, per) + 1
    s = starts[poly_idx].astype('i8')
    tri_loops = np.stack([s, s + k, s + k + 1], axis=1).astype('i4')
    tris = loop_verts[tri_loops]
    return tri_loops, tris, poly_idx


def _poly_centers(mesh):
    npoly = len(mesh.polygons)
    if npoly == 0:
        return np.zeros((0, 3), 'f8')
    try:
        c = _fget(mesh.polygons, "center", npoly * 3, 'f8')
        return c.reshape(-1, 3)
    except Exception:
        starts, totals, lverts = _poly_arrays(mesh)
        V = _fget(mesh.vertices, "co", len(mesh.vertices) * 3, 'f8').reshape(-1, 3)
        out = np.zeros((npoly, 3), 'f8')
        acc = np.zeros((npoly, 3), 'f8')
        pidx = np.repeat(np.arange(npoly), totals)
        np.add.at(acc, pidx, V[lverts])
        out[:] = acc / np.maximum(totals, 1)[:, None]
        return out


def _material_ok(mat):
    if mat is None:
        return False
    try:
        mat.name
        return True
    except ReferenceError:
        return False


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class Snapshot:
    """Standalone (numpy-only) copy of everything worth preserving."""

    def __init__(self):
        self.object_name = ""
        self.mesh_name = ""
        self.warnings = []

        # geometry -------------------------------------------------------
        self.verts = np.zeros((0, 3), 'f8')        # surface positions (BVH)
        self.base_verts = np.zeros((0, 3), 'f8')   # base (shape-key space)
        self.loop_starts = np.zeros(0, 'i4')
        self.loop_totals = np.zeros(0, 'i4')
        self.loop_verts = np.zeros(0, 'i4')
        self.edges = np.zeros((0, 2), 'i4')
        self.tri_loops = np.zeros((0, 3), 'i4')
        self.tris = np.zeros((0, 3), 'i4')
        self.tri_poly = np.zeros(0, 'i4')

        # materials ------------------------------------------------------
        self.materials = []            # list of (Material|None, name|"")
        self.poly_material = np.zeros(0, 'i4')

        # uvs ------------------------------------------------------------
        self.uv_layers = []            # [{'name','uv'(nl,2)f4,'active','active_render'}]

        # vertex groups --------------------------------------------------
        self.vgroups = []              # [{'name','weights'(nv,)f4,'lock'}]

        # shape keys -----------------------------------------------------
        self.shape_keys = []           # [{...,'delta'(nv,3)f4}]
        self.key_name = ""
        self.key_use_relative = True
        self.key_reference_name = ""

        # edge / vertex scalar data -------------------------------------
        self.crease_edge = None
        self.crease_vert = None
        self.bweight_edge = None
        self.bweight_vert = None
        self.sharp_edge = None

        self.has_custom_normals = False

        self._bvh = None

    # -- derived ---------------------------------------------------------
    @property
    def nverts(self):
        return self.verts.shape[0]

    @property
    def nloops(self):
        return self.loop_verts.shape[0]

    @property
    def npolys(self):
        return self.loop_starts.shape[0]

    def bvh(self):
        """Lazily built BVH over the fan-triangulated snapshot surface."""
        if self._bvh is None:
            if self.tris.shape[0] == 0:
                return None
            self._bvh = BVHTree.FromPolygons(
                self.verts.tolist(),
                [tuple(t) for t in self.tris.tolist()],
                all_triangles=True,
            )
        return self._bvh

    def free(self):
        self._bvh = None

    def __repr__(self):
        return (
            "<qf.Snapshot %r verts=%d polys=%d uv=%d vg=%d keys=%d mats=%d>"
            % (self.object_name, self.nverts, self.npolys, len(self.uv_layers),
               len(self.vgroups), len(self.shape_keys), len(self.materials))
        )


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

def capture(obj, use_evaluated=True):
    """Snapshot everything preservable on ``obj`` before it gets remeshed.

    Geometry, UVs, vertex groups and shape keys are read from the *base* mesh so
    that every array shares one vertex indexing.  When the object additionally
    has purely deforming modifiers (evaluated mesh has identical vertex / loop
    counts) the evaluated positions are used for the BVH surface, so the mapping
    matches the surface the remesher actually saw.
    """
    snap = Snapshot()
    if obj is None or obj.type != 'MESH' or obj.data is None:
        snap.warnings.append("capture: not a mesh object")
        return snap

    me = obj.data
    snap.object_name = obj.name
    snap.mesh_name = me.name

    nv = len(me.vertices)
    nl = len(me.loops)
    ne = len(me.edges)
    npo = len(me.polygons)

    # --- geometry -------------------------------------------------------
    base = _fget(me.vertices, "co", nv * 3, 'f8').reshape(-1, 3)
    snap.base_verts = base
    snap.verts = base
    snap.loop_starts, snap.loop_totals, snap.loop_verts = _poly_arrays(me)
    if ne:
        snap.edges = _fget(me.edges, "vertices", ne * 2, 'i4').reshape(-1, 2)
    snap.tri_loops, snap.tris, snap.tri_poly = _triangulate(
        snap.loop_starts, snap.loop_totals, snap.loop_verts)

    # --- evaluated surface (deform-only modifiers) ----------------------
    if use_evaluated and obj.modifiers:
        try:
            dg = bpy.context.evaluated_depsgraph_get()
            ev = obj.evaluated_get(dg)
            em = ev.to_mesh()
            if em is not None and len(em.vertices) == nv and len(em.loops) == nl:
                snap.verts = _fget(em.vertices, "co", nv * 3, 'f8').reshape(-1, 3)
            elif em is not None:
                snap.warnings.append(
                    "modifiers change topology (%d->%d verts); mapping uses the "
                    "base mesh" % (nv, len(em.vertices)))
            ev.to_mesh_clear()
        except Exception as ex:                                  # pragma: no cover
            snap.warnings.append("evaluated mesh unavailable: %s" % ex)

    # --- materials ------------------------------------------------------
    for slot in obj.material_slots:
        mat = slot.material
        snap.materials.append((mat, mat.name if _material_ok(mat) else ""))
    if not snap.materials:
        for mat in me.materials:
            snap.materials.append((mat, mat.name if _material_ok(mat) else ""))
    if npo:
        snap.poly_material = _fget(me.polygons, "material_index", npo, 'i4')

    # --- uv layers ------------------------------------------------------
    active_name = me.uv_layers.active.name if me.uv_layers.active else ""
    render_name = ""
    for uvl in me.uv_layers:
        if getattr(uvl, "active_render", False):
            render_name = uvl.name
    for uvl in me.uv_layers:
        uv = np.empty(nl * 2, 'f4')
        uvl.data.foreach_get("uv", uv)
        snap.uv_layers.append({
            'name': uvl.name,
            'uv': uv.reshape(-1, 2),
            'active': uvl.name == active_name,
            'active_render': uvl.name == render_name,
        })

    # --- vertex groups --------------------------------------------------
    ngroups = len(obj.vertex_groups)
    if ngroups and nv:
        W = np.zeros((ngroups, nv), 'f4')
        for i, v in enumerate(me.vertices):
            for g in v.groups:
                gi = g.group
                if 0 <= gi < ngroups:
                    W[gi, i] = g.weight
        for gi, vg in enumerate(obj.vertex_groups):
            snap.vgroups.append({
                'name': vg.name,
                'weights': W[gi],
                'lock': bool(vg.lock_weight),
            })

    # --- shape keys -----------------------------------------------------
    key = me.shape_keys
    if key is not None and key.key_blocks and nv:
        snap.key_name = key.name
        snap.key_use_relative = bool(key.use_relative)
        ref = key.reference_key
        snap.key_reference_name = ref.name if ref else key.key_blocks[0].name
        basis = np.empty(nv * 3, 'f4')
        (ref or key.key_blocks[0]).data.foreach_get("co", basis)
        basis = basis.reshape(-1, 3)
        for kb in key.key_blocks:
            co = np.empty(nv * 3, 'f4')
            kb.data.foreach_get("co", co)
            co = co.reshape(-1, 3)
            snap.shape_keys.append({
                'name': kb.name,
                'is_reference': (ref is not None and kb == ref),
                'delta': (co - basis),
                'value': float(kb.value),
                'slider_min': float(kb.slider_min),
                'slider_max': float(kb.slider_max),
                'mute': bool(kb.mute),
                'interpolation': kb.interpolation,
                'relative_key': kb.relative_key.name if kb.relative_key else "",
                'vertex_group': kb.vertex_group or "",
            })

    # --- creases / bevel weights / sharp --------------------------------
    snap.crease_edge = _attr_array(me, CREASE_EDGE, 'EDGE', ne)
    snap.crease_vert = _attr_array(me, CREASE_VERT, 'POINT', nv)
    snap.bweight_edge = _attr_array(me, BWEIGHT_EDGE, 'EDGE', ne)
    snap.bweight_vert = _attr_array(me, BWEIGHT_VERT, 'POINT', nv)

    at = me.attributes.get("sharp_edge")
    if at is not None and at.domain == 'EDGE' and at.data_type == 'BOOLEAN' and ne:
        b = np.empty(ne, '?')
        at.data.foreach_get("value", b)
        snap.sharp_edge = b if b.any() else None

    try:
        snap.has_custom_normals = bool(me.has_custom_normals)
    except AttributeError:
        snap.has_custom_normals = me.attributes.get("custom_normal") is not None

    return snap


# ---------------------------------------------------------------------------
# surface mapping
# ---------------------------------------------------------------------------

def _nearest_tris(snap, points):
    """Nearest-surface query.  points (n,3) -> (tri_idx (n,) i4, loc (n,3) f8).

    tri_idx is -1 where nothing was hit.
    """
    n = points.shape[0]
    tri_idx = np.full(n, -1, 'i4')
    loc = np.array(points, dtype='f8', copy=True)
    bvh = snap.bvh()
    if bvh is None or n == 0:
        return tri_idx, loc
    fn = bvh.find_nearest
    ti = tri_idx
    lo = loc
    for i, p in enumerate(points.tolist()):
        r = fn(p)
        if r is not None and r[2] is not None:
            ti[i] = r[2]
            lo[i] = r[0]
    return tri_idx, loc


def _bary(snap, tri_idx, points):
    """Barycentric coords of the point on each triangle closest to ``points``.

    This is the exact closest-point-on-triangle solve (Ericson, *Real-Time
    Collision Detection*, §5.1.5) vectorised over all points.  Using it instead
    of "project onto the plane and clip the weights" matters a lot: clipping and
    renormalising slides the sample along the wrong direction whenever a point
    falls outside its triangle, which happens constantly where the source mesh
    is dense and curved (sphere poles, creased corners).

    Returns (n,3) f8 weights that are always >= 0 and sum to 1.
    """
    n = points.shape[0]
    w = np.zeros((n, 3), 'f8')
    w[:, 0] = 1.0
    if n == 0:
        return w
    ok = tri_idx >= 0
    if not ok.any():
        return w

    t = snap.tris[tri_idx[ok]]
    V = snap.verts
    a = V[t[:, 0]]
    b = V[t[:, 1]]
    c = V[t[:, 2]]
    p = points[ok]

    ab = b - a
    ac = c - a
    dot = lambda x, y: np.einsum('ij,ij->i', x, y)      # noqa: E731

    ap = p - a
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def _div(num, den):
        return num / np.where(np.abs(den) < _EPS, 1.0, den)

    den = va + vb + vc
    iv = _div(vb, den)
    iw = _div(vc, den)
    sub = np.stack([1.0 - iv - iw, iv, iw], axis=1)     # interior (default)

    z = np.zeros(d1.shape, 'f8')
    one = np.ones(d1.shape, 'f8')
    e_ab = np.clip(_div(d1, d1 - d3), 0.0, 1.0)
    e_ac = np.clip(_div(d2, d2 - d6), 0.0, 1.0)
    e_bc = np.clip(_div(d4 - d3, (d4 - d3) + (d5 - d6)), 0.0, 1.0)

    regions = (
        ((d1 <= 0) & (d2 <= 0),                          np.stack([one, z, z], 1)),
        ((d3 >= 0) & (d4 <= d3),                         np.stack([z, one, z], 1)),
        ((d6 >= 0) & (d5 <= d6),                         np.stack([z, z, one], 1)),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0),              np.stack([1.0 - e_ab, e_ab, z], 1)),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0),              np.stack([1.0 - e_ac, z, e_ac], 1)),
        ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), np.stack([z, 1.0 - e_bc, e_bc], 1)),
    )
    # Later regions are overwritten by earlier ones -> apply in reverse order.
    for cond, val in reversed(regions):
        sub = np.where(cond[:, None], val, sub)

    np.clip(sub, 0.0, None, out=sub)
    ssum = sub.sum(axis=1)
    degen = ~np.isfinite(ssum) | (ssum < _EPS)
    sub[degen] = (1.0, 0.0, 0.0)
    ssum = np.where(degen, 1.0, ssum)
    sub /= ssum[:, None]

    w[ok] = sub
    return w


def _interp_vert(snap, tri_idx, bary, arr):
    """Barycentric interpolation of a per-vertex array (nv,) or (nv,k)."""
    safe = np.where(tri_idx >= 0, tri_idx, 0)
    t = snap.tris[safe]
    if arr.ndim == 1:
        return np.einsum('ij,ij->i', arr[t].astype('f8'), bary)
    return np.einsum('ijk,ij->ik', arr[t].astype('f8'), bary)


def _interp_loop(snap, tri_idx, bary, arr):
    """Barycentric interpolation of a per-loop array (nl,k) using the hit
    triangle's *own* corners — this is what keeps UV seams crisp."""
    safe = np.where(tri_idx >= 0, tri_idx, 0)
    t = snap.tri_loops[safe]
    return np.einsum('ijk,ij->ik', arr[t].astype('f8'), bary)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def apply(snapshot, new_obj, s=None):
    """Re-project ``snapshot`` onto ``new_obj`` (in place).

    ``s`` is an ``obj.quadforge`` settings group (or anything exposing the same
    ``preserve_*`` booleans); ``None`` means "preserve everything".
    """
    t0 = time.perf_counter()
    rep = {
        'ok': False,
        'uvs': False,
        'uv_layers': 0,
        'weights': 0,
        'shape_keys': 0,
        'materials': 0,
        'creases': 0,
        'creases_skipped': 0,
        'bevel_weights': 0,
        'bevel_weights_skipped': 0,
        'unmapped': 0,
        'has_custom_normals': False,
        'time_s': 0.0,
        'warnings': list(getattr(snapshot, 'warnings', ())),
    }

    def _flag(name):
        return True if s is None else bool(getattr(s, name, True))

    if snapshot is None or new_obj is None or new_obj.type != 'MESH':
        rep['warnings'].append("apply: invalid target object")
        rep['time_s'] = time.perf_counter() - t0
        return rep

    me = new_obj.data
    nv = len(me.vertices)
    nl = len(me.loops)
    ne = len(me.edges)
    npo = len(me.polygons)
    rep['has_custom_normals'] = bool(snapshot.has_custom_normals)

    if snapshot.nverts == 0 or snapshot.tris.shape[0] == 0:
        rep['warnings'].append("apply: empty snapshot, nothing transferred")
        rep['time_s'] = time.perf_counter() - t0
        return rep
    if nv == 0:
        rep['warnings'].append("apply: target mesh is empty")
        rep['time_s'] = time.perf_counter() - t0
        return rep

    NV = _fget(me.vertices, "co", nv * 3, 'f8').reshape(-1, 3)

    # ---- per-vertex mapping -------------------------------------------
    v_tri, v_loc = _nearest_tris(snapshot, NV)
    rep['unmapped'] = int((v_tri < 0).sum())
    # Barycentric coords of the *vertex itself* projected onto its triangle
    # (not of the projected point) so interpolation follows the vertex, and is
    # exact for vertices that lie on the source surface.
    v_bary = _bary(snapshot, v_tri, NV)

    # ---- vertex groups -------------------------------------------------
    if _flag('preserve_weights') and snapshot.vgroups:
        for g in snapshot.vgroups:
            w = _interp_vert(snapshot, v_tri, v_bary, g['weights'])
            np.clip(w, 0.0, 1.0, out=w)
            vg = new_obj.vertex_groups.get(g['name'])
            if vg is None:
                vg = new_obj.vertex_groups.new(name=g['name'])
            idx = np.nonzero(w > _WEIGHT_EPS)[0]
            add = vg.add
            for i in idx.tolist():
                add((i,), float(w[i]), 'REPLACE')
            try:
                vg.lock_weight = g['lock']
            except Exception:
                pass
            rep['weights'] += 1

    # ---- shape keys ----------------------------------------------------
    if _flag('preserve_shape_keys') and snapshot.shape_keys:
        rep['shape_keys'] = _apply_shape_keys(snapshot, new_obj, me, nv,
                                              NV, v_tri, v_bary, rep)

    # ---- UVs (per loop, seam aware) ------------------------------------
    if _flag('preserve_uvs') and snapshot.uv_layers and nl:
        _apply_uvs(snapshot, me, nl, npo, NV, rep)

    # ---- materials -----------------------------------------------------
    if _flag('preserve_materials') and snapshot.materials:
        rep['materials'] = _apply_materials(snapshot, new_obj, me, npo, rep)

    # ---- creases / bevel weights ---------------------------------------
    do_cr = _flag('preserve_creases')
    do_bw = _flag('preserve_bevel_weights')
    if (do_cr or do_bw) and ne:
        EV = _fget(me.edges, "vertices", ne * 2, 'i4').reshape(-1, 2)
        if do_cr and snapshot.crease_edge is not None:
            n_ok, n_skip = _transfer_edge_scalar(
                snapshot, me, EV, NV, snapshot.crease_edge, CREASE_EDGE)
            rep['creases'] += n_ok
            rep['creases_skipped'] += n_skip
        if do_bw and snapshot.bweight_edge is not None:
            n_ok, n_skip = _transfer_edge_scalar(
                snapshot, me, EV, NV, snapshot.bweight_edge, BWEIGHT_EDGE)
            rep['bevel_weights'] += n_ok
            rep['bevel_weights_skipped'] += n_skip
    if do_cr and snapshot.crease_vert is not None:
        rep['creases'] += _transfer_vert_scalar(
            snapshot, me, NV, snapshot.crease_vert, CREASE_VERT)
    if do_bw and snapshot.bweight_vert is not None:
        rep['bevel_weights'] += _transfer_vert_scalar(
            snapshot, me, NV, snapshot.bweight_vert, BWEIGHT_VERT)

    me.update()
    rep['ok'] = True
    rep['time_s'] = time.perf_counter() - t0
    return rep


# ---------------------------------------------------------------------------

def _apply_shape_keys(snap, new_obj, me, nv, NV, v_tri, v_bary, rep):
    """Rebuild the whole shape-key stack from basis-relative deltas."""
    try:
        new_obj.shape_key_clear()
    except Exception:
        pass

    ref_name = snap.key_reference_name or "Basis"
    basis_src = None
    for k in snap.shape_keys:
        if k['is_reference'] or k['name'] == ref_name:
            basis_src = k
            break

    basis_kb = new_obj.shape_key_add(name=ref_name, from_mix=False)
    # Basis == current mesh positions; shape_key_add already does that, but be
    # explicit so we control the array we add deltas to.
    flat = NV.astype('f4').ravel()
    basis_kb.data.foreach_set("co", flat)
    if basis_src is not None:
        basis_kb.slider_min = basis_src['slider_min']
        basis_kb.slider_max = basis_src['slider_max']
        basis_kb.interpolation = basis_src['interpolation']

    made = {}
    order = []
    for k in snap.shape_keys:
        if k is basis_src:
            made[k['name']] = basis_kb
            order.append((k, basis_kb))
            continue
        delta = _interp_vert(snap, v_tri, v_bary, k['delta'])
        kb = new_obj.shape_key_add(name=k['name'], from_mix=False)
        kb.data.foreach_set("co", (NV + delta).astype('f4').ravel())
        made[k['name']] = kb
        order.append((k, kb))

    key = me.shape_keys
    if key is not None:
        if snap.key_name:
            try:
                key.name = snap.key_name
            except Exception:
                pass
        key.use_relative = snap.key_use_relative

    # second pass: relative_key / value / range / mute / vertex group
    for k, kb in order:
        rel = made.get(k['relative_key'])
        if rel is not None and rel != kb:
            try:
                kb.relative_key = rel
            except Exception:
                pass
        smin, smax = k['slider_min'], k['slider_max']
        for setter in (
            lambda: setattr(kb, 'slider_max', max(smax, kb.slider_min)),
            lambda: setattr(kb, 'slider_min', smin),
            lambda: setattr(kb, 'slider_max', smax),
        ):
            try:
                setter()
            except Exception:
                pass
        try:
            kb.value = k['value']
        except Exception:
            pass
        kb.mute = k['mute']
        try:
            kb.interpolation = k['interpolation']
        except Exception:
            pass
        vgn = k['vertex_group']
        if vgn and new_obj.vertex_groups.get(vgn) is not None:
            try:
                kb.vertex_group = vgn
            except Exception:
                pass

    n = len(order)
    if basis_src is None and n:
        rep['warnings'].append("snapshot had no reference key; synthesised Basis")
        n += 1
    return n


def _apply_uvs(snap, me, nl, npo, NV, rep):
    """Per-loop UV transfer.

    The lookup point for a loop is its vertex nudged towards the face centre so
    that loops on opposite sides of a UV seam pick different source triangles;
    the barycentric weights are then evaluated at the *vertex* position on that
    triangle, so the transferred UV is not shifted inwards.
    """
    starts, totals, lverts = _poly_arrays(me)
    centers = _poly_centers(me)
    lp = np.repeat(np.arange(npo, dtype='i4'), totals) if npo else np.zeros(0, 'i4')
    lco = NV[lverts]
    sample = lco + (centers[lp] - lco) * _LOOP_INSET

    l_tri, _ = _nearest_tris(snap, sample)
    l_bary = _bary(snap, l_tri, lco)

    for layer in snap.uv_layers:
        uvl = me.uv_layers.get(layer['name'])
        if uvl is None:
            try:
                uvl = me.uv_layers.new(name=layer['name'], do_init=False)
            except TypeError:
                uvl = me.uv_layers.new(name=layer['name'])
        if uvl is None:
            rep['warnings'].append("could not create UV layer %r" % layer['name'])
            continue
        uv = _interp_loop(snap, l_tri, l_bary, layer['uv'])
        uvl.data.foreach_set("uv", uv.astype('f4').ravel())
        rep['uv_layers'] += 1
        if layer['active']:
            try:
                me.uv_layers.active = uvl
            except Exception:
                pass
        if layer['active_render']:
            try:
                uvl.active_render = True
            except Exception:
                pass
    rep['uvs'] = rep['uv_layers'] > 0


def _apply_materials(snap, new_obj, me, npo, rep):
    me.materials.clear()
    remap = []
    for mat, name in snap.materials:
        m = mat if _material_ok(mat) else (bpy.data.materials.get(name) if name else None)
        me.materials.append(m)
        remap.append(m)
    nslots = len(remap)
    if nslots and npo:
        centers = _poly_centers(me)
        f_tri, _ = _nearest_tris(snap, centers)
        safe = np.where(f_tri >= 0, f_tri, 0)
        src_poly = snap.tri_poly[safe]
        mi = snap.poly_material[src_poly] if snap.poly_material.size else np.zeros(npo, 'i4')
        mi = np.clip(mi, 0, nslots - 1).astype('i4')
        me.polygons.foreach_set("material_index", mi)
    return nslots


def _transfer_edge_scalar(snap, me, EV, NV, src_vals, attr_name):
    """Conservative nearest-edge transfer of a per-edge float.

    Only source edges with a non-zero value are candidates; a new edge picks one
    up when its midpoint is close to the source edge midpoint (relative to the
    two edge lengths) and the two edges point roughly the same way.
    """
    src_idx = np.nonzero(src_vals > 0.0)[0]
    if src_idx.size == 0 or EV.shape[0] == 0:
        return 0, 0
    SV = snap.verts
    se = snap.edges[src_idx]
    sa, sb = SV[se[:, 0]], SV[se[:, 1]]
    smid = (sa + sb) * 0.5
    sdir = sb - sa
    slen = np.linalg.norm(sdir, axis=1)
    sdir = sdir / np.maximum(slen, _EPS)[:, None]

    kd = KDTree(src_idx.size)
    for i, p in enumerate(smid.tolist()):
        kd.insert(p, i)
    kd.balance()

    na, nb = NV[EV[:, 0]], NV[EV[:, 1]]
    nmid = (na + nb) * 0.5
    ndir = nb - na
    nlen = np.linalg.norm(ndir, axis=1)
    ndir = ndir / np.maximum(nlen, _EPS)[:, None]

    ne = EV.shape[0]
    hit = np.full(ne, -1, 'i4')
    dist = np.zeros(ne, 'f8')
    find = kd.find
    for i, p in enumerate(nmid.tolist()):
        _co, idx, d = find(p)
        if idx is not None:
            hit[i] = idx
            dist[i] = d

    ok = hit >= 0
    h = np.where(ok, hit, 0)
    tol = 0.25 * (slen[h] + nlen)
    align = np.abs(np.einsum('ij,ij->i', ndir, sdir[h]))
    good = ok & (dist <= tol) & (align >= 0.5)   # within ~60 degrees

    out = np.zeros(ne, 'f4')
    out[good] = src_vals[src_idx][h[good]]
    n_ok = int(good.sum())
    n_skip = int(src_idx.size - np.unique(h[good]).size) if n_ok else int(src_idx.size)
    if n_ok:
        at = _ensure_float_attr(me, attr_name, 'EDGE', ne)
        if at is None:
            return 0, int(src_idx.size)
        at.data.foreach_set("value", out)
    return n_ok, max(n_skip, 0)


def _transfer_vert_scalar(snap, me, NV, src_vals, attr_name):
    src_idx = np.nonzero(src_vals > 0.0)[0]
    nv = NV.shape[0]
    if src_idx.size == 0 or nv == 0:
        return 0
    SV = snap.verts[src_idx]
    kd = KDTree(src_idx.size)
    for i, p in enumerate(SV.tolist()):
        kd.insert(p, i)
    kd.balance()

    # tolerance: half the mean source edge length (conservative)
    if snap.edges.shape[0]:
        el = np.linalg.norm(snap.verts[snap.edges[:, 1]] - snap.verts[snap.edges[:, 0]], axis=1)
        tol = 0.5 * float(el.mean())
    else:
        tol = float(np.linalg.norm(snap.verts.max(0) - snap.verts.min(0))) * 0.01

    out = np.zeros(nv, 'f4')
    find = kd.find
    n_ok = 0
    for i, p in enumerate(NV.tolist()):
        _co, idx, d = find(p)
        if idx is not None and d <= tol:
            out[i] = src_vals[src_idx[idx]]
            n_ok += 1
    if n_ok:
        at = _ensure_float_attr(me, attr_name, 'POINT', nv)
        if at is None:
            return 0
        at.data.foreach_set("value", out)
    return n_ok
