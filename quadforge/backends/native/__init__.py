"""QuadForge native backend - experimental field-aligned quad remesher.

Public entry point (backend contract, see CONTRACTS.md):

    remesh(context, work_obj, s, face_target: int) -> None

The mesh of ``work_obj`` is replaced **in place** with the quad-dominant
result.  Everything runs on ``numpy`` only (no scipy, no C extensions), so it
works in a factory-startup headless Blender.

Failure mode
------------
Anything the solver cannot handle raises :class:`QuadForgeNativeError`
(a subclass of ``RuntimeError``).  ``pipeline.py`` is expected to catch it and
either fall back to the QuadriFlow backend or report the message to the user::

    from .backends.native import QuadForgeNativeError
    try:
        native.remesh(context, work_obj, s, face_target)
    except QuadForgeNativeError as exc:
        ...  # fall back / report str(exc)

The input mesh is never modified before the solver succeeds, so a caught
error leaves ``work_obj`` exactly as it was.
"""

from __future__ import annotations

import numpy as np

__all__ = ["remesh", "QuadForgeNativeError"]


class QuadForgeNativeError(RuntimeError):
    """Native solver could not produce a usable mesh."""


# --------------------------------------------------------------------------
# mesh -> numpy
# --------------------------------------------------------------------------

def _read_vertices(me):
    n = len(me.vertices)
    buf = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", buf)
    return buf.reshape(n, 3)


def _read_loops(me):
    """Return ``(loop_start, loop_total, loop_vert)`` as int64 arrays."""
    npoly = len(me.polygons)
    nloop = len(me.loops)
    ls = np.empty(npoly, dtype=np.int32)
    lt = np.empty(npoly, dtype=np.int32)
    me.polygons.foreach_get("loop_start", ls)
    try:
        me.polygons.foreach_get("loop_total", lt)
    except (AttributeError, RuntimeError, TypeError):
        # Blender 4.1+ dropped `loop_total` on some builds: derive it
        lt = np.diff(np.append(ls.astype(np.int64), nloop)).astype(np.int32)
    lv = np.empty(nloop, dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    return ls.astype(np.int64), lt.astype(np.int64), lv.astype(np.int64)


def _triangulate(V, ls, lt, lv):
    """Triangulate a polygon mesh for the solver.

    Triangles pass through, quads are split along a diagonal, n-gons (n >= 5)
    get a centroid vertex and are fanned from it - a naive corner fan would
    create a huge-valence hub that wrecks the orientation field.

    Returns ``(V2, F, src_poly, cen_rows, cen_cols)`` where ``V2`` is ``V``
    plus the centroid vertices and ``(cen_rows, cen_cols)`` describes which
    original vertices average into each centroid (for per-vertex data).
    Fully vectorised - no Python loop over polygons.
    """
    n = V.shape[0]
    npoly = len(ls)
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if npoly == 0:
        return V, np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64), *empty

    Fs = []
    Ss = []

    m3 = lt == 3
    if m3.any():
        idx = np.nonzero(m3)[0]
        Fs.append(lv[ls[idx][:, None] + np.arange(3)])
        Ss.append(idx)

    m4 = lt == 4
    if m4.any():
        idx = np.nonzero(m4)[0]
        q = lv[ls[idx][:, None] + np.arange(4)]
        # split along the shorter diagonal
        d02 = np.einsum("ij,ij->i", V[q[:, 2]] - V[q[:, 0]], V[q[:, 2]] - V[q[:, 0]])
        d13 = np.einsum("ij,ij->i", V[q[:, 3]] - V[q[:, 1]], V[q[:, 3]] - V[q[:, 1]])
        a = d02 <= d13
        t1 = np.where(a[:, None], q[:, [0, 1, 2]], q[:, [0, 1, 3]])
        t2 = np.where(a[:, None], q[:, [0, 2, 3]], q[:, [1, 2, 3]])
        Fs.append(np.concatenate([t1, t2], axis=0))
        Ss.append(np.concatenate([idx, idx]))

    V2 = V
    rows = np.zeros(0, dtype=np.int64)
    cols = np.zeros(0, dtype=np.int64)
    mn = lt >= 5
    if mn.any():
        idx = np.nonzero(mn)[0]
        cnt = lt[idx]
        grp = np.repeat(np.arange(len(idx), dtype=np.int64), cnt)
        off = np.cumsum(cnt) - cnt
        k = np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(off, cnt)
        base = np.repeat(ls[idx], cnt)
        cr = np.repeat(cnt, cnt)
        a = lv[base + k]
        b = lv[base + (k + 1) % cr]
        cen = np.zeros((len(idx), 3))
        for c in range(3):
            cen[:, c] = np.bincount(grp, weights=V[a][:, c], minlength=len(idx))
        cen /= cnt[:, None]
        cid = n + grp
        Fs.append(np.stack([cid, a, b], axis=1))
        Ss.append(np.repeat(idx, cnt))
        V2 = np.concatenate([V, cen], axis=0)
        rows, cols = grp, a

    if not Fs:
        return V, np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64), *empty
    return (V2, np.concatenate(Fs, axis=0).astype(np.int64),
            np.concatenate(Ss).astype(np.int64), rows, cols)


def _read_edge_verts(me):
    ne = len(me.edges)
    if ne == 0:
        return np.zeros((0, 2), dtype=np.int64)
    buf = np.empty(ne * 2, dtype=np.int32)
    try:
        me.edges.foreach_get("vertices", buf)
    except (RuntimeError, TypeError, AttributeError):
        attr = me.attributes.get(".edge_verts")
        if attr is None:
            return np.zeros((0, 2), dtype=np.int64)
        attr.data.foreach_get("value", buf)
    return buf.reshape(ne, 2).astype(np.int64)


def _read_sharp_edges(me, s):
    """Edges flagged sharp / creased / seam, as (k, 2) vertex-index pairs."""
    ne = len(me.edges)
    if ne == 0:
        return np.zeros((0, 2), dtype=np.int64)
    flag = np.zeros(ne, dtype=bool)

    def _bool_attr(name, prop):
        buf = np.zeros(ne, dtype=bool)
        attr = me.attributes.get(name)
        if attr is not None and attr.domain == "EDGE":
            try:
                attr.data.foreach_get("value", buf)
                return buf
            except (RuntimeError, TypeError):
                pass
        try:
            me.edges.foreach_get(prop, buf)
            return buf
        except (RuntimeError, TypeError, AttributeError):
            return np.zeros(ne, dtype=bool)

    flag |= _bool_attr("sharp_edge", "use_edge_sharp")
    if s is not None and getattr(s, "use_marked_sharp", False):
        flag |= _bool_attr("uv_seam", "use_seam")
        try:
            crease = np.zeros(ne, dtype=np.float32)
            attr = me.attributes.get("crease_edge")
            if attr is not None:
                attr.data.foreach_get("value", crease)
                flag |= crease > 0.25
        except (RuntimeError, TypeError, AttributeError):
            pass

    if not flag.any():
        return np.zeros((0, 2), dtype=np.int64)
    return _read_edge_verts(me)[flag]


def _read_density(me, n, s):
    """Optional float POINT attribute ``qf_density`` (0..2, 1 = neutral).

    ``core.analysis.build_density_attr`` only writes it when curvature
    adaptivity and/or painted density are requested, so its mere presence is
    taken as the instruction to use it.
    """
    attr = me.attributes.get("qf_density")
    if attr is None or attr.domain != "POINT":
        return None
    buf = np.zeros(n, dtype=np.float32)
    try:
        attr.data.foreach_get("value", buf)
    except (RuntimeError, TypeError):
        return None
    d = buf.astype(np.float64)
    if not np.isfinite(d).all():
        d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)
    if np.allclose(d, d[0] if len(d) else 1.0):
        return None
    return d


def _read_guides(me, src_poly, nfaces_tri):
    attr = me.attributes.get("qf_guide")
    if attr is None or attr.domain != "FACE":
        return None
    npoly = len(me.polygons)
    buf = np.zeros(npoly * 3, dtype=np.float32)
    for prop in ("vector", "value", "color"):
        try:
            attr.data.foreach_get(prop, buf)
            break
        except (RuntimeError, TypeError, ValueError):
            continue
    else:
        return None
    g = buf.reshape(npoly, 3).astype(np.float64)
    if not np.any(np.abs(g) > 1e-9):
        return None
    if nfaces_tri == 0:
        return None
    return g[src_poly]


# --------------------------------------------------------------------------
# numpy -> mesh
# --------------------------------------------------------------------------

def _rebuild(me, VQ, FQ):
    me.clear_geometry()
    verts = [tuple(map(float, v)) for v in VQ]
    faces = [tuple(int(i) for i in f) for f in FQ]
    me.from_pydata(verts, [], faces)
    me.update(calc_edges=True)
    me.validate(verbose=False, clean_customdata=True)
    me.update()


# --------------------------------------------------------------------------
# backend entry point
# --------------------------------------------------------------------------

def remesh(context, work_obj, s, face_target):
    """Replace ``work_obj``'s mesh with a field-aligned quad-dominant remesh."""
    from .solver import solve, SolveError

    me = work_obj.data
    if me is None or len(me.vertices) < 4 or len(me.polygons) < 1:
        raise QuadForgeNativeError("native backend: mesh is too small")

    try:
        V0 = _read_vertices(me)
        ls, lt, lv = _read_loops(me)
        V, F, src_poly, cen_rows, cen_cols = _triangulate(V0, ls, lt, lv)
    except Exception as exc:  # pragma: no cover - defensive
        raise QuadForgeNativeError(f"native backend: mesh read failed: {exc}")

    if len(F) < 2:
        raise QuadForgeNativeError("native backend: no triangles to work with")
    if not np.isfinite(V).all():
        raise QuadForgeNativeError("native backend: mesh contains NaN/Inf coordinates")

    sharp = _read_sharp_edges(me, s)
    density = _read_density(me, len(V0), s)
    if density is not None and len(V) > len(V0):
        # extend the per-vertex data onto the added n-gon centroids
        ncen = len(V) - len(V0)
        cnt = np.bincount(cen_rows, minlength=ncen).astype(np.float64)
        ext = np.bincount(cen_rows, weights=density[cen_cols],
                          minlength=ncen) / np.maximum(cnt, 1.0)
        density = np.concatenate([density, ext])
    guides = _read_guides(me, src_poly, len(F))

    sym = (bool(getattr(s, "symmetry_x", False)),
           bool(getattr(s, "symmetry_y", False)),
           bool(getattr(s, "symmetry_z", False))) if s is not None else (False,) * 3

    params = {
        "target_faces": max(12, int(face_target)),
        "adaptive": float(getattr(s, "adaptive_size", 0.0) or 0.0) if s else 0.0,
        "sharp_edges": sharp if len(sharp) else None,
        "guide_dirs": guides,
        "density": density,
        "symmetry": sym,
        "seed": int(getattr(s, "seed", 0) or 0) if s else 0,
        "preserve_boundaries": bool(getattr(s, "preserve_boundaries", True)) if s else True,
        "use_opening_rings": bool(getattr(s, "use_opening_rings", False)) if s else False,
    }

    try:
        VQ, FQ = solve(V, F, params)
    except SolveError as exc:
        raise QuadForgeNativeError(f"native backend: {exc}")
    except MemoryError:
        raise QuadForgeNativeError("native backend: out of memory")
    except Exception as exc:  # pragma: no cover - defensive
        raise QuadForgeNativeError(f"native backend: solver failed: {exc}")

    if len(VQ) < 4 or len(FQ) < 2:
        raise QuadForgeNativeError("native backend: solver returned an empty mesh")
    if not np.isfinite(VQ).all():
        raise QuadForgeNativeError("native backend: solver produced NaN vertices")

    try:
        _rebuild(me, VQ, FQ)
    except Exception as exc:  # pragma: no cover - defensive
        raise QuadForgeNativeError(f"native backend: mesh rebuild failed: {exc}")

    if len(me.polygons) < 2:
        raise QuadForgeNativeError(
            "native backend: result did not survive mesh validation"
        )
