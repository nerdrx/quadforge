"""Mesh quality metrics for QuadForge.

Pure data-API + numpy; no context access, safe to import headless.

``mesh_report(obj)`` returns the dict described in CONTRACTS.md::

    faces, quads, tris, ngons, quad_pct, poles_3, poles_5plus,
    non_manifold_edges, symmetry_error_x/y/z, area

plus a few cheap extras (``verts``, ``edges``, ``boundary_edges``).

All measurements are taken in the object's *local* space, so the symmetry
errors are measured against the planes through the object origin -- which is
what the remesher's symmetry options operate on.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mesh_report", "REPORT_KEYS"]

REPORT_KEYS = (
    "faces", "quads", "tris", "ngons", "quad_pct",
    "poles_3", "poles_5plus", "non_manifold_edges",
    "symmetry_error_x", "symmetry_error_y", "symmetry_error_z", "area",
)

#: Above this vertex count the symmetry scan uses a deterministic subsample of
#: query points (the KD-tree itself always contains every vertex).
SYMMETRY_MAX_SAMPLES = 60000


def _empty_report() -> dict:
    return {
        "verts": 0, "edges": 0,
        "faces": 0, "quads": 0, "tris": 0, "ngons": 0, "quad_pct": 0.0,
        "poles_3": 0, "poles_5plus": 0,
        "non_manifold_edges": 0, "boundary_edges": 0,
        "symmetry_error_x": 0.0, "symmetry_error_y": 0.0, "symmetry_error_z": 0.0,
        "area": 0.0,
    }


def _face_stats(me, rep: dict) -> None:
    nf = len(me.polygons)
    rep["faces"] = nf
    if not nf:
        return
    sides = np.empty(nf, dtype=np.int32)
    me.polygons.foreach_get("loop_total", sides)
    rep["tris"] = int(np.count_nonzero(sides == 3))
    rep["quads"] = int(np.count_nonzero(sides == 4))
    rep["ngons"] = int(np.count_nonzero(sides > 4))
    rep["quad_pct"] = round(100.0 * rep["quads"] / nf, 3)

    areas = np.empty(nf, dtype=np.float32)
    me.polygons.foreach_get("area", areas)
    rep["area"] = float(areas.astype(np.float64).sum())


def _valence_stats(me, rep: dict) -> None:
    """Vertex valence (edges per vertex) -> pole counts."""
    nv = len(me.vertices)
    ne = len(me.edges)
    rep["verts"] = nv
    rep["edges"] = ne
    if not nv or not ne:
        return
    ev = np.empty(ne * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", ev)
    valence = np.bincount(ev, minlength=nv)
    rep["poles_3"] = int(np.count_nonzero(valence == 3))
    rep["poles_5plus"] = int(np.count_nonzero(valence >= 5))


def _edge_face_counts(me):
    """Return an int array: number of faces using each edge (or None)."""
    ne = len(me.edges)
    nl = len(me.loops)
    if not ne:
        return None
    if nl:
        try:
            le = np.empty(nl, dtype=np.int32)
            me.loops.foreach_get("edge_index", le)
            return np.bincount(le, minlength=ne)
        except Exception:
            pass
    # Fallback: rebuild loop->edge from polygon topology using an edge-key map.
    try:
        ev = np.empty(ne * 2, dtype=np.int32)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        key_to_edge = {}
        for i, (a, b) in enumerate(ev):
            key_to_edge[(a, b) if a < b else (b, a)] = i
        counts = np.zeros(ne, dtype=np.int64)
        for poly in me.polygons:
            for key in poly.edge_keys:
                idx = key_to_edge.get(key)
                if idx is not None:
                    counts[idx] += 1
        return counts
    except Exception:
        return None


def _manifold_stats(me, rep: dict) -> None:
    counts = _edge_face_counts(me)
    if counts is None:
        return
    rep["boundary_edges"] = int(np.count_nonzero(counts == 1))
    # Wire edges (0 faces) and edges shared by 3+ faces are non-manifold.
    # A boundary edge (exactly 1 face) is manifold-with-boundary and is
    # reported separately.
    rep["non_manifold_edges"] = int(
        np.count_nonzero(counts == 0) + np.count_nonzero(counts > 2)
    )


def _symmetry_stats(me, rep: dict) -> None:
    nv = len(me.vertices)
    if nv < 2:
        return
    try:
        from mathutils.kdtree import KDTree
    except Exception:
        return

    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)

    kd = KDTree(nv)
    for i in range(nv):
        v = co[i]
        kd.insert((float(v[0]), float(v[1]), float(v[2])), i)
    kd.balance()

    if nv > SYMMETRY_MAX_SAMPLES:
        step = max(1, nv // SYMMETRY_MAX_SAMPLES)
        sample = np.arange(0, nv, step)
    else:
        sample = np.arange(nv)

    for axis, key in enumerate(("symmetry_error_x", "symmetry_error_y", "symmetry_error_z")):
        mirrored = co[sample].copy()
        mirrored[:, axis] *= -1.0
        worst = 0.0
        for p in mirrored:
            _co, _idx, dist = kd.find((float(p[0]), float(p[1]), float(p[2])))
            if dist is not None and dist > worst:
                worst = dist
        rep[key] = float(worst)


def mesh_report(obj) -> dict:
    """Quality metrics for ``obj`` (a mesh object). Never raises."""
    rep = _empty_report()
    me = getattr(obj, "data", None)
    if obj is None or getattr(obj, "type", None) != 'MESH' or me is None:
        return rep

    _face_stats(me, rep)
    _valence_stats(me, rep)
    _manifold_stats(me, rep)
    _symmetry_stats(me, rep)
    return rep


def format_summary(rep: dict) -> str:
    """One-line human readable digest of a report dict."""
    return (
        "{faces} faces, {quad_pct:.1f}% quads "
        "({tris} tris, {ngons} ngons), poles {poles_3}/{poles_5plus}, "
        "non-manifold {non_manifold_edges}".format(**rep)
    )
