"""Guide projection.

Curve / Grease Pencil objects are sampled into polylines, projected onto the
working mesh and turned into

* sharp edge paths, and
* a per-face float-vector attribute ``qf_guide`` holding the desired flow
  direction, consumed by the native backend.

Only the native backend consumes either of these. Blender's QuadriFlow
operator hands the solver bare vertex/triangle arrays - ``use_preserve_sharp``
merely enables QuadriFlow's internal dihedral-angle crease detection, so
flag-marked sharp edges on smooth geometry never reach it (measured:
bit-identical output with/without marks). ``pipeline.run_remesh`` therefore
reroutes guided QuadriFlow solves to the native backend.
"""

from __future__ import annotations

import heapq

import numpy as np

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import analysis

GUIDE_ATTR = "qf_guide"

_MAX_SAMPLES_PER_GUIDE = 4000
_MAX_PATH_NODES = 200000


# ---------------------------------------------------------------------------
# guide sampling
# ---------------------------------------------------------------------------


def _polylines_from_curve(obj, depsgraph):
    """Sample an evaluated curve object into world space polylines."""
    lines = []
    eval_obj = obj.evaluated_get(depsgraph)
    me = None
    try:
        me = eval_obj.to_mesh()
    except Exception:
        me = None
    if me is None or len(me.vertices) == 0:
        if me is not None:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass
        return lines

    mw = obj.matrix_world
    co = analysis.verts_co(me)
    ev = analysis.edge_verts(me) if len(me.edges) else np.zeros((0, 2), dtype=np.int32)

    # build adjacency and walk chains so we get ordered polylines
    adj = {}
    for a, b in ev:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    if not adj:
        pts = [mw @ Vector(map(float, c)) for c in co]
        if len(pts) >= 2:
            lines.append(pts)
        try:
            eval_obj.to_mesh_clear()
        except Exception:
            pass
        return lines

    visited_edges = set()

    def _walk(start):
        chain = [start]
        cur = start
        prev = None
        while True:
            nxt = None
            for n in adj.get(cur, ()):
                key = (min(cur, n), max(cur, n))
                if key in visited_edges or n == prev:
                    continue
                nxt = n
                visited_edges.add(key)
                break
            if nxt is None:
                break
            chain.append(nxt)
            prev, cur = cur, nxt
            if cur == start:
                break
        return chain

    ends = [v for v, n in adj.items() if len(n) == 1]
    for v in ends + sorted(adj.keys()):
        if all((min(v, n), max(v, n)) in visited_edges for n in adj[v]):
            continue
        chain = _walk(v)
        if len(chain) >= 2:
            lines.append([mw @ Vector(map(float, co[i])) for i in chain])

    try:
        eval_obj.to_mesh_clear()
    except Exception:
        pass
    return lines


def _polylines_from_gp(obj, depsgraph):
    """Sample Grease Pencil (v3) strokes into world space polylines."""
    lines = []
    mw = obj.matrix_world
    data = obj.evaluated_get(depsgraph).data
    layers = getattr(data, "layers", None)
    if layers is None:
        return lines
    for layer in layers:
        frames = getattr(layer, "frames", None)
        if not frames:
            continue
        # only the first key of each layer - guides are static
        frame = frames[0]
        drawing = getattr(frame, "drawing", None)
        if drawing is None:
            continue
        strokes = getattr(drawing, "strokes", None)
        got = False
        if strokes is not None:
            try:
                for st in strokes:
                    pts = [mw @ Vector(p.position) for p in st.points]
                    if len(pts) >= 2:
                        lines.append(pts)
                got = True
            except Exception:
                got = False
        if got:
            continue
        # fallback: raw position attribute + curve offsets
        try:
            attr = drawing.attributes.get("position")
            if attr is None:
                continue
            n = len(attr.data)
            raw = np.empty(n * 3, dtype=np.float64)
            attr.data.foreach_get("vector", raw)
            raw = raw.reshape(n, 3)
            offs = [int(c.value) for c in drawing.curve_offsets]
            if len(offs) < 2:
                offs = [0, n]
            for i in range(len(offs) - 1):
                seg = raw[offs[i]:offs[i + 1]]
                if len(seg) >= 2:
                    lines.append([mw @ Vector(map(float, p)) for p in seg])
        except Exception:
            continue
    return lines


def _polylines_from_mesh(obj, depsgraph):
    lines = []
    eval_obj = obj.evaluated_get(depsgraph)
    me = eval_obj.data
    if me is None or not len(me.edges):
        return lines
    mw = obj.matrix_world
    co = analysis.verts_co(me)
    for a, b in analysis.edge_verts(me):
        lines.append([mw @ Vector(map(float, co[a])), mw @ Vector(map(float, co[b]))])
    return lines


def sample_guides(guide_objects, depsgraph):
    """Return a list of world-space polylines (lists of Vector)."""
    lines = []
    for obj in guide_objects or ():
        if obj is None:
            continue
        try:
            t = obj.type
            if t in {"CURVE", "SURFACE", "FONT", "CURVES"}:
                lines.extend(_polylines_from_curve(obj, depsgraph))
            elif t == "GREASEPENCIL":
                lines.extend(_polylines_from_gp(obj, depsgraph))
            elif t == "MESH":
                lines.extend(_polylines_from_mesh(obj, depsgraph))
        except Exception:
            continue
    return [ln for ln in lines if len(ln) >= 2]


def _resample(points, step):
    """Resample a polyline so successive points are at most `step` apart."""
    if step <= 0.0:
        return points
    out = [points[0]]
    for i in range(1, len(points)):
        a = out[-1]
        b = points[i]
        d = (b - a).length
        if d <= 1e-12:
            continue
        n = int(d / step)
        for k in range(1, n + 1):
            out.append(a.lerp(b, k / (n + 1.0)))
        out.append(b)
        if len(out) > _MAX_SAMPLES_PER_GUIDE:
            break
    return out


# ---------------------------------------------------------------------------
# surface walking
# ---------------------------------------------------------------------------


class _SurfaceGraph:
    def __init__(self, mesh):
        self.co = analysis.verts_co(mesh)
        ev = analysis.edge_verts(mesh)
        nv = len(self.co)
        self.nv = nv
        self.adj = [[] for _ in range(nv)]
        for ei in range(len(ev)):
            a = int(ev[ei, 0])
            b = int(ev[ei, 1])
            w = float(np.linalg.norm(self.co[a] - self.co[b]))
            self.adj[a].append((b, w, ei))
            self.adj[b].append((a, w, ei))
        self.mean_edge = 0.0
        if len(ev):
            d = np.linalg.norm(self.co[ev[:, 0]] - self.co[ev[:, 1]], axis=1)
            self.mean_edge = float(d.mean())

    def path_edges(self, src, dst):
        """A* over the edge graph; returns list of edge indices (may be empty)."""
        if src == dst:
            return []
        goal = self.co[dst]
        dist = {src: 0.0}
        prev = {}
        heap = [(float(np.linalg.norm(self.co[src] - goal)), 0.0, src)]
        seen = set()
        expanded = 0
        while heap:
            _f, g, v = heapq.heappop(heap)
            if v in seen:
                continue
            seen.add(v)
            expanded += 1
            if v == dst:
                break
            if expanded > _MAX_PATH_NODES:
                return []
            for nb, w, ei in self.adj[v]:
                ng = g + w
                if ng < dist.get(nb, 1e30) - 1e-12:
                    dist[nb] = ng
                    prev[nb] = (v, ei)
                    h = float(np.linalg.norm(self.co[nb] - goal))
                    heapq.heappush(heap, (ng + h, ng, nb))
        if dst not in prev and dst != src:
            return []
        out = []
        cur = dst
        while cur != src:
            p = prev.get(cur)
            if p is None:
                return []
            out.append(p[1])
            cur = p[0]
        out.reverse()
        return out


def _face_vert_arrays(mesh):
    npoly = len(mesh.polygons)
    ltot = np.empty(npoly, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", ltot)
    lstart = np.empty(npoly, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", lstart)
    nl = len(mesh.loops)
    lv = np.empty(nl, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", lv)
    return lstart, ltot, lv


def _bvh_from_mesh(mesh):
    co = analysis.verts_co(mesh)
    lstart, ltot, lv = _face_vert_arrays(mesh)
    polys = []
    for i in range(len(lstart)):
        s = int(lstart[i])
        n = int(ltot[i])
        polys.append(tuple(int(x) for x in lv[s:s + n]))
    verts = [tuple(map(float, c)) for c in co]
    return BVHTree.FromPolygons(verts, polys, all_triangles=False), co, polys


def _write_guide_attr(mesh, dirs):
    attr = mesh.attributes.get(GUIDE_ATTR)
    if attr is not None and (attr.domain != "FACE" or attr.data_type != "FLOAT_VECTOR"):
        mesh.attributes.remove(attr)
        attr = None
    if attr is None:
        attr = mesh.attributes.new(GUIDE_ATTR, "FLOAT_VECTOR", "FACE")
    attr = mesh.attributes[GUIDE_ATTR]
    attr.data.foreach_set("vector", np.ascontiguousarray(dirs.ravel(), dtype=np.float32))


def project_guides(work_obj, guide_objects, s) -> int:
    """Project guide curves onto ``work_obj``.

    Marks the traversed edges sharp and stores per-face guide directions in the
    ``qf_guide`` FLOAT_VECTOR face attribute. Returns the number of edges marked.
    """
    mesh = work_obj.data
    if mesh is None or len(mesh.polygons) == 0 or len(mesh.edges) == 0:
        return 0

    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except Exception:
        depsgraph = None
    if depsgraph is None:
        return 0

    lines = sample_guides(guide_objects, depsgraph)
    if not lines:
        return 0

    try:
        bvh, co, polys = _bvh_from_mesh(mesh)
    except Exception:
        return 0

    graph = _SurfaceGraph(mesh)
    step = graph.mean_edge if graph.mean_edge > 0.0 else 0.01

    inv = work_obj.matrix_world.inverted_safe()
    npoly = len(polys)
    dirs = np.zeros((npoly, 3), dtype=np.float64)
    dir_hits = np.zeros(npoly, dtype=np.int32)

    marked = set()

    for line in lines:
        pts = [inv @ p for p in line]
        pts = _resample(pts, step)
        proj = []
        for p in pts:
            hit = bvh.find_nearest(p)
            if hit is None or hit[0] is None:
                continue
            loc, _nor, fidx, _dist = hit
            if fidx is None or fidx >= npoly:
                continue
            face = polys[fidx]
            best = None
            bestd = 1e30
            lv = np.array(loc)
            for vi in face:
                d = float(np.linalg.norm(co[vi] - lv))
                if d < bestd:
                    bestd = d
                    best = vi
            if best is None:
                continue
            if proj and proj[-1][0] == best:
                continue
            proj.append((best, fidx, Vector(loc)))

        for i in range(len(proj) - 1):
            v0, f0, l0 = proj[i]
            v1, f1, l1 = proj[i + 1]
            eids = graph.path_edges(v0, v1)
            for e in eids:
                marked.add(e)
            tangent = l1 - l0
            if tangent.length > 1e-9:
                t = np.array(tangent.normalized())
                for fi in (f0, f1):
                    if dir_hits[fi] and float(np.dot(dirs[fi], t)) < 0.0:
                        dirs[fi] -= t
                    else:
                        dirs[fi] += t
                    dir_hits[fi] += 1

    if dir_hits.any():
        norms = np.linalg.norm(dirs, axis=1)
        nz = norms > 1e-9
        dirs[nz] /= norms[nz][:, None]
        _write_guide_attr(mesh, dirs)

    if not marked:
        mesh.update()
        return 0

    sharp = np.zeros(len(mesh.edges), dtype=bool)
    mesh.edges.foreach_get("use_edge_sharp", sharp)
    idx = np.fromiter(marked, dtype=np.int64, count=len(marked))
    idx = idx[(idx >= 0) & (idx < len(sharp))]
    sharp[idx] = True
    mesh.edges.foreach_set("use_edge_sharp", sharp)
    mesh.update()
    return int(len(idx))
