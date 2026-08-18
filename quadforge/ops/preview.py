"""Flow preview -- see the edge flow before committing to a full solve.

``quadforge.preview_flow``   -- run the Native *field* stage only and draw it
``quadforge.clear_preview``  -- delete the preview object again

A full remesh is a field solve followed by a position solve and a graph
extraction; only the first of those decides where the edge loops go, and it is
a small fraction of the total time.  This operator stops right after it and
turns the result into a throwaway line mesh:

* one short stroke per sampled input face, pointing along the 4-RoSy field's
  principal direction (both members of the class are the same edge flow, so a
  single stroke is drawn rather than a cross -- a cross reads as noise at any
  useful stroke count);
* stroke *length* is the local ``rho``, the target quad edge length, so the
  preview shows the sizing field as well as the flow;
* strokes are sampled proportionally to the predicted quad density, so a region
  that will get small quads also gets more strokes.

Everything that shapes the field is honoured, because the input to the solve is
produced by the real pipeline stages: ``pipeline.make_work_object`` (modifiers)
and ``pipeline.preprocess`` (hard edges, marked sharp, materials, UV seams,
guides, painted/curvature density), and the solver parameters are assembled
exactly the way ``backends/native/__init__.py`` assembles them (adaptive size,
detail range, input prior, opening rings, symmetry, seed, boundaries).

The preview object lives in its own collection, is tagged, and is filtered out
of every operator that takes mesh objects as input (see
``ops/remesh.selected_meshes``).
"""

from __future__ import annotations

import time

import bpy
import numpy as np

from .remesh import alive, ensure_collection, move_to_collection

#: Collection every preview object is parked in.
PREVIEWS_COLLECTION = "QuadForge Previews"
#: Custom property on a preview object, holding the source object's name.
PREVIEW_KEY = "quadforge_preview"
#: Appended to the source object's name.
PREVIEW_SUFFIX = " Flow Preview"


# ---------------------------------------------------------------------------
# preview object identification (used by ops/remesh.selected_meshes)
# ---------------------------------------------------------------------------

def is_preview(obj) -> bool:
    """True for a QuadForge flow-preview object.

    Three independent tells, because a user may rename or re-link one and it
    still must never be fed to the solver: the tag property, the name suffix,
    and membership of the previews collection.
    """
    if obj is None:
        return False
    try:
        if PREVIEW_KEY in obj.keys():
            return True
        if obj.name.endswith(PREVIEW_SUFFIX):
            return True
        for coll in obj.users_collection:
            if coll.name.startswith(PREVIEWS_COLLECTION):
                return True
    except (ReferenceError, AttributeError):
        return False
    return False


def preview_name(obj) -> str:
    return "%s%s" % (obj.name, PREVIEW_SUFFIX)


def find_preview(obj):
    """The existing preview for ``obj``, or None."""
    if not alive(obj):
        return None
    name = obj.name
    for other in bpy.data.objects:
        if other is obj:
            continue
        try:
            if other.get(PREVIEW_KEY) == name:
                return other
        except ReferenceError:
            continue
    return bpy.data.objects.get(preview_name(obj))


def all_previews():
    return [o for o in bpy.data.objects if is_preview(o)]


def remove_preview(obj) -> bool:
    """Delete a preview object and its mesh. Returns True if something went."""
    if not alive(obj):
        return False
    me = obj.data if obj.type == 'MESH' else None
    try:
        bpy.data.objects.remove(obj, do_unlink=True)
    except Exception:
        return False
    if me is not None and getattr(me, "users", 1) == 0:
        try:
            bpy.data.meshes.remove(me)
        except Exception:
            pass
    return True


def _drop_empty_collection():
    coll = bpy.data.collections.get(PREVIEWS_COLLECTION)
    if coll is None or coll.objects or coll.children:
        return
    try:
        bpy.data.collections.remove(coll)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# field stage
# ---------------------------------------------------------------------------

class PreviewError(RuntimeError):
    """The field stage could not run on this object."""


def _read_solver_input(context, obj, s):
    """Pipeline preprocess + native mesh read, on a throwaway working copy.

    Returns ``(V, F, params, info)`` with ``params`` assembled exactly the way
    ``backends.native.remesh`` assembles them, so the preview honours every
    setting that shapes the field.
    """
    from .. import pipeline
    from ..backends import native as _native

    work = pipeline.make_work_object(context, obj)
    try:
        # 'NATIVE' skips preprocess's QuadriFlow-can't-do-flags warning: the
        # preview always shows the native field, whatever the backend is set to.
        report = {"backend": "NATIVE", "warnings": [], "limitations": []}
        pipeline.preprocess(context, work, s, report)
        face_target = pipeline.face_target_from_settings(obj, work.data, s)

        me = work.data
        if me is None or len(me.vertices) < 4 or len(me.polygons) < 1:
            raise PreviewError("mesh is too small for a field solve")
        V0 = _native._read_vertices(me)
        ls, lt, lv = _native._read_loops(me)
        V, F, src_poly, cen_rows, cen_cols = _native._triangulate(V0, ls, lt, lv)
        if len(F) < 2:
            raise PreviewError("no triangles to work with")
        if not np.isfinite(V).all():
            raise PreviewError("mesh contains NaN/Inf coordinates")

        sharp = _native._read_sharp_edges(me, s)
        density = _native._read_density(me, len(V0), s)
        if density is not None and len(V) > len(V0):
            ncen = len(V) - len(V0)
            cnt = np.bincount(cen_rows, minlength=ncen).astype(np.float64)
            ext = np.bincount(cen_rows, weights=density[cen_cols],
                              minlength=ncen) / np.maximum(cnt, 1.0)
            density = np.concatenate([density, ext])
        guides = _native._read_guides(me, src_poly, len(F))
    finally:
        pipeline.discard_object(work)

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
        "detail_range": float(getattr(s, "detail_range", 3.0) or 3.0) if s else 3.0,
        "use_input_density": bool(getattr(s, "use_input_density", False)) if s else False,
    }
    info = {
        "target_faces": params["target_faces"],
        "input_faces": int(report.get("input_faces", 0) or 0),
        "hard_edges": int(report.get("hard_edges", 0) or 0),
        "guide_edges": int(report.get("guide_edges", 0) or 0),
        "warnings": list(report.get("warnings") or []),
    }
    return V, F, params, info


def _shape_rho(sol, V, F, p):
    """Mirror of the ``rho`` shaping ``solver.solve`` applies after
    ``solve_fields`` and before extraction.

    The field solve alone does not produce the sizing the quads will really
    have: ``solve()`` boosts the density in a graph-distance band around every
    feature edge and, when a wider size band was requested, re-limits and
    re-normalises the field afterwards.  Skipping that here would draw strokes
    at creases that are up to twice as long as the quads that land there, so
    the same two steps are repeated -- read-only, on our own copy of ``sol``.
    Kept in one function so the duplication is visible: if ``solver.solve``
    changes its shaping, this is the single place that has to follow.
    """
    from ..backends.native import fields as _f
    from ..backends.native import solver as _solver

    rho = np.asarray(sol.rho, dtype=np.float64).copy()

    # 1. guide-interior edges are not creases (solver.solve drops them before
    #    the boost, so the boost must not see them either)
    sharp_feat = p.get("sharp_edges")
    if sharp_feat is not None and len(sharp_feat):
        sharp_feat = np.asarray(sharp_feat, dtype=np.int64).reshape(-1, 2)
    else:
        sharp_feat = None
    gm = getattr(sol, "guide_mask", None)
    if (gm is not None and sharp_feat is not None and len(sharp_feat)
            and np.any(gm) and bool(p.get("guides_win", True))):
        keep = ~(gm[sharp_feat[:, 0]] & gm[sharp_feat[:, 1]])
        sharp_feat = sharp_feat[keep]
        if not len(sharp_feat):
            sharp_feat = None

    # 2. feature-density boost, rescaled back onto the pre-boost cell budget
    fb = float(p.get("feature_density", 2.0))
    if fb > 1.0 and sharp_feat is not None:
        plateau = float(p.get("feature_density_plateau", 2.0))
        decay = max(float(p.get("feature_density_decay", 1.5)), 1e-3)
        nvv = V.shape[0]
        ed_all = _f.build_edges(F)
        d = np.full(nvv, np.inf)
        seed = np.unique(sharp_feat)
        d[seed] = 0.0
        cur = np.zeros(nvv, dtype=bool)
        cur[seed] = True
        for k in range(1, 12):
            m = cur[ed_all[:, 0]] | cur[ed_all[:, 1]]
            nxt = np.zeros(nvv, dtype=bool)
            nxt[ed_all[m].ravel()] = True
            newly = nxt & ~np.isfinite(d)
            if not newly.any():
                break
            d[newly] = float(k)
            cur = nxt
        d[~np.isfinite(d)] = 1e9
        w = np.exp(-np.maximum(d - plateau, 0.0) / decay)
        boosted = rho / (1.0 + (fb - 1.0) * w)
        wa = _f.vertex_areas(V, F, V.shape[0])
        pre = float(np.sum(wa / np.maximum(rho, 1e-12) ** 2))
        rho = boosted * _f.budget_scale(boosted, wa, pre)

    # 3. wide-sizing re-limit (legacy 3x solves never enter here)
    wide = getattr(_solver, "_wide_sizing", None)
    if wide is not None and wide(p):
        wa = _f.vertex_areas(V, F, V.shape[0])
        pre = float(np.sum(wa / np.maximum(rho, 1e-12) ** 2))
        lim = _f.limit_size_gradient(rho, V, _f.build_edges(F))
        rho = lim * _f.budget_scale(lim, wa, pre)

    return rho


def solve_preview_field(context, obj, s):
    """Native field stage only.

    Returns ``(V, F, sol, rho, info)``: the (possibly pre-subdivided) triangle
    mesh the field was solved on, the :class:`FieldSolution`, the shaped target
    edge length, and a dict of diagnostics.  Raises :class:`PreviewError`.
    """
    from ..backends.native import fields as _f
    from ..backends.native import solver as _solver

    t0 = time.perf_counter()
    V, F, params, info = _read_solver_input(context, obj, s)
    V = np.ascontiguousarray(np.asarray(V, dtype=np.float64).reshape(-1, 3))
    F = np.ascontiguousarray(np.asarray(F, dtype=np.int64).reshape(-1, 3))
    if V.shape[0] < 4 or len(F) < 2:
        raise PreviewError("mesh too small for the native field solver")

    target = int(params["target_faces"])
    sharp_in = params["sharp_edges"]
    dens_in = params["density"]
    guide_in = params["guide_dirs"]

    # the input prior is a statement about the mesh the user authored, so it is
    # measured before the refinement below -- exactly as solver.solve does
    prior_in = None
    if params["use_input_density"]:
        prior_in = _f.input_detail_prior(V, F, V.shape[0])

    # same refinement gate as solver.solve: the field is solved on the same
    # sampling the real remesh would use, so the preview cannot be sharper (or
    # blunter) than what it is previewing
    for _ in range(int(getattr(_solver, "_MAX_SUBDIV", 4))):
        if V.shape[0] >= float(getattr(_solver, "_MIN_SAMPLES_PER_QUAD", 2.5)) * target:
            break
        if V.shape[0] > int(getattr(_solver, "_MAX_VERTS", 1_000_000)) // 4:
            break
        V, F, sharp_in, dens_in, guide_in, prior_in = _solver._subdivide(
            V, F, sharp_in, dens_in, guide_in, prior_in)

    p = dict(params)
    p["sharp_edges"] = sharp_in
    p["density"] = dens_in
    p["guide_dirs"] = guide_in
    p["input_prior"] = prior_in
    p.setdefault("curvature_align", 0.7)

    try:
        sol = _f.solve_fields(V, F, p)
    except Exception as exc:
        raise PreviewError("field solve failed: %s: %s" % (type(exc).__name__, exc))

    rho = _shape_rho(sol, V, F, p)
    if not np.isfinite(rho).all() or float(np.min(rho)) <= 0.0:
        rho = np.asarray(sol.rho, dtype=np.float64)

    # the exact parameter dict the field was solved with, so a caller (the
    # test suite) can run the *real* solver on the very same input and check
    # that the preview is showing the flow it claims to be showing
    info["params"] = p
    info.update({
        "solve_verts": int(V.shape[0]),
        "solve_tris": int(len(F)),
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "ring_openings": int((sol.stats or {}).get("ring_openings", 0)),
        "time_s": time.perf_counter() - t0,
    })
    return V, F, sol, rho, info


# ---------------------------------------------------------------------------
# field -> line segments
# ---------------------------------------------------------------------------

def sample_faces(V, F, rho, max_segments):
    """Face indices to draw a stroke on, sampled by predicted quad density.

    A stroke per input face would show the *input* tessellation; sampling the
    inverse CDF of ``area / rho**2`` (the number of output quads a face will
    carry) makes the stroke density follow the sizing field instead, so a
    region that gets small quads also gets more strokes.  Deterministic.
    """
    from ..backends.native import fields as _f

    nf = len(F)
    want = int(max(1, min(int(max_segments), nf)))
    if want >= nf:
        return np.arange(nf, dtype=np.int64)
    _, areas = _f.face_normals_areas(V, F)
    rf = np.mean(rho[F], axis=1)
    w = np.asarray(areas, dtype=np.float64) / np.maximum(rf, 1e-12) ** 2
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(w.sum())
    if total <= 0.0:
        step = max(1, nf // want)
        return np.arange(0, nf, step, dtype=np.int64)[:want]
    cdf = np.cumsum(w)
    picks = (np.arange(want, dtype=np.float64) + 0.5) * (total / want)
    idx = np.searchsorted(cdf, picks, side="left")
    return np.unique(np.clip(idx, 0, nf - 1)).astype(np.int64)


def _comb_sweeps(Q, N, src, dst, D, iters):
    """Jacobi combing sweeps on one level's graph (see :func:`comb_field`)."""
    from ..backends.native import fields as _f

    n = len(Q)
    for _ in range(int(iters)):
        rep = _f.rosy4_representative(Q[dst], N[dst], D[src])
        acc = np.empty((n, 3), dtype=np.float64)
        for c in range(3):
            acc[:, c] = np.bincount(dst, weights=rep[:, c], minlength=n)
        acc += 0.7 * D                       # damping: no 2-cycles
        nxt = _f.rosy4_representative(Q, N, acc)
        weak = np.einsum("ij,ij->i", acc, acc) < 1e-24
        if weak.any():
            nxt[weak] = D[weak]
        if np.allclose(nxt, D, atol=1e-12):
            break
        D = nxt
    return D


def comb_field(V, F, sol, iters=12):
    """Pick one representative per vertex out of each 4-RoSy class so that
    neighbours agree *as vectors*.

    The solver's ``Q`` is a cross field: at every vertex the four directions
    ``+-Q``, ``+-(N x Q)`` are the same answer, and which of them ends up in
    the array is arbitrary.  Drawing that array directly gives a stroke image
    where half the strokes sit at 90 degrees to their neighbours -- correct,
    and unreadable; it looks like static rather than flow.  Measured as the
    share of adjacent drawn strokes more than 60 degrees apart: Suzanne
    9.8% -> 4.1%, the Dinasty head 6.7% -> 4.7%.

    Combing keeps every vertex's class exactly as it is and only chooses the
    member of it that agrees with the choice already made next door, by the
    same extrinsic matching the orientation solver uses:

        rep = argmax_{r in class(Q_j)} <r, D_i>   for every directed edge i->j
        D_j = the member of class(Q_j) closest to the sum of those votes

    so the field is never blurred or moved -- only the arrow head is.

    A flat sweep propagates one ring at a time, which is far too slow to comb a
    50k-vertex head; the solver's own multiresolution hierarchy is already in
    the :class:`FieldSolution`, so the choice is made on the coarsest level
    (a few hundred vertices, globally consistent in a handful of sweeps) and
    prolonged down.  The 90-degree jumps that survive are the field's real
    singularities, where no consistent choice exists anywhere.
    """
    from ..backends.native import fields as _f
    from ..backends.native import solver as _solver

    N = np.asarray(sol.N, dtype=np.float64)
    Q = np.asarray(sol.Q, dtype=np.float64)
    n = V.shape[0]

    levels = getattr(sol, "levels", None)
    if levels:
        try:
            if int(levels[0].get("n", -1)) != n:
                raise ValueError("hierarchy does not match the fine mesh")
            Qs = _solver._restrict_orientations(levels, Q)
            D = None
            for li in range(len(levels) - 1, -1, -1):
                lv = levels[li]
                Ql, Nl = Qs[li], lv["N"]
                if D is None:
                    Dl = Ql.copy()
                else:
                    # levels[li]['parent'] maps this level onto the coarser one
                    Dl = _f.rosy4_representative(Ql, Nl, D[lv["parent"]])
                D = _comb_sweeps(Ql, Nl, lv["src"], lv["indices"], Dl,
                                 iters if li else 2 * iters)
            return D
        except Exception:
            pass                              # fall back on the flat sweep

    edges = sol.edges if getattr(sol, "edges", None) is not None else _f.build_edges(F)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(edges):
        return Q
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    return _comb_sweeps(Q, N, src, dst, Q.copy(), 12 * iters)


def build_segments(V, F, sol, rho, max_segments=12000, length_scale=0.85,
                   lift=0.06, comb=True):
    """(k, 2, 3) endpoints of the flow strokes, in the source object's space."""
    from ..backends.native import fields as _f

    sel = sample_faces(V, F, rho, max_segments)
    Fs = F[sel]
    N = np.asarray(sol.N, dtype=np.float64)
    Q = comb_field(V, F, sol) if comb else np.asarray(sol.Q, dtype=np.float64)

    cen = (V[Fs[:, 0]] + V[Fs[:, 1]] + V[Fs[:, 2]]) / 3.0
    fn, _areas = _f.face_normals_areas(V, Fs)
    fn = _f.normalize(fn)

    # 4-RoSy average of the three corners: fold each onto the first corner's
    # class before summing, or a 90-degree disagreement cancels to zero
    ref = Q[Fs[:, 0]]
    acc = ref.copy()
    for k in (1, 2):
        acc = acc + _f.rosy4_representative(Q[Fs[:, k]], N[Fs[:, k]], ref)

    # onto the face plane
    d = acc - fn * _f._dot(acc, fn)[:, None]
    ln = np.sqrt(np.einsum("ij,ij->i", d, d))
    weak = ln < 1e-9
    if weak.any():                       # degenerate corner: fall back on the
        alt = ref - fn * _f._dot(ref, fn)[:, None]   # first corner alone
        d[weak] = alt[weak]
        ln = np.sqrt(np.einsum("ij,ij->i", d, d))
    good = ln > 1e-9
    d = d[good] / ln[good][:, None]
    cen, fn, Fs = cen[good], fn[good], Fs[good]
    if not len(d):
        return np.zeros((0, 2, 3), dtype=np.float64), sel[:0]

    half = 0.5 * float(length_scale) * np.mean(rho[Fs], axis=1)
    off = cen + fn * (float(lift) * 2.0 * half)[:, None]
    seg = np.empty((len(d), 2, 3), dtype=np.float64)
    seg[:, 0, :] = off - d * half[:, None]
    seg[:, 1, :] = off + d * half[:, None]
    return seg, sel[good]


# ---------------------------------------------------------------------------
# preview object
# ---------------------------------------------------------------------------

def _make_line_mesh(name, seg):
    me = bpy.data.meshes.new(name)
    k = len(seg)
    verts = seg.reshape(k * 2, 3).tolist()
    edges = [(2 * i, 2 * i + 1) for i in range(k)]
    me.from_pydata(verts, edges, [])
    me.update()
    return me


def make_preview_object(context, obj, seg, info=None):
    """Create (or replace) ``'<obj> Flow Preview'`` and return it."""
    name = preview_name(obj)
    me = _make_line_mesh(name, seg)

    prev = find_preview(obj)
    if alive(prev) and prev.type == 'MESH':
        old = prev.data
        prev.data = me
        if old is not None and getattr(old, "users", 1) == 0:
            try:
                bpy.data.meshes.remove(old)
            except Exception:
                pass
        prev.name = name
    else:
        prev = bpy.data.objects.new(name, me)

    prev[PREVIEW_KEY] = obj.name
    for key, value in (info or {}).items():
        if isinstance(value, (int, float, str)):
            prev["qf_preview_" + key] = value

    coll = ensure_collection(context, PREVIEWS_COLLECTION)
    move_to_collection(prev, coll)

    # follow the source object exactly: the segments are in its local space,
    # so an identity basis under it resolves to the source's world matrix
    from mathutils import Matrix
    prev.parent = obj
    prev.matrix_parent_inverse = Matrix.Identity(4)
    prev.matrix_basis = Matrix.Identity(4)

    # obviously disposable: wire-only, unselectable, never rendered
    prev.display_type = 'WIRE'
    prev.hide_select = True
    prev.hide_render = True
    prev.show_in_front = False
    prev.color = (0.04, 0.04, 0.05, 1.0)
    for flag in ("hide_probe_volume", "visible_shadow", "visible_diffuse"):
        try:
            setattr(prev, flag, False)
        except (AttributeError, TypeError):
            pass
    return prev


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def _preview_source(context):
    obj = getattr(context, "object", None)
    if obj is not None and obj.type == 'MESH' and not is_preview(obj):
        return obj
    for other in getattr(context, "selected_objects", []) or []:
        if other.type == 'MESH' and not is_preview(other):
            return other
    return None


class QUADFORGE_OT_preview_flow(bpy.types.Operator):
    bl_idname = "quadforge.preview_flow"
    bl_label = "Preview Flow"
    bl_description = (
        "Solve only the orientation field and draw it as short strokes along "
        "the edge flow, sized by the quads it would produce. Much faster than "
        "a full remesh. Shows the Native field; QuadriFlow+ follows its own flow"
    )
    bl_options = {'REGISTER', 'UNDO'}

    max_segments: bpy.props.IntProperty(
        name="Strokes",
        description="Upper bound on the number of flow strokes drawn",
        default=12000, min=100, soft_max=60000, max=400000,
    )
    length_scale: bpy.props.FloatProperty(
        name="Stroke Length",
        description="Stroke length as a fraction of the local target quad size",
        default=0.85, min=0.1, max=2.0,
    )

    @classmethod
    def poll(cls, context):
        if getattr(context, "mode", 'OBJECT') != 'OBJECT':
            return False
        obj = _preview_source(context)
        return obj is not None and obj.data is not None and len(obj.data.polygons) > 0

    def execute(self, context):
        obj = _preview_source(context)
        if obj is None:
            self.report({'ERROR'}, "No mesh object selected")
            return {'CANCELLED'}
        s = getattr(obj, "quadforge", None)
        if s is None:
            self.report({'ERROR'}, "QuadForge settings are not registered")
            return {'CANCELLED'}

        try:
            V, F, sol, rho, info = solve_preview_field(context, obj, s)
        except PreviewError as exc:
            self.report({'ERROR'}, "Preview Flow: %s" % exc)
            return {'CANCELLED'}
        except Exception as exc:  # never take the UI down with the solver
            self.report({'ERROR'}, "Preview Flow: %s: %s" % (type(exc).__name__, exc))
            return {'CANCELLED'}

        seg, _sel = build_segments(V, F, sol, rho,
                                   max_segments=int(self.max_segments),
                                   length_scale=float(self.length_scale))
        if len(seg) == 0:
            self.report({'ERROR'}, "Preview Flow: the field produced no usable strokes")
            return {'CANCELLED'}

        info["segments"] = int(len(seg))
        prev = make_preview_object(context, obj, seg, info)

        for text in info.get("warnings") or []:
            self.report({'WARNING'}, str(text))

        backend_note = ""
        if getattr(s, "backend", 'QUADRIFLOW') != 'NATIVE':
            backend_note = " (Native field; QuadriFlow+ follows its own flow)"
        self.report(
            {'INFO'},
            "Preview Flow: %d strokes, target %d faces, quad size %.4g-%.4g, "
            "%.2fs%s" % (len(seg), info["target_faces"], info["rho_min"],
                         info["rho_max"], info["time_s"], backend_note),
        )
        # keep the source active: the preview is scenery, not a selection
        try:
            context.view_layer.objects.active = obj
            obj.select_set(True)
            prev.select_set(False)
        except Exception:
            pass
        return {'FINISHED'}


class QUADFORGE_OT_clear_preview(bpy.types.Operator):
    bl_idname = "quadforge.clear_preview"
    bl_label = "Clear Flow Preview"
    bl_description = "Delete the flow preview object"
    bl_options = {'REGISTER', 'UNDO'}

    all_objects: bpy.props.BoolProperty(
        name="All Objects",
        description="Delete every flow preview in the file, not just this one",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        # the panel calls this on every redraw, so check the collection first
        # and only walk the scene when there isn't one
        coll = bpy.data.collections.get(PREVIEWS_COLLECTION)
        if coll is not None and coll.objects:
            return True
        return bool(all_previews())

    def execute(self, context):
        targets = []
        obj = getattr(context, "object", None)
        if is_preview(obj):
            targets = [obj]
        elif not self.all_objects:
            for src in ([obj] if obj is not None else []) + list(
                    getattr(context, "selected_objects", []) or []):
                prev = find_preview(src) if (src is not None
                                             and src.type == 'MESH') else None
                if prev is not None and prev not in targets:
                    targets.append(prev)
        if self.all_objects or not targets:
            targets = all_previews()

        gone = sum(1 for p in targets if remove_preview(p))
        _drop_empty_collection()
        if not gone:
            self.report({'INFO'}, "No flow preview to clear")
            return {'CANCELLED'}
        self.report({'INFO'}, "Cleared %d flow preview%s"
                    % (gone, "" if gone == 1 else "s"))
        return {'FINISHED'}


CLASSES = (
    QUADFORGE_OT_preview_flow,
    QUADFORGE_OT_clear_preview,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
