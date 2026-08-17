"""QuadForge native field-aligned quad remesher - orchestration.

``solve(V, F, params) -> (VQ, FQ)`` where ``FQ`` is a list of index tuples of
length 3 or 4 (quad-dominant).

Pipeline
--------
1. topology + area-weighted vertex normals
2. hard constraints: sharp edges, mesh boundary, per-face guide directions
3. target edge length ``rho`` from the requested face count, modulated by the
   optional per-vertex ``qf_density`` attribute
4. randomised vertex-clustering multiresolution hierarchy
5. 4-RoSy orientation field, solved coarse -> fine
6. lattice-compatible position field, solved coarse -> fine
7. graph extraction (``extract.extract_quads``)
8. up to two cheap re-runs of steps 6-7 with a rescaled ``rho`` to land the
   face count near the target (the orientation field is scale-free, so it is
   reused unchanged)

Symmetry
--------
``params['symmetry']`` is accepted but **not** enforced here.  Exact symmetry
is produced by the pipeline's ``exact_symmetry`` bisect path (bisect ->
remesh half -> mirror + weld), which is mathematically exact; approximating
it inside the field solver would only be a soft constraint and would fight
the bisect path.  The value is stored in the returned stats for traceability.
"""

from __future__ import annotations

import time

import numpy as np

from . import fields as _f
from .extract import extract_quads

__all__ = ["solve", "SolveError"]


class SolveError(RuntimeError):
    """Raised when the input cannot be remeshed by the native solver."""


# a quad of side rho covers rho^2 of surface area, so rho = sqrt(area/target)
# is the right first guess; the feedback loop at the end of solve() corrects
# the residual bias from collapsed cells and incomplete cycles, so this stays
# at 1.0 (it is here as a single knob should a global bias ever show up)
_RHO_CALIB = 1.0

# the position lattice is read off the input graph, so the input needs
# roughly this many vertices per output quad; below that the solver
# midpoint-subdivides the input first (surface-preserving)
_MIN_SAMPLES_PER_QUAD = 2.5
_MAX_SUBDIV = 4
# refinement stops once a further 1-to-4 split would exceed this vertex count
_MAX_VERTS = 1_000_000

_DEFAULTS = {
    "target_faces": 5000,
    "adaptive": 0.0,
    "sharp_edges": None,
    "guide_dirs": None,
    "density": None,
    "symmetry": (False, False, False),
    "seed": 0,
    "orient_iters": 20,
    "pos_iters": 20,
    "preserve_boundaries": True,
    "verbose": False,
}


def _subdivide(V, F, sharp, density, guides):
    """One 1-to-4 midpoint subdivision (the piecewise-linear surface is
    unchanged, only the sampling density grows).  Constraint data is carried
    along."""
    n = V.shape[0]
    e = _f.build_edges(F)
    key = e[:, 0] * np.int64(n) + e[:, 1]
    korder = np.argsort(key, kind="stable")
    skey = key[korder]

    def eid(a, b):
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        k = lo * np.int64(n) + hi
        return n + korder[np.searchsorted(skey, k)]

    Vm = 0.5 * (V[e[:, 0]] + V[e[:, 1]])
    V2 = np.concatenate([V, Vm], axis=0)

    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    ab, bc, ca = eid(a, b), eid(b, c), eid(c, a)
    F2 = np.concatenate([
        np.stack([a, ab, ca], axis=1),
        np.stack([b, bc, ab], axis=1),
        np.stack([c, ca, bc], axis=1),
        np.stack([ab, bc, ca], axis=1),
    ], axis=0)

    sharp2 = None
    if sharp is not None and len(sharp):
        sa, sb = sharp[:, 0], sharp[:, 1]
        sm = eid(sa, sb)
        sharp2 = np.concatenate([
            np.stack([sa, sm], axis=1), np.stack([sm, sb], axis=1)
        ], axis=0)

    dens2 = None
    if density is not None:
        dens2 = np.concatenate(
            [density, 0.5 * (density[e[:, 0]] + density[e[:, 1]])])

    guides2 = None
    if guides is not None:
        guides2 = np.tile(np.asarray(guides, dtype=np.float64), (4, 1))

    return V2, F2, sharp2, dens2, guides2


def _guides_to_vertices(F, n, guide_dirs, N):
    """Per-face guide vectors -> per-vertex tangent constraint directions."""
    g = np.asarray(guide_dirs, dtype=np.float64).reshape(-1, 3)
    if len(g) != len(F):
        return None
    ln = np.sqrt(np.einsum("ij,ij->i", g, g))
    live = ln > 1e-9
    if not live.any():
        return None
    gd = np.zeros((n, 3))
    gu = g[live] / ln[live][:, None]
    fi = np.nonzero(live)[0]
    for k in range(3):
        vi = F[fi, k]
        # 4-RoSy-agnostic accumulation: align to the first contribution
        for c in range(3):
            gd[:, c] += np.bincount(vi, weights=gu[:, c], minlength=n)
    return gd


def solve(V, F, params=None):
    """Field-aligned quad remesh of a triangle mesh.

    Parameters
    ----------
    V : (n, 3) float array
    F : (m, 3) int array
    params : dict, see module docstring / CONTRACTS.md

    Returns
    -------
    (VQ, FQ) : (k, 3) float64 array and a list of 3/4-tuples of vertex indices.
    """
    p = dict(_DEFAULTS)
    for k, v in (params or {}).items():
        p[k] = v
    for k in ("target_faces", "orient_iters", "pos_iters", "seed", "adaptive"):
        if p.get(k) is None:
            p[k] = _DEFAULTS[k]

    t0 = time.time()
    V = np.ascontiguousarray(np.asarray(V, dtype=np.float64).reshape(-1, 3))
    F = np.ascontiguousarray(np.asarray(F, dtype=np.int64).reshape(-1, 3))
    n = V.shape[0]
    if n < 4 or len(F) < 2:
        raise SolveError("mesh too small for the native solver")
    if F.max() >= n or F.min() < 0:
        raise SolveError("triangle indices out of range")

    target = max(12, int(p["target_faces"]))
    rng = np.random.default_rng(int(p["seed"]) & 0x7FFFFFFF)

    # ---- refine the input if it is too coarse for the request ----------
    # The position lattice can only be read off the input graph, so we need
    # roughly `MIN_SAMPLES_PER_QUAD` input vertices per output quad.  Midpoint
    # subdivision keeps the surface identical, it only adds samples.
    sharp_in = p.get("sharp_edges")
    if sharp_in is not None and len(sharp_in):
        sharp_in = np.asarray(sharp_in, dtype=np.int64).reshape(-1, 2)
    else:
        sharp_in = None
    dens_in = p.get("density")
    if dens_in is not None:
        dens_in = np.asarray(dens_in, dtype=np.float64).ravel()
        if dens_in.size != n:
            dens_in = None
    guide_in = p.get("guide_dirs")
    if guide_in is not None:
        guide_in = np.asarray(guide_in, dtype=np.float64).reshape(-1, 3)
        if len(guide_in) != len(F):
            guide_in = None

    for _ in range(_MAX_SUBDIV):
        if V.shape[0] >= _MIN_SAMPLES_PER_QUAD * target:
            break
        if V.shape[0] > _MAX_VERTS // 4:
            break
        V, F, sharp_in, dens_in, guide_in = _subdivide(
            V, F, sharp_in, dens_in, guide_in)
    n = V.shape[0]

    # ---- v2 path: curvature-aligned fields + robust extraction ----------
    # (falls back to the v1 pipeline below on any failure)
    if bool(p.get("use_v2", True)):
        try:
            from . import fields as _f2
            from . import extract as _e2
            p2 = dict(p)
            p2["sharp_edges"] = sharp_in
            p2["density"] = dens_in
            p2["guide_dirs"] = guide_in
            p2.setdefault("curvature_align", 0.7)
            sol = _f2.solve_fields(V, F, p2)
            # feature-density boost: thin rims and creases need denser quads
            # than flat regions or their silhouettes alias into jagged steps
            fb = float(p2.get("feature_density", 1.5))
            if fb > 1.0 and sharp_in is not None and len(sharp_in):
                near = np.zeros(V.shape[0], dtype=bool)
                near[np.unique(sharp_in)] = True
                ed_all = _f.build_edges(F)
                for _ in range(2):
                    m = near[ed_all[:, 0]] | near[ed_all[:, 1]]
                    near[ed_all[m].ravel()] = True
                rho2 = np.asarray(sol.rho, dtype=np.float64).copy()
                rho2[near] /= fb
                sol.rho = rho2
            VQ, FQ = _e2.extract(V, F, sol, p2)
            # accept only a result that plausibly covers the input surface:
            # a collapsed extraction (fragments of the input) must fall back
            if len(FQ) >= max(4, 0.3 * target):
                VQa = np.asarray(VQ, dtype=np.float64)
                _, in_areas = _f.face_normals_areas(V, F)
                out_area = 0.0
                for f in FQ:
                    pts = VQa[list(f)]
                    for i in range(1, len(pts) - 1):
                        out_area += 0.5 * np.linalg.norm(
                            np.cross(pts[i] - pts[0], pts[i + 1] - pts[0]))
                if out_area >= 0.6 * float(in_areas.sum()):
                    return VQa, list(FQ)
        except Exception:
            if bool(p.get("v2_strict", False)):
                raise
            # fall through to v1

    # ---- topology ----------------------------------------------------
    N = _f.vertex_normals(V, F)
    edges = _f.build_edges(F)
    if len(edges) == 0:
        raise SolveError("mesh has no edges")
    indptr, indices, _src = _f.build_csr(edges, n)

    _, areas = _f.face_normals_areas(V, F)
    area = float(areas.sum())
    if not np.isfinite(area) or area <= 0.0:
        raise SolveError("degenerate input mesh (zero surface area)")

    # ---- constraints --------------------------------------------------
    sharp_list = []
    if sharp_in is not None and len(sharp_in):
        sharp_list.append(sharp_in)
    be = _f.boundary_edges(F)
    bnd_verts = np.zeros(n, dtype=bool)
    if len(be):
        bnd_verts[be.ravel()] = True
        if p.get("preserve_boundaries", True):
            sharp_list.append(be)
    sharp_all = np.concatenate(sharp_list, axis=0) if sharp_list else None

    gverts = None
    if guide_in is not None:
        gverts = _guides_to_vertices(F, n, guide_in, N)

    con_mask, con_dir = _f.build_constraints(V, N, n, sharp_all, gverts)
    # only creases / boundaries pin the *position* field; guides steer the
    # orientation field but must leave the lattice free to slide
    if sharp_all is not None and len(sharp_all):
        pin_mask, _ = _f.build_constraints(V, N, n, sharp_all, None)
        pin_mask &= con_mask
    else:
        pin_mask = np.zeros(n, dtype=bool)

    # ---- target edge length ------------------------------------------
    rho0 = _RHO_CALIB * np.sqrt(area / float(target))
    rho = np.full(n, rho0, dtype=np.float64)
    if dens_in is not None:
        d = dens_in
        if d.size == n:
            d = np.clip(np.nan_to_num(d, nan=1.0), 0.25, 4.0)
            rho = rho0 / d
            # renormalise so that the *mean* cell area still matches the
            # requested face count (density only redistributes detail)
            mean_rho = float(np.mean(rho))
            if mean_rho > 1e-12:
                rho *= rho0 / mean_rho

    # ---- hierarchy -----------------------------------------------------
    levels = _f.build_hierarchy(V, N, indptr, indices, rho, con_mask, con_dir,
                                rng, pin_mask=pin_mask)
    t_hier = time.time()

    # ---- orientation field (coarse -> fine) ----------------------------
    top = levels[-1]
    Q = _f.random_tangents(top["N"], rng)
    oit = int(p["orient_iters"])
    for li in range(len(levels) - 1, -1, -1):
        lv = levels[li]
        it = oit if li == 0 else max(6, oit // 2)
        Q = _f.smooth_orientations(Q, lv["N"], lv["src"], lv["indices"],
                                   lv["con_mask"], lv["con_dir"], it)
        if li > 0:
            Q = _f.prolong_orientations(Q, levels[li - 1]["parent"],
                                        levels[li - 1]["N"])
    t_orient = time.time()

    # ---- position field + extraction (with a scale feedback loop) -------
    pit = int(p["pos_iters"])
    Qs = _restrict_orientations(levels, Q)
    best = None
    scale_fac = 1.0
    for _attempt in range(3):
        cur_rho = rho * scale_fac
        lev_rho = [cur_rho]
        for li in range(1, len(levels)):
            par = levels[li - 1]["parent"]
            cnt = np.bincount(par, minlength=levels[li]["n"]).astype(np.float64)
            lev_rho.append(
                np.bincount(par, weights=lev_rho[-1], minlength=levels[li]["n"]) / cnt
            )

        start_lv = len(levels) - 1
        O = levels[start_lv]["P"].copy()
        for li in range(start_lv, -1, -1):
            lv = levels[li]
            it = pit if li == 0 else max(5, pit // 2)
            O = _f.smooth_positions(O, lv["P"], Qs[li], lv["N"], lev_rho[li],
                                    lv["src"], lv["indices"], it,
                                    con_mask=lv["pin_mask"],
                                    con_dir=lv["con_dir"])
            if li > 0:
                par = levels[li - 1]["parent"]
                lf = levels[li - 1]
                O = _f.prolong_positions(O, par, lf["P"], Qs[li - 1], lf["N"],
                                          lev_rho[li - 1])

        VQ, FQ = extract_quads(O, Qs[0], N, cur_rho, edges,
                               bnd_verts=bnd_verts)
        nf = len(FQ)
        nq = sum(1 for f in FQ if len(f) == 4)
        score = abs(np.log(max(nf, 1) / float(target))) - 0.25 * (
            nq / float(max(nf, 1))
        )
        if best is None or score < best[0]:
            best = (score, VQ, FQ)
        if nf == 0:
            scale_fac *= 0.6
            continue
        ratio = nf / float(target)
        if 0.75 <= ratio <= 1.35:
            break
        # rho ~ 1/sqrt(face count)
        scale_fac *= float(np.clip(np.sqrt(ratio), 0.55, 1.8))

    _, VQ, FQ = best
    if len(FQ) == 0:
        raise SolveError(
            "field extraction produced no faces (input may be too coarse "
            "relative to the requested face count)"
        )

    if p.get("verbose"):
        t1 = time.time()
        nq = sum(1 for f in FQ if len(f) == 4)
        print(
            f"[quadforge.native] n={n} tris={len(F)} target={target} "
            f"levels={len(levels)} -> verts={len(VQ)} faces={len(FQ)} "
            f"quads={nq} ({100.0 * nq / max(len(FQ), 1):.1f}%) "
            f"hier={t_hier - t0:.2f}s orient={t_orient - t_hier:.2f}s "
            f"pos+extract={t1 - t_orient:.2f}s total={t1 - t0:.2f}s"
        )
    return VQ, FQ


def _restrict_orientations(levels, Q_fine):
    """Orientation per level: ``Q_fine`` is the finest-level field; build the
    coarse restrictions by averaging children (4-RoSy matched) so that the
    position solve has a frame at every level."""
    out = [None] * len(levels)
    out[0] = Q_fine
    for li in range(1, len(levels)):
        par = levels[li - 1]["parent"]
        nc = levels[li]["n"]
        Nf = levels[li - 1]["N"]
        Qf = out[li - 1]
        # reference = lowest-index child
        ref = np.zeros((nc, 3))
        rev = np.arange(len(par))[::-1]
        ref[par[rev]] = Qf[rev]
        Nc = levels[li]["N"]
        refc = ref - Nc * np.einsum("ij,ij->i", ref, Nc)[:, None]
        rl = np.sqrt(np.einsum("ij,ij->i", refc, refc))
        bad = rl < 1e-9
        if bad.any():
            refc[bad] = _f.random_tangents(Nc[bad], np.random.default_rng(0))
            rl = np.sqrt(np.einsum("ij,ij->i", refc, refc))
        refc = refc / np.maximum(rl, 1e-12)[:, None]

        # 4-RoSy accumulation of the children onto the reference
        perp = np.cross(Nf, Qf)
        r = refc[par]
        d0 = np.einsum("ij,ij->i", Qf, r)
        d1 = np.einsum("ij,ij->i", perp, r)
        use1 = np.abs(d1) > np.abs(d0)
        rep = np.where(use1[:, None], perp, Qf)
        sg = np.where(use1, d1, d0)
        rep = rep * np.where(sg < 0.0, -1.0, 1.0)[:, None]
        acc = np.empty((nc, 3))
        for c in range(3):
            acc[:, c] = np.bincount(par, weights=rep[:, c], minlength=nc)
        acc -= Nc * np.einsum("ij,ij->i", acc, Nc)[:, None]
        al = np.sqrt(np.einsum("ij,ij->i", acc, acc))
        good = al > 1e-9
        Qc = refc.copy()
        Qc[good] = acc[good] / al[good][:, None]
        out[li] = Qc
    return out
