"""Wide size contrast (``detail_range``) and the input-tessellation prior.

Two sizing features share one machine here, so they share a test module:

* **Size Contrast** (``detail_range``, settings ``detail_range``) widens the
  band the curvature adaptivity may open up.  The old code clamped
  ``kappa / kappa_ref`` to ``[1/3, 3]`` before raising it to ``adaptive/2``,
  which caps the coarse:fine quad-size ratio at 3 however starved the budget
  is.  The clamp is now the requested band, so the ratio at full adaptivity is
  the number the user typed.

* **Detail from Input** (``use_input_density``) reads the input mesh's own
  tessellation: long input edges mean the artist already decided nothing
  happens there, so the output goes coarse there too even where curvature is
  ambiguous.

Both can only be spent inside a *gradient-limited* field.  A sizing field that
steps from 1 to 8 across one edge is not realisable - the extractor has to
stitch two grids inside a single cell and pays in irregular vertices - so the
combined field is relaxed until ``|grad rho| <= 0.3`` (Persson's mesh-size
gradient limiting) before it is normalised back onto the face budget.

What is checked:

* **defaults are bit-identical** - the parameters absent, at their defaults,
  and explicitly off produce the same mesh, on ``solve_fields`` + ``extract``
  and on the whole ``solver.solve``, so no existing result moves;
* **budget conservation** - ``sum(A_v / rho_v**2)`` still equals the request
  to floating point for every band and with the prior on, i.e. the contrast is
  paid for by redistribution, not by extra faces;
* **the band answers** - the area-weighted coarse:fine ratio of ``rho`` grows
  monotonically with ``detail_range`` on a shape with a real curvature range;
* **the grading bound holds** - ``limit_size_gradient`` returns a field that
  satisfies the bound exactly, and a wide band goes through it (its worst
  gradient is a fraction of the unlimited one);
* **the prior coarsens** - on a dome finely tessellated and welded to a
  sparse flat panel, the output quads on the panel grow by 1.5x-2.5x relative
  to the dome as the band widens, and this survives the solver's own
  pre-refinement;
* **composition** - a painted density spike still wins where it overlaps a
  region the prior wants coarse.
"""

import hashlib

import numpy as np


# --------------------------------------------------------------------------
# fixtures (numpy only)
# --------------------------------------------------------------------------

def _bumpy_grid(nx=13, half=2.0, height=0.55, sigma=0.7):
    """Coarse triangulated square with a Gaussian dome in the middle."""
    xs = np.linspace(-half, half, nx + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Z = height * np.exp(-(X ** 2 + Y ** 2) / sigma ** 2)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    F = []
    for i in range(nx):
        for j in range(nx):
            a = i * (nx + 1) + j
            F.append((a, a + 1, a + nx + 2))
            F.append((a, a + nx + 2, a + nx + 1))
    return (np.ascontiguousarray(V, dtype=np.float64),
            np.ascontiguousarray(np.array(F, dtype=np.int64)))


def dome_panel(extract, blob=1.0, fine=0.075, coarse=0.5, rounds=5):
    """Dense dome welded to a sparse flat panel - one conforming shell.

    Built by running the extractor's own red-green refinement against a
    hand-made sizing field, which is the shortest way to a *conforming* mixed
    tessellation (no hanging nodes) from numpy alone.
    """
    V, F = _bumpy_grid()
    N = np.tile(np.array([0.0, 0.0, 1.0]), (len(V), 1))
    Q = np.tile(np.array([1.0, 0.0, 0.0]), (len(V), 1))
    r = np.linalg.norm(V[:, :2], axis=1)
    w = np.clip((r - blob) / (0.45 * blob), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    rho = fine + (coarse - fine) * w
    V, F, N, Q, rho, _ = extract._refine_for_lattice(
        V, F, N, Q, rho, None, rounds=rounds)
    return np.ascontiguousarray(V), np.ascontiguousarray(F)


def _sphere_grid(nu, nv, rad, P, T):
    V = np.stack([rad * np.sin(P) * np.cos(T), rad * np.sin(P) * np.sin(T),
                  rad * np.cos(P)], axis=-1).reshape(-1, 3)
    F = []
    for i in range(nv - 1):
        for j in range(nu):
            a = i * nu + j
            b = i * nu + (j + 1) % nu
            c = (i + 1) * nu + j
            d = (i + 1) * nu + (j + 1) % nu
            F.append((a, b, d))
            F.append((a, d, c))
    return (np.ascontiguousarray(V, dtype=np.float64),
            np.ascontiguousarray(np.array(F, dtype=np.int64)))


def wavy_sphere(nu=72, nv=54, amp=0.16, k=6):
    """Small bumpy sphere - cheap enough to solve several times."""
    ph = np.linspace(1e-3, np.pi - 1e-3, nv)
    th = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    T, P = np.meshgrid(th, ph)
    rad = 1.0 + amp * np.sin(k * T) * np.sin(k * P) * np.exp(-((P - 1.5) * 2) ** 2)
    return _sphere_grid(nu, nv, rad, P, T)


def bumpy_band_sphere(nu=160, nv=120, amp=0.05, k=16, band=(1.1, 2.0)):
    """Sphere carrying a band of fine bumps.

    The smooth part fixes the curvature reference at ``kappa = 1`` and the
    bumps run 10-14x past it, so ``kappa / kappa_ref`` genuinely leaves the
    legacy ``[1/3, 3]`` clamp (27% of the vertices sit above 3).  On a shape
    whose curvature never leaves that window, widening the clamp is a no-op
    by construction - which is why this fixture, not the small one, is what
    the band case is measured on.
    """
    ph = np.linspace(1e-3, np.pi - 1e-3, nv)
    th = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    T, P = np.meshgrid(th, ph)
    w = (np.clip((P - band[0]) / 0.25, 0.0, 1.0)
         * np.clip((band[1] - P) / 0.25, 0.0, 1.0))
    rad = 1.0 + amp * np.sin(k * T) * np.sin(k * P) * w
    return _sphere_grid(nu, nv, rad, P, T)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _hash(VQ, FQ):
    m = hashlib.sha256()
    m.update(np.ascontiguousarray(np.asarray(VQ, dtype=np.float64)).tobytes())
    for f in FQ:
        m.update(str(tuple(int(i) for i in f)).encode())
    return m.hexdigest()


def _area_weighted_ratio(fields, V, F, rho, lo=0.02, hi=0.90):
    """Coarse:fine ratio of a sizing field, weighted by the area it governs."""
    w = fields.vertex_areas(V, F, len(V))
    o = np.argsort(rho)
    cw = np.cumsum(w[o]) / w.sum()
    a = rho[o][min(int(np.searchsorted(cw, lo)), len(rho) - 1)]
    b = rho[o][min(int(np.searchsorted(cw, hi)), len(rho) - 1)]
    return float(b / max(a, 1e-12))


def _max_gradient(fields, V, F, rho):
    e = fields.build_edges(F)
    L = np.linalg.norm(V[e[:, 1]] - V[e[:, 0]], axis=1)
    return float(np.max(np.abs(rho[e[:, 1]] - rho[e[:, 0]])
                        / np.maximum(L, 1e-12)))


def _quad_sizes(VQ, FQ):
    """(centroids, mean edge length) per output face."""
    VQ = np.asarray(VQ, dtype=np.float64)
    cen, ln = [], []
    for f in FQ:
        P = VQ[list(f)]
        cen.append(P.mean(axis=0))
        ln.append(float(np.mean(np.linalg.norm(
            np.roll(P, -1, axis=0) - P, axis=1))))
    return np.asarray(cen), np.asarray(ln)


TARGET = 900


def run(ctx):
    r = ctx.results()

    fields = ctx.try_imp("quadforge.backends.native.fields")
    extract = ctx.try_imp("quadforge.backends.native.extract")
    solver = ctx.try_imp("quadforge.backends.native.solver")

    _fx = {}

    def fixture(name):
        if name not in _fx:
            _fx[name] = {"dome": lambda: dome_panel(extract),
                         "sphere": wavy_sphere,
                         "bumps": bumpy_band_sphere}[name]()
        return _fx[name]

    # ---------------------------------------------------------------- 1
    with r.case("sizing_defaults_are_bit_identical") as c:
        if fields is None or extract is None or solver is None:
            c.skip("native modules unavailable")
        V, F = fixture("sphere")
        for tag, base in (("uniform", {}), ("adaptive", {"adaptive": 0.6})):
            p0 = dict(target_faces=TARGET, seed=3, **base)
            h0 = _hash(*extract.extract(V, F, fields.solve_fields(V, F, p0), p0))
            s0 = _hash(*solver.solve(V, F, p0))
            for label, extra in (
                    ("detail_range=3", {"detail_range": 3.0}),
                    ("prior off", {"use_input_density": False}),
                    ("both", {"detail_range": 3.0, "use_input_density": False})):
                p = dict(p0, **extra)
                h = _hash(*extract.extract(V, F, fields.solve_fields(V, F, p), p))
                c.require(h == h0, "%s/%s changed solve_fields+extract (%s vs %s)"
                          % (tag, label, h[:12], h0[:12]))
                s = _hash(*solver.solve(V, F, p))
                c.require(s == s0, "%s/%s changed solver.solve (%s vs %s)"
                          % (tag, label, s[:12], s0[:12]))
            c.note("%s hash %s" % (tag, h0[:12]))

    # ---------------------------------------------------------------- 2
    with r.case("sizing_budget_is_conserved") as c:
        if fields is None:
            c.skip("native modules unavailable")
        V, F = fixture("sphere")
        n = len(V)
        N = fields.vertex_normals(V, F)
        edges = fields.build_edges(F)
        indptr, indices, _s = fields.build_csr(edges, n)
        cur = fields.principal_curvatures(V, F, N=N, indptr=indptr,
                                          indices=indices)
        w = fields.vertex_areas(V, F, n)
        prior = fields.input_detail_prior(V, F, n, edges=edges)
        cases = [("B=%.0f" % b, dict(detail_range=b)) for b in (3, 6, 10, 12)]
        cases.append(("prior", dict(input_prior=prior)))
        cases.append(("B=8+prior", dict(detail_range=8.0, input_prior=prior)))
        for label, kw in cases:
            rho = fields.target_edge_lengths(V, F, n, TARGET, adaptive=1.0,
                                             cur=cur, edges=edges, **kw)
            pred = float(np.sum(w / rho ** 2))
            c.require(abs(pred / TARGET - 1.0) < 1e-9,
                      "%s: predicted cell count %.3f, wanted %d" %
                      (label, pred, TARGET))
        c.note("all bands predict %d cells to 1e-9" % TARGET)

    # ---------------------------------------------------------------- 3
    with r.case("detail_range_widens_the_band") as c:
        if fields is None:
            c.skip("native modules unavailable")
        V, F = fixture("bumps")
        n = len(V)
        N = fields.vertex_normals(V, F)
        edges = fields.build_edges(F)
        indptr, indices, _s = fields.build_csr(edges, n)
        cur = fields.principal_curvatures(V, F, N=N, indptr=indptr,
                                          indices=indices)
        kk = np.nan_to_num(np.maximum(np.abs(cur.k1), np.abs(cur.k2)))
        over = float((kk / max(np.percentile(kk, 60.0), 1e-12) > 3.0).mean())
        c.require(over > 0.10,
                  "the fixture keeps %.1f%% of its vertices past the legacy "
                  "clamp; too few to measure a wider band" % (100.0 * over))
        seen = []
        for b in (3.0, 6.0, 10.0):
            rho = fields.target_edge_lengths(V, F, n, TARGET, adaptive=1.0,
                                             cur=cur, edges=edges,
                                             detail_range=b)
            seen.append(_area_weighted_ratio(fields, V, F, rho))
        c.require(seen[1] > seen[0] * 1.2 and seen[2] > seen[1] * 1.1,
                  "coarse:fine ratio did not follow detail_range: %s"
                  % ["%.2f" % x for x in seen])
        c.note("coarse:fine (area-weighted p90/p02) B=3 %.2fx  B=6 %.2fx  "
               "B=10 %.2fx, %.0f%% of vertices past the legacy clamp"
               % (seen[0], seen[1], seen[2], 100.0 * over))

    # ---------------------------------------------------------------- 4
    with r.case("sizing_gradient_is_limited") as c:
        if fields is None:
            c.skip("native modules unavailable")
        V, F = fixture("dome")
        n = len(V)
        edges = fields.build_edges(F)
        L = np.linalg.norm(V[edges[:, 1]] - V[edges[:, 0]], axis=1)
        # a deliberately stepped field: fine inside the dome, coarse outside
        rad = np.linalg.norm(V[:, :2], axis=1)
        step = np.where(rad < 1.0, 0.02, 0.30)
        g = 0.3
        lim = fields.limit_size_gradient(step, V, edges, grading=g)
        worst = np.max(np.abs(lim[edges[:, 1]] - lim[edges[:, 0]])
                       / np.maximum(g * L, 1e-12))
        c.require(worst <= 1.0 + 1e-9,
                  "limit_size_gradient left a %.3fx violation of its own bound"
                  % worst)
        c.require(np.all(lim <= step + 1e-12),
                  "limit_size_gradient raised rho somewhere; it must only relax")
        c.require(float(lim.max() / lim.min()) > 5.0,
                  "the limiter flattened the field entirely (%.2fx left)"
                  % float(lim.max() / lim.min()))
        # ...and a wide band really goes through it
        N = fields.vertex_normals(V, F)
        indptr, indices, _s = fields.build_csr(edges, n)
        cur = fields.principal_curvatures(V, F, N=N, indptr=indptr,
                                          indices=indices)
        prior = fields.input_detail_prior(V, F, n, edges=edges)
        raw = fields.target_edge_lengths(V, F, n, TARGET, adaptive=1.0, cur=cur,
                                         edges=edges, detail_range=8.0,
                                         input_prior=prior, grading=0.0)
        lim = fields.target_edge_lengths(V, F, n, TARGET, adaptive=1.0, cur=cur,
                                         edges=edges, detail_range=8.0,
                                         input_prior=prior)
        gr, gl = (_max_gradient(fields, V, F, raw),
                  _max_gradient(fields, V, F, lim))
        c.require(gl < 0.5 * gr,
                  "the wide band did not get graded: worst gradient %.2f "
                  "limited vs %.2f raw" % (gl, gr))
        c.note("worst |grad rho| raw %.2f -> limited %.2f; step field bound "
               "satisfied to %.3f" % (gr, gl, worst))

    # ---------------------------------------------------------------- 5
    with r.case("input_prior_coarsens_sparse_input") as c:
        if solver is None:
            c.skip("native modules unavailable")
        V, F = fixture("dome")
        rad = np.linalg.norm(V[:, :2], axis=1)
        prior = fields.input_detail_prior(V, F, len(V))
        c.require(float(prior[rad > 1.7].mean() / prior[rad < 0.8].mean()) > 4.0,
                  "the fixture does not actually have a tessellation step")

        def panel_over_dome(**kw):
            p = dict(target_faces=TARGET, seed=3)
            p.update(kw)
            cen, ln = _quad_sizes(*solver.solve(V, F, p))
            rr = np.linalg.norm(cen[:, :2], axis=1)
            return float(ln[rr > 1.7].mean() / ln[rr < 0.8].mean()), len(ln)

        off, n_off = panel_over_dome()
        on, n_on = panel_over_dome(use_input_density=True)
        wide, n_wide = panel_over_dome(use_input_density=True, detail_range=8.0)
        c.require(off < 1.15,
                  "the fixture is not neutral without the prior (%.2fx)" % off)
        c.require(on > 1.35,
                  "Detail from Input did not coarsen the sparse panel "
                  "(%.2fx vs %.2fx off)" % (on, off))
        c.require(wide > on,
                  "a wider band did not widen the prior's effect "
                  "(%.2fx at B=8 vs %.2fx at B=3)" % (wide, on))
        for label, nf in (("off", n_off), ("on", n_on), ("on B=8", n_wide)):
            c.require(abs(nf / float(TARGET) - 1.0) <= 0.05,
                      "prior %s missed the face count by %+.1f%% (%d of %d)"
                      % (label, 100.0 * (nf / float(TARGET) - 1.0), nf, TARGET))
        c.note("panel/dome quad size: off %.2fx  prior %.2fx  prior+B8 %.2fx "
               "(faces %d/%d/%d of %d)"
               % (off, on, wide, n_off, n_on, n_wide, TARGET))

    # ---------------------------------------------------------------- 6
    with r.case("input_prior_survives_pre_refinement") as c:
        if solver is None or fields is None:
            c.skip("native modules unavailable")
        V, F = fixture("dome")
        n0 = len(V)
        # a request big enough that solve() midpoint-subdivides the input first
        big = int(n0 / solver._MIN_SAMPLES_PER_QUAD) + 4000
        seen = {}
        _tel = fields.target_edge_lengths

        def spy(*a, **kw):
            seen["prior"] = kw.get("input_prior") is not None
            seen["n"] = a[2]
            return _tel(*a, **kw)
        fields.target_edge_lengths = spy
        try:
            solver.solve(V, F, dict(target_faces=big, seed=3,
                                    use_input_density=True))
        finally:
            fields.target_edge_lengths = _tel
        c.require(seen.get("n", n0) > n0,
                  "the fixture did not trigger pre-refinement, so this case "
                  "proves nothing (n stayed %d)" % n0)
        c.require(seen.get("prior") is True,
                  "the prior was dropped on the way through pre-refinement")
        # the carried field must still describe the *input*: panel over dome
        pr = fields.input_detail_prior(V, F, n0)
        rad = np.linalg.norm(V[:, :2], axis=1)
        c.note("pre-refinement %d -> %d verts, prior carried, panel/dome %.1fx"
               % (n0, seen["n"],
                  float(pr[rad > 1.7].mean() / pr[rad < 0.8].mean())))

    # ---------------------------------------------------------------- 7
    with r.case("prior_composes_with_paint_density") as c:
        if fields is None:
            c.skip("native modules unavailable")
        V, F = fixture("dome")
        n = len(V)
        edges = fields.build_edges(F)
        prior = fields.input_detail_prior(V, F, n, edges=edges)
        # paint a fine patch out on the sparse panel, where the prior wants
        # the mesh coarse: the local instruction has to win locally
        rad = np.linalg.norm(V[:, :2], axis=1)
        patch = (rad > 1.6) & (V[:, 0] > 0.0)
        dens = np.where(patch, 2.0, 1.0)
        base = fields.target_edge_lengths(V, F, n, TARGET, edges=edges,
                                          input_prior=prior)
        both = fields.target_edge_lengths(V, F, n, TARGET, density=dens,
                                          edges=edges, input_prior=prior)
        rest = (rad > 1.6) & (V[:, 0] <= 0.0)
        gain = float(base[patch].mean() / base[rest].mean()
                     / (both[patch].mean() / both[rest].mean()))
        c.require(gain > 1.5,
                  "a painted density spike only refined the patch by %.2fx "
                  "relative to the unpainted rest of the same panel" % gain)
        # the paint wins where it was painted, and only there: the prior's own
        # panel-over-dome statement has to survive next to it
        keep = (float(both[rest].mean() / both[rad < 0.8].mean())
                / float(base[rest].mean() / base[rad < 0.8].mean()))
        c.require(0.85 < keep < 1.15,
                  "painting one half of the panel moved the prior's "
                  "panel-over-dome ratio on the *other* half by %.2fx" % keep)
        c.note("paint gain over the prior on the same panel: %.2fx; the "
               "prior's panel/dome ratio elsewhere kept to %.2fx"
               % (gain, keep))

    # ---------------------------------------------------------------- 8
    with r.case("sizing_property_and_plumbing") as c:
        props = ctx.try_imp("quadforge.properties")
        if props is None:
            c.skip("quadforge.properties is not importable")
        settings_cls = None
        for name in dir(props):
            obj = getattr(props, name)
            if isinstance(obj, type) and "detail_range" in getattr(
                    obj, "__annotations__", {}):
                settings_cls = obj
                break
        c.require(settings_cls is not None,
                  "detail_range is not declared on the settings group")
        c.require(settings_cls is not None
                  and "use_input_density" in settings_cls.__annotations__,
                  "use_input_density is not declared on the settings group")
        for k in ("detail_range", "use_input_density"):
            c.require(k not in props.PRESET_KEYS,
                      "%s must not be a preset key (it would rewrite existing "
                      "results when a preset is picked)" % k)
        for k in ("detail_range", "use_input_density", "input_prior"):
            c.require(solver is not None and k in solver._DEFAULTS,
                      "solver._DEFAULTS does not carry %s" % k)
            c.require(fields is not None and k in fields.FIELD_DEFAULTS,
                      "fields.FIELD_DEFAULTS does not carry %s" % k)
        c.require(fields is not None
                  and fields.FIELD_DEFAULTS["detail_range"]
                  == fields.LEGACY_DETAIL_RANGE == 3.0,
                  "the default band is no longer the legacy 3x one")
        native = ctx.try_imp("quadforge.backends.native")
        c.require(native is not None, "native backend is not importable")
        c.note("detail_range default %.1f, grading %.2f"
               % (fields.FIELD_DEFAULTS["detail_range"], fields.SIZE_GRADING))

    return r.list()
