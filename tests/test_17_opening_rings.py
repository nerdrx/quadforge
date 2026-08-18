"""Opening rings: concentric edge loops around eye sockets and mouth rims.

The feature (``quadforge.backends.native.rings``) turns every small closed
boundary loop of the solve mesh into a ring instruction for the orientation
field, and a resolution request so those loops can actually close.  All of
it is checked here on synthetic fixtures built in numpy, so the module can be
exercised without touching a sculpt:

* **detection** - a disc with a hole reports exactly one opening (its outer
  border is too long to be one), a symmetry-bisect cut is not an opening, and
  a closed mesh reports none;
* **alignment** - with rings on, the extracted edges inside a 4-quad band
  around the hole line up with the iso-distance tangent far better than
  without.  Measured medians on the two fixtures, 6 seeds each:

      fixture   rings off        rings on
      disc      17.9-20.8 deg    4.3-6.1 deg
      sphere    15.1-21.6 deg    4.1-6.5 deg

  the thresholds below are set with headroom around those numbers;
* **resolution** - a direction is not enough at a game budget.  The band
  density boost gives every opening ``ring_min_quads`` quads around it (32 by
  default, capped at 2x), paid for out of the same cell budget: the predicted
  cell count ``sum(A_v / rho_v**2)`` is *identical* with the boost on and off,
  which is checked here to floating-point equality.  Measured quads around the
  hole, 4 seeds: disc 21-22 -> 32-35, sphere 19-21 -> 34-37;
* **closed loops** - the first offset contour is handed to the extractor as a
  feature curve, so the loop exists by construction.  Best loop purity over
  the first two rings (fraction of the ring's vertices with exactly two
  neighbours inside it) goes from 0.22-0.71 to 0.74-1.00;
* **off is off** - with ``use_opening_rings`` False the extracted mesh is
  bit-identical to the same solve with the parameter absent entirely, on both
  the ``solve_fields`` + ``extract`` path and the whole ``solver.solve``
  pipeline, so the flag cannot cost anything when it is not asked for.

The angles are 4-RoSy angles: an edge that runs *across* the ring (a radial
spoke) is as aligned as one that runs along it, because both belong to the
same cross field.
"""

import hashlib

import numpy as np


# --------------------------------------------------------------------------
# fixtures (numpy only - no bpy, no addon state)
# --------------------------------------------------------------------------

def _grid_faces(nr, nt, wrap=True):
    F = []
    for i in range(nr - 1):
        for j in range(nt if wrap else nt - 1):
            a = i * nt + j
            b = i * nt + (j + 1) % nt
            c = (i + 1) * nt + j
            d = (i + 1) * nt + (j + 1) % nt
            F.append((a, b, d))
            F.append((a, d, c))
    return np.asarray(F, dtype=np.int64)


def annulus(R=1.0, r=0.15, nr=44, nt=96):
    """Flat disc in z = 0 with a circular hole at the origin."""
    rad = r * (R / r) ** np.linspace(0.0, 1.0, nr)
    th = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    T, Rr = np.meshgrid(th, rad)
    V = np.stack([Rr * np.cos(T), Rr * np.sin(T), np.zeros_like(T)],
                 axis=-1).reshape(-1, 3)
    return V, _grid_faces(nr, nt)


def holed_sphere(cap_deg=15.0, nu=64, nv=48):
    """Unit sphere with the north polar cap removed."""
    ph = np.linspace(np.radians(cap_deg), np.pi, nv)
    th = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    T, P = np.meshgrid(th, ph)
    V = np.stack([np.sin(P) * np.cos(T), np.sin(P) * np.sin(T),
                  np.cos(P)], axis=-1).reshape(-1, 3)
    F = _grid_faces(nv, nu)
    last = (nv - 1) * nu                     # weld the degenerate south ring
    keep = np.arange(len(V))
    keep[last:] = last
    F = keep[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    used = np.unique(F)
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def _half_ball(radius=0.08, nu=32, nv=13):
    """Sphere with its pole on +x, capped there and cut off exactly at x = 0.

    The single boundary is therefore one closed loop that lies *exactly* in
    the x = 0 plane - a stand-in for whatever the pipeline's exact-symmetry
    bisect leaves behind on a small closed part.
    """
    phi = np.linspace(0.0, 0.5 * np.pi, nv)[1:]        # drop the pole ring
    th = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    T, P = np.meshgrid(th, phi)
    V = radius * np.stack([np.cos(P), np.sin(P) * np.cos(T),
                           np.sin(P) * np.sin(T)], axis=-1).reshape(-1, 3)
    F = list(map(tuple, _grid_faces(len(phi), nu)))
    pole = len(V)
    V = np.vstack([V, [radius, 0.0, 0.0]])
    F += [(pole, (j + 1) % nu, j) for j in range(nu)]
    return V, np.asarray(F, dtype=np.int64)


# --------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------

def _edges(FQ):
    ep = set()
    for f in FQ:
        k = len(f)
        for i in range(k):
            a, b = int(f[i]), int(f[(i + 1) % k])
            ep.add((min(a, b), max(a, b)))
    return np.asarray(sorted(ep), dtype=np.int64)


def rosy4_median(VQ, FQ, ref_fn, sel_fn):
    """Median 4-RoSy angle (deg) between output edges in the band and the
    reference ring direction at their midpoints."""
    VQ = np.asarray(VQ, dtype=np.float64)
    e = _edges(FQ)
    mid = 0.5 * (VQ[e[:, 0]] + VQ[e[:, 1]])
    d = VQ[e[:, 1]] - VQ[e[:, 0]]
    ln = np.sqrt(np.einsum("ij,ij->i", d, d))
    ok = ln > 1e-12
    mid, d, ln = mid[ok], d[ok], ln[ok]
    sel = sel_fn(mid)
    if not sel.any():
        return float("nan"), 0
    dd = d[sel] / ln[sel][:, None]
    ref = ref_fn(mid[sel])
    c = np.clip(np.abs(np.einsum("ij,ij->i", dd, ref)), 0.0, 1.0)
    ang = np.degrees(np.arccos(c))
    return float(np.median(np.minimum(ang, 90.0 - ang))), int(sel.sum())


def _tangent_about_z(P):
    t = np.stack([-P[:, 1], P[:, 0], np.zeros(len(P))], axis=1)
    return t / np.maximum(np.linalg.norm(t, axis=1), 1e-12)[:, None]


def _hash(VQ, FQ):
    m = hashlib.sha256()
    m.update(np.ascontiguousarray(np.asarray(VQ, dtype=np.float64)).tobytes())
    for f in FQ:
        m.update(str(tuple(int(i) for i in f)).encode("ascii"))
    return m.hexdigest()


def _max_valence(VQ, FQ):
    val = np.zeros(len(VQ), dtype=np.int64)
    for a, b in _edges(FQ):
        val[a] += 1
        val[b] += 1
    live = val > 0
    return int(val[live].max()) if live.any() else 0


def _adjacency(VQ, FQ):
    adj = [set() for _ in range(len(VQ))]
    for a, b in _edges(FQ):
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _rim_edges(VQ, FQ, center, tol):
    """Output boundary edges belonging to the hole nearest ``center``."""
    use = {}
    for f in FQ:
        k = len(f)
        for i in range(k):
            a, b = int(f[i]), int(f[(i + 1) % k])
            key = (min(a, b), max(a, b))
            use[key] = use.get(key, 0) + 1
    be = np.asarray([k for k, v in use.items() if v == 1], dtype=np.int64)
    if not len(be):
        return np.zeros((0, 2), dtype=np.int64)
    VQ = np.asarray(VQ, dtype=np.float64)
    mid = 0.5 * (VQ[be[:, 0]] + VQ[be[:, 1]])
    return be[np.linalg.norm(mid - np.asarray(center), axis=1) < tol]


def _quads_around(VQ, FQ, center, tol):
    return int(len(_rim_edges(VQ, FQ, center, tol)))


def _loop_purity(VQ, FQ, center, tol, nrings=2):
    """Per-ring fraction of vertices that have exactly two neighbours inside
    their own ring - i.e. how much of the ring is a genuine closed loop.

    Ring *k* is the set of output vertices at graph distance *k* from the
    hole's own boundary loop, so ring 1 is what an artist would call the first
    eyelid loop.  A ring that is really a loop scores 1.0; a ring that the
    grid merely clips scores well below.
    """
    rim = _rim_edges(VQ, FQ, center, tol)
    if not len(rim):
        return []
    adj = _adjacency(VQ, FQ)
    seen = set(int(v) for v in np.unique(rim))
    cur = set(seen)
    out = []
    for _k in range(nrings):
        nxt = set()
        for v in cur:
            nxt |= adj[v]
        nxt -= seen
        if not nxt:
            break
        out.append(sum(1 for v in nxt if len(adj[v] & nxt) == 2) / float(len(nxt)))
        seen |= nxt
        cur = nxt
    return out


# --------------------------------------------------------------------------

TARGET = 1500


def run(ctx):
    r = ctx.results()

    rings = ctx.try_imp("quadforge.backends.native.rings")
    fields = ctx.try_imp("quadforge.backends.native.fields")
    extract = ctx.try_imp("quadforge.backends.native.extract")

    _cache = {}

    def solve(V, F, on, seed=1, _key=None, **kw):
        # the fixtures are rebuilt per call but are deterministic, so a key of
        # (fixture name, flags) is enough to reuse a solve across cases
        key = (_key, on, seed, tuple(sorted(kw.items()))) if _key else None
        if key is not None and key in _cache:
            return _cache[key]
        p = dict(target_faces=TARGET, seed=seed, use_opening_rings=on)
        p.update(kw)
        sol = fields.solve_fields(V, F, p)
        VQ, FQ = extract.extract(V, F, sol, p)
        out = (sol, np.asarray(VQ), FQ)
        if key is not None:
            _cache[key] = out
        return out

    # (fixture key, builder, hole centre, radius that isolates the hole)
    FIXTURES = (("disc", annulus, np.zeros(3), 0.5),
                ("sphere", holed_sphere, np.array([0.0, 0.0, 1.0]), 0.6))

    # ---------------------------------------------------------------- 1
    with r.case("rings_detect_openings") as c:
        if rings is None or fields is None:
            c.skip("quadforge.backends.native.rings is not importable")
        V, F = annulus()
        rho = np.sqrt(np.pi * (1.0 - 0.15 ** 2) / TARGET)
        ops = rings.detect_openings(V, F, rho_mean=rho)
        c.require(len(ops) == 1,
                  "disc with a hole: expected 1 opening, got %d" % len(ops))
        op = ops[0]
        c.require(abs(op.perimeter - 2 * np.pi * 0.15) < 0.05,
                  "opening perimeter %.4f, expected ~%.4f"
                  % (op.perimeter, 2 * np.pi * 0.15))
        c.require(float(np.linalg.norm(op.center)) < 0.02,
                  "opening centre %s is not the hole" % (op.center,))
        c.note("perimeter=%.4f verts=%d" % (op.perimeter, len(op.verts)))

    with r.case("rings_ignore_symmetry_cut") as c:
        if rings is None:
            c.skip("rings module unavailable")
        # A big flat sheet plus a small sphere bisected at x = 0: the sphere's
        # cut rim is a small closed loop lying entirely in the symmetry plane,
        # which is exactly what the pipeline's exact-symmetry bisect leaves
        # behind and exactly what must NOT be ringed.  It passes the size
        # filter (the sheet dominates the area), so only the plane test can
        # reject it.
        gx, gy = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(-1, 1, 40))
        Vs = np.stack([gx, gy, np.full_like(gx, -0.5)], axis=-1).reshape(-1, 3)
        Fs = _grid_faces(40, 40, wrap=False)
        Vb, Fb = _half_ball(0.08, nu=32, nv=13)
        V = np.vstack([Vs, Vb])
        F = np.vstack([Fs, Fb + len(Vs)])
        no_sym = rings.detect_openings(V, F, symmetry=(False, False, False))
        with_sym = rings.detect_openings(V, F, symmetry=(True, False, False))
        c.require(len(no_sym) >= 1,
                  "the bisected sphere's rim should look like an opening "
                  "when no symmetry is declared (got %d)" % len(no_sym))
        c.require(len(with_sym) == 0,
                  "a loop lying in the declared symmetry plane is the bisect "
                  "cut, not an opening (got %d)" % len(with_sym))
        # ...and a closed mesh has none at all
        Vc, Fc = holed_sphere(cap_deg=0.5, nu=32, nv=24)
        Vc = np.vstack([Vc, [0.0, 0.0, 1.0]])
        rim = np.arange(32)
        cap = np.stack([np.full(32, len(Vc) - 1), np.roll(rim, -1), rim],
                       axis=1)
        ops_closed = rings.detect_openings(Vc, np.vstack([Fc, cap]))
        c.require(len(ops_closed) == 0,
                  "closed mesh reported %d openings" % len(ops_closed))
        c.note("cut fixture: %d openings without symmetry, %d with"
               % (len(no_sym), len(with_sym)))

    # ---------------------------------------------------------------- 2
    with r.case("rings_align_disc") as c:
        if rings is None or extract is None:
            c.skip("native modules unavailable")
        V, F = annulus()
        rh = 0.15
        rho = np.sqrt(np.pi * (1.0 - rh ** 2) / TARGET)
        band = 4.0 * rho
        rad = lambda P: np.linalg.norm(P[:, :2], axis=1)          # noqa: E731
        sel = lambda P: (rad(P) > rh * 1.02) & (rad(P) < rh + band)  # noqa: E731
        _, VQ0, FQ0 = solve(V, F, False, _key="disc")
        _, VQ1, FQ1 = solve(V, F, True, _key="disc")
        off, n0 = rosy4_median(VQ0, FQ0, _tangent_about_z, sel)
        on, n1 = rosy4_median(VQ1, FQ1, _tangent_about_z, sel)
        c.note("band edges %d/%d  off=%.2f deg  on=%.2f deg" % (n0, n1, off, on))
        c.require(n1 > 40, "only %d edges in the 4-quad band" % n1)
        c.require(on < 8.0, "rings on: 4-RoSy median %.2f deg (want < 8)" % on)
        c.require(on < 0.45 * off,
                  "rings on (%.2f deg) is not a clear win over off (%.2f deg)"
                  % (on, off))
        c.require(_max_valence(VQ1, FQ1) <= 12,
                  "rings on produced a valence-%d vertex: the hole collapsed"
                  % _max_valence(VQ1, FQ1))

    with r.case("rings_align_sphere") as c:
        if rings is None or extract is None:
            c.skip("native modules unavailable")
        V, F = holed_sphere()
        cap = np.radians(15.0)
        area = 2 * np.pi * (1 + np.cos(cap))
        rho = np.sqrt(area / TARGET)
        band = 4.0 * rho
        pol = lambda P: np.arccos(np.clip(                        # noqa: E731
            P[:, 2] / np.maximum(np.linalg.norm(P, axis=1), 1e-12), -1, 1))
        sel = lambda P: (pol(P) > cap * 1.02) & (pol(P) < cap + band)  # noqa
        _, VQ0, FQ0 = solve(V, F, False, _key="sphere")
        _, VQ1, FQ1 = solve(V, F, True, _key="sphere")
        off, n0 = rosy4_median(VQ0, FQ0, _tangent_about_z, sel)
        on, n1 = rosy4_median(VQ1, FQ1, _tangent_about_z, sel)
        c.note("band edges %d/%d  off=%.2f deg  on=%.2f deg" % (n0, n1, off, on))
        c.require(n1 > 40, "only %d edges in the 4-quad band" % n1)
        c.require(on < 8.0, "rings on: 4-RoSy median %.2f deg (want < 8)" % on)
        c.require(on < 0.45 * off,
                  "rings on (%.2f deg) is not a clear win over off (%.2f deg)"
                  % (on, off))
        c.require(_max_valence(VQ1, FQ1) <= 14,
                  "rings on produced a valence-%d vertex: the hole collapsed"
                  % _max_valence(VQ1, FQ1))

    # ---------------------------------------------------------------- 2b
    with r.case("rings_density_conserves_budget") as c:
        # The band boost is a *reallocation*: sum(A_v / rho_v**2) - the number
        # of rho-sized cells the surface carries, i.e. the face budget the
        # field encodes - must come out of the boost unchanged, or the density
        # profile silently sets the face count (see solver.solve's identical
        # rescale for the feature-density boost).
        if rings is None or fields is None:
            c.skip("native modules unavailable")
        for name, build, _ctr, _tol in FIXTURES:
            V, F = build()
            wa = fields.vertex_areas(V, F, len(V))
            s0, _, _ = solve(V, F, False, _key=name)
            s1, _, _ = solve(V, F, True, _key=name)
            b0 = float(np.sum(wa / np.maximum(s0.rho, 1e-12) ** 2))
            b1 = float(np.sum(wa / np.maximum(s1.rho, 1e-12) ** 2))
            c.require(abs(b1 / b0 - 1.0) < 1e-9,
                      "%s: the ring boost moved the cell budget by %.3f%% "
                      "(%.1f -> %.1f)" % (name, 100.0 * (b1 / b0 - 1.0), b0, b1))
            gain = s1.stats.get("ring_boost_max", 1.0)
            cap = rings.RING_DEFAULTS["ring_density_max"]
            c.require(1.0 < gain <= cap + 1e-9,
                      "%s: boost %.3f outside (1, %.2f]" % (name, gain, cap))
            # ...and the boost really is local: the band got finer, the rest
            # of the surface paid for it by getting very slightly coarser
            ratio = s1.rho / np.maximum(s0.rho, 1e-12)
            c.require(ratio.min() < 0.95 and ratio.max() > 1.0,
                      "%s: rho ratio %.3f..%.3f is not a local reallocation"
                      % (name, ratio.min(), ratio.max()))
            c.note("%s budget %.1f (x%.7f) boost=%.2f rho ratio %.3f..%.3f"
                   % (name, b0, b1 / b0, gain, ratio.min(), ratio.max()))

    with r.case("rings_density_budget_cap") as c:
        # An absurd quota on a mesh whose openings are big must not be allowed
        # to drag the whole budget into the bands.
        if rings is None or fields is None:
            c.skip("native modules unavailable")
        V, F = annulus()
        wa = fields.vertex_areas(V, F, len(V))
        s0, _, _ = solve(V, F, False, _key="disc")
        b0 = float(np.sum(wa / np.maximum(s0.rho, 1e-12) ** 2))
        cap = 0.05
        p = dict(target_faces=TARGET, seed=1, use_opening_rings=True,
                 ring_min_quads=400.0, ring_density_max=20.0,
                 ring_density_budget=cap)
        s1 = fields.solve_fields(V, F, p)
        dem = s1.stats.get("ring_boost_demand", 1.0)
        b1 = float(np.sum(wa / np.maximum(s1.rho, 1e-12) ** 2))
        c.require(dem <= 1.0 + cap + 1e-6,
                  "the band claimed %.3fx the budget with a %.2f cap" % (dem, cap))
        c.require(abs(b1 / b0 - 1.0) < 1e-9,
                  "capped boost moved the cell budget by %.3f%%"
                  % (100.0 * (b1 / b0 - 1.0)))
        c.note("demand=%.4f (cap %.2f) boost=%.2f budget x%.7f"
               % (dem, cap, s1.stats.get("ring_boost_max", 1.0), b1 / b0))

    with r.case("rings_quads_around_opening") as c:
        # The whole point of the boost: an opening a 12k-face head gives ten-odd
        # quads cannot carry a loop, so the band is refined until it can.
        if rings is None or extract is None:
            c.skip("native modules unavailable")
        quota = rings.RING_DEFAULTS["ring_min_quads"]
        for name, build, ctr, tol in FIXTURES:
            V, F = build()
            _, V0, F0 = solve(V, F, False, _key=name)
            _, V1, F1 = solve(V, F, True, _key=name)
            n0 = _quads_around(V0, F0, ctr, tol)
            n1 = _quads_around(V1, F1, ctr, tol)
            c.require(n0 > 0 and n1 > 0, "%s: no hole in the output" % name)
            c.require(n1 >= 0.85 * quota,
                      "%s: %d quads around the opening, quota is %g"
                      % (name, n1, quota))
            c.require(n1 > 1.3 * n0,
                      "%s: %d quads around with rings vs %d without is not a "
                      "boost" % (name, n1, n0))
            c.note("%s quads around the hole %d -> %d (quota %g)"
                   % (name, n0, n1, quota))

    # ---------------------------------------------------------------- 2c
    with r.case("rings_pinned_loop_exists") as c:
        # ring_pin_segments hands the extractor the first offset contour as a
        # feature curve.  It has to be a genuine cycle: the extractor treats a
        # feature vertex of degree != 2 as a corner that cannot slide, so a
        # contour with junctions freezes the lattice onto the staircase
        # instead of laying a loop along it.
        if rings is None or fields is None:
            c.skip("native modules unavailable")
        for name, build, _ctr, _tol in FIXTURES:
            V, F = build()
            sol, _, _ = solve(V, F, True, _key=name)
            c.require(getattr(sol, "ring_dist", None) is not None,
                      "%s: solve_fields did not publish ring_dist" % name)
            segs = rings.ring_pin_segments(V, F, sol.ring_dist, sol.rho,
                                           params=dict(target_faces=TARGET))
            c.require(len(segs) >= 8,
                      "%s: only %d pin segments" % (name, len(segs)))
            deg = np.bincount(segs.ravel())
            deg = deg[deg > 0]
            c.require(int(deg.min()) == 2 and int(deg.max()) == 2,
                      "%s: pin contour is not a closed cycle (degrees %d..%d)"
                      % (name, deg.min(), deg.max()))
            # ...and it is far enough out that a row of cells fits between it
            # and the pinned rim
            dq = sol.ring_dist[np.unique(segs)] / np.maximum(
                sol.rho[np.unique(segs)], 1e-12)
            c.require(float(np.min(dq)) >= rings.RING_DEFAULTS["ring_pin_min_gap"],
                      "%s: pin contour comes within %.2f quads of the rim"
                      % (name, float(np.min(dq))))
            c.note("%s pin cycle: %d segments at %.2f-%.2f quads out"
                   % (name, len(segs), float(np.min(dq)), float(np.max(dq))))

    with r.case("rings_loops_close") as c:
        if rings is None or extract is None:
            c.skip("native modules unavailable")
        for name, build, ctr, tol in FIXTURES:
            V, F = build()
            offs, ons = [], []
            for seed in (1, 2, 3):
                _, V0, F0 = solve(V, F, False, seed=seed, _key=name)
                _, V1, F1 = solve(V, F, True, seed=seed, _key=name)
                p0 = _loop_purity(V0, F0, ctr, tol)
                p1 = _loop_purity(V1, F1, ctr, tol)
                c.require(p0 and p1, "%s: no rings around the hole" % name)
                offs.append(max(p0))
                ons.append(max(p1))
            m0, m1 = float(np.median(offs)), float(np.median(ons))
            c.require(m1 >= 0.72,
                      "%s: best loop purity %.2f with rings on (want >= 0.72)"
                      % (name, m1))
            c.require(m1 > m0 + 0.10,
                      "%s: loop purity %.2f on vs %.2f off is not a clear win"
                      % (name, m1, m0))
            c.note("%s loop purity off=%.2f on=%.2f (medians of 3 seeds)"
                   % (name, m0, m1))
        # and the pin is really what does it: switching it off changes the mesh
        V, F = annulus()
        _, Vp, Fp = solve(V, F, True, _key="disc")
        _, Vn, Fn = solve(V, F, True, _key="disc", ring_pin=False)
        c.require(_hash(Vp, Fp) != _hash(Vn, Fn),
                  "ring_pin=False produced the same mesh: the pin is not wired "
                  "through to the extractor")

    # ---------------------------------------------------------------- 3
    with r.case("rings_off_is_a_no_op") as c:
        if fields is None or extract is None:
            c.skip("native modules unavailable")
        solver = ctx.try_imp("quadforge.backends.native.solver")
        for name, (V, F) in (("disc", annulus()), ("sphere", holed_sphere())):
            p_absent = dict(target_faces=TARGET, seed=3)
            sol = fields.solve_fields(V, F, p_absent)
            h_absent = _hash(*extract.extract(V, F, sol, p_absent))
            p_off = dict(p_absent, use_opening_rings=False)
            sol = fields.solve_fields(V, F, p_off)
            h_off = _hash(*extract.extract(V, F, sol, p_off))
            c.require(h_absent == h_off,
                      "%s: use_opening_rings=False changed the output "
                      "(%s vs %s)" % (name, h_absent[:12], h_off[:12]))
            # ...and the same through the whole backend, which is where the
            # extractor's ring-pin hook and the feature-density boost live
            if solver is not None:
                s_absent = _hash(*solver.solve(V, F, p_absent))
                s_off = _hash(*solver.solve(V, F, p_off))
                c.require(s_absent == s_off,
                          "%s: solver.solve differs with use_opening_rings "
                          "present-but-False (%s vs %s)"
                          % (name, s_absent[:12], s_off[:12]))
            c.note("%s hash %s" % (name, h_off[:12]))

    # ---------------------------------------------------------------- 4
    with r.case("rings_property_and_plumbing") as c:
        props = ctx.try_imp("quadforge.properties")
        if props is None:
            c.skip("quadforge.properties is not importable")
        settings_cls = getattr(props, "QuadForgeSettings", None)
        if settings_cls is None:
            for name in dir(props):
                obj = getattr(props, name)
                if isinstance(obj, type) and hasattr(obj, "__annotations__") \
                        and "use_opening_rings" in getattr(
                            obj, "__annotations__", {}):
                    settings_cls = obj
                    break
        c.require(settings_cls is not None
                  and "use_opening_rings" in settings_cls.__annotations__,
                  "use_opening_rings is not declared on the settings group")
        c.require("use_opening_rings" not in props.PRESET_KEYS,
                  "use_opening_rings must not be a preset key")
        native = ctx.try_imp("quadforge.backends.native")
        solver = ctx.try_imp("quadforge.backends.native.solver")
        c.require(solver is not None
                  and "use_opening_rings" in solver._DEFAULTS,
                  "solver._DEFAULTS does not carry use_opening_rings")
        c.require(fields is not None
                  and "use_opening_rings" in fields.FIELD_DEFAULTS,
                  "fields.FIELD_DEFAULTS does not carry use_opening_rings")
        c.require(native is not None, "native backend is not importable")
        missing = [k for k in ("ring_falloff", "ring_plateau", "ring_strength",
                               "ring_min_quads", "ring_density_max",
                               "ring_density_plateau", "ring_density_decay",
                               "ring_density_budget", "ring_pin_offset",
                               "ring_pin_min_gap")
                   if rings is None or k not in rings.RING_DEFAULTS]
        c.require(not missing,
                  "rings.RING_DEFAULTS is missing %s" % (missing,))
        c.note("plumbed through properties -> native -> solver -> fields; "
               "%d ring knobs" % (0 if rings is None else len(rings.RING_DEFAULTS)))

    return r.list()
