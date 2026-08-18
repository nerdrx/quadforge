"""Opening rings: concentric edge loops around eye sockets and mouth rims.

The feature (``quadforge.backends.native.rings``) turns every small closed
boundary loop of the solve mesh into a ring instruction for the orientation
field.  Three things are checked here, all on synthetic fixtures built in
numpy so the module can be exercised without touching a sculpt:

* **detection** - a disc with a hole reports exactly one opening (its outer
  border is too long to be one), a symmetry-bisect cut is not an opening, and
  a closed mesh reports none;
* **alignment** - with rings on, the extracted edges inside a 4-quad band
  around the hole line up with the iso-distance tangent far better than
  without.  Measured medians on the two fixtures, 6 seeds each:

      fixture   rings off        rings on
      disc      17.9-20.8 deg    7.4-9.0 deg
      sphere    15.1-21.6 deg    7.4-9.5 deg

  the thresholds below are set with headroom around those numbers;
* **off is off** - with ``use_opening_rings`` False the extracted mesh is
  bit-identical to the same solve with the parameter absent entirely, so the
  flag cannot cost anything when it is not asked for.

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


# --------------------------------------------------------------------------

TARGET = 1500


def run(ctx):
    r = ctx.results()

    rings = ctx.try_imp("quadforge.backends.native.rings")
    fields = ctx.try_imp("quadforge.backends.native.fields")
    extract = ctx.try_imp("quadforge.backends.native.extract")

    def solve(V, F, on, seed=1, **kw):
        p = dict(target_faces=TARGET, seed=seed, use_opening_rings=on)
        p.update(kw)
        sol = fields.solve_fields(V, F, p)
        VQ, FQ = extract.extract(V, F, sol, p)
        return sol, np.asarray(VQ), FQ

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
        _, VQ0, FQ0 = solve(V, F, False)
        _, VQ1, FQ1 = solve(V, F, True)
        off, n0 = rosy4_median(VQ0, FQ0, _tangent_about_z, sel)
        on, n1 = rosy4_median(VQ1, FQ1, _tangent_about_z, sel)
        c.note("band edges %d/%d  off=%.2f deg  on=%.2f deg" % (n0, n1, off, on))
        c.require(n1 > 40, "only %d edges in the 4-quad band" % n1)
        c.require(on < 10.0, "rings on: 4-RoSy median %.2f deg (want < 10)" % on)
        c.require(on < 0.7 * off,
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
        _, VQ0, FQ0 = solve(V, F, False)
        _, VQ1, FQ1 = solve(V, F, True)
        off, n0 = rosy4_median(VQ0, FQ0, _tangent_about_z, sel)
        on, n1 = rosy4_median(VQ1, FQ1, _tangent_about_z, sel)
        c.note("band edges %d/%d  off=%.2f deg  on=%.2f deg" % (n0, n1, off, on))
        c.require(n1 > 40, "only %d edges in the 4-quad band" % n1)
        c.require(on < 11.0, "rings on: 4-RoSy median %.2f deg (want < 11)" % on)
        c.require(on < 0.7 * off,
                  "rings on (%.2f deg) is not a clear win over off (%.2f deg)"
                  % (on, off))
        c.require(_max_valence(VQ1, FQ1) <= 14,
                  "rings on produced a valence-%d vertex: the hole collapsed"
                  % _max_valence(VQ1, FQ1))

    # ---------------------------------------------------------------- 3
    with r.case("rings_off_is_a_no_op") as c:
        if fields is None or extract is None:
            c.skip("native modules unavailable")
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
        c.note("plumbed through properties -> native -> solver -> fields")

    return r.list()
