"""Regressions for the Keep Small Shells = OFF path (campaign finding F2).

With ``preserve_small_shells`` on, small separate shells are lifted out before
the solver and rejoined verbatim, so the whole flag-off path — every shell
going through the solver together — was effectively untested. It was also
broken, in three independent ways:

* QuadriFlow reports success and still hands back a TORN quad mesh for a shell
  far below its useful resolution (a 24-face hair card). Joined into the
  result those tears read as open seams, and the exact-symmetry mirror cannot
  close them because they are nowhere near the symmetry plane.
* the native solver's per-shell rho floor (every shell keeps at least
  MIN_SHELL_QUADS quads) is a promise the face budget has to pay for. On a
  170-shell avatar it exceeds the whole budget; honouring it drove rho on the
  tiny shells 18x below the global value, exploded the conforming refinement
  and starved the body — Dinasty at target 2000 collapsed from 1290 to 546
  faces on its main shell.
* shells too thin for the lattice were dropped outright and never grafted
  back, because the lost-region rescue only looked at regions of >= 16 faces:
  8 of 9 shells on the plate fixture below, 100 of 172 on Dinasty.

The fixtures are the two-shell object from test_10 (with the shell limit
raised so the flag actually applies to the small sphere) and a hair-card
variant of it, both straddling the symmetry plane.
"""

import os

import bmesh
import bpy

DINASTY = "/home/nerdrx/Documents/Dinasty_no piercings.blend"


def _shells(obj):
    """[(face count, surface area)] per connected shell, largest area first."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    seen = set()
    out = []
    for f0 in bm.faces:
        if f0.index in seen:
            continue
        n = 0
        area = 0.0
        stack = [f0]
        seen.add(f0.index)
        while stack:
            f = stack.pop()
            n += 1
            area += f.calc_area()
            for e in f.edges:
                for nf in e.link_faces:
                    if nf.index not in seen:
                        seen.add(nf.index)
                        stack.append(nf)
        out.append((n, area))
    bm.free()
    out.sort(key=lambda t: -t[1])
    return out


def _boundary_edges(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    bm.free()
    return n


def _join(objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return objs[0]


def _two_shell_object():
    """test_10's fixture: one big sphere + one small sphere crossing X=0."""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
    big = bpy.context.active_object
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.3,
                                         location=(0.0, 0.0, 1.15))
    return _join([big, bpy.context.active_object])


def _card_shell_object():
    """The same sphere wearing 8 thin hair cards across the symmetry plane.

    A card is 6 faces and 0.024 thick — three orders below the target quad
    size. This is the shape both backends used to destroy: QuadriFlow tore the
    cards open, the native lattice could not resolve them at all.
    """
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
    objs = [bpy.context.active_object]
    for i in range(8):
        bpy.ops.mesh.primitive_cube_add(size=0.3,
                                        location=(0.0, 0.25 * i - 0.9, 1.1))
        card = bpy.context.active_object
        card.scale = (1.0, 0.25, 0.08)
        bpy.ops.object.transform_apply(location=False, rotation=False,
                                       scale=True)
        objs.append(card)
    return _join(objs)


def _run(ctx, make_obj, backend, seed, preserve, **extra):
    ctx.fresh_scene()
    obj = make_obj()
    s = ctx.settings(obj, target_count=1200, symmetry_x=True,
                     exact_symmetry=True, keep_original=False,
                     backend=backend, seed=seed,
                     preserve_small_shells=preserve, **extra)
    from quadforge import pipeline
    return pipeline.run_remesh(bpy.context, obj, s)


def _check_closed_multishell(c, res, want_shells, label):
    c.require(res.get("ok"), "%s: run failed: %r" % (label, res.get("error")))
    out = res["object"]
    stats = res.get("stats") or {}
    seam = stats.get("seam_open_edges")
    c.require(seam == 0,
              "%s: seam_open_edges=%r (want 0)" % (label, seam))
    holes = _boundary_edges(out)
    c.require(holes == 0,
              "%s: watertight input came back with %d boundary edges"
              % (label, holes))
    shells = _shells(out)
    c.require(len(shells) >= want_shells,
              "%s: %d of %d input shells survived"
              % (label, len(shells), want_shells))
    return out, shells


def run(ctx):
    r = ctx.results()

    # ---- the two-shell fixture, flag off, both backends, 3 seeds ----------
    # small_shell_limit is raised past the small sphere's 128 faces so the flag
    # genuinely changes the path (the automatic limit for this fixture is 64,
    # which would leave the sphere on the solver side either way).
    for backend in ("QUADRIFLOW", "NATIVE"):
        with r.case("two_shell_flag_off_closed_%s" % backend.lower()) as c:
            notes = []
            for seed in (0, 1, 2):
                res = _run(ctx, _two_shell_object, backend, seed, False,
                           small_shell_limit=200)
                out, shells = _check_closed_multishell(
                    c, res, 2, "%s seed %d" % (backend, seed))
                notes.append("s%d:%df/%dsh" % (seed, len(out.data.polygons),
                                               len(shells)))
            c.note(" ".join(notes))

    # ---- thin hair cards, the shape that actually broke -------------------
    for backend in ("QUADRIFLOW", "NATIVE"):
        with r.case("hair_cards_flag_off_closed_%s" % backend.lower()) as c:
            notes = []
            for seed in (0, 1):
                res = _run(ctx, _card_shell_object, backend, seed, False)
                out, shells = _check_closed_multishell(
                    c, res, 9, "%s seed %d" % (backend, seed))
                notes.append("s%d:%df/%dsh" % (seed, len(out.data.polygons),
                                               len(shells)))
            c.note(" ".join(notes))

    # ---- flag ON must stay closed too (the fixes must not have moved it) --
    with r.case("hair_cards_flag_on_still_closed") as c:
        for backend in ("QUADRIFLOW", "NATIVE"):
            res = _run(ctx, _card_shell_object, backend, 0, True)
            _check_closed_multishell(c, res, 9, "%s flag-on" % backend)

    # ---- the body must not be starved by the shells' minimum budgets ------
    with r.case("dinasty_flag_off_main_shell_survives") as c:
        if not os.path.exists(DINASTY):
            c.skip("Dinasty asset not present at %s" % DINASTY)

        def dinasty_obj():
            bpy.ops.wm.open_mainfile(filepath=DINASTY)
            meshes = [o for o in bpy.data.objects
                      if o.type == 'MESH' and len(o.data.polygons)]
            if not meshes:
                c.skip("no mesh objects in %s" % DINASTY)
            meshes.sort(key=lambda o: -len(o.data.polygons))
            return meshes[0]

        areas = {}
        faces = {}
        for preserve in (True, False):
            # fresh_scene() is pointless here - the fixture opens a .blend
            obj = dinasty_obj()
            s = ctx.settings(obj, target_count=2000, symmetry_x=True,
                             exact_symmetry=True, keep_original=False,
                             backend="NATIVE", seed=0,
                             preserve_small_shells=preserve)
            from quadforge import pipeline
            res = pipeline.run_remesh(bpy.context, obj, s)
            c.require(res.get("ok"), "preserve=%s: run failed: %r"
                      % (preserve, res.get("error")))
            main = _shells(res["object"])[0]
            faces[preserve] = main[0]
            areas[preserve] = main[1]
        c.require(areas[True] > 0.0, "flag-on main shell has no area")
        cover = areas[False] / areas[True]
        c.note("main shell area %.4f -> %.4f (%.0f%%), faces %d -> %d"
               % (areas[True], areas[False], 100.0 * cover,
                  faces[True], faces[False]))
        c.require(cover >= 0.90,
                  "flag-off main shell covers only %.0f%% of the flag-on "
                  "result's area (%d vs %d faces)"
                  % (100.0 * cover, faces[False], faces[True]))

    return r.list()
