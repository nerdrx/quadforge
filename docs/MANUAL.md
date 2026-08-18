# QuadForge — User Manual

QuadForge turns any mesh into a clean quad mesh and carries your UVs,
materials, vertex groups and shape keys onto the new topology.

This manual documents **v0.5.4**. Everything below was checked against the
code, so where a number appears (a default, a threshold, a clamp) it is the
number the addon actually uses.

- [Quick start](#quick-start)
- [Presets](#presets)
- [Setting reference](#setting-reference)
- [Workflows](#workflows)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Defaults at a glance](#defaults-at-a-glance)

---

## Quick start

1. Install the zip (Edit → Preferences → Add-ons → Install from Disk).
2. Select a mesh object. Open the 3D Viewport sidebar (`N`) → **QuadForge**.
3. Pick a **Preset** — `Game Avatar` for a character, `Hard Surface` for
   mechanical parts, `Quick Draft` to just see what happens.
4. Press **Remesh**.

The result is a new object named `<source>_quad`. The original is not destroyed:
it is moved to a hidden collection called *QuadForge Originals*, and
**Toggle Original** in the Results box flips between the two.

Settings live **per object** (`object.quadforge`), so two objects in the same
scene can have completely different setups, and a result object inherits the
settings that produced it.

---

## Presets

The **Preset** dropdown sits at the top of the panel. Choosing one writes its
values into the 16 properties listed below, once. After that it gets out of the
way: edit anything you like, nothing is reverted, and the dropdown keeps
showing the preset name as a label. It is a starting point, not a mode.

| Property | Game Avatar | Sculpt Cleanup | Hard Surface | Quick Draft |
|---|---|---|---|---|
| Backend | Native | Native | QuadriFlow+ | QuadriFlow+ |
| Target Mode | Faces | Faces | Faces | Faces |
| Quad Count | 15 000 | 20 000 | 8 000 | 4 000 |
| Adaptive Size | 40 % | 60 % | 0 % | 0 % |
| Strict Count | off | off | off | off |
| Painted Density | off | off | off | off |
| Detect Hard Edges | **on** | off | **on** | off |
| Use Marked Edges | **on** | off | **on** | off |
| Material Boundaries | off | off | **on** | off |
| Follow UV Islands | **on** | off | off | off |
| Use Guides | off | off | off | off |
| Symmetry X | **on** | off | off | off |
| Symmetry Y / Z | off | off | off | off |
| Exact Symmetry | on | on | on | **off** |
| Keep Small Shells | on | **off** | on | on |

**What no preset ever touches**: the whole *Preserve* box (Boundaries, UVs,
Vertex Groups, Shape Keys, Materials, Creases, Bevel Weights), *Keep Original*,
*Seed*, *Small Shell Limit*, *Hang-Safe Solver*, *Opening Rings*, the guide
collection and the LOD targets. Those either protect your data, protect your
Blender session, or (in the case of Opening Rings) reallocate your quad budget
in a way you should be choosing deliberately — a workflow preset has no
business flipping them.

Notes on individual choices:

- **Sculpt Cleanup** turns *Use Marked Edges* off as well as *Detect Hard
  Edges*: a raw sculpt rarely carries meaningful sharp/seam/crease marks, and
  the flow is better off driven purely by curvature. It also turns *Keep Small
  Shells* off, because on a sculpt the small separate pieces are usually
  dyntopo debris or blocked-in parts you *do* want remeshed — the opposite of
  the authored hair cards on a finished avatar.
- **Exact Symmetry** is inert unless at least one symmetry axis is on. Quick
  Draft turns it off anyway, because if you do enable an axis, the solver's own
  approximate symmetry is the faster path — and a draft does not need a
  vertex-exact mirror.
- **Quick Draft leaves the Hang-Safe Solver on.** Turning it off would save
  about a second, at the price of a solver stall being able to freeze Blender.
  Not a trade a preset should make for you.

---

## Setting reference

The order below follows the panel.

### Target

#### Target Mode — `mode` (default `Faces`)
How the quad budget is expressed. All three end up as one absolute face count
before anything else happens, clamped to the range 12 … 8 000 000.

- **Faces** — use *Quad Count* directly.
- **Ratio** — `input face count × Ratio`. Handy for "half of whatever this is".
- **Edge Length** — `world surface area ÷ (Edge Length)²`. This is the mode to
  use when several objects must share a texel/quad density: give them all the
  same edge length rather than guessing per-object counts. It is measured in
  **world** units, so object scale counts.

#### Quad Count — `target_count` (default 5000)
Approximate quad count **of the finished object**, not of the solve. If *Keep
Small Shells* set some shells aside, their faces are subtracted from the
solver's budget so the total lands near your number. The subtraction is capped:
when the preserved shells alone come to more than 60 % of the target, the
budget is left alone, the result overshoots, and the report warns
*"preserved shells (N faces) approach or exceed the target; total output will
overshoot"*.

The count is approximate by design. See *Strict Count* if you need it tight,
and the [FAQ](#faq) for why it can land high.

#### Strict Count — `strict_count` (default off)
Re-solves up to **3** more times, correcting the request from the measured
result each round, and stops as soon as the count is within **10 %** of the
target. **QuadriFlow+ only** — the Native backend ignores it. Each retry is a
full solve, so this is the most expensive checkbox in the panel. Turn it on for
final LODs and hard face budgets; leave it off while iterating.

#### Adaptive Size — `adaptive_size` (default 0 %)
Spends more of the budget where the surface curves and less where it is flat.
It **redistributes** the budget rather than adding to it. The two backends
deliver it very differently, and this matters:

- **Native** — real adaptivity. Target edge length shrinks with curvature
  (`ρ ∝ κ^(-a/2)`, clamped to the *Size Contrast* band, 3× by default) and the
  field is renormalised so the predicted face count stays on target. Curved
  regions genuinely get more, smaller quads.
- **QuadriFlow+** — QuadriFlow takes no density input at all, so QuadForge
  applies adaptivity *afterwards* as a density-weighted relaxation: quad
  **size** varies, quad **count** and topology do not. Plus, if *Adapt Quad
  Count* is on, a conservative all-quad edge-loop removal pass in the flat
  regions.

Values around 30–50 % are a good general setting; 60 %+ is for sculpts with
strong detail/flat contrast. At 0 % the density attribute is uniform and both
backends solve uniformly.

#### Size Contrast — `detail_range` (default 3.0, Native only)
How much coarser the flattest parts of the mesh are allowed to get than the
busiest ones, as a ratio of **quad edge length** (so 6 means 6× the edge, 36×
the area). It is the clamp on the curvature term above, and the full ratio is
only reached at *Adaptive Size* 100 % — the achievable spread is
`Size Contrast ^ (adaptive/100)`, so at 40 % adaptivity a setting of 6 buys
about 2×, not 6×. Greyed out at *Adaptive Size* 0.

Leave it at 3 and nothing changes: 3 is exactly what QuadForge did before this
setting existed, down to the last bit. Raise it when the budget is too small to
carry the whole surface at one quad size — a character at 3 000 faces, a prop
that is mostly flat panel with one detailed fitting. On the Dinasty avatar at a
3 000-face request the whole-mesh coarse:fine ratio went from 7.0× to 8.3× as
this moved from 3 to 10, and the face that had collapsed into a handful of
triangle fans came back with recognisable eye and muzzle loops.

Above 3 the sizing field is also **gradient-limited**: quad size may change by
at most ~30 % of itself per quad, so it ramps between the fine and coarse
regions instead of stepping. That costs a little of the nominal contrast and
buys a mesh the extractor can actually build — without it, the wide bands
produce size seams, and on a heavily creased model they make the solve collapse
outright. Feature and *Opening Rings* densification still win locally: they are
applied on top of the graded field, so a hard edge crossing a coarse panel
keeps its denser band, just measured against that panel's size.

Values much past 8 buy little on an organic mesh — the curvature signal itself
runs out — but they are there for hard-surface work where one small fitting
sits on a large flat body.

#### Detail from Input — `use_input_density` (default off, Native only)
Reads the **input mesh's own tessellation** as a detail hint. Where the artist
left long edges they have already said "nothing happens here", and curvature
cannot see that: a decimated flat panel and a smooth blob both read as κ ≈ 0.
With this on, each vertex's mean incident input edge length (smoothed in the
log domain) multiplies into the sizing field, bounded by *Size Contrast*, so
sparsely tessellated regions come out coarse and densely sculpted ones fine.

It is a **no-op on evenly tessellated input** — the measure is taken relative
to the mesh's own median, so a uniform mesh has ratio 1 everywhere — and it is
measured before the solver's internal refinement, so the solver's own
subdivision cannot wash it out.

Turn it on for sculpts assembled from parts of very different densities
(a ZBrush blob welded to a box-modelled panel, a scan glued to hand geometry),
and for retopology of an already-good mesh whose density you want to keep. It
does **not** help on production game meshes that are already evenly
tessellated — on the Dinasty avatar it changed the quad distribution without
improving it, because that mesh's density variation is not where the detail is.
**Native backend only**, and no preset turns it on.

#### Adapt Quad Count — `adapt_quad_count` (default on)
Asks the solver for up to **10 % extra faces** (`1 + 0.10 × adaptive`) so the
QuadriFlow+ adaptive pass has slack to remove whole edge loops from the flat
regions. **No effect** at *Adaptive Size* 0, and no effect on the Native
backend. Turn it off when the face count matters more than the flow.

#### Painted Density — `use_paint_density` (default off)
Multiplies the density field by a map you paint on the mesh. Mid grey is
neutral, brighter is denser, darker is coarser. See
[How do I paint density?](#how-do-i-paint-density) for the full workflow.
Painted density and *Adaptive Size* multiply together, and the combined field
is clamped to 0.05 … 4.0.

### Edge Loops

Most of this box does the same thing mechanically — it marks edges on the
working copy as **hard**, which both backends treat as a feature the edge flow
must follow, and the settings differ only in where the marks come from. The
last two, *Opening Rings* and *Use Guides*, are different in kind: they steer
the Native solver's orientation field directly.

#### Detect Hard Edges — `detect_hard_edges` (default on)
Marks edges whose dihedral angle exceeds *Hard Edge Angle*. Non-manifold edges
(3+ faces) are always treated as hard whatever this is set to.

#### Hard Edge Angle — `hard_edge_angle` (default 40°)
The angle threshold. Lower values catch softer creases and produce many more of
them; on an organic sculpt, going much below 30° tends to shatter the flow into
short chains. 40–60° suits most hard-surface work.

#### Use Marked Edges — `use_marked_sharp` (default on)
Also honours what is already authored on the mesh: **sharp** edges, **UV
seams**, and **creases above 0.5**. When this is **off**, existing sharp flags
are actively cleared on the working copy, so only angle detection steers the
flow — that is the point of turning it off on a scan or an imported mesh with
junk marks.

#### Material Boundaries — `use_materials` (default off)
Marks every edge between two different material slots as hard, so a clean edge
loop runs along each material border. Essential when materials define panel
seams; pointless (and costly in quads) when materials are just colour variation.

#### Follow UV Islands — `use_uv_seams` (default off)
Treats UV island boundaries and marked seams as feature edges, so the flow
aligns along them and the texture seams survive. Turn this on whenever the
source object is already textured and you intend to keep the texture — without
it, the UV transfer still runs, but island borders can end up cutting across
quads diagonally.

#### Opening Rings — `use_opening_rings` (default off, Native only)
Runs concentric edge loops around small closed holes — eye sockets, the rim of
a mouth bag, ear canals — instead of letting the curvature-aligned flow run
straight past them. QuadForge finds every closed boundary loop shorter than
15 % of the perimeter of a square patch with the same surface area as the mesh
(so large open borders and the symmetry-bisect cut are skipped), then does
three things around each one: steers the orientation field along that loop's
offset rings over a band roughly six quads wide, **refines the mesh inside that
band** so a ring has enough segments to be a ring rather than a polygon, and
pins the first offset loop about 1.5 quads out so it exists as a real edge run
rather than by luck. Hand-drawn guides and hard edges still win wherever they
overlap the band, and the rim edges themselves are pinned as before. **Native
backend only** — QuadriFlow has no channel to receive any of this. Off by
default, and no preset turns it on.

**It spends quads, but not extra ones.** The refinement is budget-conserving:
the socket's new quads are paid for by the rest of the mesh going marginally
coarser (about 2 % on a typical head), not by a bigger face count — face counts
were measured within ±2 % of target at budgets from 1 500 to 12 500. The
bands together can never claim more than 30 % of the budget, so a mesh with two
hundred little tubes cannot starve its own body. Expect quad sizes to vary more
across the mesh than with the option off; that is the feature working.

Measured on a disc with a hole and a sphere with a hole, the edges inside a
four-quad band go from 15–21° off the ring direction to 4–6°. On a real avatar
head at a 12 000-face whole-body budget, the quads around an eye socket go from
25 to 46 and the first three rings go from mostly broken (loop purity
0.54 / 0.26 / 0.23) to mostly closed (0.71 / 0.65 / 0.61) — the topology
clearly acknowledges the eye. More budget still means cleaner rings, but you no
longer need a head-only or 25k+ remesh for it to do anything. It remains
experimental: roughly one ring vertex in three is still a T-junction, so this
is not hand-drawn lid topology.

**If nothing happens, look at Keep Small Shells first.** If the eyes, mouth bag
or ear pieces are separate small shells, *Keep Small Shells* sets them aside
untouched and their openings never reach the solver — on a typical avatar head
most of the holes are exactly that (on the author's test avatar, 13 of 15).
QuadForge now counts both sides and says so: the report carries
`ring_openings_preserved` and `ring_openings_solved`, and when the preserved
side wins you get a warning naming the count (see
[Troubleshooting](#warnings-about-your-settings)). To ring them, turn *Keep
Small Shells* off or raise *Small Shell Limit* — bearing in mind that remeshing
authored eyes and teeth is usually a worse trade than leaving their openings
unringed.

#### Use Guides — `use_guides` (default off) / Guide Collection — `guide_collection`
Projects the curve and Grease Pencil objects in the guide collection onto the
surface and steers the edge flow along them. **New Guide** creates an empty
curve at the 3D cursor in a *QuadForge Guides* collection, ticks *Use Guides*,
and drops you into Edit Mode with the Draw tool active — draw the stroke, then
Tab back to Object Mode.

Guides need the Native solver: it reads the projected direction field
(`qf_guide`, per-face) and steers the whole orientation field with it.
Blender's QuadriFlow operator has no constraint channel at all (measured:
bit-identical output with and without guide marks), so when guides land on
the surface a **QuadriFlow+** solve automatically switches to the Native
backend for that run — the report says so in `warnings` and
`backend: NATIVE`.

### Symmetry

#### X / Y / Z — `symmetry_x`, `symmetry_y`, `symmetry_z` (all default off)
Symmetry about the object-space planes `x=0`, `y=0`, `z=0`. Any combination is
allowed. **Object space** — if the mesh is not centred on the axis in its own
local coordinates, this does nothing useful.

#### Exact — `exact_symmetry` (default on)
With Exact on, QuadForge bisects the mesh, solves one half, trims the cut back
and mirror-welds it: the result is symmetric to the last bit, and per-axis
symmetry error in the report reads 0. The cut is padded by about three target
edge lengths before solving and trimmed afterwards, because a pinned cut
boundary flattens features within a couple of edges of the plane.

With Exact off, the backend's own approximate symmetry mode is used. Faster,
and *nearly* symmetric — but not vertex-for-vertex, which is visible on a
character's face and makes mirrored weight painting unreliable.

Small shells are lifted out before the bisect and rejoined after the weld:
bisecting a thin centreline hair plate shreds it into pinholes.

### Preserve

This box is the reason to use QuadForge instead of the bare QuadriFlow
operator. All seven default to **on** and there is rarely a reason to change
them — turning one off does not speed the remesh up meaningfully, it just
discards data.

| Setting | What it does |
|---|---|
| `preserve_boundaries` | Pins open boundary edges (holes, borders) so an open mesh keeps its silhouette. Also passed to the solver, not just the transfer. |
| `preserve_uvs` | Re-projects every UV layer, keeping island borders crisp. |
| `preserve_weights` | Rebuilds all vertex groups and weights, side-aware so mirrored limbs do not bleed into each other. |
| `preserve_shape_keys` | Rebuilds the whole key stack; slider values are restored afterwards. The remesh always runs on the **rest** shape. |
| `preserve_materials` | Keeps the slots and re-assigns every face to the material it sat on. |
| `preserve_creases` | Transfers subdivision creases onto the matching new edges. |
| `preserve_bevel_weights` | Transfers bevel weights onto the matching new edges. |

### Output

#### Keep Original — `keep_original` (default on)
Moves the source into the hidden *QuadForge Originals* collection instead of
deleting it. **Toggle Original** flips visibility between a result and its
source. With it off, the source object is deleted at the end of a successful
run — there is no undo inside the pipeline.

#### Backend — `backend` (default QuadriFlow+)
See [Which backend should I use?](#which-backend-should-i-use).

#### Seed — `seed` (default 0)
Randomisation seed. The Native solver is **bit-identical** for a given seed, so
changing it is a real way to shop for a different flow on a difficult mesh.
QuadriFlow is not reproducible even at a fixed seed on some inputs, so re-runs
can differ whatever you set here.

#### Hang-Safe Solver — `solver_isolation` (default on)
Runs QuadriFlow in a separate, killable Blender process per shell, with a
timeout and jittered retries (rescale / target / seed) if it stalls. This works
around a genuine upstream non-convergence bug: without it, a stalled solve
hangs your Blender session with no way out but killing it. It costs about a
second of process startup per solve. **QuadriFlow+ only** — the Native solver
runs in-process and cannot hang this way. Leave it on unless you are debugging.

#### Keep Small Shells — `preserve_small_shells` (default on)
Small separate shells — hair cards, feathers, teeth, eyes, piercings — are kept
at their **original topology** instead of being remeshed. They are usually
already hand-authored and far below the solver's useful resolution, so
remeshing them at a reduced budget is strictly destructive.

The largest shell is **never** preserved, and on a single-shell mesh this
setting does nothing at all.

#### Small Shell Limit — `small_shell_limit` (default 0 = automatic)
Shells with fewer faces than this are the ones kept. At 0 the limit is
`max(64, 2 % of the input face count)`. Raise it when authored detail is being
remeshed anyway (large hair plates on a light mesh); lower it when blocked-in
parts you *want* remeshed are being skipped.

#### LOD Targets — `lod_targets` (default `8000,2000,500`)
Comma-separated face counts for **Generate LODs**. Each entry produces one
object named `<source>_LOD0`, `_LOD1`, … in a *QuadForge LODs* collection.
Entries below 12 and duplicates are dropped. The source object is untouched —
every LOD is solved on a throwaway duplicate.

**Batch Remesh** runs the pipeline on every selected mesh object, each with its
own per-object settings.

### Results

Reads back the report of the last run: face count, quad %, tris/ngons, poles
(valence 3 and 5+), non-manifold edges, time, per-axis symmetry error and the
backend used, followed by up to four **warnings** (red) and four
**limitations** (blue). **Quality Report** and **Symmetry** recompute the
metrics on the active object without remeshing.

---

## Workflows

### 1. Game avatar (preset: Game Avatar)

For a rigged, textured, shape-keyed character with authored detail shells.

1. Select the character body. If it is a multi-object character, do the body
   first — hair and clothes usually want their own targets.
2. Preset → **Game Avatar**. That gives you Native, 15 000 quads, exact X
   symmetry, 40 % adaptivity, UV islands followed and small shells preserved.
3. Sanity-check two things the preset cannot know:
   - **Is the mesh centred on X in object space?** If not, symmetry does
     nothing useful. Fix the origin first.
   - **Is 15 000 the right budget?** Set your real number, or switch Target
     Mode to *Ratio* if you are reducing a known mesh.
4. Remesh. On a 30k-face rig expect roughly 15–45 s.
5. Read the Results box. `Sym X` should be `0`. `seam_open_edges` in the report
   should be 0 — anything else means the mirror weld left holes.
6. Check the transfer where it is hardest: pose the rig, and move a couple of
   the largest shape keys. Rest-pose metrics look fine even when weights are
   wrong (this is exactly how the historic mirrored-weight leak hid).
7. If the count came out well above 15 000, that is almost always the preserved
   shells — see the [FAQ](#why-is-my-face-count-higher-than-the-target).

### 2. Sculpt cleanup (preset: Sculpt Cleanup)

For a dense multires/dyntopo sculpt with no authored marks, no UVs and no rig.

1. Preset → **Sculpt Cleanup**: Native, 20 000 quads, 60 % adaptivity, no
   symmetry, no feature edges, small shells *not* preserved.
2. If the sculpt is symmetric and you want to keep it that way, turn **X** on
   and leave **Exact** on. The preset leaves symmetry off because dyntopo
   sculpts are frequently not symmetric in object space even when they look it.
3. Remesh, then look at the silhouette rather than the numbers. If detail
   areas (nostrils, fingers, ear folds) are mushy, the budget is the problem
   first and adaptivity second: raise *Quad Count* before raising *Adaptive
   Size* past 60 %.
4. If flat areas are still over-tessellated, turn *Adapt Quad Count* on and
   switch to QuadriFlow+ for the edge-loop-removal pass — that pass does not
   exist on Native.
5. Sculpt again on the result: it is a clean base mesh, not a final asset.

### 3. Hard surface (preset: Hard Surface)

For mechanical parts, boolean output, CAD-ish imports.

1. Assign materials to the panels you want separated, if they are not already.
2. Preset → **Hard Surface**: QuadriFlow+, 8 000 quads, uniform density, hard
   edges from angle + existing marks + material boundaries.
3. Tune *Hard Edge Angle* to the part. 40° (the default) catches most chamfers;
   raise it towards 60° if soft filleted transitions are being marked and
   splintering the flow.
4. Turn **Strict Count** on for the final run if the budget is contractual.
5. Remesh, then look for the classic failure: a sharp corner that got rounded.
   The fix is usually a real crease or sharp mark on that edge plus *Use Marked
   Edges* — angle detection alone cannot see a feature that the tessellation
   already rounded off.
6. Adaptive Size stays at 0 here on purpose. Curvature adaptivity on a
   mechanical part starves the flats, which is where the panel loops need to be
   uniform.

### 4. Quick draft (preset: Quick Draft)

For "will this mesh remesh at all", and for setting up before committing time.

1. Preset → **Quick Draft**: QuadriFlow+, 4 000 quads, every feature toggle off.
2. Remesh. This is the fastest configuration that still runs the full
   protection stack (hang-safe workers, cavity restore, data transfer).
3. Read the warnings, not the mesh. A draft run is the cheapest way to discover
   that shells are being dropped, that the target is below the per-shell floor,
   or that the solver refuses parts of the mesh.
4. Then switch to the preset you actually want and raise the count. Nothing you
   learned is wasted — the preset only rewrites the 16 covered properties.

---

## Troubleshooting

These are the messages the pipeline actually emits, in the Results box and in
`object.quadforge.last_report`.

### Warnings about the result's shape

**"preserved shells (N faces) approach or exceed the target; total output will
overshoot"**
The shells being kept verbatim come to more than 60 % of your target, so
QuadForge refused to starve the solved surface to pay for them. Either raise
*Quad Count*, or lower *Small Shell Limit* so fewer shells are preserved, or
turn *Keep Small Shells* off if those shells really should be remeshed.

**"target N is below what M separate shells can express; expect roughly K+
faces"**
QuadriFlow cancels below about 24 faces per shell, so a mesh made of many
loose parts has a hard floor. Raise the target, or turn *Keep Small Shells* on
so the small parts stop consuming the 24-face minimum each.

**"solver dropped N faces of interior/nested geometry; original topology was
restored for those regions"**
QuadriFlow silently discards interior cavities (mouth bags, nested shells) when
the mesh has open boundaries. QuadForge detected it by reverse coverage and
grafted the original geometry back. The result is valid but those regions are
not remeshed — they carry their input topology. Try the Native backend, or
close the open boundaries first.

**"N boundary edge(s) the solver tore into a watertight input were sealed"**
The input was closed and the output was not; the holes were filled. Usually
harmless, but check the filled area for a topology mess if N is large.

**"N degenerate shell(s) (M faces) refused by the solver; their original
topology was kept"**
Some shells are geometry QuadriFlow cannot handle (coincident card stacks,
zero-area faces). They kept their input topology. The Native backend rescue
pass already tried and also failed. Clean those shells by hand, or accept them.

**"exact symmetry: N boundary edges remain near the seam"**
The mirror weld left open edges on the symmetry plane. Usually a mesh that is
not actually symmetric in object space, or a shell straddling the plane with
authored asymmetry. Check the seam in edit mode; re-running with *Keep Small
Shells* on often fixes it, since that keeps centreline shells out of the cut.

**"exact symmetry: bisecting left no geometry, falling back to solver
symmetry"**
Everything was on one side of the plane — the mesh is not centred on that axis
in object space. Fix the origin or turn the axis off.

**"exact symmetry: post-solve trim failed; seam may be off-plane"**
The padded band could not be trimmed back cleanly. The result is usable but the
seam may sit slightly off the plane; the reported symmetry error will show it.

### Warnings about your settings

**"Use Guides is on but the guide collection is empty"** — set *Guide
Collection*, or press **New Guide**.

**"guides produced no surface paths (too far from the mesh?)"** — the guide
curves did not project onto the surface. Guides are projected by nearest point;
draw them on or very near the surface, and check the guide object's transform.

**"Opening Rings: N of M openings sit on shells that Preserve Small Shells
keeps verbatim, so they cannot be ringed - turn that option off (or raise Small
Shell Limit) to ring them"**
Exactly what it says: the eye sockets and mouth rim belong to small shells that
are kept at their original topology, so the solver never sees those openings
and *Opening Rings* has nothing to work on. The report also carries the raw
counts as `ring_openings_preserved` and `ring_openings_solved`. Decide which
you want more — authored eyes kept verbatim, or their openings ringed.

**"native backend unavailable, used QuadriFlow"** — the Native module failed to
import (a broken install, or NumPy missing from the Blender Python). Reinstall
the addon.

### Limitations (blue, informational)

**"adaptive post-pass only runs on the QuadriFlow backend"** — you set
*Adaptive Size* with Native selected. Not a problem: Native does adaptivity
inside the solver and does not need the post-pass.

**"QuadriFlow takes no density input, so adaptivity is delivered by
density-weighted relaxation (quad size varies, quad count does not). No edge
loop could be removed without breaking the all-quad output."**
Working as designed. If you want the count to move with the density, use the
Native backend, or turn on *Adapt Quad Count* and accept ~10 % more faces.

**"adaptivity had no effect: the curvature/paint density field was flat over
this mesh"** — the mesh has no curvature contrast (a cube, a plane) or you did
not actually paint anything. Check that *Painted Density* is on **and** the
density map has non-neutral areas.

### Things that are known limits, not bugs

From the project's own honest-limitations list (see
[PAPER.md](PAPER.md) §7):

- **Semantic loop placement.** Curvature alignment follows forms; it does not
  *plan* loops. Eyelid rings, mouth loops and the loops an artist would draw
  around a joint are not planned for you. Use guides, or retopologise those
  areas by hand. *Opening Rings* is a partial answer for the holes
  specifically — it rings eye sockets and mouth rims and buys them the
  resolution to close, but roughly one ring vertex in three is still a
  T-junction.
- **Thin shells are resolution-bound.** Fingers, fur cards, cloth edges below
  the target quad size will lose their silhouette. Raise the count, paint
  density there, or keep them as preserved small shells.
- **Face-count adherence drifts under strong density fields**, measured at
  −15 % … +1 %. Use *Strict Count* on QuadriFlow+ if the number must be tight.
- **QuadriFlow remains weather.** It is not reproducible for a fixed seed on
  some inputs; two identical runs can differ. The Native solver is
  bit-deterministic per seed.

### Nothing happened / it errored

The pipeline never raises: a failure comes back as an error string in the
report and an operator error in the status bar. If a run fails outright, the
fastest triage is the Quick Draft preset — if that also fails, the input is the
problem (check for zero-area faces, NaN coordinates, or an extreme object
scale). Blender's console (Window → Toggle System Console) carries the full
report JSON.

---

## FAQ

### Why is my face count higher than the target?

Most often: **preserved small shells**. *Quad Count* is the target for the
whole finished object, and QuadForge subtracts the preserved shells' faces from
the solver's budget to hit it — but only while that leaves the solved surface a
healthy budget (the preserved set must stay under 60 % of the target). Beyond
that it refuses to starve the body, overshoots, and warns you.

The other causes, in rough order of frequency:

1. **Per-shell floor.** QuadriFlow cancels below ~24 faces per shell, so a mesh
   with many loose parts cannot go below `24 × shells`. Look for the *"target N
   is below what M separate shells can express"* warning.
2. **Adapt Quad Count**, which asks for up to 10 % extra on purpose.
3. **The solve is approximate.** Both solvers hit a target within a band, not
   exactly. *Strict Count* narrows it to 10 % on QuadriFlow+.
4. **Ratio / Edge Length modes** compute a count from your input; a
   surprising area or scale gives a surprising count. The resolved number shows
   up in the report as `target_faces`.

### Which backend should I use?

**Native** for organic and character work: curvature-following flow (median
alignment ~3° vs QuadriFlow's 7.1° on the benchmark ellipsoid), true density
and adaptive reallocation, real guide steering, and bit-determinism per seed.
It is the newer of the two and still labelled experimental.

**QuadriFlow+** for hard-surface, mechanical and boolean input, and whenever
you need *Strict Count* or the adaptive edge-loop-removal pass. It is Blender's
bundled solver wrapped in QuadForge's protections: per-shell killable workers,
T-junction and pinched-vertex repair, cavity-loss detection and restore.

Feature matrix:

| | QuadriFlow+ | Native |
|---|---|---|
| Strict Count | yes | ignored |
| Adaptive Size | post-pass (size varies, count does not) | in-solver, real reallocation |
| Adapt Quad Count edge-loop removal | yes | no |
| Painted density | post-pass relaxation | in-solver |
| Guides | auto-switches the solve to Native | full directional steering |
| Opening Rings | no | yes (experimental) |
| Deterministic per seed | no | yes |
| Hang-Safe Solver applies | yes | n/a (cannot hang this way) |

Both support exact symmetry, all the Preserve options, small-shell
preservation, and hard-edge marking.

### How do I paint density?

1. Press **Paint** in the Target box. This creates two attributes and enters
   Vertex Paint: `qf_density` (float, the one the solver reads) and
   `qf_density_col` (colour, the one you can actually paint — Blender cannot
   paint float attributes interactively).
2. Paint in greyscale. **Mid grey (0.5) is neutral.** The brush is preset to
   white primary / black secondary.
3. The convention is `qf_density = red × 2.0`, so red 0.0 → density 0 (coarsest),
   0.5 → 1.0 (neutral), 1.0 → 2.0 (twice as dense). **Only the red channel is
   read** — green and blue are written to match so the viewport shows a
   readable greyscale, but they are ignored.
4. Leave Vertex Paint, make sure **Painted Density** is ticked, and Remesh. The
   colour attribute is synced into the float attribute automatically at the
   start of the run, and the colour attribute always wins when both exist.
5. **Clear** resets everything to neutral.

Painted density multiplies with *Adaptive Size*; the combined field is clamped
to 0.05 … 4.0. Like adaptivity, it **redistributes** the budget — painting
everything white does not double your face count, it just does nothing.

### Can I remesh a rigged, shape-keyed character and keep working?

Yes — that is the design centre. Vertex groups, the whole shape-key stack, UVs,
materials, creases and bevel weights are re-projected onto the new topology.
The remesh always runs on the **rest** shape (non-zero key sliders are zeroed
for the solve and restored afterwards), so a posed or mixed viewport state does
not get baked in. Verify by posing, not by looking at the rest pose.

### The panel is empty / says "Select a mesh object"

The panel only draws for an active **mesh** object. Curves, empties, armatures
and multi-object selections with a non-mesh active object all show that line.

### Do settings apply to all objects?

No. Settings are stored per object. **Batch Remesh** uses each object's own
settings; a result object inherits the settings that produced it.

### Where did my original go?

Into the hidden *QuadForge Originals* collection, unless you turned *Keep
Original* off. **Toggle Original** switches visibility between the result and
its source.

---

## Defaults at a glance

Verified against `quadforge/properties.py` at v0.5.4.

| Property | Default |
|---|---|
| `preset` | `CUSTOM` |
| `mode` | `FACES` |
| `target_count` | 5000 (min 12) |
| `target_ratio` | 1.0 |
| `target_edge_length` | 0.1 m |
| `adaptive_size` | 0 % |
| `detail_range` | 3.0 (3 … 12) |
| `use_input_density` | off |
| `adapt_quad_count` | on |
| `strict_count` | off |
| `use_paint_density` | off |
| `detect_hard_edges` | on |
| `hard_edge_angle` | 40° |
| `use_marked_sharp` | on |
| `use_materials` | off |
| `use_uv_seams` | off |
| `use_opening_rings` | off |
| `use_guides` | off |
| `symmetry_x` / `_y` / `_z` | off |
| `exact_symmetry` | on |
| `preserve_boundaries` | on |
| `preserve_uvs` | on |
| `preserve_weights` | on |
| `preserve_shape_keys` | on |
| `preserve_materials` | on |
| `preserve_creases` | on |
| `preserve_bevel_weights` | on |
| `keep_original` | on |
| `backend` | `QUADRIFLOW` |
| `seed` | 0 |
| `preserve_small_shells` | on |
| `small_shell_limit` | 0 (automatic) |
| `solver_isolation` | on |
| `lod_targets` | `8000,2000,500` |

Internal constants worth knowing: strict-count tolerance 10 % with 3 retries;
per-shell QuadriFlow floor 24 faces; automatic small-shell limit
`max(64, 2 % of input faces)`; preserved-shell budget deduction capped at 60 %
of the target; adaptive boost `1 + 0.10 × adaptive`; density clamp 0.05 … 4.0;
resolved face target clamped to 12 … 8 000 000.
