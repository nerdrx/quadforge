# QuadForge: Building a Production Auto-Retopology System through
# Agentic Development and Adversarial Visual QA

**nerdrx & Claude (Anthropic Fable 5, with parallel Opus 5 implementation agents)**
*August 9–18, 2026 — 22 releases, ~13,000 lines, 162 automated checks*

---

## Abstract

We describe the construction, over roughly nine calendar days, of QuadForge — a
free, open-source (GPL-3.0) automatic retopology add-on for Blender intended as
a replacement for the commercial Quad Remesher. The system comprises two
solving backends (a hardened wrapper around Blender's bundled QuadriFlow, and a
from-scratch pure-NumPy field-aligned quad remesher), a data-preservation layer
that survives remeshing with shape keys, vertex weights, UVs and materials
intact, and a defensive pipeline that treats real-world game meshes — with
their card stacks, T-junctions, interior cavities and authored debris — as the
norm rather than the exception. We catalogue 25+ root-caused defects, several
of them in Blender itself, and draw methodological conclusions about benchmark
fixtures versus real assets, metric-driven versus perception-driven quality
work, and the practical dynamics of delegating implementation to parallel LLM
agents under falsifiable acceptance gates.

---

## 1. Introduction

Automatic retopology converts arbitrary triangle soup — sculpts, scans,
boolean output, game rips — into clean, animation-ready quad meshes. The
commercial state of the art (Exoside Quad Remesher) is closed-source and
licensed; Blender's bundled QuadriFlow operator is free but exposes none of
the workflow features that make retopology practical (density control, guides,
exact symmetry, data preservation), and, as we document below, harbors several
serious defects on real-world input.

QuadForge began with a one-line user request ("write me a replacement for
Quad Remesher, I don't wanna pay for it — self test as much as possible") and
evolved through twenty-two releases driven almost entirely by a single feedback
channel: **the user looking at wireframe renders of their own hand-made
avatars and saying what looked wrong.** Every major quality advance in this
project traces back to one of those complaints. We consider this the paper's
central finding and return to it in §7.

## 2. System architecture

```
pipeline.run_remesh(context, obj, settings)      single entry point, never raises
 ├─ transfer.capture()        numpy snapshot: UVs, weights, 640-key stacks, materials
 ├─ shape keys zeroed         rest shape is remeshed; slider values restored after
 ├─ preprocess                hard-edge marking (dihedral), material/UV-island
 │                            boundaries, guide-curve projection, density attrs
 ├─ small shells set aside    authored hair/teeth/eyes below ~2% of input faces
 │                            are never remeshed (they are already good)
 ├─ exact symmetry            bisect with ~3-edge padding past the plane → solve
 │                            half → cut back → mirror → weld → pinhole fill
 ├─ backend.remesh            QUADRIFLOW+ (per-shell isolated workers) or NATIVE
 ├─ restore_lost_regions      reverse-BVH detection of solver-dropped geometry,
 │                            grafted back from the input with a warning
 ├─ fix_orientation           per-shell normal vote against the source surface
 ├─ transfer.apply            side-aware, vertex-identity-exact data restoration
 └─ report                    quad%, poles, seam integrity, symmetry error, warnings
```

Key design positions, each earned through a concrete failure:

**D1 — The solver is untrusted.** Blender's QuadriFlow (a) hangs
non-deterministically on input/target combinations with no pattern we could
find, (b) raises `RuntimeError` rather than cancelling on degenerate shells,
(c) silently discards interior cavities (mouth bags) when the mesh has open
boundaries, and (d) is not reproducible even for a fixed seed. QuadForge
therefore executes it **per connected shell, in a killable child Blender
process** with timeouts, jittered retries (rescale/target/seed), per-shell
give-up, and a reverse-coverage check that grafts back anything the solver
dropped. A stall can no longer freeze the host session — which the commercial
competitor also guarantees, by the same architecture (external engine
process).

**D2 — Don't touch what shouldn't be touched.** Testing on a fully hand-made
avatar showed that remeshing authored detail shells (hair plates, feather
ruffs, piercings, teeth, eyes) at any reduced budget is strictly destructive.
`preserve_small_shells` keeps sub-threshold shells byte-identical, on every
backend, including through the symmetry mirror (bisecting a centerline hair
plate shreds it; setting it aside does not).

**D3 — Data preservation is the product.** No other free tool re-attaches a
640-shape-key stack, 150 vertex groups and seam-crisp UVs onto new topology.
The transfer layer maps by nearest surface point with exact
closest-point-on-triangle barycentrics, disambiguates thin double-walls by
normal agreement, and — critically — takes data *verbatim by vertex identity*
wherever geometry survived unchanged. Measured end-to-end: posed-deformation
error p95 0.06% of the bounding diagonal; shape-key reconstruction error 0.00%
on the largest keys.

## 3. The native solver (v2)

The native backend is an Instant-Meshes-family field-aligned remesher in pure
NumPy (no SciPy), extended in four directions:

**3.1 Curvature-tensor-aligned orientation field.** Per-vertex shape operators
are fitted by least squares over 1–2 rings, then smoothed **as tensors**
(mean + spin-2 deviatoric parts) rather than as unit direction fields — noise
cancels instead of renormalising, which took alignment on a noisy ellipsoid
from 16.7° to 3.5° and is the single change that makes quad flow follow brows,
muzzles and muscle forms. The 4-RoSy field is then soft-aligned to the k1
direction with weight `curvature_align · smoothstep(anisotropy) · confidence`,
where confidence (curvature × edge length) prevents flat-panel float noise
from reporting anisotropy ≈ 1. Analytic fixtures: torus 0.19°, ellipsoid 0.10°
median alignment error where anisotropy > 0.3 (v1 baseline: 23.8°).

**3.2 Sampling-robust extraction.** The position lattice is read off the input
graph, which silently fails on real sculpts: edge lengths vary 5× and any
input edge longer than ~1.5ρ contributes no lattice sample. On the first real
test model this deleted the entire smoothly-sculpted torso while every uniform
benchmark fixture passed (§6, L2). The fix is conforming red–green
pre-refinement to lattice density (bit-identical surface, midpoint splits),
plus per-shell ρ clamps (every shell keeps ≥16 quads) and coverage-dominant
retry scoring. Result on the failing shell: area coverage 16.6% → 98.9%.

**3.3 Repair to watertightness.** Rotation-system orbit extraction, then:
graph-edge-only welding (never by distance — coincident card stacks must not
fuse), half-edge hole tracing with pinch-safe figure-eight decomposition,
quad-dominant hole fills, staged triangle-annihilation walks (escalating only
while below a 97% quad floor — unconstrained walks minted valence-3 clutter),
doublet collapse, and tangent-space relaxation with reprojection. All closed
benchmark fixtures extract at 100% quads, watertight, deterministic per seed.

**3.4 Regularization and feature handling.** A 120-iteration polish blends
per-quad closed-form square fitting (corners as complex `a·i^k`), edge springs
toward a rest length that is *diffused mesh-wide* (springs targeting each
vertex's own mean edge length are fixed-point-stable at any locally-uniform
size — the original "2× neighboring quads" bug), and a light umbrella term,
reprojected onto the input every few iterations. Feature chains (creases,
boundaries) are faired **along** their curves and snapped to an arc-length
resampled, capped-Laplacian-smoothed copy of the input feature polyline —
snapping to the raw polyline reproduces the sculpt's triangulation zigzag at
machine precision, which is exactly what one does not want. Corners anchor
exactly onto input corners; symmetry-plane vertices fair in-plane. Feature
neighborhoods receive a graded density boost (plateau-then-decay falloff over
graph distance; see §5 for how the amplitude of this boost produced the
project's subtlest regression).

## 4. Verification apparatus

- **162 assertion checks** across 17 headless test modules (registration,
  remeshing accuracy, symmetry exactness, sharp preservation, data transfer,
  density/guides, reporting, native gates, LODs/batch, orientation/seam
  regressions, mirrored-weight regressions, a 34-case pathological-input
  gauntlet, UV-island following, seam-debris regressions, the preserve-off
  path, presets, opening rings), with crash-detecting runner, per-module
  isolation mode, and flake-hardened assertions.
- **A quality benchmark** (`tests/bench_native.sh`) comparing both backends on
  six procedural fixtures across: quad%, watertightness, poles/1k, count
  error, surface deviation, **flow alignment to principal curvature
  directions** (the "pretty edge flow" number), density response, and wall
  time — with per-fixture wireframe renders for eyeballing.
- **Real-asset validation** on two rigged avatars: a 29k-face commercial-style
  game rig of 116 shells with 640 shape keys (Rexouium), and a 46.8k-face
  fully hand-authored model (NX-Dinasty) whose creator was the reviewer.
- **Visual review as a first-class gate.** Headless Workbench renders
  (clay + wireframe overlays, deviation heatmaps baked to vertex colors,
  posed comparisons), rendered from scripted cameras and judged by a human —
  and by the orchestrating model reading the images — every round.

## 5. Defect catalogue (selected, all root-caused)

**In Blender / QuadriFlow (upstream):**
- U1. Non-convergence (indefinite spin) on chaotic input/target combinations;
  no seed reproducibility even when it converges.
- U2. `quadriflow_remesh` **raises** on degenerate shells instead of returning
  `CANCELLED` — and Blender exits with code 0 despite the Python traceback,
  so naive automation reads crashes as success.
- U3. Interior cavities silently discarded when solving meshes with open
  boundaries — including inside its own mesh-symmetry mode.
- U4. Refuses meshes whose 1st-percentile edge length falls under an absolute
  epsilon (~real-world VRChat scale); refuses face targets below ~24.
- U5. Boundary-constrained solves flatten features within ~2 edge lengths of
  the boundary (the "missing inner toes").
- U7. `use_preserve_sharp` derives features from geometric dihedral angle
  only: sharp *flags* on flat geometry are ignored entirely (verified by
  identical output hashes) — so any flag-based flow-control scheme
  (guides, material boundaries, UV seams marked as sharp) is silently
  inert on this solver wherever the surface is flat.
- U6. `bpy.data.libraries.load` mutates the requested-names list in place;
  `Object.copy()` carries a stale bound box until a depsgraph update; objects
  in excluded collections do not evaluate their armature modifiers (a trap
  that manufactured a phantom 15%-error weight-transfer "regression" in our
  own harness); the Wireframe modifier's even-offset explodes at degenerate
  corners; `shape_key_clear` semantics on shared meshes; ARG_MAX limits on
  worker command lines at 200+ shells.

**In QuadForge (ours, found by tests or the user's eye):**
- Q1. Exact-symmetry weight leak: `bmesh.ops.mirror` copies deform layers
  un-swapped; transfer then wrote correct weights per group without clearing
  stale ones → mirrored limbs weighted 1.0 to *both* sides, moving halfway
  when posed. Found only by posing the rig (metrics on the rest pose scored
  it 99.9% fine).
- Q2. Solver-dropped mouth cavities reached the user because nothing verified
  reverse coverage; now `restore_lost_regions` grafts and warns.
- Q3. The regularizer's rest-length self-reference and the triangle-walk pole
  minting (§3.4) — both "the polish pass manufactured the ugliness".
- Q4. Feature pins that could slide along the 4-RoSy *perpendicular*;
  cluster centroids averaging pinned with unpinned samples (the literal
  stair-step); a segment projector that snapped every query to an endpoint
  because polylines were encoded as degenerate triangles.
- Q5. The v0.4.4 density regression (below).
- Q6. `_fill_holes` skipped a traced loop only when **every** corner carried a
  boundary flag. A genuine large opening whose extraction clusters happened to
  miss a single input boundary vertex therefore failed that test, and the
  filler fanned one centroid across the entire hole — valence-67 poles, spiral
  meshes. Pre-existing since the filler was written, surfaced only by the
  opening-ring fixtures (§8), and measured firing on ~40% of config×seed
  combinations on a bordered flat sheet. Loops longer than one lattice orbit
  are now treated as real openings at **≥90%** boundary corners; the filler's
  actual job — a missing lattice cell, never more than an orbit long, all of
  whose corners are interior samples — is untouched. Outputs shifted slightly,
  for the better (benchmark grid control poles/1k 36.2 → 26.3), so v0.5.2 is
  deliberately not bit-identical to v0.5.1. The lesson generalises past this
  filler: an all-corners predicate over individually noisy per-vertex flags is
  a conjunction of *n* unreliable tests, and it fails catastrophically rather
  than gracefully — the right shape for noisy evidence is a supermajority
  threshold.

**The v0.4.4/v0.4.5 confusion — a case study in attribution.** To fix rim
aliasing, v0.4.4 raised the feature-density amplitude to 2.5×. Metrics local
to the rim improved and the release shipped. In fact, at 2.5× the "feature
band" had grown to swallow 61–84% of all output vertices — the boost had
degenerated into a near-global rescale whose rebalance starved every
non-feature region (interiors up to 34.6% coarser, evenness +36%, squareness
+41%). v0.4.5 then *improved* the falloff shape, but the user — comparing
against their memory of v0.4.3 — correctly reported "seems worse". An
overnight 8-configuration sweep with region-split metrics identified the true
culprit one release earlier than assumed, refuted the initially-favored
hypothesis about v0.4.5, and selected 2.0×/plateau-2/decay-1.5, which beats
the regressed default on every axis simultaneously. Two morals: (a) per-release
A/Bs against the immediate predecessor cannot catch slow regressions — keep an
anchored baseline; (b) when a user says "worse", believe the perception and
instrument until the numbers explain it.

## 6. Methodological lessons

- **L1. Real assets are the benchmark.** Five separate catastrophic failures
  (fragment collapse, cavity loss, weight leak, fur-card refusal, toe
  flattening) passed every synthetic fixture and appeared within minutes on a
  real rigged model. Uniform tessellation, watertightness, single shells and
  rest poses are all lies that fixtures tell.
- **L2. Silence is not success.** The worker that crashed after 48 of 60
  shells returned exit code 0. The solver that covered 16.6% of the surface
  reported 100% quads and 0 boundary edges — of the fragment it kept. Every
  wrapper needs positive evidence of completeness (coverage checks, per-item
  done-markers), not absence of errors.
- **L3. Perception-driven QA outperformed metric-driven QA.** The user's
  five-word complaints ("the ears are fucked", "quads so uneven", "still a bit
  jagged", "seems worse") each localized a defect class that the metric suite
  scored as healthy. The productive loop was: complaint → targeted render →
  hypothesis → *instrument until a number reproduces the perception* → fix
  against that number → re-render for the eye that complained.
- **L4. Agents under falsifiable gates, with permission to fail.** Eleven
  Opus-class implementation agents worked in parallel lanes with numeric
  acceptance gates. The system worked best when agents reported failure
  honestly — and they did: one declined to implement a requested feature
  after measuring its premise false (zero lattice-line breaks), one reported
  its own bridging feature as net-negative and shipped it off-by-default, one
  missed its acceptance targets and said so in the first line. Each of those
  honest failures redirected the effort profitably. Conversely, the biggest
  wasted hours came from the orchestrator (this author) trusting green
  fixtures over a red-flagged perception.
- **L5. Determinism must be designed for and measured.** QuadriFlow's
  reproducibility turned out to be *input-dependent* — bit-identical across
  five runs on some meshes, genuinely varying on others (a symmetric UV
  sphere) — which first leaked into flaky tests, was amplified by
  order-dependent set iteration in our own weld code, and finally required a
  statistically sized test sample rather than a 30-edge coin flip. The native solver is
  bit-deterministic per seed by construction, which made every regression in
  it bisectable.
- **L6. Ship small, keep an anchor.** Twenty-two releases in nine days meant
  every regression window was one release wide — but §5's case study shows
  windows of one are still windows. The benchmark now keeps per-version
  anchors (`abbench_*`).

## 7. Results

*(Numbers as of v0.4.6; the overnight campaign of 2026-08-18 ran the full
suite 10x across two Blender binaries and two code states, three benchmark
repetitions, isolated-config zip installs, four end-to-end avatar/backend
combinations with posed-rig verification, a 25-run leak soak, and
determinism measurement.)*

- Suite: **123/123** then, **162/162** at v0.5.4; pathological-input gauntlet
  34/34 with the invariant *valid mesh or clean error, never a crash or hang*.
- Benchmark gates: native **5/6** fixtures pass everything (the 6th requires a
  watertight Suzanne; Blender's own Suzanne has 168 boundary edges).
- Flow alignment (median angle to principal curvature, ellipsoid): native
  **~3°** vs QuadriFlow 7.1°. Density response: native reallocates budget
  (≈159% of ideal response) where QuadriFlow's post-hoc relaxation reaches 65%.
- Rexouium end-to-end (29k faces, 116 shells, exact symmetry): ~99.7% quads,
  watertight seam (seam_open = 0 in every campaign run), all 640 shape keys /
  150 groups / UVs preserved **exactly**, posed-deformation error p95
  0.121–0.161% (max 0.7–2.7%) of the bbox diagonal across backends,
  ~14–45 s depending on backend.
- NX-Dinasty (46.8k hand-made): main-shell coverage 99%+, authored hair,
  ruff, teeth and eyes byte-identical (25 keys / 118 groups / UVs exact),
  posed-deformation p95 0.074–0.161%, exact mirror symmetry with zero open
  seam edges attributable to the pipeline.
- Resource envelope: 25 consecutive remeshes in one session leak nothing
  (all datablock growth is reclaimable zero-user orphans; peak RSS 646 MB) —
  a lesson bought expensively, since an earlier *unserialised* test campaign
  exhausted the host's 60 GB of RAM and took the orchestrating session down
  with it.

What remains honestly unsolved: semantic loop placement (eyelid rings, mouth
loops as an artist would draw them) — our curvature alignment follows forms
but does not *plan* loops (§8 attacks the ring half of this directly: the
blocker turned out to be resolution rather than alignment, and buying
resolution locally moves it from unsolved to partly solved); thin-shell
silhouettes remain resolution-bound;
face-count adherence under strongly non-uniform density fields drifts
(-15%..+1%); and Blender's QuadriFlow remains, at the deepest level, weather. (Two
further items closed post-campaign in v0.4.7: the off-default
`preserve_small_shells=False` path — four independent defects, from workers
accepting torn output to whole shells evading the restore check, a path that
had been fragile since inception because nothing exercised it: every
untested path rots — and a per-process nondeterminism in exact-symmetry
cleanup caused by iterating an id()-hashed BMFace set.)

## 8. Opening rings, and what actually blocks semantic loop placement

The first item on §7's unsolved list is the one we then attacked directly, and
the result is worth reporting precisely because it is half a success.

**Motivation.** A hand-retopologist rings an eye socket, a mouth and a nostril
with concentric loops, because that is the topology that deforms correctly and
that is what makes a model read as authored. A 4-RoSy field aligned to
principal curvature has nothing useful to say there: the collar of surface
around a hole cut into a smooth form is nearly developable, so its principal
directions are either isotropic (undefined, and the anisotropy gate correctly
suppresses them) or inherited from the *larger* form the collar sits on. The
field therefore walks straight past the hole, the extractor produces a grid
that is merely **clipped** by the opening, and the quads shear diagonally
through the socket instead of ringing it. Nothing in the pipeline was wrong;
the field was simply answering a different question than the artist asks.

**Method.** No new solver machinery. `detect_openings` accepts a *closed*
boundary loop shorter than 15% of `4·√area` (the perimeter of a square patch
covering that fraction of the surface — a plane's outer border, an open
cylinder's ends and the symmetry-bisect cut all sit well above it, eye sockets
and mouth rims well below) and longer than three output quads. Loops lying
entirely in a declared symmetry plane are the exact-symmetry bisect cut, not
openings, and are dropped; a rim that merely *touches* the plane (a mouth bag
halved by the bisect) is kept, with its in-plane part excluded from the
distance source so the ring field stays mirror-symmetric. From the union of
those rims we run a multi-source Bellman–Ford geodesic distance over the
length-weighted edge graph, clamped to the band width so cost stays local, a
few Jacobi passes to take the kinks out of the piecewise-linear result, then
the per-triangle gradient, area-averaged onto vertices, projected into the
tangent plane and rotated 90°: the tangent of the *offset ring* through each
vertex, which is the direction a hand-placed loop takes. That direction is
written into the **existing** soft alignment channel — the one the guide
system already owns, already 4-RoSy aware, already restricted onto every
hierarchy level — in place of the curvature target, with a weight decaying
over `ring_falloff` = 6 output quads behind a `ring_plateau` = 2 full-strength
core, at `ring_strength` = 0.6 at the rim. The rim vertices themselves are
left alone (they are boundary vertices, already *hard* constrained to the same
direction) and user guides keep theirs (a drawn stroke is an explicit
instruction and outranks an inferred one). Two calibration details earned
their keep: the strength is deliberately below 1, because the ring count *has*
to change as the band widens, which needs irregular vertices, which the field
can only place where it is free to deviate from the ideal polar direction; and
the geodesic replaced a hop-limited BFS whose hop budget had to be guessed
from a mean edge length — the mesh around an opening is exactly where a sculpt
is finest, so a 6-quad band silently stopped at 3.7 quads and the outer half
of it got no instruction at all.

**Stage one (v0.5.2): the instruction lands, and nothing visible happens.** On
the two synthetic fixtures (a disc with a hole, a sphere with a polar hole) the
median 4-RoSy angle of extracted edges inside a four-quad band to the true ring
direction fell from 15–21° to **7.4–9.0°**, with no valence blow-up at the rim,
and the orientation field's *own* residual against the ideal ring direction
inside the band was **0.04°** — the field had done as it was told, essentially
exactly. On the real avatar, at the budget characters actually ship at, the
result was invisible. At 12k faces a Dinasty eye socket carries only a dozen to
sixteen quads around its rim (perimeter ÷ local target edge length): a
concentric loop made of sixteen segments, which must *also* grow its ring count
outwards, is a lumpy polygon, and it reads as noise rather than as an eyelid.
At **≥25k** the same solve clearly read — concentric arcs under the lower lid,
a ringed nose tip. So the honest v0.5.2 finding was a negative one, and the
0.04° residual is what made it diagnostic rather than merely disappointing: for
semantic loop placement, **resolution, not alignment, is the binding
constraint.** The field was already right; the mesh had nowhere to put what the
field knew. We deliberately did not chase the symptom by raising
`ring_strength` until *something* appeared — a stronger ring instruction at 12k
does not buy loops, it buys distortion.

**Stage two (v0.5.4): buy the resolution, locally, without buying faces.** The
diagnosis named its own fix, and the fix is the interesting part.

1. **Band density boost** (`rings.ring_density`). Every detected opening is
   given a quota — `ring_min_quads` = 32 quads around its rim — as a
   per-opening factor `clip(ring_min_quads · ρ_rim / perimeter, 1,
   ring_density_max=2)`, applied over the band with the same
   plateau-then-decay profile (1.5 / 2.5 quads) the feature-density machinery
   of §3.4 uses, for the same reason: a hard step in ρ stitches a dense collar
   onto a coarse interior and the seam shows up as a band of extra poles. The
   boosted field is then **rescaled back onto the pre-boost cell budget**
   `Σ A_v/ρ_v²`, so the socket's new quads are paid for by an interior that
   goes about 2% coarser rather than by a larger mesh: face counts held to
   ±2% at budgets 1560 / 3120 / 6240 / 12500, and the test asserts the
   pre-boost budget to floating-point equality. A `ring_density_budget` = 0.30
   cap, enforced by bisection on the boost itself, stops a head with two
   hundred hair-strand tubes from starving everything else — measured demand
   on the messiest real input was only 1.043–1.11, so the cap never binds; it
   exists for the input we have not seen.
2. **Pinned first offset loop** (`rings.ring_pin_segments`). The field solution
   now carries its geodesic distance forward, and the extractor marches the
   iso-contour at `ring_pin_offset` = 1.5 output quads, folds it into the sharp
   feature list before lattice refinement, and thereby makes the first ring
   exist *by construction* instead of by hoping alignment survives extraction.
   Two negative results here are worth more than the feature: an unordered bag
   of snapped segment pairs — the obvious first implementation — bought nothing
   and visibly scarred the collar, because the extractor treats a feature
   vertex of degree ≠ 2 as a **corner that cannot slide**, and a per-triangle
   bag manufactures such junctions by the dozen, freezing the lattice onto the
   staircase; walking the contour into an *ordered cycle* instead took
   first-ring loop purity 0.70 → 0.81. And the nominal offset of 1.0 quads (the
   place the first lattice line wants to be) is wrong because the rim is pinned
   too: a contour a bare quad out asks for a row of cells in a strip thinner
   than one, and on the disc fixture it duly collapsed into a valence-18 fan.
   Hence 1.5, plus a `ring_pin_min_gap` = 0.9 guard for the contour drifting
   inwards where the band is coarse.
3. **Retune, which measured as "change nothing".** With density in play, the
   widened `ring_falloff` of 8–9 that stage one had suggested is *worse* and
   brittle — valence 19–20 on the fixtures at strength 0.6 — so 6 / plateau 2 /
   strength 0.6 stand, and 0.6 remained the most robust of {0.6, 0.7, 0.75,
   0.8} over six seeds × two fixtures.

**Results (v0.5.4 against rings off, and against v0.5.2).**

| | off | v0.5.2 | v0.5.4 |
|---|---|---|---|
| disc, 4-RoSy band alignment | 17.9–20.8° | 7.4–9.0° | **4.3–6.1°** |
| sphere, 4-RoSy band alignment | 15.1–21.6° | 7.4–9.5° | **4.1–6.5°** |
| quads around a Dinasty eye socket @12k (counted on the output) | 25 | — | **46** |
| Dinasty loop purity, rings 1 / 2 / 3 | 0.54 / 0.26 / 0.23 | — | **0.71 / 0.65 / 0.61** |
| max valence, 12 fixture runs | — | — | **≤ 8** |

The costs are visible and small: global edge-length CV rises 0.381 → 0.436,
which is the boost doing exactly its job (the field is deliberately less
uniform now); eye-local quad aspect *improves* 1.43 → 1.27; quad share 98.2% →
98.1%; wall time unchanged; and with the feature off, output is bit-identical
to the previous release, verified by hash on three fixtures through both the
`solve_fields`+`extract` path and the whole `solver.solve` path.

**The ablation is the headline.** Splitting the two mechanisms apart: the
density boost alone captured most of the win, while field steering alone barely
moved the extracted result (loop purity 0.63 with steering only, against 0.54
with the feature off). Stage one's thesis is therefore not merely restated but
*proven constructively* — at character budgets the field already knew the
answer, and the only thing that helped was giving it somewhere to write it. The
generalisable form: when a correct instruction produces no visible effect,
measure whether the medium can express it before strengthening the
instruction.

**What it is honestly not.** First-ring loop purity of ~0.71 means roughly one
in three ring vertices is still a T-junction: the topology now clearly
*acknowledges* the eye, which is a different claim from artist-quality lids.
Much of the band sits on the socket's hidden inner wall rather than on the
visible lid, and the obvious remedy — widening the profile until it reaches the
lid — measured worse (purity 0.71 → 0.57), so it was not taken. The feature
stays **off by default and out of every preset** (the suite asserts the preset
exclusion), deliberately: it reallocates the user's quad budget, and a
workflow preset has no business making that trade silently.

**The fourth lever became a warning instead of a feature.** Preserved small
shells never reach the backend, so an opening that belongs to one cannot be
ringed however the feature is tuned — on the author's own avatar **13 of 15**
openings are exactly that, with 2 reaching the solver. Rather than special-case
the pipeline, `_preserved_opening_warning` counts openings on both sides of the
split (reading the boundary straight off the Blender mesh, filtered by the same
rule the solver uses), reports `ring_openings_preserved` /
`ring_openings_solved`, and tells the user which knob is holding it. This is a
general shape worth naming: in a pipeline mostly made of defensive exceptions,
a correct feature can measure as inert because an earlier, also-correct stage
removed its input — and the cheapest fix is usually not to remove the exception
but to make the interaction legible to whoever ticked the box.

## 9. Conclusion

A production-grade retopology tool was built in nine days not because any
single algorithm was novel — most components are recognizable descendants of
Instant Meshes and standard mesh-repair practice — but because the
*verification structure* around them was unusually aggressive: hostile
fixtures, real rigged assets, quantitative gates on every merge, renders in
front of a human who owned the ground truth, and implementation agents whose
reports were trusted precisely because they were willing to say "I measured
my idea and it is wrong." The complete system, its 162-check suite, its
benchmark, and this history are free software:
**github.com/nerdrx/quadforge**.

---

*Appendix A — Release timeline:* v0.1.0 initial (79 checks) → v0.2.x
robustness (hang isolation, weight-leak fix, 34-case gauntlet, T-edge repair)
→ v0.3.x workflow (Keep Small Shells, UV islands, padded symmetry) → v0.4.x
native solver v2 (curvature fields, watertight extraction, regularization,
feature fairing, graded density) → v0.4.6 density-regression correction →
v0.4.7 preserve-off path repaired, honest face budgets, full determinism →
v0.4.8 face counts land on target (budget-conserving density, secant search)
→ v0.5.0 presets, guide-constraint fix, 2.4–2.8× native speedup → v0.5.1
QuadriFlow flag-blindness warnings (upstream defect U7) → v0.5.2 Opening
Rings, experimental, plus the hole-filler correctness fix (§5 Q6) → v0.5.3
guided QuadriFlow solves auto-route to the Native backend → v0.5.4 Opening
Rings that read at a game budget (§8): band density boost, pinned first
offset loop, preserved-opening warning.

*Appendix B — Reproduction:* `./tests/run_all.sh` (full suite),
`tests/bench_native.sh` (quality shootout), both headless against Blender
5.2 LTS.
