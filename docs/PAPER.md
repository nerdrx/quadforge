# QuadForge: Building a Production Auto-Retopology System through
# Agentic Development and Adversarial Visual QA

**nerdrx & Claude (Anthropic Fable 5, with parallel Opus 5 implementation agents)**
*August 9–18, 2026 — 15 releases, ~10,700 lines, 123 automated checks*

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
evolved through fifteen releases driven almost entirely by a single feedback
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

- **123 assertion checks** across 14 headless test modules (registration,
  remeshing accuracy, symmetry exactness, sharp preservation, data transfer,
  density/guides, reporting, native gates, LODs/batch, orientation/seam
  regressions, mirrored-weight regressions, a 34-case pathological-input
  gauntlet, UV-island following, seam-debris regressions), with crash-detecting
  runner, per-module isolation mode, and flake-hardened assertions.
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
- **L6. Ship small, keep an anchor.** Fifteen releases in nine days meant
  every regression window was one release wide — but §5's case study shows
  windows of one are still windows. The benchmark now keeps per-version
  anchors (`abbench_*`).

## 7. Results

*(Numbers as of v0.4.6; the overnight campaign of 2026-08-18 ran the full
suite 10x across two Blender binaries and two code states, three benchmark
repetitions, isolated-config zip installs, four end-to-end avatar/backend
combinations with posed-rig verification, a 25-run leak soak, and
determinism measurement.)*

- Suite: **123/123**; pathological-input gauntlet 34/34 with the invariant
  *valid mesh or clean error, never a crash or hang*.
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
but does not *plan* loops; thin-shell silhouettes remain resolution-bound;
face-count adherence under strongly non-uniform density fields drifts
(-15%..+1%); and Blender's QuadriFlow remains, at the deepest level, weather. (Two
further items closed post-campaign in v0.4.7: the off-default
`preserve_small_shells=False` path — four independent defects, from workers
accepting torn output to whole shells evading the restore check, a path that
had been fragile since inception because nothing exercised it: every
untested path rots — and a per-process nondeterminism in exact-symmetry
cleanup caused by iterating an id()-hashed BMFace set.)

## 8. Conclusion

A production-grade retopology tool was built in nine days not because any
single algorithm was novel — most components are recognizable descendants of
Instant Meshes and standard mesh-repair practice — but because the
*verification structure* around them was unusually aggressive: hostile
fixtures, real rigged assets, quantitative gates on every merge, renders in
front of a human who owned the ground truth, and implementation agents whose
reports were trusted precisely because they were willing to say "I measured
my idea and it is wrong." The complete system, its 123-check suite, its
benchmark, and this history are free software:
**github.com/nerdrx/quadforge**.

---

*Appendix A — Release timeline:* v0.1.0 initial (79 checks) → v0.2.x
robustness (hang isolation, weight-leak fix, 34-case gauntlet, T-edge repair)
→ v0.3.x workflow (Keep Small Shells, UV islands, padded symmetry) → v0.4.x
native solver v2 (curvature fields, watertight extraction, regularization,
feature fairing, graded density) → v0.4.6 density-regression correction.

*Appendix B — Reproduction:* `./tests/run_all.sh` (full suite),
`tests/bench_native.sh` (quality shootout), both headless against Blender
5.2 LTS.
