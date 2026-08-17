# Native solver v2 — make it the primary backend

Goal: beat the QuadriFlow backend on *topology quality*, not just robustness:

1. **Curvature-aligned flow** — 4-RoSy orientation field aligned to principal
   curvature directions (weighted by anisotropy |k1-k2|/(|k1|+|k2|)). This is
   what produces natural loops around eyes, mouths, ears, muscle forms.
2. **True density reallocation** — per-vertex target edge length from
   `qf_density` and curvature adaptivity; a 2.5x painted head really gets
   ~6x more quads per area. (The v1 solver already does this — keep it.)
3. **Clean extraction** — the current weak point. Acceptance gates below.
4. Guides and sharp edges keep constraining the field as in v1.

## Module contract (package quadforge/backends/native/)

```python
# fields.py  (owner: agent-fields)
solve_fields(V: (n,3)f64, F: (m,3)i32, params) -> FieldSolution
#   FieldSolution: dataclass with
#     N   (n,3) f64  vertex normals
#     Q   (n,3) f64  unit tangent dir (representative of the 4-RoSy class)
#     rho (n,)  f64  per-vertex target edge length
#   params keys: target_faces:int, density:(n,)|None (1=neutral, >1 denser),
#     adaptive:float 0..1, sharp_edges:(e,2)i32, guide_dirs:{vidx: (3,)f64},
#     curvature_align:float 0..1 (weight of principal-dir alignment), seed:int
#   Must be deterministic for a given seed. numpy only, no scipy.

# extract.py (owner: agent-extract)
extract(V, F, sol: FieldSolution, params) -> (VQ (k,3)f64, FQ list[tuple 3|4])
#   Gates (enforced by the benchmark, measured after mesh.validate()):
#     closed input  -> closed output (0 boundary edges), 0 non-manifold edges
#     quad_pct >= 97 on organic closed shells at economy budgets
#     face count within 25% of target (before strict-count retries)
#     no vertex farther than 1.5% bbox diag from the input surface
#   Repair passes are fair game (hole fill, tri fusion, pole relax) as long as
#   they preserve the two manifoldness gates.

# solver.py (owner: integrator) orchestrates; __init__.py adapts to Blender.
```

## Benchmark (tests/bench_native.py, owner: agent-bench)

Headless: fixtures = sphere, cube(sharp), torus, Suzanne-subdiv, dense head
region (Dinasty head crop is NOT in repo — use Suzanne head as proxy),
half-density/full-density painted sphere. For native v2 AND quadriflow backend:
  quad_pct, boundary_edges, non_manifold, poles_5plus per 1k faces,
  face-count error, surface deviation p95, **flow alignment**: median angle
  between quad edge directions and principal curvature directions where
  anisotropy > 0.3 (the metric that shows loop quality), density correlation,
  wall time. Prints a comparison table; exits nonzero if native regresses on
  its gates. Renders one wire close-up per fixture pair to /tmp for eyeballing.

## Notes for agents

- Blender for tests: /run/media/nerdrx/Lex/claude/quadwild_tools/blender-5.2.0-linux-x64/blender
  (`--background --factory-startup --python x.py`); bundled numpy 2.3. The
  numpy core can also be developed against system python3 (numpy 2.5).
- v1 files exist and WORK (95-98% quads on spheres, holes on complex meshes)
  — read them first; keep public signatures of __init__.remesh unchanged.
- Principal curvature directions: estimate per-vertex via least-squares fit of
  the shape operator in the tangent plane over the 1-ring (or 2-ring for
  stability); smooth the resulting cross field a few iterations before mixing
  into the 4-RoSy energy with weight curvature_align * anisotropy.
- Do NOT touch files outside your ownership. Scratch dir:
  /tmp/claude-1000/-run-media-nerdrx-Lex-claude/cbcfcc9a-ddfd-4fbb-8582-1c3b129cb280/scratchpad/<agent>/
