# QuadForge — Module Contracts

Free Quad-Remesher-class auto-retopology addon for Blender 5.2+ (works headless).
Target Blender: 5.2 LTS. Test binary: `/run/media/nerdrx/Lex/claude/quadwild_tools/blender-5.2.0-linux-x64/blender`

## Layout (file ownership — do not edit files owned by other modules)

```
quadforge/                     addon package
  __init__.py                  registration (OWNER: integrator; agents don't touch)
  properties.py                QF_Settings PropertyGroup (OWNER: integrator — READ ONLY contract)
  pipeline.py                  orchestration                 (OWNER: agent-pipeline)
  core/
    analysis.py                curvature, hard edges, density (OWNER: agent-pipeline)
    guides.py                  guide projection, material loops (OWNER: agent-pipeline)
    transfer.py                data preservation              (OWNER: agent-transfer)
    report.py                  mesh quality metrics           (OWNER: agent-ui)
  backends/
    quadriflow.py              built-in solver wrapper        (OWNER: agent-pipeline)
    native/                    field-based solver             (OWNER: agent-native)
      __init__.py  solver.py  fields.py  extract.py
  ops/
    remesh.py paint.py batch.py lods.py                       (OWNER: agent-ui)
  ui/
    panel.py                                                  (OWNER: agent-ui)
tests/
  run_tests.py  test_*.py                                     (OWNER: agent-tests)
```

## Settings contract (properties.py — final, code against these names)

`obj.quadforge` → `QF_Settings` with fields:

- preset: Enum ['CUSTOM','GAME_AVATAR','SCULPT_CLEANUP','HARD_SURFACE','QUICK_DRAFT']
  default CUSTOM — selecting one writes `properties.PRESET_KEYS` once via its update
  callback (apply-on-select, never reverted; see `properties.PRESETS` and docs/MANUAL.md).
  Settings copies must wrap writes in `properties.suppress_preset_update()`.
- target_count: Int, default 5000, min 12
- mode: Enum ['FACES','RATIO','EDGE'] default FACES
- target_ratio: Float default 1.0; target_edge_length: Float default 0.1
- adaptive_size: Float 0..100 (%, curvature adaptivity; 0 = uniform)
- adapt_quad_count: Bool (True = allow count drift for adaptivity)
- strict_count: Bool (iterate solver up to 3x to land within 10% of target)
- use_paint_density: Bool — read float point-attribute `qf_density` (0..2, 1=neutral)
- detect_hard_edges: Bool default True; hard_edge_angle: Float radians default radians(40)
- use_marked_sharp: Bool (respect existing sharp/crease/seam as hard edges)
- use_materials: Bool (material boundaries become preserved edge loops)
- use_uv_seams: Bool (UV island boundaries become feature edges)
- use_guides: Bool; guide_collection: PointerProperty(Collection) — curve/GP objects steer flow
- symmetry_x/symmetry_y/symmetry_z: Bool
- exact_symmetry: Bool (bisect + remesh half + mirror weld = mathematically exact)
- preserve_boundaries: Bool default True
- preserve_uvs / preserve_weights / preserve_shape_keys / preserve_materials /
  preserve_creases / preserve_bevel_weights: Bool, all default True
- keep_original: Bool default True (original moved to 'QuadForge Originals' collection, hidden)
- backend: Enum ['QUADRIFLOW','NATIVE'] default QUADRIFLOW
- seed: Int
- preserve_small_shells: Bool default True (small separate shells keep their topology)
- small_shell_limit: Int default 0 (0 = auto: max(64, 2% of input faces))
- solver_isolation: Bool default True (QuadriFlow in a killable child process)
- lod_targets: String, comma-separated face counts, e.g. "8000,2000,500"
- last_report: String (JSON of last run stats, set by pipeline)

## Pipeline contract (pipeline.py)

```python
def run_remesh(context, obj, s) -> dict
# s = obj.quadforge. Returns {'ok': bool, 'error': str|None, 'object': new_obj|None,
#   'stats': {'faces': int, 'quads': int, 'quad_pct': float, 'time_s': float, ...}}
# Steps: validate → snapshot (transfer.capture) → duplicate working mesh →
#   preprocess (triangulate ngons>4 only if needed, apply modifiers on copy, analysis, guides)
#   → backend.remesh → postprocess (exact symmetry mirror, transfer.apply, report) →
#   swap into scene (respect keep_original).
# MUST be callable headless (no bpy.ops that need UI context beyond object mode basics).
```

## Backend contract

```python
# backends/quadriflow.py
def remesh(context, work_obj, s, face_target: int) -> None      # in-place on work_obj
# backends/native/__init__.py
def remesh(context, work_obj, s, face_target: int) -> None      # in-place; numpy only
# native internals: solver.solve(V:(n,3)f64, F:(m,3)i32, params: dict) -> (VQ:(k,3), FQ_list)
#   FQ_list: list of index tuples, each of length 3 or 4 (quad-dominant; >=70% quads expected)
# params: {'target_faces', 'adaptive', 'sharp_edges': (e,2)i32, 'guide_dirs': per-face unit vec or None,
#          'density': per-vertex float or None, 'symmetry': (bool,bool,bool), 'seed'}
```

## Analysis / guides contract (core/analysis.py, core/guides.py)

```python
def mark_hard_edges(work_obj, s) -> int            # marks edge sharp; returns count
def build_density_attr(work_obj, s) -> None        # writes float point attr 'qf_density' (curvature × paint)
def material_boundaries_to_sharp(work_obj) -> int
def project_guides(work_obj, guide_objects, s) -> int  # nearest-surface projection of curve/GP polylines;
    # marks path edges sharp AND stores per-face guide direction in 'qf_guide' 3-float face attr
```

## Transfer contract (core/transfer.py)

```python
def capture(obj) -> Snapshot        # UVs, materials+poly assignments, vgroups, shape keys,
                                    # creases, bevel weights, custom normals flag; original mesh copy
def apply(snapshot, new_obj, s) -> dict   # surface-nearest mapping (build BVH on original mesh);
    # shape keys: transfer basis-relative deltas via barycentric surface mapping.
    # returns {'uvs': bool, 'weights': int, 'shape_keys': int, 'materials': int, ...}
```

## Report contract (core/report.py)

```python
def mesh_report(obj) -> dict   # faces, quads, tris, ngons, quad_pct, poles_3, poles_5plus,
                               # non_manifold_edges, symmetry_error_x/y/z (max mirror mismatch), area
```

## Ops (ops/*.py) — bl_idnames

`quadforge.remesh`, `quadforge.paint_density`, `quadforge.clear_density`,
`quadforge.remesh_batch`, `quadforge.generate_lods`, `quadforge.quality_report`,
`quadforge.toggle_original`, `quadforge.guides_new`, `quadforge.symmetry_check`

Panel: View3D sidebar, category "QuadForge".

## Testing rules

- All tests run headless: `blender --background --factory-startup --python tests/run_tests.py`
- run_tests.py discovers `tests/test_*.py`, each exposes `def run(ctx) -> list[tuple[name, ok, msg]]`
- Addon dir added to sys.path + registered via `quadforge.register()` in run_tests.py.
- Exit code non-zero on any failure; print `RESULT ok/total` summary line.

## Style

Python only, no external deps beyond bundled numpy. No `bpy.ops` where direct data API works.
Every module must import cleanly without UI (no context access at import time).
