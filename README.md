# QuadForge

Free auto-retopology addon for Blender 5.2+ — a Quad Remesher replacement.

Converts any mesh (sculpt, scan, boolean soup, triangulated import) into a clean,
quad, animation-ready mesh — and unlike most remeshers, it brings your **UVs,
materials, vertex weights and shape keys along** to the new topology.

## Features

**Remeshing**
- Target quad count / ratio / edge length, with optional **Strict Count** (iterates until within ~10%)
- **Adaptive Size** — concentrate quads in curved areas
- **Painted density** — paint where you want more/less resolution (`qf_density` attribute)
- Backends: **QuadriFlow+** (Blender's built-in solver wrapped with QuadForge pre/post passes)
  and an experimental **Native** field-based solver (Instant-Meshes-style, pure numpy)

**Edge flow control**
- Hard-edge detection by angle, plus existing sharp / crease / seam edges
- **Material boundaries** become preserved edge loops
- **Guide curves** — draw Bezier/Grease-Pencil strokes; QuadForge projects them onto the
  surface and aligns edge flow along them
- Open-boundary preservation

**Symmetry**
- X / Y / Z toggles, any combination
- **Exact mode**: bisect → remesh half → mirror-weld = mathematically perfect symmetry

**Data preservation (the good stuff)**
- UVs (seam-aware), materials + face assignments, vertex groups (weights),
  **shape keys**, creases, bevel weights — all re-projected onto the new topology.
  Remesh a rigged, shape-keyed character and keep working.

**Workflow**
- Non-destructive: original is kept in a hidden `QuadForge Originals` collection (one-click toggle)
- **Batch remesh** all selected objects
- **LOD generation** — one click, multiple face targets → `_LOD0/1/2` set
- **Quality report** — quad %, poles, non-manifold edges, per-axis symmetry error

## Install

1. Zip the `quadforge/` folder (or use `./package.sh` → `quadforge-<version>.zip`)
2. Blender → Edit → Preferences → Add-ons → Install from Disk → pick the zip
3. Panel appears in the 3D Viewport sidebar (N) → **QuadForge** tab

## Test

```bash
tests/run_all.sh            # headless full suite
QF_ONLY=symmetry tests/run_all.sh   # filter
```

## License

GPL-3.0 (same as Blender). Do whatever you want, keep it free.
