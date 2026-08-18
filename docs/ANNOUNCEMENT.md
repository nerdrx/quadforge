# QuadForge — a free, open-source Quad Remesher alternative for Blender

*(draft for BlenderArtists / Reddit / social — post it, edit it, or ignore it; your call)*

I got tired of the Quad Remesher licence, so (with heavy help from AI agents)
I built a replacement and I'm giving it away: **QuadForge**, GPL-3.0, for
Blender 4.2+ / 5.2.

**github.com/nerdrx/quadforge** — grab the zip from Releases, install from
disk, find it in the N-sidebar.

What it does that might interest you:

- **Auto-retopology with two engines**: a hardened QuadriFlow wrapper (solver
  stalls can never freeze Blender — everything runs in killable worker
  processes) and a from-scratch field-aligned solver whose edge flow follows
  curvature (brows, muzzles, muscle forms) and beats QuadriFlow on a
  flow-alignment benchmark that ships in the repo.
- **Your rig survives.** Shape keys (tested with a 640-key stack), vertex
  weights, seam-crisp UVs, materials — all re-attached to the new topology.
  Posed-deformation error measured at ~0.1% of the body diagonal.
- **Hand-authored detail is sacred**: hair cards, teeth, eyes, piercings and
  other small shells keep their original topology instead of becoming blobs.
- **Exact symmetry** — mathematically mirrored, watertight seam, and features
  near the centerline (inner toes, nose tips) survive the cut.
- Painted density, guide strokes (Native backend), UV-island following,
  hard-edge detection, LOD generation, batch, one-click presets, quality
  reports.
- **151 automated tests**, three benchmark suites, deterministic per seed,
  and a research paper in the repo documenting every bug we found — several
  of them in Blender's own QuadriFlow.

It was built in ~9 days of adversarial testing against real VRChat-style
avatars. It is not a 1:1 Quad Remesher clone — its facial loop placement is
curvature-driven rather than artist-semantic — but for game avatars, sculpt
cleanup and LOD work it holds its own, and it's free forever.

Feedback and hostile test meshes very welcome.
