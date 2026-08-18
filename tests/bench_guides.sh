#!/usr/bin/env bash
# QuadForge guide-quality benchmark (does the edge flow follow the guide?).
#
#   tests/bench_guides.sh                 all 3 fixtures x 2 backends x guides/control
#   tests/bench_guides.sh --quick         sphere + grid only
#   tests/bench_guides.sh --no-render     skip the PNGs
#   QF_BLENDER=/path/to/blender tests/bench_guides.sh
#   QF_GUIDE_BENCH_OUT=/somewhere tests/bench_guides.sh
#
# Writes the banded alignment table to stdout, plus bench_guides.json and
# <fixture>_<backend>_<guides|control>.png into $QF_GUIDE_BENCH_OUT.
set -u

BLENDER="${QF_BLENDER:-/run/media/nerdrx/Lex/claude/quadwild_tools/blender-5.2.0-linux-x64/blender}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
BENCH="$HERE/bench_guides.py"
export QF_GUIDE_BENCH_OUT="${QF_GUIDE_BENCH_OUT:-/tmp/claude-1000/-run-media-nerdrx-Lex-claude/cbcfcc9a-ddfd-4fbb-8582-1c3b129cb280/scratchpad/guides}"

if [ ! -x "$BLENDER" ]; then
    echo "!! blender binary not found or not executable: $BLENDER" >&2
    echo "   set QF_BLENDER=/path/to/blender" >&2
    exit 2
fi
if [ ! -f "$BENCH" ]; then
    echo "!! missing $BENCH" >&2
    exit 2
fi

mkdir -p "$QF_GUIDE_BENCH_OUT" || exit 2
cd "$ROOT" || exit 2

echo ".. blender  $BLENDER"
echo ".. out dir  $QF_GUIDE_BENCH_OUT"

"$BLENDER" --background --factory-startup --python "$BENCH" -- "$@" 2>&1
status="${PIPESTATUS[0]}"

if [ "$status" -ne 0 ]; then
    echo "!! blender exited with $status (crash before the bench finished?)" >&2
fi
exit 0
