#!/usr/bin/env bash
# QuadForge native-vs-QuadriFlow quality benchmark.
#
#   tests/bench_native.sh              full run (6 fixtures x 2 backends)
#   tests/bench_native.sh --quick      sphere + Suzanne only
#   QF_BLENDER=/path/to/blender tests/bench_native.sh
#   QF_BENCH_OUT=/somewhere tests/bench_native.sh
#
# Writes the table to stdout, bench_results.json and <fixture>_<backend>.png
# to $QF_BENCH_OUT.  Always exits 0 while the native solver is still v1:
# the run is a baseline measurement, not a pass/fail test.
set -u

BLENDER="${QF_BLENDER:-/run/media/nerdrx/Lex/claude/quadwild_tools/blender-5.2.0-linux-x64/blender}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
BENCH="$HERE/bench_native.py"
export QF_BENCH_OUT="${QF_BENCH_OUT:-/tmp/claude-1000/-run-media-nerdrx-Lex-claude/cbcfcc9a-ddfd-4fbb-8582-1c3b129cb280/scratchpad/bench}"

if [ ! -x "$BLENDER" ]; then
    echo "!! blender binary not found or not executable: $BLENDER" >&2
    echo "   set QF_BLENDER=/path/to/blender" >&2
    exit 2
fi
if [ ! -f "$BENCH" ]; then
    echo "!! missing $BENCH" >&2
    exit 2
fi

mkdir -p "$QF_BENCH_OUT" || exit 2
cd "$ROOT" || exit 2

echo ".. blender  $BLENDER"
echo ".. out dir  $QF_BENCH_OUT"

"$BLENDER" --background --factory-startup --python "$BENCH" -- "$@" 2>&1
status="${PIPESTATUS[0]}"

if [ "$status" -ne 0 ]; then
    echo "!! blender exited with $status (crash before the bench finished?)" >&2
fi
exit 0
