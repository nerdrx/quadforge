#!/usr/bin/env bash
# QuadForge headless test suite.
#
#   tests/run_all.sh                       run everything in one Blender process
#   QF_ONLY=test_02 tests/run_all.sh       run only matching test modules
#   QF_ISOLATE=1 tests/run_all.sh          one Blender process per test module
#                                          (survives a hard crash in the addon)
#   QF_BLENDER=/path/to/blender            override the Blender binary
#
# Exit code is non-zero if any check failed or Blender died.
set -u

BLENDER="${QF_BLENDER:-/run/media/nerdrx/Lex/claude/quadwild_tools/blender-5.2.0-linux-x64/blender}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
RUNNER="$HERE/run_tests.py"

if [ ! -x "$BLENDER" ]; then
    echo "!! blender binary not found or not executable: $BLENDER" >&2
    echo "   set QF_BLENDER=/path/to/blender" >&2
    exit 2
fi

cd "$ROOT" || exit 2

LOG="$(mktemp -t quadforge-tests-XXXXXX.log)"
trap 'rm -f "$LOG"' EXIT

# Runs blender once and streams its output; returns blender's exit status.
run_blender() {
    "$BLENDER" --background --factory-startup --python "$RUNNER" 2>&1 | tee "$LOG"
    return "${PIPESTATUS[0]}"
}

# Emits a synthetic FAIL line when Blender died before printing RESULT.
# $1 = blender exit status
report_crash() {
    local rc="$1" last
    last="$(grep -E '^BEGIN ' "$LOG" | tail -n 1 | sed 's/^BEGIN //')"
    [ -z "$last" ] && last="<startup>"
    echo "FAIL ${last}::process — blender died with exit ${rc} before finishing"
    echo "     (a segfault in the addon cannot be trapped from Python;"
    echo "      see /tmp/blender.crash.txt, or re-run with QF_ISOLATE=1)"
}

total_ok=0
total_all=0

if [ "${QF_ISOLATE:-0}" = "1" ]; then
    modules=()
    for f in "$HERE"/test_*.py; do
        [ -e "$f" ] || continue
        base="$(basename "$f" .py)"
        if [ -n "${QF_ONLY:-}" ] && [[ "$base" != *"${QF_ONLY}"* ]]; then
            continue
        fi
        modules+=("$base")
    done
    if [ "${#modules[@]}" -eq 0 ]; then
        echo "!! no test modules matched QF_ONLY=${QF_ONLY:-}" >&2
        exit 2
    fi
    echo "== QuadForge test suite (isolated: ${#modules[@]} processes) =="
    for m in "${modules[@]}"; do
        QF_ONLY="$m" "$BLENDER" --background --factory-startup \
            --python "$RUNNER" > "$LOG" 2>&1
        rc=$?
        grep -E '^(PASS|FAIL) ' "$LOG"
        line="$(grep -E '^RESULT ' "$LOG" | tail -n 1)"
        if [ -z "$line" ]; then
            report_crash "$rc"
            total_all=$((total_all + 1))
        else
            ok="${line#RESULT }"; ok="${ok%%/*}"
            all="${line##*/}"
            total_ok=$((total_ok + ok))
            total_all=$((total_all + all))
        fi
    done
    echo "RESULT ${total_ok}/${total_all}"
    [ "$total_ok" -eq "$total_all" ] && [ "$total_all" -gt 0 ]
    status=$?
else
    run_blender
    status=$?
    if ! grep -qE '^RESULT ' "$LOG"; then
        report_crash "$status"
        ok="$(grep -cE '^PASS ' "$LOG")"
        all=$(( ok + $(grep -cE '^FAIL ' "$LOG") + 1 ))
        echo "RESULT ${ok}/${all}"
        status=1
    fi
fi

echo "run_all.sh: exit ${status}"
exit "$status"
