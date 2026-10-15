#!/usr/bin/env bash
# Self-test for tests/check_contracts.py: good fixtures must pass (exit 0),
# every broken fixture must fail (exit 1).
# Usage: bash tests/selftest.sh   (override interpreter via PYTHON=/path/to/python)
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

if ! "$PY" -c "import numpy" 2>/dev/null; then
  echo "error: '$PY' has no numpy; set PYTHON=/path/to/python with numpy" >&2
  exit 2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== generating fixtures =="
"$PY" tests/make_fixtures.py "$TMP/fx" || exit 2
echo

pass=0
fail=0
expect() {
  local want="$1" desc="$2"; shift 2
  "$@" >"$TMP/last.log" 2>&1
  local got=$?
  if [ "$got" -eq "$want" ]; then
    echo "ok    $desc"
    pass=$((pass + 1))
  else
    echo "FAIL  $desc (want exit $want, got $got)"
    sed 's/^/      /' "$TMP/last.log"
    fail=$((fail + 1))
  fi
}

echo "== good fixtures (expect exit 0) =="
expect 0 "clips dir"        "$PY" tests/check_contracts.py clips     "$TMP/fx/good/clips"
expect 0 "hand_pose dir"    "$PY" tests/check_contracts.py hand_pose "$TMP/fx/good/hand_pose"
expect 0 "hand_pose file"   "$PY" tests/check_contracts.py hand_pose "$TMP/fx/good/hand_pose/demo01_001.npz"
expect 0 "pose6d dir"       "$PY" tests/check_contracts.py pose6d    "$TMP/fx/good/pose6d"
expect 0 "pose6d file"      "$PY" tests/check_contracts.py pose6d    "$TMP/fx/good/pose6d/demo01_001.npz"
expect 0 "samples dir"      "$PY" tests/check_contracts.py samples   "$TMP/fx/good/samples"
expect 0 "samples file"     "$PY" tests/check_contracts.py samples   "$TMP/fx/good/samples/train/s000.npz"
echo

echo "== broken fixtures (expect exit 1) =="
expect 1 "hand_pose: partial-frame NaN" "$PY" tests/check_contracts.py hand_pose "$TMP/fx/broken/handpose_partial_nan/hand_pose"
expect 1 "hand_pose: float64 dtype"     "$PY" tests/check_contracts.py hand_pose "$TMP/fx/broken/handpose_bad_dtype/hand_pose"
expect 1 "hand_pose: missing conf key"  "$PY" tests/check_contracts.py hand_pose "$TMP/fx/broken/handpose_missing_conf/hand_pose"
expect 1 "hand_pose: conf/NaN mismatch" "$PY" tests/check_contracts.py hand_pose "$TMP/fx/broken/handpose_conf_mismatch/hand_pose"
expect 1 "pose6d: non-orthonormal R"    "$PY" tests/check_contracts.py pose6d    "$TMP/fx/broken/pose6d_bad_rotation/pose6d"
expect 1 "pose6d: partial NaN"          "$PY" tests/check_contracts.py pose6d    "$TMP/fx/broken/pose6d_partial_nan/pose6d"
expect 1 "pose6d: duplicate object_ids" "$PY" tests/check_contracts.py pose6d    "$TMP/fx/broken/pose6d_dup_ids/pose6d"
expect 1 "samples: label id out of range" "$PY" tests/check_contracts.py samples "$TMP/fx/broken/samples_bad_label/samples"
expect 1 "samples: slot/id mismatch"    "$PY" tests/check_contracts.py samples   "$TMP/fx/broken/samples_bad_slot/samples"
expect 1 "samples: missing valid masks" "$PY" tests/check_contracts.py samples   "$TMP/fx/broken/samples_missing_keys/samples"
expect 1 "samples: no label keys"       "$PY" tests/check_contracts.py samples   "$TMP/fx/broken/samples_no_label/samples"
expect 1 "samples: missing index"       "$PY" tests/check_contracts.py samples   "$TMP/fx/broken/samples_missing_index/samples"
expect 1 "samples: split leak"          "$PY" tests/check_contracts.py samples   "$TMP/fx/broken/samples_split_leak/samples"
expect 1 "clips: duplicate clip_id"     "$PY" tests/check_contracts.py clips     "$TMP/fx/broken/clips_duplicate_id/clips"
expect 1 "clips: frames mismatch"       "$PY" tests/check_contracts.py clips     "$TMP/fx/broken/clips_frames_mismatch/clips"
expect 1 "clips: extra csv column"      "$PY" tests/check_contracts.py clips     "$TMP/fx/broken/clips_extra_column/clips"
echo

echo "== usage errors (expect exit 2) =="
expect 2 "unknown stage"   "$PY" tests/check_contracts.py nope "$TMP/fx/good/clips"
expect 2 "missing path"    "$PY" tests/check_contracts.py clips "$TMP/fx/does_not_exist"
echo

if command -v ffprobe >/dev/null 2>&1; then
  echo "note: ffprobe found; good-clips check included real ffprobe spot-checks"
else
  echo "note: ffprobe not found; ffprobe spot-check path not exercised"
fi

echo "======================================"
echo "selftest: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
