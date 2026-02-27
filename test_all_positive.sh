#!/bin/bash
# Test ALL IM factories, ALL animodes, positive samples only.
# Runs 4 GPU workers in parallel. Auto-finds valid seed per factory.
#
# Flow: precompute splits → read metadata → render only valid animodes
set -e

WORK_DIR="/mnt/data/yurh/Infinigen-Sim"
BLENDER="/mnt/data/yurh/blender-3.6.0-linux-x64/blender"
PYTHON="/mnt/data/yurh/miniconda3/envs/partpacker_wan/bin/python"
BASE="/mnt/data/yurh/Infinite-Mobility"
OUT_DIR="$WORK_DIR/test_output"
PRECOMPUTE_DIR="$WORK_DIR/precompute"
N_GPUS=4
DURATION=2.0
SAMPLES=16
RES=256

cd "$WORK_DIR"

# Factory list (all IM)
FACTORIES=(
  DishwasherFactory BeverageFridgeFactory MicrowaveFactory OvenFactory
  KitchenCabinetFactory ToiletFactory WindowFactory LiteDoorFactory
  OfficeChairFactory TapFactory LampFactory PotFactory
  BottleFactory BarChairFactory PanFactory TVFactory
)

# Find a valid seed for each factory (one with scene.urdf)
declare -A VALID_SEED
for factory in "${FACTORIES[@]}"; do
  found=""
  for s in $(ls "$BASE/outputs/$factory/" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | head -10); do
    if [ -f "$BASE/outputs/$factory/$s/scene.urdf" ]; then
      found="$s"
      break
    fi
  done
  if [ -z "$found" ]; then
    echo "WARNING: No valid seed for $factory (no scene.urdf found), skipping"
  else
    VALID_SEED[$factory]=$found
  fi
done

# ── Stage 1: Precompute splits ──
echo "============================================================"
echo "  Stage 1: Precompute splits"
echo "============================================================"
n_precompute=0
n_skip=0
for factory in "${!VALID_SEED[@]}"; do
  seed=${VALID_SEED[$factory]}
  meta="$PRECOMPUTE_DIR/$factory/$seed/metadata.json"
  if [ -f "$meta" ]; then
    echo "  $factory/$seed: cached"
    continue
  fi
  echo "  $factory/$seed: computing..."
  $PYTHON split_precompute.py --factory "$factory" --seed "$seed" \
    --base "$BASE" --output_dir "$PRECOMPUTE_DIR" 2>&1 | sed 's/^/    /'
  if [ -f "$meta" ]; then
    n_precompute=$((n_precompute + 1))
  else
    echo "    SKIP: no movable joints"
    n_skip=$((n_skip + 1))
  fi
done
echo "  Precomputed: $n_precompute, Skipped (no joints): $n_skip"
echo ""

# ── Stage 2: Build job list from metadata ──
# Factory -> max animode (fallback when metadata doesn't exist)
declare -A ANIMODES
ANIMODES=(
  [DishwasherFactory]=2 [BeverageFridgeFactory]=2
  [MicrowaveFactory]=2 [OvenFactory]=2
  [KitchenCabinetFactory]=2 [ToiletFactory]=3
  [WindowFactory]=4 [LiteDoorFactory]=0
  [OfficeChairFactory]=2 [TapFactory]=2
  [LampFactory]=3 [PotFactory]=4
  [BottleFactory]=3 [BarChairFactory]=2
  [PanFactory]=0 [TVFactory]=2
)

# Read metadata to get valid animodes per (factory, seed)
JOBS=()
for factory in "${!VALID_SEED[@]}"; do
  seed=${VALID_SEED[$factory]}
  meta="$PRECOMPUTE_DIR/$factory/$seed/metadata.json"
  if [ ! -f "$meta" ]; then
    echo "  SKIP $factory/$seed: no precompute metadata (no movable joints)"
    continue
  fi
  # Get valid animodes from metadata using Python
  valid_anims=$($PYTHON -c "
import json, sys
meta = json.load(open('$meta'))
joints = meta.get('joints', [])
if not joints:
    sys.exit(0)
joint_types = {j['type'] for j in joints}
max_anim = ${ANIMODES[$factory]}
STYPE = {0: 'revolute', 1: 'prismatic', 2: 'continuous'}
valid = []
for a in range(max_anim + 1):
    if a in STYPE:
        if STYPE[a] in joint_types:
            valid.append(a)
    else:
        valid.append(a)
# Fallback: non-standard factory mapping, include all animodes
if not valid and joint_types:
    valid = list(range(max_anim + 1))
for a in valid:
    print(a)
")
  for a in $valid_anims; do
    JOBS+=("$factory:$seed:$a")
  done
done

echo "============================================================"
echo "  Stage 2: Positive Sample Test: ${#JOBS[@]} jobs across $N_GPUS GPUs"
echo "============================================================"
for factory in $(echo "${!VALID_SEED[@]}" | tr ' ' '\n' | sort); do
  echo "  $factory seed=${VALID_SEED[$factory]}"
done
echo ""
echo "  Duration: ${DURATION}s, Res: ${RES}, Samples: $SAMPLES"
echo "  Views: hemi_00, hemi_08, orbit_00"
echo ""

# Results tracking
RESULT_DIR="$OUT_DIR/_results"
mkdir -p "$RESULT_DIR"
> "$RESULT_DIR/summary.txt"

run_job() {
  local job="$1"
  local gpu="$2"
  IFS=':' read -r factory seed animode <<< "$job"
  local outdir="$OUT_DIR/$factory/$seed"
  local logfile="$RESULT_DIR/${factory}_s${seed}_anim${animode}.log"

  local animode_suffix=""
  if [ "$animode" -gt 0 ]; then
    animode_suffix="_anim${animode}"
  fi

  CUDA_VISIBLE_DEVICES=$gpu $BLENDER --background --python render_articulation.py -- \
    --factory "$factory" --seed "$seed" --device 0 \
    --base "$BASE" \
    --animode "$animode" \
    --views hemi_00 hemi_08 \
    --moving_views orbit_00 \
    --output_dir "$outdir" \
    --duration "$DURATION" --samples "$SAMPLES" --resolution "$RES" \
    --skip_bg \
    > "$logfile" 2>&1

  local exit_code=$?

  # Count output videos
  local n_videos=0
  local n_static=0
  for vname in hemi_00 hemi_08 orbit_00; do
    local mp4="$outdir/${vname}${animode_suffix}_nobg.mp4"
    if [ -f "$mp4" ]; then
      n_videos=$((n_videos + 1))
    fi
  done
  n_static=$(grep -c "STATIC detected" "$logfile" 2>/dev/null || true)

  local status="OK"
  if [ $exit_code -ne 0 ]; then
    # Check if it was "No matching joints" (expected, not a failure)
    if grep -q "No matching joints\|exiting without rendering" "$logfile" 2>/dev/null; then
      status="SKIP(no_joints)"
    else
      status="FAIL(exit=$exit_code)"
    fi
  fi

  printf "%-30s anim%-2s seed=%-2s %s videos=%d static=%d\n" \
    "$factory" "$animode" "$seed" "$status" "$n_videos" "$n_static" >> "$RESULT_DIR/summary.txt"
  echo "[GPU $gpu] $factory s$seed anim$animode: $status (${n_videos}v ${n_static}s)"
}

# Parallel execution with N_GPUS workers
job_idx=0
total=${#JOBS[@]}

while [ $job_idx -lt $total ]; do
  pids=()
  # Launch up to N_GPUS jobs
  for ((g=0; g<N_GPUS && job_idx<total; g++)); do
    run_job "${JOBS[$job_idx]}" "$g" &
    pids+=($!)
    job_idx=$((job_idx + 1))
  done

  # Wait for this batch
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
done

echo ""
echo "============================================================"
echo "  RESULTS SUMMARY"
echo "============================================================"
sort "$RESULT_DIR/summary.txt"
echo ""

# Count totals
n_ok=$(grep -c " OK " "$RESULT_DIR/summary.txt" 2>/dev/null || true)
n_skip=$(grep -c "SKIP" "$RESULT_DIR/summary.txt" 2>/dev/null || true)
n_fail=$(grep -c "FAIL" "$RESULT_DIR/summary.txt" 2>/dev/null || true)
echo "Total: $n_ok OK, $n_skip SKIP, $n_fail FAIL (of ${#JOBS[@]} jobs)"
echo "Results: $RESULT_DIR/summary.txt"
echo "Videos: $OUT_DIR/{Factory}/{seed}/"
