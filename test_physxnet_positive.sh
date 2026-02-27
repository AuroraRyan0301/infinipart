#!/bin/bash
# Test PhysXNet + PhysX_mobility rendering: prepare scene + render all animodes
set -e

WORK_DIR="/mnt/data/yurh/Infinigen-Sim"
BLENDER="/mnt/data/yurh/blender-3.6.0-linux-x64/blender"
PYTHON="/mnt/data/yurh/miniconda3/envs/partpacker_wan/bin/python"
OUT_DIR="$WORK_DIR/test_output"
N_GPUS=4
DURATION=2.0
SAMPLES=16
RES=256

cd "$WORK_DIR"

# Test cases: factory seed max_animode
declare -a TESTS=(
  "FurniturePhysXNetFactory:0:4"
  "ElectronicsPhysXNetFactory:0:3"
  "LightingPhysXNetFactory:0:3"
  "ContainerPhysXNetFactory:0:3"
  "PlumbingPhysXNetFactory:0:3"
  "ElectronicsPhysXMobilityFactory:0:3"
  "FurniturePhysXMobilityFactory:0:2"
  "PlumbingPhysXMobilityFactory:0:3"
  "KitchenPhysXMobilityFactory:0:3"
  "ToolPhysXMobilityFactory:0:2"
)

echo "============================================================"
echo "  PhysXNet + PhysX_mobility Positive Sample Test"
echo "============================================================"

# Results tracking
RESULT_DIR="$OUT_DIR/_results_physx"
mkdir -p "$RESULT_DIR"
> "$RESULT_DIR/summary.txt"

# Step 1: Prepare all scenes (non-Blender, fast)
echo ""
echo "=== Stage 1: Preparing scenes ==="
for test in "${TESTS[@]}"; do
  IFS=':' read -r factory seed max_anim <<< "$test"
  echo "  Preparing $factory seed=$seed ..."
  $PYTHON -c "
from physxnet_loader import prepare_physxnet_scene
import sys
try:
    info = prepare_physxnet_scene('$factory', $seed)
    print(f'    OK: obj_id={info[\"obj_id\"]}, dataset={info[\"dataset\"]}')
except Exception as e:
    print(f'    FAIL: {e}')
    sys.exit(1)
"
  if [ $? -ne 0 ]; then
    echo "    ERROR: Scene preparation failed for $factory"
    echo "$factory seed=$seed PREPARE_FAIL" >> "$RESULT_DIR/summary.txt"
  fi
done

# Step 2: Build job list and render in parallel
echo ""
echo "=== Stage 2: Rendering ==="

JOBS=()
for test in "${TESTS[@]}"; do
  IFS=':' read -r factory seed max_anim <<< "$test"
  for ((a=0; a<=max_anim; a++)); do
    JOBS+=("$factory:$seed:$a")
  done
done

echo "  ${#JOBS[@]} render jobs across $N_GPUS GPUs"
echo ""

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
    --base "$WORK_DIR" \
    --animode "$animode" \
    --views hemi_00 hemi_08 \
    --moving_views orbit_00 \
    --output_dir "$outdir" \
    --duration "$DURATION" --samples "$SAMPLES" --resolution "$RES" \
    --skip_bg \
    > "$logfile" 2>&1

  local exit_code=$?
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
    if grep -q "No matching joints\|exiting without rendering" "$logfile" 2>/dev/null; then
      status="SKIP(no_joints)"
    else
      status="FAIL(exit=$exit_code)"
    fi
  fi

  printf "%-45s anim%-2s seed=%-2s %s videos=%d static=%d\n" \
    "$factory" "$animode" "$seed" "$status" "$n_videos" "$n_static" >> "$RESULT_DIR/summary.txt"
  echo "[GPU $gpu] $factory s$seed anim$animode: $status (${n_videos}v ${n_static}s)"
}

# Parallel execution
job_idx=0
total=${#JOBS[@]}
while [ $job_idx -lt $total ]; do
  pids=()
  for ((g=0; g<N_GPUS && job_idx<total; g++)); do
    run_job "${JOBS[$job_idx]}" "$g" &
    pids+=($!)
    job_idx=$((job_idx + 1))
  done
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

n_ok=$(grep -c " OK " "$RESULT_DIR/summary.txt" 2>/dev/null || true)
n_skip=$(grep -c "SKIP" "$RESULT_DIR/summary.txt" 2>/dev/null || true)
n_fail=$(grep -c "FAIL" "$RESULT_DIR/summary.txt" 2>/dev/null || true)
echo "Total: $n_ok OK, $n_skip SKIP, $n_fail FAIL (of ${#JOBS[@]} jobs)"
echo "Results: $RESULT_DIR/summary.txt"
