#!/usr/bin/env bash
# Render negative-sample articulation videos for every factory/seed under outputs/.
#
# Usage:
#   bash run_negative_all.sh                  # all factories, one sample each
#   bash run_negative_all.sh --dry-run        # print commands without running
#   bash run_negative_all.sh --all-seeds      # all factories, all seeds
#   bash run_negative_all.sh --seeds 0,2      # specific seeds only
#   bash run_negative_all.sh --gpu 0,1        # use two GPUs in parallel
#   bash run_negative_all.sh --factory X,Y    # specific factories only
#
# Requirements:
#   - Blender 3.6  (BLENDER env var or default path below)
#   - ffmpeg       (from conda blender_test env)
#   - CUDA GPU

set -euo pipefail

# ── Defaults (override via env vars or flags) ──
BASE="${BASE:-/mnt/workspace/fulian/work/infinipart}"
BLENDER="${BLENDER:-/mnt/workspace/fulian/blender-3.6.0-linux-x64/blender}"
CONDA_ENV="${CONDA_ENV:-/mnt/workspace/fulian/miniforge3/envs/blender_test}"
ENVMAP="${ENVMAP:-${BASE}/envmap/indoor/brown_photostudio_06_2k.exr}"
RESOLUTION="${RESOLUTION:-512}"
SAMPLES="${SAMPLES:-8}"
DURATION="${DURATION:-1.0}"
FPS="${FPS:-30}"
NEG_VIEWS="${NEG_VIEWS:-front threequarter}"
GPUS="0"
SEEDS=""       # empty = auto-detect from folder
ALL_SEEDS=false  # default: one sample per factory
DRY_RUN=false
FACTORY_FILTER=""

# ── Parse CLI flags ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --gpu)       GPUS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --all-seeds) ALL_SEEDS=true; shift ;;
        --base)      BASE="$2"; shift 2 ;;
        --blender)   BLENDER="$2"; shift 2 ;;
        --envmap)    ENVMAP="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --samples)   SAMPLES="$2"; shift 2 ;;
        --duration)  DURATION="$2"; shift 2 ;;
        --fps)       FPS="$2"; shift 2 ;;
        --factory)   FACTORY_FILTER="$2"; shift 2 ;;
        *)           echo "Unknown flag: $1"; exit 1 ;;
    esac
done

OUTPUTS_DIR="${BASE}/outputs"
SCRIPT="${BASE}/render_negative_samples.py"
SCRIPT_POS="${BASE}/render_articulation.py"
CONDA_LIB="${CONDA_ENV}/lib"
CONDA_BIN="${CONDA_ENV}/bin"

# Convert comma-separated GPU list to array
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_GPUS=${#GPU_ARR[@]}

# ── Sanity checks ──
if [[ ! -f "$BLENDER" ]]; then
    echo "ERROR: Blender not found at $BLENDER"
    exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: render_negative_samples.py not found at $SCRIPT"
    exit 1
fi
if [[ ! -f "$ENVMAP" ]]; then
    echo "ERROR: Environment map not found at $ENVMAP"
    exit 1
fi

# ── Compute thread count per GPU (60% of cores / NUM_GPUS, minimum 1) ──
TOTAL_CORES=$(nproc)
THREADS=$(( TOTAL_CORES * 60 / 100 / NUM_GPUS ))
if [[ "$THREADS" -lt 1 ]]; then THREADS=1; fi

# ── Discover factories ──
FACTORIES=()
if [[ -n "$FACTORY_FILTER" ]]; then
    IFS=',' read -ra FACTORIES <<< "$FACTORY_FILTER"
else
    for d in "$OUTPUTS_DIR"/*/; do
        factory_name="$(basename "$d")"
        # Skip non-factory dirs (e.g. motion_test_v3)
        if [[ "$factory_name" == "motion_test_v3" ]]; then
            continue
        fi
        FACTORIES+=("$factory_name")
    done
fi

echo "========================================"
echo "  Positive + Negative Sample Batch Renderer"
echo "========================================"
echo "  Base:       $BASE"
echo "  Blender:    $BLENDER"
echo "  Envmap:     $ENVMAP"
echo "  GPUs:       $GPUS (${NUM_GPUS} devices, parallel)"
echo "  Resolution: ${RESOLUTION}, Samples: ${SAMPLES}"
echo "  Duration:   ${DURATION}s @ ${FPS} fps"
echo "  Threads:    ${THREADS} per GPU / ${TOTAL_CORES} cores total"
echo "  Factories:  ${#FACTORIES[@]}"
echo "========================================"
echo

# ── Build job list: (factory, seed) pairs ──
JOB_FACTORIES=()
JOB_SEEDS=()

for factory in "${FACTORIES[@]}"; do
    if [[ -n "$SEEDS" ]]; then
        IFS=',' read -ra seed_arr <<< "$SEEDS"
    else
        seed_arr=()
        for sd in "$OUTPUTS_DIR/$factory"/*/; do
            s="$(basename "$sd")"
            if [[ "$s" =~ ^[0-9]+$ ]]; then
                seed_arr+=("$s")
            fi
        done
        IFS=$'\n' seed_arr=($(sort -n <<< "${seed_arr[*]}")); unset IFS
        if ! $ALL_SEEDS && [[ ${#seed_arr[@]} -gt 0 ]]; then
            seed_arr=("${seed_arr[0]}")
        fi
    fi
    for seed in "${seed_arr[@]}"; do
        JOB_FACTORIES+=("$factory")
        JOB_SEEDS+=("$seed")
    done
done

TOTAL=${#JOB_FACTORIES[@]}
echo "Total jobs: $TOTAL"
echo

if $DRY_RUN; then
    for (( i=0; i<TOTAL; i++ )); do
        GPU_IDX=$(( i % NUM_GPUS ))
        GPU="${GPU_ARR[$GPU_IDX]}"
        factory="${JOB_FACTORIES[$i]}"
        seed="${JOB_SEEDS[$i]}"
        OUTPUT_DIR="$OUTPUTS_DIR/motion_test_v3/$factory/$seed"
        echo "[$(( i+1 ))/$TOTAL] $factory seed=$seed  (GPU $GPU)"
        echo "  [dry-run] POS: CUDA_VISIBLE_DEVICES=$GPU $BLENDER --background --python $SCRIPT_POS -- --factory $factory --seed $seed --output_dir $OUTPUT_DIR/positives"
        echo "  [dry-run] NEG: CUDA_VISIBLE_DEVICES=$GPU $BLENDER --background --python $SCRIPT -- --factory $factory --seed $seed --output_dir $OUTPUT_DIR"
    done
    exit 0
fi

# ── Result tracking ──
RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR"' EXIT

# ── Worker function: runs pos+neg for one (factory, seed) on a given GPU ──
run_job() {
    local job_idx="$1"
    local factory="$2"
    local seed="$3"
    local gpu="$4"
    local result_file="$RESULTS_DIR/job_${job_idx}"

    local scene_dir="$OUTPUTS_DIR/$factory/$seed"
    local urdf="$scene_dir/scene.urdf"

    if [[ ! -f "$urdf" ]]; then
        echo "[$(( job_idx+1 ))/$TOTAL] SKIP $factory seed=$seed (no scene.urdf)"
        echo "skipped" > "$result_file"
        return
    fi

    local output_dir="$OUTPUTS_DIR/motion_test_v3/$factory/$seed"
    mkdir -p "$output_dir/positives" "$output_dir/negatives"

    echo "[$(( job_idx+1 ))/$TOTAL] START $factory seed=$seed  (GPU $gpu)"

    # ── Positive ──
    local pos_log="$output_dir/positives/render.log"
    if env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PATH="$CONDA_BIN:$PATH" \
        LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}" \
        "$BLENDER" --background --threads "$THREADS" \
        --python "$SCRIPT_POS" -- \
        --factory "$factory" \
        --seed "$seed" \
        --device 0 \
        --base "$BASE" \
        --envmap "$ENVMAP" \
        --resolution "$RESOLUTION" \
        --samples "$SAMPLES" \
        --duration "$DURATION" \
        --fps "$FPS" \
        --output_dir "$output_dir/positives" \
        --views $NEG_VIEWS \
        > "$pos_log" 2>&1; then
        echo "[$(( job_idx+1 ))/$TOTAL] OK  positive $factory seed=$seed -> $output_dir/positives/"
    else
        echo "[$(( job_idx+1 ))/$TOTAL] FAIL positive $factory seed=$seed (see $pos_log)"
        echo "failed $factory seed=$seed positive (see $pos_log)" > "$result_file"
        return
    fi

    # ── Negative ──
    local neg_log="$output_dir/negatives/render.log"
    if env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PATH="$CONDA_BIN:$PATH" \
        LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}" \
        "$BLENDER" --background --threads "$THREADS" \
        --python "$SCRIPT" -- \
        --factory "$factory" \
        --seed "$seed" \
        --device 0 \
        --base "$BASE" \
        --envmap "$ENVMAP" \
        --resolution "$RESOLUTION" \
        --samples "$SAMPLES" \
        --duration "$DURATION" \
        --fps "$FPS" \
        --output_dir "$output_dir" \
        --neg_views $NEG_VIEWS \
        > "$neg_log" 2>&1; then
        echo "[$(( job_idx+1 ))/$TOTAL] OK  negative $factory seed=$seed -> $output_dir/negatives/"
        echo "done" > "$result_file"
    else
        echo "[$(( job_idx+1 ))/$TOTAL] FAIL negative $factory seed=$seed (see $neg_log)"
        echo "failed $factory seed=$seed negative (see $neg_log)" > "$result_file"
    fi
}

# ── FIFO-based semaphore for parallel GPU scheduling ──
FIFO=$(mktemp -u)
mkfifo "$FIFO"
exec 3<>"$FIFO"
rm "$FIFO"

# Pre-fill semaphore with NUM_GPUS tokens
for (( i=0; i<NUM_GPUS; i++ )); do
    echo "$i" >&3
done

PIDS=()

for (( i=0; i<TOTAL; i++ )); do
    # Block until a GPU slot is available
    read -r GPU_SLOT <&3

    GPU="${GPU_ARR[$GPU_SLOT]}"
    factory="${JOB_FACTORIES[$i]}"
    seed="${JOB_SEEDS[$i]}"

    (
        run_job "$i" "$factory" "$seed" "$GPU"
        # Return GPU slot to semaphore
        echo "$GPU_SLOT" >&3
    ) &
    PIDS+=($!)
done

# Wait for all background jobs
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

exec 3>&-

# ── Summarize results ──
DONE=0
FAILED=0
SKIPPED=0
FAIL_LIST=""

for (( i=0; i<TOTAL; i++ )); do
    result_file="$RESULTS_DIR/job_${i}"
    if [[ ! -f "$result_file" ]]; then
        FAILED=$(( FAILED + 1 ))
        FAIL_LIST="$FAIL_LIST  - ${JOB_FACTORIES[$i]} seed=${JOB_SEEDS[$i]} (no result)\n"
    else
        result=$(cat "$result_file")
        case "$result" in
            done)    DONE=$(( DONE + 1 )) ;;
            skipped) SKIPPED=$(( SKIPPED + 1 )) ;;
            failed*) FAILED=$(( FAILED + 1 )); FAIL_LIST="$FAIL_LIST  - ${result#failed }\n" ;;
        esac
    fi
done

echo
echo "========================================"
echo "  Batch complete"
echo "  Done:    $DONE"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo "  Total:   $TOTAL"
echo "========================================"

if [[ -n "$FAIL_LIST" ]]; then
    echo
    echo "Failed jobs:"
    echo -e "$FAIL_LIST"
fi
