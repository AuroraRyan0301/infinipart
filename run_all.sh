#!/bin/bash
# ============================================================
# Cluster deployment script for Infinigen-Sim pipeline
# Designed for 100-card 4090 cluster
#
# Usage:
#   # Full pipeline on single node (4 GPUs)
#   bash run_all.sh
#
#   # Sharded across N nodes (run on each node with different SHARD_ID)
#   SHARD_ID=0 N_SHARDS=25 bash run_all.sh
#
#   # Skip generation (assets already exist)
#   SKIP_GENERATE=1 bash run_all.sh
#
#   # Only specific source
#   SOURCE=physxnet bash run_all.sh
#
#   # Dry run
#   DRY_RUN=1 bash run_all.sh
# ============================================================
set -e

# ── Configuration (override via environment) ──
WORK_DIR="${WORK_DIR:-/mnt/data/yurh/Infinigen-Sim}"
PYTHON="${PYTHON:-/mnt/data/yurh/miniconda3/envs/partpacker_wan/bin/python}"
BLENDER="${BLENDER:-/mnt/data/yurh/blender-3.6.0-linux-x64/blender}"

N_GPUS="${N_GPUS:-4}"              # GPUs per node
N_SEEDS="${N_SEEDS:-50}"           # Seeds per IM factory
SOURCE="${SOURCE:-all}"            # im, physxnet, physxmob, all
SHARD_ID="${SHARD_ID:-}"           # e.g. 0 (empty = no sharding)
N_SHARDS="${N_SHARDS:-}"           # e.g. 25 (empty = no sharding)
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_POSITIVE="${SKIP_POSITIVE:-0}"
SKIP_NEGATIVE="${SKIP_NEGATIVE:-0}"
DRY_RUN="${DRY_RUN:-0}"

cd "$WORK_DIR"
export PYTHONUNBUFFERED=1

# ── Shard args ──
SHARD_ARG=""
SHARD_LABEL="full"
if [ -n "$SHARD_ID" ] && [ -n "$N_SHARDS" ]; then
    SHARD_ARG="--shard ${SHARD_ID}/${N_SHARDS}"
    SHARD_LABEL="shard_${SHARD_ID}_of_${N_SHARDS}"
fi

# ── Logging ──
LOG="pipeline_${SHARD_LABEL}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "  Infinigen-Sim Pipeline"
echo "============================================================"
echo "  Date:      $(date)"
echo "  Node:      $(hostname)"
echo "  GPUs:      $N_GPUS"
echo "  Source:    $SOURCE"
echo "  Seeds:     $N_SEEDS"
echo "  Shard:     ${SHARD_LABEL}"
echo "  Log:       $LOG"
echo "  Dry run:   $DRY_RUN"
echo "============================================================"

DRY_FLAG=""
if [ "$DRY_RUN" = "1" ]; then
    DRY_FLAG="--dry_run"
fi

# ── Stage 1: Generate assets + split precompute + verify ──
if [ "$SKIP_GENERATE" != "1" ]; then
    echo ""
    echo "=== Stage 1: Asset Generation ==="
    echo "Started: $(date)"
    $PYTHON run_pipeline.py \
        --stage generate \
        --source "$SOURCE" \
        --n_seeds "$N_SEEDS" \
        --n_workers "$N_GPUS" \
        $SHARD_ARG \
        $DRY_FLAG
    echo "Generate done: $(date)"
else
    echo ""
    echo "=== Stage 1: SKIPPED (SKIP_GENERATE=1) ==="
fi

# ── Stage 2: Positive sample rendering ──
if [ "$SKIP_POSITIVE" != "1" ]; then
    echo ""
    echo "=== Stage 2: Positive Rendering ==="
    echo "Started: $(date)"
    $PYTHON run_pipeline.py \
        --stage render_positive \
        --source "$SOURCE" \
        --n_gpus "$N_GPUS" \
        $SHARD_ARG \
        $DRY_FLAG
    echo "Positive render done: $(date)"
else
    echo ""
    echo "=== Stage 2: SKIPPED (SKIP_POSITIVE=1) ==="
fi

# ── Stage 3: Negative sample rendering ──
if [ "$SKIP_NEGATIVE" != "1" ]; then
    echo ""
    echo "=== Stage 3: Negative Rendering ==="
    echo "Started: $(date)"
    $PYTHON run_pipeline.py \
        --stage render_negative \
        --source "$SOURCE" \
        --n_gpus "$N_GPUS" \
        $SHARD_ARG \
        $DRY_FLAG
    echo "Negative render done: $(date)"
else
    echo ""
    echo "=== Stage 3: SKIPPED (SKIP_NEGATIVE=1) ==="
fi

echo ""
echo "============================================================"
echo "  Pipeline complete: $(date)"
echo "  Log: $WORK_DIR/$LOG"
echo "============================================================"
