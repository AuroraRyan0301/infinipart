#!/bin/bash
# Render timing ablation: DishwasherFactory/0, 2 views
# Configs: default(32spp), 16spp, eevee
set -e
cd /mnt/data/yurh/Infinigen-Sim
export PYTHONUNBUFFERED=1

BLENDER=/mnt/data/yurh/blender-3.6.0-linux-x64/blender
GPU=0
FACTORY=DishwasherFactory
SEED=0
BASE=/mnt/data/yurh/Infinite-Mobility
LOG=ablation.log

exec > >(tee "$LOG") 2>&1

echo "=========================================="
echo "Render Ablation - $(date)"
echo "=========================================="

# Config: name, extra_args
run_config() {
    local NAME=$1
    shift
    local EXTRA="$@"

    echo ""
    echo "============================================================"
    echo "Config: $NAME - started $(date)"
    echo "============================================================"

    OUT_DIR="benchmark_output/ablation/${NAME}"
    rm -rf "$OUT_DIR" 2>/dev/null
    mkdir -p "$OUT_DIR"

    CUDA_VISIBLE_DEVICES=$GPU $BLENDER --background --python-use-system-env \
        --python render_articulation.py -- \
        --factory $FACTORY --seed $SEED --device 0 \
        --output_dir "$OUT_DIR" \
        --base $BASE \
        --resolution 512 \
        --duration 4.0 --fps 30 \
        --animode 0 \
        --views hemi_00 \
        --moving_views orbit_00 \
        $EXTRA \
        2>&1

    echo "--- $NAME finished $(date) ---"
}

run_config "cycles_32spp" --samples 32
run_config "cycles_16spp" --samples 16
run_config "eevee" --samples 32 --engine eevee

echo ""
echo "=========================================="
echo "All done - $(date)"
echo "=========================================="
