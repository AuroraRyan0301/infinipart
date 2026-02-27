#!/bin/bash
# Benchmark full pipeline: generate → split_precompute → render (1 animode, 32 views)
# Test 3 sources: IM, PhysXNet, PhysXMobility
set -e
cd /mnt/data/yurh/Infinigen-Sim
export PYTHONUNBUFFERED=1

PYTHON=/mnt/data/yurh/miniconda3/envs/partpacker_wan/bin/python
BLENDER=/mnt/data/yurh/blender-3.6.0-linux-x64/blender
BENCH_DIR=benchmark_output
GPU=0

mkdir -p $BENCH_DIR

echo "=========================================="
echo "Pipeline Benchmark - $(date)"
echo "=========================================="

# Test cases: (source, factory, seed, base_dir)
# IM: DishwasherFactory/0 - already generated
# PhysXNet: AppliancePhysXNetFactory/10007 - already generated (json→urdf done)
# PhysXMobility: FurniturePhysXMobilityFactory/100520 - already generated

declare -a SOURCES=("IM" "PhysXNet" "PhysXMobility")
declare -a FACTORIES=("DishwasherFactory" "AppliancePhysXNetFactory" "FurniturePhysXMobilityFactory")
declare -a SEEDS=("0" "10007" "100520")
declare -a BASES=("/mnt/data/yurh/Infinite-Mobility" "/mnt/data/yurh/Infinigen-Sim" "/mnt/data/yurh/Infinigen-Sim")

for i in 0 1 2; do
    SRC=${SOURCES[$i]}
    FACTORY=${FACTORIES[$i]}
    SEED=${SEEDS[$i]}
    BASE=${BASES[$i]}

    echo ""
    echo "============================================================"
    echo "[$SRC] $FACTORY / $SEED"
    echo "============================================================"

    # ── Stage 1: Generate (Blender) ──
    URDF="$BASE/outputs/$FACTORY/$SEED/scene.urdf"
    if [ -f "$URDF" ]; then
        echo "[Stage 1: Generate] SKIPPED (already exists: $URDF)"
        echo "  Time: 0s (cached)"
    else
        echo "[Stage 1: Generate] Running..."
        T1_START=$(date +%s)
        CUDA_VISIBLE_DEVICES=$GPU $BLENDER --background --python-use-system-env \
            --python infinigen_examples/generate_individual_assets.py -- \
            --output_folder "outputs/$FACTORY" -f "$FACTORY" -n 1 --seed "$SEED" \
            2>&1 | tail -5
        T1_END=$(date +%s)
        echo "  Time: $((T1_END - T1_START))s"
    fi

    # ── Stage 2: Split Precompute ──
    echo ""
    SPLIT_OUT="$BENCH_DIR/precompute"
    echo "[Stage 2: Split Precompute] Running..."
    T2_START=$(date +%s)

    if [[ "$FACTORY" == *"PhysX"* ]]; then
        # PhysXNet/PhysXMobility source detection
        if [[ "$FACTORY" == *"PhysXMobility"* ]]; then
            SRC_ARG="physxmob"
        else
            SRC_ARG="physxnet"
        fi
        $PYTHON split_precompute.py --factory "$FACTORY" --seed "$SEED" \
            --source "$SRC_ARG" --output_dir "$SPLIT_OUT" --force 2>&1
    else
        $PYTHON split_precompute.py --factory "$FACTORY" --seed "$SEED" \
            --output_dir "$SPLIT_OUT" --force 2>&1
    fi

    T2_END=$(date +%s)
    N_SPLITS=$(ls -d "$SPLIT_OUT/$FACTORY/$SEED"/anim* 2>/dev/null | wc -l)
    echo "  Splits: $N_SPLITS animodes"
    echo "  Time: $((T2_END - T2_START))s"

    # ── Stage 3: Render (1 animode, 32 views, bg+nobg via compositor) ──
    echo ""
    RENDER_OUT="$BENCH_DIR/renders/$FACTORY/$SEED"
    mkdir -p "$RENDER_OUT"
    echo "[Stage 3: Render] animode=0, 32 views (bg+nobg single pass)..."
    T3_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPU $BLENDER --background --python-use-system-env \
        --python render_articulation.py -- \
        --factory "$FACTORY" --seed "$SEED" --device 0 \
        --output_dir "$RENDER_OUT" \
        --base "$BASE" \
        --resolution 512 --samples 32 \
        --duration 4.0 --fps 30 \
        --animode 0 \
        --views hemi_00 hemi_01 hemi_02 hemi_03 hemi_04 hemi_05 hemi_06 hemi_07 \
                hemi_08 hemi_09 hemi_10 hemi_11 hemi_12 hemi_13 hemi_14 hemi_15 \
        --moving_views orbit_00 orbit_01 orbit_02 orbit_03 orbit_04 orbit_05 orbit_06 orbit_07 \
                       sweep_00 sweep_01 sweep_02 sweep_03 sweep_04 sweep_05 sweep_06 sweep_07 \
        2>&1 | grep -E "Rendering view|Saved|Total|STATIC|collision|time|Error|ERROR" | tail -40

    T3_END=$(date +%s)
    N_NOBG=$(find "$RENDER_OUT" -name "*_nobg.mp4" 2>/dev/null | wc -l)
    N_BG=$(find "$RENDER_OUT" -name "*_bg.mp4" -o -name "*_bg_*.mp4" 2>/dev/null | wc -l)
    echo "  Videos: $N_NOBG nobg + $N_BG bg"
    echo "  Time: $((T3_END - T3_START))s"

    echo ""
    echo "--- [$SRC] Summary ---"
    echo "  Generate:   cached"
    echo "  Precompute: $((T2_END - T2_START))s ($N_SPLITS splits)"
    echo "  Render:     $((T3_END - T3_START))s ($N_NOBG nobg + $N_BG bg videos)"
done

echo ""
echo "=========================================="
echo "Benchmark done - $(date)"
echo "Results in: $BENCH_DIR/"
echo "=========================================="
