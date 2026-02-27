#!/bin/bash
# Batch collision response tests across multiple factories
# Usage: bash test_collision_batch.sh <group_id>  (0-3)
# Run 4 groups in parallel across 4 tmux panes

BLENDER="/mnt/data/yurh/blender-3.6.0-linux-x64/blender"
SCRIPT="/mnt/data/yurh/Infinigen-Sim/render_articulation.py"
BASE="/mnt/data/yurh/Infinite-Mobility"
OUTBASE="/mnt/data/yurh/Infinigen-Sim/test_physics/collision_batch"

GROUP=$1
mkdir -p "$OUTBASE"

run_test() {
    local factory=$1
    local seed=$2
    local animode=$3
    local tag="${factory}_s${seed}_anim${animode}"
    local outdir="${OUTBASE}/${tag}"

    echo "========================================"
    echo "  TEST: ${tag}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES="" "$BLENDER" --background --python "$SCRIPT" -- \
        --factory "$factory" --seed "$seed" --animode "$animode" \
        --views front --skip_bg \
        --output_dir "$outdir" --base "$BASE" 2>&1 | \
        grep -E "Phase|Rest|collision|PROXIMITY|target|done|WARNING|ERROR|Passive|Animated|one-way|anticipation|keyframed|skipped|No matching|No passive"

    echo "  -> Output: ${outdir}"
    echo ""
}

case $GROUP in
    0)
        # Group 0: BeverageFridge + Dishwasher animode 1 (prismatic push door)
        run_test BeverageFridgeFactory 0 1
        run_test BeverageFridgeFactory 1 1
        run_test DishwasherFactory 0 1
        ;;
    1)
        # Group 1: Dishwasher seeds 1,2 + KitchenCabinet seed 0
        run_test DishwasherFactory 1 1
        run_test DishwasherFactory 2 1
        run_test KitchenCabinetFactory 0 1
        ;;
    2)
        # Group 2: KitchenCabinet seeds 1,2 + Microwave seed 0
        run_test KitchenCabinetFactory 1 1
        run_test KitchenCabinetFactory 2 1
        run_test MicrowaveFactory 0 1
        ;;
    3)
        # Group 3: Microwave seeds 1,2 + Oven seed 2
        run_test MicrowaveFactory 1 1
        run_test MicrowaveFactory 2 1
        run_test OvenFactory 2 1
        ;;
esac

echo "======== Group $GROUP DONE ========"
