#!/bin/bash
set -e
source /mnt/cpfs/yurh/miniconda3/etc/profile.d/conda.sh
conda activate partpacker_wan

CKPT="/mnt/cpfs/yurh/Infinigen-Sim/checkpoints/diff_jepa_filtered_full/step_70000.pt"
SCRIPT="/mnt/cpfs/yurh/Infinigen-Sim/infer_quick.py"
CFGS="0.0 1.0 2.0 4.0 5.0 7.9 11.0"
SEEDS="123 777"

# Assign tasks to GPUs round-robin
GPU=0
PIDS=()

for seed in $SEEDS; do
    for cfg in $CFGS; do
        outdir="/mnt/cpfs/yurh/Infinigen-Sim/output/infer_seed${seed}_cfg${cfg}"
        echo "GPU=$GPU seed=$seed cfg=$cfg -> $outdir"
        CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT \
            --ckpt $CKPT \
            --output_dir "$outdir" \
            --n_samples 8 --cfg_scale $cfg --seed $seed --no_render \
            > /dev/null 2>&1 &
        PIDS+=($!)
        GPU=$(( (GPU + 1) % 4 ))
    done
done

echo "Launched ${#PIDS[@]} tasks, waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait $pid; then
        echo "PID $pid failed"
        FAIL=1
    fi
done

if [ $FAIL -ne 0 ]; then
    echo "Some tasks failed"
    exit 1
fi
echo "All inference done!"
