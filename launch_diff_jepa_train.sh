#!/bin/bash
set -e

CONDA_SH="/mnt/cpfs/yurh/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate partpacker_wan

REPO="/mnt/cpfs/yurh/Infinigen-Sim"
CKPT_DIR="${REPO}/checkpoints/diff_jepa_filtered_full"
mkdir -p "$CKPT_DIR"

echo "============================================"
echo "Stage 1: Encode diff JEPA (4 GPU parallel)"
echo "============================================"

PIDS=()
for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard python "${REPO}/encode_diff_jepa.py" \
        --shard $shard --num_shards 4 --batch_size 8 --num_workers 8 \
        > "${REPO}/logs/encode_diff_jepa_shard${shard}.txt" 2>&1 &
    PIDS+=($!)
    echo "  Shard $shard started (PID ${PIDS[-1]})"
done

echo "Waiting for all 4 encoding shards..."
FAIL=0
for i in 0 1 2 3; do
    if wait ${PIDS[$i]}; then
        echo "  Shard $i done (PID ${PIDS[$i]})"
    else
        echo "  Shard $i FAILED (PID ${PIDS[$i]})"
        FAIL=1
    fi
done

if [ $FAIL -ne 0 ]; then
    echo "ERROR: Some encoding shards failed. Check logs. Aborting training."
    exit 1
fi

echo ""
echo "============================================"
echo "Stage 2: Train (4 GPU DDP, 110k steps)"
echo "============================================"

cd /mnt/cpfs/yurh/PartPacker

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_partnet_vjepa_ddp.py \
    --data_root /mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_solidified \
    --jepa_root /mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_diff_jepa_filtered \
    --output_dir "$CKPT_DIR" \
    --batch_size 8 \
    --steps 110000 \
    --warmup_proj_steps 10000 \
    --lr 1e-4 \
    --init_from_pretrained \
    --exclude_categories "PhysXNet,PhysXNet_PhysXnet" \
    --wandb_project infinipart \
    --wandb_run_name diff_jepa_filtered_full \
    2>&1 | tee "${REPO}/logs/train_diff_jepa_filtered_full.txt"
