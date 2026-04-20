#!/bin/bash
set -e

export NETRC=/mnt/data/yurh/.netrc
export HOME=/mnt/data/yurh
CONDA_SH="/mnt/cpfs/yurh/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate partpacker_wan

REPO="/mnt/cpfs/yurh/Infinigen-Sim"
CKPT_DIR="${REPO}/checkpoints/diff_jepa_v2_selfattn"
mkdir -p "$CKPT_DIR" "${REPO}/logs"

echo "============================================"
echo "Stage 1: Encode v2 features (4 GPU parallel)"
echo "  orig_sub(5120) + diff_sub(4480) = 9600 tokens x 1408 dim"
echo "  hemi only, no box, ratio [0.2, 3.0]"
echo "============================================"

PIDS=()
for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard python "${REPO}/encode_diff_jepa_v2.py" \
        --shard $shard --num_shards 4 --batch_size 8 --num_workers 8 \
        > "${REPO}/logs/encode_v2_shard${shard}.txt" 2>&1 &
    PIDS+=($!)
    echo "  Shard $shard started (PID ${PIDS[-1]})"
done

echo "Waiting for encoding..."
FAIL=0
for i in 0 1 2 3; do
    if wait ${PIDS[$i]}; then
        echo "  Shard $i done"
    else
        echo "  Shard $i FAILED"
        FAIL=1
    fi
done

if [ $FAIL -ne 0 ]; then
    echo "ERROR: Encoding failed. Aborting."
    exit 1
fi

echo ""
echo "============================================"
echo "Stage 2: Train v2 (4 GPU DDP)"
echo "  Projector: MLP + 2-layer Self-Attention"
echo "  DiT: from step_70000, proj: fresh"
echo "  Warmup proj 10k, full train 100k"
echo "============================================"

cd /mnt/cpfs/yurh/PartPacker

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_partnet_vjepa_ddp.py \
    --data_root /mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_solidified \
    --jepa_root /mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_jepa_v2 \
    --manifest "${REPO}/checkpoints/diff_jepa_v2_selfattn/manifest.json" \
    --output_dir "$CKPT_DIR" \
    --batch_size 8 \
    --steps 110000 \
    --warmup_proj_steps 10000 \
    --lr 1e-4 \
    --resume "${REPO}/checkpoints/diff_jepa_filtered_full/step_70000.pt" \
    --resume_dit_only \
    --proj_type self_attn \
    --proj_layers 2 \
    --proj_heads 8 \
    --cond_dim 1408 \
    --cond_tokens 9600 \
    --exclude_categories "PhysXNet,PhysXNet_PhysXnet,box" \
    --wandb_project infinipart \
    --wandb_run_name diff_jepa_v2_selfattn \
    2>&1 | tee "${REPO}/logs/train_v2_selfattn.txt"
