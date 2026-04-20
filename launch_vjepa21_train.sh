#!/bin/bash
set -e

export NETRC=/mnt/data/yurh/.netrc
export HOME=/mnt/data/yurh
source /mnt/cpfs/yurh/miniconda3/etc/profile.d/conda.sh
conda activate partpacker_wan

REPO="/mnt/cpfs/yurh/Infinigen-Sim"
CKPT_DIR="${REPO}/checkpoints/vjepa21_mlp"
mkdir -p "$CKPT_DIR" "${REPO}/logs"

echo "============================================"
echo "Stage 1: Encode V-JEPA 2.1 features (4 GPU)"
echo "  ViT-g @384 → [23040, 1408] per video"
echo "  hemi only, no box, ratio [0.2, 3.0]"
echo "============================================"

N_ENCODED=$(find /mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_vjepa21/ -name "*.pt" 2>/dev/null | wc -l)
if [ "$N_ENCODED" -gt 3000 ]; then
    echo "  Already encoded ($N_ENCODED files), skipping"
else
    PIDS=()
    for shard in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$shard python "${REPO}/encode_vjepa21.py" \
            --shard $shard --num_shards 4 --batch_size 4 --num_workers 8 \
            > "${REPO}/logs/encode_vjepa21_shard${shard}.txt" 2>&1 &
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
        echo "ERROR: Encoding failed."
        exit 1
    fi
fi

echo ""
echo "============================================"
echo "Stage 2: Train MLP projector (4 GPU DDP)"
echo "  V-JEPA 2.1 [23040, 1408] → MLP → [23040, 1536]"
echo "  DiT from Exp1 step_70000"
echo "  Warmup 10k, full 100k"
echo "============================================"

cd "${REPO}"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_partpacker_v3_mlp.py \
    --manifest "${CKPT_DIR}/manifest.json" \
    --output_dir "$CKPT_DIR" \
    --resume "${REPO}/checkpoints/diff_jepa_filtered_full/step_70000.pt" \
    --batch_size 8 \
    --steps 110000 \
    --warmup_proj_steps 10000 \
    --lr 1e-4 \
    --cond_dim 1408 \
    --cond_tokens 23040 \
    --wandb_project infinipart \
    --wandb_run_name vjepa21_mlp \
    2>&1 | tee "${REPO}/logs/train_vjepa21_mlp.txt"
