#!/bin/bash
set -e

# === CONFIGURE THESE FOR YOUR MACHINE ===
export NETRC=/mnt/data/yurh/.netrc
export HOME=/mnt/data/yurh
CONDA_SH="/mnt/cpfs/yurh/miniconda3/etc/profile.d/conda.sh"
# =========================================

source "$CONDA_SH"
conda activate partpacker_wan

REPO="/mnt/cpfs/yurh/Infinigen-Sim"
CKPT_DIR="${REPO}/checkpoints/adaln_cond"
mkdir -p "$CKPT_DIR" "${REPO}/logs"

echo "============================================"
echo "Stage 1: Encode v2 features (4 GPU parallel)"
echo "  orig_sub(5120) + diff_sub(4480) = 9600 tokens x 1408 dim"
echo "  hemi only, no box, ratio [0.2, 3.0]"
echo "============================================"

# Check if encoding already done
N_ENCODED=$(find /mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_jepa_v2/ -name "*.pt" 2>/dev/null | wc -l)
if [ "$N_ENCODED" -gt 3000 ]; then
    echo "  Already encoded ($N_ENCODED files), skipping"
else
    PIDS=()
    for shard in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$shard python "${REPO}/encode_diff_jepa_v2.py" \
            --shard $shard --num_shards 4 --batch_size 8 --num_workers 8 \
            > "${REPO}/logs/encode_adaln_shard${shard}.txt" 2>&1 &
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
fi

echo ""
echo "============================================"
echo "Stage 2: AdaLN Condition Training (4 GPU DDP)"
echo "  DiT: from Exp1 step_70000"
echo "  Proj: MLP(1408->1536->1536) + AdaLN pool MLP"
echo "  Warmup 10k steps, full 100k steps"
echo "============================================"

# Generate manifest if not exists
MANIFEST="${CKPT_DIR}/manifest.json"
if [ ! -f "$MANIFEST" ]; then
    # Copy from v2 selfattn manifest (same data split)
    cp "${REPO}/checkpoints/diff_jepa_v2_selfattn/manifest.json" "$MANIFEST"
    echo "  Copied manifest"
fi

cd "${REPO}"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_partpacker_adaln.py \
    --data_root /mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_solidified \
    --jepa_root /mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_jepa_v2 \
    --manifest "$MANIFEST" \
    --output_dir "$CKPT_DIR" \
    --resume "${REPO}/checkpoints/diff_jepa_filtered_full/step_70000.pt" \
    --resume_dit_only \
    --batch_size 8 \
    --steps 110000 \
    --warmup_proj_steps 10000 \
    --lr 1e-4 \
    --cond_dim 1408 \
    --cond_tokens 9600 \
    --wandb_project infinipart \
    --wandb_run_name adaln_cond \
    2>&1 | tee "${REPO}/logs/train_adaln_cond.txt"
