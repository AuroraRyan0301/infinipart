# Prompt for Claude Code on New 4-GPU Machine

Copy the following prompt to start a new Claude Code session on the new machine. It contains all context needed to understand the project and run the AdaLN experiment.

---

## Prompt (copy below)

```
I need you to run an experiment on this machine. Read the project docs first, then execute.

## Project Context

This is the Infinigen-Sim project at `/mnt/cpfs/yurh/Infinigen-Sim/`. The goal is to train a network that predicts dual-volume (part0 + part1) mesh of articulated objects from video.

**Pipeline**: Video → VJEPA2 encode → Feature → PartPacker Flow DiT → VAE decode → dual-volume mesh

Read these docs for full context:
1. `CLAUDE.md` — project overview, architecture, data pipeline
2. `claude_docs/TRAINING_STATUS_20260411.md` — current training status, experiments, key findings
3. `claude_docs/ADALN_CONDITION_DESIGN.md` — the AdaLN design we're about to run

## Key Finding from Previous Experiments

The main bottleneck is **weak conditioning**. The PartPacker DiT uses cross-attention to inject video features, but the model learns to mostly ignore it — requiring CFG=11 (very high) to get decent results. CFG comparison grids are in `output/cfg_grid_seed123/` and `output/cfg_grid_seed777/`.

Root cause: DiT's AdaLN only uses timestep, NOT video condition. Cross-attention is "soft" — the model can ignore it. AdaLN is "hard" — directly modulates every layer's activations.

## What to Run: AdaLN Video Condition Injection

The script `train_partpacker_adaln.py` adds pooled video features to the AdaLN pathway:
- `cond_pool_mlp`: pools projected video features → [B, 1536] → adds to timestep embedding
- Every DiT layer's scale/shift/gate now depends on video condition
- Zero-init output so pretrained DiT behavior is preserved initially
- Warmup phase: AdaLN path DISABLED, only cross-attention trains
- Phase 2: AdaLN path ENABLED, all parameters train

## Launch Command

The launch script handles everything (encode features if needed + train):

```bash
cd /mnt/cpfs/yurh/Infinigen-Sim
bash launch_adaln_train.sh
```

Run it in tmux so it persists. Use `tmux new-session -s adaln_train`.

## Before Running, Check:

1. `nvidia-smi` — need 4 GPUs free
2. `conda activate partpacker_wan` — verify env exists
3. Verify data paths exist:
   - `/mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_solidified/` (gt_latent)
   - `/mnt/data/yurh/Infinigen-Sim/data_ssd/precompute/` (videos for encoding)
   - `/mnt/cpfs/yurh/Infinigen-Sim/checkpoints/diff_jepa_filtered_full/step_70000.pt` (DiT checkpoint)
   - `/mnt/cpfs/yurh/PartPacker/` (PartPacker repo with flow/modules/dit.py)
   - `/mnt/cpfs/yurh/vjepa2/` (VJEPA2 model)
4. If data paths differ on this machine, update the paths in `launch_adaln_train.sh`
5. Verify `NETRC` path in `launch_adaln_train.sh` points to the correct wandb credentials

## Expected Behavior

- Stage 1 (encoding): ~6 min, 4 GPU parallel, outputs to `data_ssd/encoded_jepa_v2/`
- Stage 2 (training): ~4-5 s/step, 110k steps total
  - Phase 1 (0-10k): warmup proj via cross-attention only, AdaLN cond disabled
  - Phase 2 (10k+): full training, AdaLN cond enabled
- wandb: project `infinipart`, run name `adaln_cond`
- Checkpoints: `checkpoints/adaln_cond/step_*.pt` every 5k steps

## How to Monitor

```bash
# Training progress
grep "avg100" logs/train_adaln_cond.txt | tail -10

# GPU utilization
nvidia-smi

# wandb link
grep "wandb.*View run" logs/train_adaln_cond.txt
```

## How to Evaluate (after some training)

```bash
# Quick inference with latest checkpoint
CUDA_VISIBLE_DEVICES=0 conda run -n partpacker_wan python infer_quick.py \
    --ckpt checkpoints/adaln_cond/latest.pt \
    --output_dir output/infer_adaln \
    --n_samples 8 --cfg_scale 5.0
```

Compare with previous experiment's CFG=11 results to see if lower CFG now works.

## Key Question to Answer

Does AdaLN injection reduce the optimal CFG? If CFG=3-5 now produces quality similar to Exp1's CFG=11, the design is working. Run CFG comparison at step 20k-30k to check early.
```

---

**Note**: The prompt above is self-contained. The new Claude Code instance should read the referenced docs, verify paths, and execute. All scripts are already written and committed to the repo.
