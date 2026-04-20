# Training Status & Analysis — 2026-04-11

## Current Pipeline

```
Video (mp4) → VJEPA2 ViT-g → Feature Encoding → Projector → PartPacker Flow DiT → VAE Decode → Mesh
```

## Data

| Item | Value |
|---|---|
| Total animodes with gt_latent | 2,785 |
| After filtering (ratio [0.2,3.0], no box, hemi only) | 968 animodes, 3,024 train views |
| Part1/Part0 area ratio median | 0.141 (heavily imbalanced) |
| Categories | PhysXMobility (797), window (76), faucet (25), dishwasher (23), cabinet (14), drawer (11), plier (10), oven (8), lamp (2), trash (2) |

### Data filtering rationale
- **Ratio [0.2, 3.0]**: Exclude extremely imbalanced splits where part1 is tiny (model can cheat by predicting all-part0)
- **No box**: Box category has severe interpenetration artifacts in rendering
- **Hemi only**: Fixed viewpoints (16 hemisphere views), exclude orbit/sweep for consistency

## Experiment 1: Linear Projector + Diff JEPA

**Config:**
- Feature: [10240, 1408] — first 5 temporal tokens original, rest diff (t - t-5)
- Projector: `nn.Linear(1408, 1536)` — per-token independent linear mapping
- DiT: 1249.5M, from PartPacker pretrained
- Data: 1036 animodes × ~6.6 views = 6832 train views (all view types)
- Training: 4×H200, bs=8/GPU, lr=1e-4, 10k warmup proj + 60k full = 70k steps total
- Checkpoints: `checkpoints/diff_jepa_filtered_full/step_*.pt`

**Results at step 70k (323 epochs):**
- avg100 loss: ~0.47 (plateau since step ~25k)
- topo_correct: 12% → 25% → 38% (slowly improving)
- train_mse ≈ test_mse (no overfitting, but also no memorization)
- Visual quality: box-like objects (dishwasher, oven, cabinet) recognizable; thin structures (faucet, plier, window) poor

**CFG Scale Analysis (step 70k):**

| CFG | Effect |
|---|---|
| 0 | Random prior shapes, no condition influence |
| 1-4 | Gradual improvement, condition starting to steer |
| 5 | Moderate quality, condition visible |
| 7.9-11 | **Best quality**, sharpest details, closest to GT |

**Conclusion: Condition IS effective but very weak — needs CFG=11 to amplify**

CFG comparison grids: `output/cfg_grid_seed123/`, `output/cfg_grid_seed777/`

## Experiment 2: Self-Attn Projector + Orig+Diff JEPA (running)

**Config:**
- Feature: [9600, 1408] — orig tokens stride-2 subsampled (5120) + diff tokens stride-2 subsampled (4480)
- Projector: MLP(1408→1536, GELU, 1536→1536) + 2-layer TransformerEncoder (8 heads, d=1536)
- DiT: loaded from Exp1 step_70000, projector fresh
- Data: 968 animodes, hemi only, no box
- Training: 4×H200, bs=8/GPU, lr=1e-4, 10k warmup proj + 100k full = 110k steps
- Speed: ~4.2s/step
- wandb: https://wandb.ai/auroraryan0301/infinipart/runs/grdxdynh
- Checkpoints: `checkpoints/diff_jepa_v2_selfattn/`

**Changes from Exp1:**
1. Feature carries both shape (orig) and motion (diff) info, not diff-only
2. Self-attention projector allows token interaction (vs independent linear mapping)
3. Cleaner data (no box, hemi only, balanced ratio)

**Expected outcome:** Stronger condition → lower CFG needed. If still needs CFG>7 → condition injection mechanism is the bottleneck, not projector quality.

## Root Cause Analysis: Why Conditioning is Weak

### The DiT Architecture

```python
# DiTLayer._forward():
# AdaLN uses ONLY timestep — video condition NOT here
t_adaln = self.adaln_linear(F.silu(t_emb))  # t_emb from timestep only
shift, scale, gate = ...  # modulates self-attn and FFN

# Video condition enters ONLY through cross-attention
h = self.norm2(x)
x = x + self.attn2(h, c)  # c = projected video features
```

**Problem:** AdaLN (scale/shift/gate) is "hard" modulation — directly changes activations. Cross-attention is "soft" — model can learn to set attention weights near zero, effectively ignoring condition.

### Evidence
- CFG=0 vs CFG=11 shows condition IS present in the model, but DiT learns to mostly ignore it during training
- 5-animode overfit experiment succeeded → model CAN learn the mapping, but with 1000+ samples the condition signal gets diluted
- Loss plateaus quickly → model converges to a weakly-conditioned prior

### Papers on Conditioning Strength

| Method | Papers | Mechanism | Strength |
|---|---|---|---|
| AdaLN modulation | DiT, SD3, TRELLIS 2 | Condition → scale/shift every LayerNorm | Strongest |
| ControlNet | 3DTopia, LN3Diff | Duplicate encoder, add outputs to main DiT | Very strong |
| Multi-layer cross-attn | InstantMesh, Zero123++ | Inject at every layer | Strong |
| Decoupled cross-attn | IP-Adapter | Separate K/V projections | Medium |
| Projector enhancement | Self-attn projector (ours) | Richer token representation | Weak |

## ROOT CAUSE IDENTIFIED: V-JEPA 2.0 Dense Features Are Poor

**This supersedes the projector/AdaLN analysis.** The weak conditioning is primarily caused by V-JEPA 2.0's poor dense feature quality, not the projector design.

| Model | ADE20K mIoU ↑ | NYU Depth RMSE ↓ | ImageNet Acc |
|---|---|---|---|
| **V-JEPA 2.0 (current)** | **22.2** | **0.682** | 82.2 |
| V-JEPA 2.1 | **47.9** | **0.307** | 85.5 |
| DINOv2 ViT-g | ~49 | ~0.31 | 86.5 |

V-JEPA 2.0 paper confirms: context tokens (unmasked) have no training loss → encoder uses them as global aggregators, discarding local spatial information. V-JEPA 2.1 fixes this with Dense Predictive Loss (all tokens contribute to loss) + Deep Self-Supervision (intermediate layer losses).

**Action: Upgrade to V-JEPA 2.1 before further architecture experiments.**

Checkpoint downloaded: `/mnt/cpfs/yurh/vjepa2/checkpoints/vjepa2_1_vitg_384.pt`

## Proposed Fix: AdaLN Video Condition Injection (deferred)

See `claude_docs/ADALN_CONDITION_DESIGN.md` for full design and implementation. To be tested AFTER V-JEPA 2.1 upgrade.

**Core idea:** Pool video features → MLP → inject into AdaLN alongside timestep at every DiT layer.

```python
# Before (timestep only):
t_adaln = self.adaln_linear(F.silu(t_emb))

# After (timestep + video condition):
combined = t_emb + video_global_emb  # or concat + MLP
t_adaln = self.adaln_linear(F.silu(combined))
```

This makes video condition **impossible to ignore** — it directly modulates every layer's activations.

## Key Scripts

| Script | Purpose | Env |
|---|---|---|
| `encode_diff_jepa_v2.py` | Encode orig+diff JEPA features | partpacker_wan |
| `infer_quick.py` | Quick inference + Blender render comparison | partpacker_wan |
| `make_cfg_grid_v2.py` | CFG comparison grid with labels | partpacker_wan |
| `analyze_mesh_area_ratio.py` | Part0/Part1 surface area analysis | partpacker_wan |
| `launch_v2_train.sh` | Full pipeline: encode + train | partpacker_wan |

## Key Paths

| Item | Path |
|---|---|
| Data (moved) | `/mnt/data/yurh/Infinigen-Sim/data_ssd/` |
| GT latents | `data_ssd/encoded_solidified/` |
| JEPA v2 features | `data_ssd/encoded_jepa_v2/` |
| Precompute (meshes) | `data_ssd/precompute_solidified/` |
| Precompute (videos) | `data_ssd/precompute/` |
| Exp1 checkpoints | `checkpoints/diff_jepa_filtered_full/` |
| Exp2 checkpoints | `checkpoints/diff_jepa_v2_selfattn/` |
| Inference output | `output/infer_quick/`, `output/cfg_grid_seed*/` |
| Area ratio CSV | `data_ssd/mesh_area_ratio_dist.csv` |
