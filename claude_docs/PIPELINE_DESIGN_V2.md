# Infinipart Pipeline Design v2 — with SLat Refiner

## End-to-End Goal
```
Input:  Video of articulated object in motion
Output: Dual volume mesh (part0.obj + part1.obj) — topology-preserving 3D reconstruction
```

## Pipeline Overview

### Key Components (disambiguation)

| Abbreviation | Full Name | What It Does | Resolution |
|---|---|---|---|
| **PP VAE** | PartPacker VAE (340M) | point cloud → latent [4096, 64] per part → occupancy field → FlexiCubes mesh | **64³** fixed |
| **SLat Encoder** | TRELLIS 2 Shape VAE Encoder (354M) | mesh → O-Voxel → sparse latent [N, 32] | **512** (LR) or **1024** (HR) |
| **SLat Decoder** | TRELLIS 2 Shape VAE Decoder (474M) | sparse latent [N, 32] → FlexiDualGrid → mesh | **512** (LR) or **1024** (HR) |
| **SLat Flow (ours)** | DualPartSLatModel (1.6B total, 311M trainable) | VJEPA cond + coords + noise → flow matching → SLat latent [N, 32] | **512** (current) |

### Training Pipeline

```
1. DATA GENERATION
   Spawn/Setup → Precompute (topo split) → Render (Blender Cycles)
   Output: part0/1.obj + metadata.json + mp4 videos

2. ENCODING (3 separate encoders, all frozen)
   ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────────┐
   │ PP VAE Encoder   │   │ V-JEPA2 ViT-g   │   │ SLat Encoder (512)   │
   │ (partpacker_wan) │   │ (partpacker_wan) │   │ (trellis2 env)       │
   └─────────────────┘   └─────────────────┘   └──────────────────────┘
          │                      │                        │
   gt_latent.pt            *_jepa.pt              slat_gt.pt
   [1, 8192, 64]           [10240, 1408]          p0_lr: {coords [N,4], feats [N,32]}  ← res=512, coords [0,31]
   (PP VAE latent,          (video features,       p1_lr: {coords [M,4], feats [M,32]}
    用于旧DiT训练+          condition for           p0_hr/p1_hr: coords [0,63]          ← res=1024, for future cascade
    生成VAE occ coords)     SLat Flow model)

3. TRAINING (DualPartSLatModel, trellis2 env)
   Input:    GT SLat coords [N,4] (from SLat Encoder 512) + Gaussian noise
   Cond:     VJEPA2 features [10240, 1408] → VJEPAProjector → [10240, 1024]
   Target:   GT SLat features [N,32] (from SLat Encoder 512)
   Loss:     Flow matching velocity MSE
   Backbone: SLatFlowModel 512 (1.3B, frozen, TRELLIS 2 pretrained)
   Trainable: VJEPAProjector + PartCrossAttention × 30 (progressive: block 29→15)
```

### Inference Pipeline

```
   Video ──▶ V-JEPA2 ViT-g ──▶ [10240, 1408]
                                      │
                                      ▼
                          VJEPAProjector (trainable)
                          1408 → 1024
                                      │
                               [10240, 1024]
                                      │
   Coords source ─────────────────────┤
   (OPEN QUESTION, see below)         │
                                      ▼
              ┌────────────────────────────────────────┐
              │  DualPartSLatModel                      │
              │  Backbone: SLatFlowModel 512 (frozen)   │
              │                                          │
              │  30 frozen blocks, part cross-attn       │
              │  on blocks 15-29 (last 15 layers)        │
              │                                          │
              │  Flow matching: 50 Euler steps           │
              │  noise(t=1) → SLat latent(t=0)           │
              └────────────────────────────────────────┘
                         │              │
                   part0 SLat      part1 SLat
                   [N, 32]         [M, 32]
                         │              │
                         ▼              ▼
              ┌───────────────┐ ┌───────────────┐
              │ SLat Decoder   │ │ SLat Decoder   │
              │ (TRELLIS 2,    │ │ (TRELLIS 2,    │
              │  frozen, 512)  │ │  frozen, 512)  │
              └───────────────┘ └───────────────┘
                         │              │
                    part0.obj      part1.obj
```
```

## Coords Source: PP VAE (Decided)

推理时 SLat Flow Model 需要知道"在哪些体素位置预测特征"。这些 coords 由 **PP VAE (PartPacker VAE)** 提供。

### 完整推理路径

```
Video → VJEPA2 → [10240, 1408]                     ← video conditioning
Video → DiT (旧 PartPacker Flow DiT) → PP VAE latent [1, 8192, 64]  ← 3D shape prediction
                                            │
                                  ┌─────────┴─────────┐
                                  │ split              │
                          part0 [1,4096,64]    part1 [1,4096,64]
                                  │                    │
                                  ▼                    ▼
                          PP VAE Decode        PP VAE Decode
                          query 32³ grid       query 32³ grid
                                  │                    │
                          occ > 0 → coords     occ > 0 → coords
                          [N, 4] range[0,31]   [M, 4] range[0,31]
                                  │                    │
                                  └────────┬───────────┘
                                           │
                                           ▼
                               DualPartSLatModel (512)
                               + VJEPA2 conditioning
                               flow matching 50 steps
                                           │
                                    ┌──────┴──────┐
                              part0 SLat    part1 SLat
                              [N, 32]       [M, 32]
                                    │             │
                              SLat Decoder  SLat Decoder
                              (512)         (512)
                                    │             │
                              part0.obj     part1.obj
```

### PP VAE 当前问题
- 43% 样本的 VAE occ coords 有一个 part 为空（decode 后 occupancy 全负）
- cabinet 全军覆没（part1 隔板/小零件完全丢失）
- 原因：PP VAE 在 64³ FlexiCubes 上训练，对多分离组件的 dual volume 编码质量差

### PP VAE 改善方向（下一步，需要用户引导）
PP VAE 改善是打通端到端推理的关键。目前 VAE 56% 的 survive rate 远远不够。改善后重新编码 → 重新生成 VAE coords → 完整推理链路可用。

### 训练时为什么用 GT coords？
训练阶段 SLat Flow Model 学的是"给定位置，预测特征"。用 GT coords 训练让模型专注学习特征预测，不受 coords 噪声干扰。推理时换用 PP VAE coords 有 distribution shift，但 coords 只是指定位置，模型应该能 generalize。

## Model Architecture

### DualPartSLatModel
- **Base**: TRELLIS 2 SLatFlowModel 1.3B (30 transformer blocks)
- **Frozen**: self-attention + cross-attention + MLP in all 30 blocks
- **Trainable** (311M total, ~160M effective):
  - `VJEPAProjector`: Linear(1408→1024) + 2-layer TransformerEncoder → projects VJEPA features to SLat cond space
  - `PartCrossAttention` × 30: cross-attention between part0 and part1 sparse tokens
- **Progressive training**: only last N layers' PartCrossAttention active
  - Start: block 29 only (1 layer)
  - Expand: +1 layer every 500 steps
  - End: blocks 15-29 (15 layers)
  - Why: inserting random cross-attn between frozen blocks disrupts pretrained features. Progressive lets later layers stabilize first.

### Key Finding: Gradient Locality
With all 30 layers active from the start, loss plateaus at ~1.8 (overfit setting).
With progressive 29→15, loss reaches 0.03 on same data. The frozen blocks attenuate gradients to early cross-attn layers.

## Training Config

| Param | Value |
|-------|-------|
| Optimizer | AdamW, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95) |
| Warmup | 500 steps |
| Loss | Flow matching MSE (velocity prediction) |
| Batch size | 1 per GPU (sparse tensors, variable token count) |
| Precision | bf16 autocast |
| Grad clip | 1.0 |
| Progressive | start=29, min=15, expand_every=500 |

## Data Flow

```
/mnt/data_ssd/infinigen-sim-data/
├── precompute/          20 categories, ~40K animodes (part0/1.obj + mp4)
│   ├── {IS_factory}/    18 types × 100 seeds
│   ├── PhysXMobility/   1888 seeds, 5877 with mp4
│   └── PhysXNet_PhysXnet/ 7615 seeds
│
├── encoded/             20 categories, 2318 animodes
│   └── {cat}/{seed}_{animode}/
│       ├── gt_latent.pt          [1, 8192, 64] PartPacker VAE latent
│       └── views/*_nobg_jepa.pt  [10240, 1408] VJEPA2 features
│
├── slat_gt/             20 categories, 2318 animodes
│   └── {cat}/{seed}_{animode}.pt
│       ├── p0_lr: {feats: [N,32], coords: [N,4]}  512-resolution
│       ├── p1_lr: {feats: [M,32], coords: [M,4]}
│       ├── p0_hr: ...                               1024-resolution (for future cascade)
│       └── p1_hr: ...
│
├── vae_coords/          VAE occupancy grid coords (56% survive rate)
│   └── {cat}/{seed}_{animode}.pt
│       ├── p0_coords: [N, 4]  range [0, 31]
│       └── p1_coords: [M, 4]
│
└── checkpoints/
    ├── slat_progressive/       overfit 8K steps (loss=0.03)
    ├── slat_progressive_full/  full 20K steps (loss=0.15-0.30)
    └── slat_overfit_long/      extended overfit 8K→38K steps
```

## Current Bottlenecks (Priority Order)

1. **PP VAE 质量改善** — 端到端推理的核心阻塞点。43% 样本 occ coords 为空，cabinet 全灭。改善后才能有完整推理链路。
2. **PhysXMobility encoding** — 5,877 rendered animodes, only 6 encoded → 正在 4 卡编码中
3. **IS precompute coverage** — 10K precomputed but only 1.1K encoded → more rendering + encoding
4. **Mesh surface quality** — flow matching noise → small holes in decoded mesh → more training steps
5. **PhysXNet rendering** — 23K precomputed, only 1.2K rendered → need more GPU time

## Two-Stage Architecture

整体是两级：PP VAE 负责粗糙的 3D shape + coords，SLat Flow 负责高质量 refine。

| Stage | Model | Input | Output | Resolution | Role |
|-------|-------|-------|--------|------------|------|
| **Stage 1** | PartPacker Flow DiT (1.25B) + PP VAE (340M) | VJEPA2 video | PP VAE latent → occ coords [0,31] | 32³ (64³ FlexiCubes) | 粗糙形状 + 体素位置 |
| **Stage 2** | DualPartSLatModel (1.6B) + SLat Decoder (474M) | coords + VJEPA2 video | SLat latent → high-res mesh | 512 (FlexiDualGrid) | 精细化几何 + 拓扑保持 |

Stage 1 给出"在哪里"（coords），Stage 2 给出"长什么样"（features → mesh）。

## vs Previous Pipeline (PP VAE only)

| Aspect | PP VAE Only (旧) | PP VAE + SLat Refiner (新) |
|--------|------------------|---------------------------|
| Decoder | FlexiCubes 64³ (fixed) | FlexiDualGrid 512³ (adaptive) |
| Topology preservation | 12.8% exact CC match | Much better (TBD quantitative) |
| Output mesh quality | ~6-12K faces, smooth but merged | ~100K-2M faces, detail preserved |
| Training target | PP VAE latent [8192, 64] | SLat latent [N, 32] |
| Inference coords | PP VAE self-contained (dense grid) | PP VAE provides coords → SLat refines |
| Conditioning | VJEPA2 video | Same |
| PP VAE 角色 | 最终输出 | 中间步骤（提供 coords） |
