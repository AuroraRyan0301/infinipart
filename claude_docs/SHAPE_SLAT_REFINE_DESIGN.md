# Shape SLat Refine Module Design

## Goal
Use TRELLIS 2's Shape SLat Flow Model as a post-processing refine step after PartPacker VAE decoding.
VAE outputs 64³ mesh (with CC topology loss) → Shape SLat refines to 512³ (with topology preservation).

## Architecture

### Original TRELLIS 2 Shape SLat Block (30 blocks)
```
x → norm1 → self_attn(sparse) → +res → norm2 → cross_attn(DinoV3_image) → +res → norm3 → MLP → +res
```

### Our Modified Block (DualPartSLatBlock)
```
part0_x → norm1 → self_attn → +res → norm_vjepa → cross_attn(VJEPA) → +res
        → norm_part → part_cross_attn(part1_x) → +res → norm3 → MLP → +res

part1_x → norm1 → self_attn → +res → norm_vjepa → cross_attn(VJEPA) → +res
        → norm_part → part_cross_attn(part0_x) → +res → norm3 → MLP → +res
```

Each block has 4 operations (vs original 3):
1. **Self-attention**: sparse tokens attend to themselves (within one part) — **frozen from pretrained**
2. **VJEPA cross-attention**: sparse tokens attend to video features — **new projection layer (1408→1024), trainable**
3. **Part cross-attention**: part0 tokens attend to part1, part1 attend to part0 — **new, trainable**
4. **MLP**: feed-forward — **frozen from pretrained**

### Conditioning
- **Original**: DinoV3 image features [N_patches, 1024]
- **Ours**: VJEPA2 video features [10240, 1408] → projection [10240, 1024] → cross-attention
- Projection layer: `nn.Linear(1408, 1024)` — trainable

### Part Cross-Attention
```python
class PartCrossAttention(nn.Module):
    def __init__(self, channels, num_heads):
        self.norm = LayerNorm32(channels)
        self.cross_attn = SparseMultiHeadAttention(
            channels, ctx_channels=channels,
            num_heads=num_heads, type="cross", attn_mode="full"
        )

    def forward(self, x: SparseTensor, other_part: SparseTensor):
        h = x.replace(self.norm(x.feats))
        h = self.cross_attn(h, other_part.feats)  # attend to other part's tokens
        return x + h
```

## Pipeline
```
Input: part0.obj (64³), part1.obj (64³), video.mp4

1. Extract VJEPA features from video → [10240, 1408]
2. Project VJEPA → [10240, 1024] via learned projection
3. Convert part0/part1 meshes to 64³ occupancy → sparse coords
4. Initialize noise SparseTensor for part0 and part1
5. Flow matching sampling (50 steps):
   - For each timestep:
     - part0_h, part1_h = DualPartSLatModel(part0_noise, part1_noise, t, vjepa_cond)
     - Update part0_noise, part1_noise
6. Decode part0_slat → 512³ mesh via SLat Decoder
7. Decode part1_slat → 512³ mesh via SLat Decoder
```

## Training
- **Frozen**: Self-attention (30 blocks), MLP (30 blocks), SLat Decoder
- **Trainable**: VJEPA projection, Part cross-attention (30 blocks), cross-attn KV projection
- **Data**: 4636 part pairs with video, encode GT high-res OBJ → SLat latent as target
- **Loss**: Flow matching loss on SLat latent space

## Model Sizes
- Shape SLat Flow Model: ~1.3B params (frozen backbone)
- New trainable params: ~30 * (part_cross_attn ~4M + vjepa_proj ~1.4M) ≈ ~160M
- SLat Decoder: frozen
- Total trainable: ~160M

## Key Questions
- Can SLat Decoder handle our dual-volume parts? (trained on single objects)
- Will 4636 training pairs be enough for 160M new params?
- Memory: 2 × 1.3B (part0 + part1) + gradients for 160M → need ~40-60GB, fits H200

## Dependencies
- TRELLIS 2 weights: `microsoft/TRELLIS.2-4B` (MIT license)
- VJEPA2 encoder (already have)
- trellis conda env

---

# 中文版：Shape SLat Refine 模块设计

## 目标
用 TRELLIS 2 的 Shape SLat Flow Model 作为 PartPacker VAE 解码后的精细化步骤。
VAE 输出 64³ mesh（拓扑 CC 丢失严重）→ Shape SLat 精细化到 512³（保留拓扑结构）。

## 背景问题
- PartPacker VAE 用 FlexiCubes 在 64³ 网格上提取 mesh，分辨率是硬限制（128/256 会产生 NaN）
- 64³ = 262K 个体素，无法表达复杂物体的多个分离组件（CC 丢失：OBJ 平均 8.34 个组件 → VAE 只剩 1.81 个）
- TRELLIS 2 的 Shape SLat 能输出 512³ 甚至 1024³ 的高质量 mesh

## 架构设计

### 原版 TRELLIS 2 Shape SLat Block（共 30 个 block）
每个 block 3 步：
```
x → 自注意力(sparse tokens 内部) → 交叉注意力(图片 DinoV3 特征) → MLP
```

### 我们的修改版 Block（DualPartSLatBlock）
每个 block 4 步：
```
part0 → 自注意力(part0 内部) → 交叉注意力(VJEPA 视频特征) → 零件交叉注意力(attend to part1) → MLP
part1 → 自注意力(part1 内部) → 交叉注意力(VJEPA 视频特征) → 零件交叉注意力(attend to part0) → MLP
```

### 为什么需要零件交叉注意力？
part0 和 part1 组成一个完整物体。如果单独精细化 part0，它不知道 part1 长什么样，可能：
- 两个 part 重叠（占据同一空间）
- 拼接处不吻合（缝隙或错位）
- 整体形状不协调

零件交叉注意力让 part0 的每个体素"看到" part1 的几何信息，反之亦然，确保两者互补。

### 哪些参数训练，哪些冻结？

| 组件 | 参数量 | 状态 | 原因 |
|------|--------|------|------|
| 自注意力（30 blocks） | ~280M | **冻结** | 保留 TRELLIS 2 学到的 3D 几何先验 |
| MLP（30 blocks） | ~750M | **冻结** | 同上 |
| DinoV3 交叉注意力 | ~240M | **冻结主体，训 KV projection** | 替换为 VJEPA 特征 |
| VJEPA 投影层 | ~1.4M | **训练** | 1408→1024 维度对齐 |
| 零件交叉注意力（30 blocks） | ~120M | **训练** | 全新模块 |
| SLat Decoder | ~200M | **冻结** | 保留解码能力 |
| **总训练参数** | **~160M** | | 总模型 ~1.5B 中只训 ~10% |

### Conditioning（条件信息）

| 原版 | 我们 |
|------|------|
| 一张图片 → DinoV3 → [N_patches, 1024] | 视频 → VJEPA2 → [10240, 1408] → 投影 → [10240, 1024] |
| 单张静态图，没有运动信息 | 81 帧视频，包含铰接运动信息 |

用 VJEPA 而不是 DinoV3 的原因：视频里的**运动信息**决定了 part0/part1 的划分方式（哪些零件一起动 = 同一个 part）。静态图片无法提供这个信息。

## 推理流程

```
输入：part0.obj (64³), part1.obj (64³), 视频

步骤：
1. 视频 → VJEPA2 → [10240, 1408] → 投影层 → [10240, 1024]
2. part0/part1 mesh → 64³ occupancy → 提取活跃体素坐标 (coords)
3. 在 coords 上初始化随机噪声 SparseTensor（每个体素 32 维）
4. Flow matching 采样（50 步）：
   每一步：
   - part0_tokens 和 part1_tokens 同时过 DualPartSLatModel
   - 两者通过零件交叉注意力交换信息
   - 更新噪声
5. part0 SLat → SLat Decoder → 512³ mesh
6. part1 SLat → SLat Decoder → 512³ mesh
```

## 训练数据
- 4636 组 part0/part1 OBJ 对 + 对应视频
- GT：高分辨率 OBJ → encode 成 SLat latent 作为监督信号
- Loss：Flow matching loss（预测速度场 vs GT 速度场）

## 资源需求
- 显存：~40-60GB（2 个 part 的 sparse tokens + 160M 可训参数的梯度）→ 单卡 H200 (143GB) 够用
- 训练数据：4636 对，160M 参数 → 约 30 个 sample/param，偏少但可以先试
- 预计训练时间：取决于 sparse token 数量和 flow matching 步数

## 待确认问题
1. SLat Decoder（冻结的）能否处理我们的 dual-volume parts？它是在单个完整物体上训的
2. 64³ occupancy 转 sparse coords 后，token 数量够不够？（原版 TRELLIS 2 最大 8192 tokens）
3. 4636 训练对够不够训 160M 新参数？
4. Flow matching 的训练目标：是直接用 GT OBJ encode 的 SLat，还是用 VAE latent decode 的 mesh 再 encode？
