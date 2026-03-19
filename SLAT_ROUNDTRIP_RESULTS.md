# SLat VAE Roundtrip Results

## Summary
TRELLIS 2 的 SLat VAE（Structured Latent VAE）在我们的 part mesh 上 roundtrip 重建质量优秀，远超 PartPacker VAE。隔板、按钮、曲面细节全部保留。

## Pipeline
```
GT OBJ → normalize to [-0.45, 0.45]
       → mesh_to_flexible_dual_grid(grid_size=128)  # O-Voxel 体素化
       → (coords, dual_vertices, intersected)       # 稀疏体素表示
       → relative_verts = dual_verts * grid_size - coords  # 相对偏移
       → SparseTensor → SLat Encoder → latent [N, 32]
       → SLat Decoder(resolution=grid_size) → mesh
```

## 定量结果

| 样本 | Grid | 体素数 | Latent | 输出面数 | 重建质量 |
|------|------|--------|--------|----------|----------|
| Cabinet part0 | 128 | 30,310 | 64 tokens | 62,898 | 优秀：5层隔板全保留 |
| Drawer part0 | 64 | 24,162 | 48 tokens | 50,608 | 优秀 |
| Faucet part0 | 128 | 21,068 | 54 tokens | 42,508 | 优秀：龙头细节保留 |
| PhysXNet 耳机 part0 | 128 | 21,492 | 52 tokens | 45,090 | 优秀：弧度和耳罩细节保留 |
| Toaster part0 | 64 | 21,061 | 64 tokens | 42,842 | 优秀：按钮、插槽保留 |

## 对比 PartPacker VAE

| 指标 | PartPacker VAE (64³ FlexiCubes) | SLat VAE (128-512 O-Voxel) |
|------|------|------|
| 分辨率 | 64³ = 262K voxels (固定) | 自适应 grid (21K-50K 活跃体素) |
| 输出面数 | ~6K-12K | ~42K-63K (5-8x more) |
| 隔板/细节 | 丢失（CC error 6.7/part） | 保留 |
| 分离组件 | 合并（CC 8.34→1.81） | 待测（roundtrip 形状好） |
| 限制 | 128/256 分辨率产生 NaN | grid_size 灵活可调 |
| Watertight | 输出 watertight | Part mesh 不 watertight（预期行为） |

## 可视化

对比图在 `output/slat_roundtrip/` 目录：
- `*_gt.png` — GT OBJ 原始 mesh
- `*_slat.png` — SLat VAE encode→decode 重建

## 环境配置

### 关键发现
系统 nvcc (CUDA 13.0) 编译的 C++ extension 和 PyTorch (CUDA 12.4) runtime **不兼容**：
- o-voxel 的 CUDA hashmap kernel：insert 后 lookup 全返回 0
- FlexGEMM 的 triton sparse conv kernel：illegal memory access

**解决方案**：用 conda 安装的 CUDA toolkit (12.9) 的 nvcc 重新编译所有 C++ extension。

### Conda 环境：`trellis2`
```bash
# 创建
conda create -n trellis2 python=3.10 -y
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# flash-attn (prebuild)
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4+cu124torch2.6-cp310-cp310-linux_x86_64.whl

# CUDA toolkit (for nvcc)
conda install -c nvidia cuda-toolkit=12.4

# 基础依赖
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers safetensors zstandard kornia timm

# C++ extensions (必须用 conda nvcc 编译)
CONDA_CUDA=/mnt/data/yurh/miniconda3/envs/trellis2
# 先 patch torch 的 CUDA 版本检查
sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE.../pass/' $CONDA_CUDA/lib/python3.10/site-packages/torch/utils/cpp_extension.py

CUDA_HOME=$CONDA_CUDA TORCH_CUDA_ARCH_LIST="9.0" pip install /tmp/extensions/FlexGEMM --no-build-isolation --no-deps
CUDA_HOME=$CONDA_CUDA TORCH_CUDA_ARCH_LIST="9.0" pip install /tmp/extensions/CuMesh --no-build-isolation --no-deps
cd /mnt/cpfs/yurh/TRELLIS.2/o-voxel && CUDA_HOME=$CONDA_CUDA TORCH_CUDA_ARCH_LIST="9.0" python setup.py build_ext --inplace

# 记得恢复 torch patch
```

### 权重路径
- TRELLIS.2-4B: `/mnt/data/yurh/TRELLIS.2-4B/`
- Shape Encoder: `ckpts/shape_enc_next_dc_f16c32_fp16.safetensors` (354M params)
- Shape Decoder: `ckpts/shape_dec_next_dc_f16c32_fp16.safetensors` (474M params)

### 运行 SLat roundtrip
```bash
cd /mnt/cpfs/yurh/TRELLIS.2
CUDA_VISIBLE_DEVICES=2 /mnt/data/yurh/miniconda3/envs/trellis2/bin/python \
  /mnt/cpfs/yurh/Infinigen-Sim/test_slat_roundtrip.py
```

## 已知问题
1. **spconv backend 不能替代 flex_gemm**：权重 key 名称不同（`.conv.weight` vs `.conv.conv.weight`），即使 remap 后 upsample 行为不同，`flexible_dual_grid_to_mesh` 面片全为 0
2. **FlexGEMM 在 H200 上有 triton kernel bug**（GitHub Issue #95），但用 conda nvcc 重编后解决
3. **Grid size 自适应**：大物体需要降到 64-128 才能保持 <50K 活跃体素，降低时注意保留小物件

## 下一步
1. 写 DualPartSLatModel finetune 训练脚本
2. 预处理 GT SLat（用 SLat Encoder encode 所有 part mesh）
3. 训练 DualPartSLatModel：VJEPA conditioning + part cross-attention
4. 评估：GT vs VAE-only vs VAE+SLat 三方对比
