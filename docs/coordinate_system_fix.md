# PartPacker Coordinate System Fix (2026-03-31)

## Problem

PartPacker flow model 的 latent 空间在 **GLB 坐标系** 下，不是 OBJ Y-up 坐标系。
app.py 中的 `TRIMESH_GLB_EXPORT` 变换 **不是** 为了 GLB 格式导出，而是 flow model latent 空间本身就在这个坐标系下。

```python
TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # xyz -> zxy
```

## Evidence

1. **Asymmetric box VAE roundtrip**: VAE 本身不做坐标变换（input xyz → output xyz 完全一致）
2. **Multi-view inference test**: 同一个物体不同视角的 flow inference，raw decode 朝向不一致；加上 `@ TRIMESH_GLB_EXPORT.T` 后所有视角都变正
3. **app.py**: decode 后统一做 `mesh.vertices = mesh.vertices @ TRIMESH_GLB_EXPORT.T`

## Root Cause

PartPacker 训练时，GT mesh 在 VAE encode **之前**做了 `@ TRIMESH_GLB_EXPORT`（OBJ Y-up → GLB 坐标系），使得 latent 空间在 GLB 坐标系下。
inference 时 flow model 输出的 latent 也在 GLB 坐标系，decode 后需要 `@ TRIMESH_GLB_EXPORT.T`（逆变换）回到 OBJ Y-up 才能正常显示。

## Impact on Our Pipeline

**之前所有 gt_latent 都在错误的坐标系下编码。** GT mesh 直接 box_normalize → VAE encode，没有先做 `@ TRIMESH_GLB_EXPORT`。
这导致：
- gt_latent 在 OBJ Y-up 坐标系
- flow model pretrained weights 期望 GLB 坐标系
- latent MSE 因坐标系不对齐而偏高（~2.0）
- finetune 时模型需要额外学习坐标系转换，浪费容量

## Fix

在 VAE encode GT mesh 之前，对 vertices 做：
```python
TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
vertices = vertices @ TRIMESH_GLB_EXPORT  # OBJ Y-up -> GLB coord for VAE
```

需要修改的文件：
- `encode_for_training.py`: VAE encode 前加变换
- 重新 encode 所有 gt_latent（IS + PhysXMobility）

Decode 后显示时做逆变换：
```python
vertices = vertices @ TRIMESH_GLB_EXPORT.T  # GLB coord -> OBJ Y-up for display
```

## Coordinate System Summary

| 环节 | 坐标系 | 变换 |
|---|---|---|
| 原始 OBJ mesh | Y-up | - |
| VAE encode 输入 | **GLB** (需要先 `@ TRIMESH_GLB_EXPORT`) | OBJ→GLB |
| VAE latent 空间 | GLB | - |
| Flow model latent | GLB | - |
| VAE decode 输出 | GLB | - |
| Blender 渲染/导出 OBJ | 需要 `@ TRIMESH_GLB_EXPORT.T` | GLB→OBJ |
| Blender obj_import | 自动 Y-up→Z-up | `(x,y,z)→(x,-z,y)` |
