# Infinigen-Sim (infinipart) Project

## Core Goal
Train a network to predict **dual-volume topological partition** of articulated objects from video only.

**ALWAYS REMEMBER: This is a TOPOLOGY problem. Every operation (splits, bridging, merging, joint classification) is on the URDF link/joint graph. Meshes are just the geometric realization — the logic lives in the topology.**

- **Input**: Video encoded by V-JEPA2
- **Output**: Dual volume (part0 + part1) — **alternating 2-coloring** on the reduced URDF graph
- **How it works**:
  1. Classify joints as active/passive/fixed for the given animode
  2. Merge all parts connected by **fixed joints** into single topological nodes (node contraction)
  3. The reduced graph: **nodes = merged part groups**, **edges = movable joints** (active + passive)
  4. Apply **bipartite 2-coloring**: adjacent nodes (connected by movable joints) get opposite colors
  5. part0 = one color, part1 = the other color
- This is NOT "moving vs static", NOT "root vs child". It is **alternating/interleaved coloring** on the joint graph.

### Example: Lamp (chain: base → arm_joint → arm → head_joint → head)

**basic_0** (active: arm_joint):
- head_joint = fixed → merge arm+head into node B
- Graph: `base —[arm_joint]— B(arm+head)`
- 2-coloring: base=part0, B=part1

**basic_1** (active: head_joint):
- arm_joint = fixed → merge base+arm into node A
- Graph: `A(base+arm) —[head_joint]— head`
- 2-coloring: A=part0, head=part1

**senior_0** (active: arm_joint + head_joint, same chain allowed):
- No fixed joints → no merging
- Graph: `base —[arm_joint]— arm —[head_joint]— head`
- 2-coloring: **base=part0, arm=part1, head=part0**
- Note: base and head get SAME color! Both "ends" of the chain are part0, middle is part1. This is alternating coloring, NOT moving/static.

### Example: Dishwasher (3 siblings: door, upper_rack, lower_rack)

**basic_1** (active: upper_rack_joint):
- BVH: rack slides out, hits door → door_joint = passive; lower_rack not hit → fixed
- Merge fixed: base+lower_rack → node A
- Graph: `upper_rack —[rack_joint]— A(base+lower) —[door_joint]— door`
- 2-coloring: **upper_rack=part0, A=part1, door=part0**
- Note: upper_rack and door get SAME color (part0), even though rack is active and door is passive!

## Data Pipeline
1. **Asset Generation**: Infinigen-Sim procedurally generates articulated objects (URDF + per-part OBJs)
   - IS factories: DishwasherFactory, LampFactory, CabinetFactory, etc. (18 factories in `infinigen/assets/sim_objects/`)
   - PhysX factories: AppliancePhysXMobilityFactory, etc. (in `outputs/`)
2. **Precompute**: For each object x each animode: classify joints → merge fixed-joint parts → 2-color the reduced graph → export part0.obj + part1.obj
   - **basic_animode**: one joint per animode (one per joint, can be many)
   - **senior_animode**: combinations of multiple joints (same-chain allowed), **randomly sampled, max 10**
   - Bridging parts handling per URDF topology rules
   - Root node defaults to **part0**
   - NO "default/all joints active" mode
   - Each split MUST output a **verify PNG** visualization alongside the OBJs
   - Output: `precompute/{Factory}/{id}/{basic_N or senior_N}/part0.obj + part1.obj + verify.png` + `metadata.json`
3. **Render**: Blender renders articulation videos for each animode
4. **Encode**: Videos -> V-JEPA2 features (network input); dual volume OBJs -> PartPacker VAE -> `[B, 8192, 64]` latent (GT)
5. **Train**: Network learns: V-JEPA2 video features -> dual volume latent

## Architecture
- PartPacker VAE: part0 + part1 each -> [B, 4096, 64] -> concat -> [B, 8192, 64]
- PartPacker Flow DiT: 1249.5M params, 1536 hidden_dim, 24 layers, 16 heads

## Key Principles
- **Dual volume is per-animode**: same object, different active joints = different part0/part1 split
- **Same-chain combinations NOW ALLOWED**: senior animodes CAN combine joints on the same kinematic chain
- **URDF correctness is CRITICAL**: the entire pipeline depends on accurate URDF joint/link topology. Wrong URDF = wrong splits, wrong bridging detection, wrong everything

### Bridging Parts Handling (based on URDF topology, NOT size)
A "bridging part" is a part connected to multiple other parts. Check its connections:
- If ANY connection crosses a **movable joint** (active/passive) AND that connection is **NOT via a joint** (rigid/fixed link) → **REMOVE** the bridging part entirely (from both part0 and part1)
- If ALL connections do **NOT cross any movable joint** → **MERGE** the bridging part into its neighbor (node contraction)
This is a topological operation based on URDF link/joint graph, NOT based on mesh size or face count.

## Joint Classification Per Animode (Active / Passive / Fixed)
Each animode defines ONE active joint (basic) or a set of active joints (senior). The remaining joints must be classified:

1. **Active joints**: defined by the animode. Drive the motion (trajectory type is per-animode: sinusoidal oscillation, one-way sinusoidal, linear, or linear oscillation).
2. **Passive joints**: joints whose connected parts are HIT by the active motion trajectory (detected via BVH collision). These joints open passively — precompute their "pre-opening" so animation looks natural.
3. **Fixed joints**: joints whose connected parts are NOT hit by the active motion. Stay locked.

**BVH collision classification is per-animode per-trajectory**: each animode has its own trajectory type, and the passive/fixed classification is specific to THAT animode's trajectory. Different animodes (even with the same active joints) can have different splits if they use different trajectory types.

### Precompute Phase
- **Normalize FIRST, then split**: normalize the entire object to unit cube, THEN do topology splits per animode
- For each animode, compute the initial active joint trajectory (e.g., sinusoidal)
- Run BVH collision detection along the trajectory
- If active part collides with a joint-connected part → that joint = **passive**, precompute its pre-opening angle/offset
- If no collision → that joint = **fixed**
- Merge parts connected by fixed joints into single topological nodes
- Apply bipartite 2-coloring on the reduced graph (edges = movable joints) → part0 and part1

### Render Simulation Phase
- Use BVH collision detection per-frame to prevent interpenetration
- Collision with **passive joint** part → update that joint per-frame (it yields/opens)
- Collision with **fixed** part → STOP all motion (cannot penetrate fixed geometry)

### Static Filtering
- 32 views = 16 fixed hemisphere + 8 orbit + 8 sweep
- If ALL 32 views show NO motion → **skip this animode entirely** (don't export, don't render)

### Metadata to Save Per Animode
- Trajectory type (sinusoidal oscillation / one-way sinusoidal / linear / linear oscillation / custom)
- Envmap path + filename used for rendering
- Joint classification (active / passive / fixed per joint)
- **Joint 12-dim vector** per active joint: `[0:3]` axis_origin (normalized), `[3:6]` axis_direction (unit), `[6:9]` type one-hot [revolute, prismatic, continuous], `[9:11]` motion range [min, max], `[11]` exists flag
- 2-coloring assignment (which merged part groups → part0 vs part1)
- Passive joint pre-opening angles

## Data Sources
- **IS factories** (from this repo): 18 procedural factories in `infinigen/assets/sim_objects/mapping.py`
- **PhysX factories**: pre-generated in `outputs/` directory
- **PartNet**: NOT used as mesh asset source, but PhysX may reference PartNet materials/mappings — that's fine
- **NO MORE**: IM (Infinite-Mobility) repo and Sapien mesh assets have been removed

## Reference Code (read-only, has bugs but useful)
- Old version: `/mnt/data/yurh/infinipart/` — has split_precompute.py, render_articulation.py, etc. **WARNING**: many URDF splits and part connections are WRONG, but some animation logic is correct
- PartPacker: `/mnt/data/yurh/PartPacker/` — part split processing reference

## Environment
- Conda env: `infinigen-sim` (has infinigen installed from this repo)
- Blender: `/mnt/data/yurh/blender-3.6.0-linux-x64/blender`
- GPU: 4x L20X (143GB each)

---

# 中文版（强化记忆）

## 核心目标
训练一个网络，**只从视频**预测铰接物体的**双色拓扑划分（dual volume）**。

**时刻牢记：这是拓扑问题。所有操作（划分、桥接、合并、关节分类）都在 URDF link/joint 图上进行。Mesh 只是几何实现，逻辑在拓扑里。**

- 输入：视频 → V-JEPA2 编码的特征
- 输出：dual volume = part0 + part1 的**交替二着色**
- **划分方法**：
  1. 对当前 animode 分类关节：active / passive / fixed
  2. 把所有 **fixed 关节**相连的 parts 合并成一个拓扑节点（节点收缩）
  3. 化简后的图：**节点 = 合并后的零件组**，**边 = 可动关节**（active + passive）
  4. 对这个图做**二部图交替着色**：可动关节两端的节点颜色不同
  5. part0 = 一种颜色，part1 = 另一种颜色
- **不是"运动 vs 静止"，不是"根侧 vs 子侧"**。是关节图上的**间隔着色**。

### 例：灯（链式：base → arm_joint → arm → head_joint → head）
- **senior_0**（active: arm_joint + head_joint）：无 fixed → 不合并
- 图：`base —[arm]— arm —[head]— head`
- 着色：**base=part0, arm=part1, head=part0** ← base 和 head 同色！

### 例：洗碗机（3个兄弟关节：door, upper_rack, lower_rack）
- **basic_1**（active: rack_joint）：rack 碰 door → door=passive, lower_rack=fixed
- 合并 fixed：base+lower_rack → A
- 图：`rack —[rack_j]— A —[door_j]— door`
- 着色：**rack=part0, A=part1, door=part0** ← rack 和 door 同色，尽管一个主动一个被动！

## 数据管线
1. **资产生成**：用 Infinigen-Sim 程序化生成带铰链的物体（URDF + 逐零件 OBJ）
   - IS 工厂：repo 自带 18 个（`infinigen/assets/sim_objects/`）
   - PhysX 工厂：已预生成在 `outputs/` 目录
2. **Precompute**：每个物体 × 每个 animode → 分类关节 → 合并 fixed 关节连接的零件 → 二着色化简图 → 导出 part0.obj + part1.obj
3. **渲染**：Blender 渲染每个 animode 的运动视频
4. **编码**：视频 → V-JEPA2 特征（网络输入）；dual volume OBJ → PartPacker VAE → `[B, 8192, 64]` latent（GT）
5. **训练**：网络学习 V-JEPA2 特征 → dual volume latent

## 绝对禁止
- **不要用 IM（Infinite-Mobility）的任何东西**：repo 已删除，路径不存在
- **不要用 Sapien/PartNet 的 mesh 资产**：已移除。但 PhysX 可能引用 PartNet 的材质/映射，这是允许的
- 只用 IS 原生工厂 + PhysX 工厂

## Animode 命名规则
- **basic_animode**：每个铰链单独一个（basic_0, basic_1, ...），数量等于铰链数
- **senior_animode**：多铰链组合（senior_0, senior_1, ...），**同链组合允许**，**随机采样，最多保留 10 个**
- Root 节点默认 **part0**
- 每个 split 必须输出 **verify.png** 可视化验证图
- **没有"全动铰链"模式**（不要 default/all joints active）

## 关键技术点
- **同链组合允许**：senior animode 可以组合同一运动链上的铰链
- **URDF 正确性至关重要**：整个管线依赖 URDF 的关节/连杆拓扑。URDF 错 = 一切都错

### 桥接零件处理（基于 URDF 拓扑，不是零件大小）
"桥接零件"= 连接多个其他零件的零件。检查它的每条连接：
- 如果**任意一条连接**指向可动关节（active/passive）的子侧零件，且该连接**不是铰链连接**（是 rigid/fixed）→ **直接删除**该桥接零件（从 part0 和 part1 都去掉）
- 如果**所有连接都不跨越可动关节**→ **合并**该桥接零件到邻居（节点收缩）
这是基于 URDF link/joint 图的拓扑操作，不看 mesh 面数或大小。
- **dual volume 是 per-animode 的**：同一物体，激活不同铰链 = 完全不同的 part0/part1

## 关节分类策略（每个 animode 的 Active / Passive / Fixed）
每个 animode 定义主动关节，其余关节需要通过碰撞检测分类：

1. **主动关节（Active）**：由 animode 定义，驱动运动（轨迹类型跟着 animode 走：正弦往复、单向正弦、线性、线性往复）
2. **被动关节（Passive）**：主动运动轨迹下，BVH 碰撞检测到被碰到的关节连接零件 → 该关节为被动，提前预计算打开角度，让动画自然
3. **固定关节（Fixed）**：主动运动轨迹下未被碰到 → 锁死不动

**BVH 碰撞分类是 per-animode per-trajectory 的**：每个 animode 有自己的轨迹模式，passive/fixed 分类只对该 animode 的轨迹成立。不同 animode 用不同的 split。

### Precompute 阶段
- **先归一化，再拆分**：先把整个物体归一化到单位立方体，然后再按 animode 做拓扑划分
- 对每个 animode，计算主动关节初始轨迹（随机选择：正弦往复/单向正弦/线性/线性往复）
- 沿轨迹做 BVH 碰撞检测
- 碰到关节连接零件 → 该关节 = passive，预计算其提前打开量
- 没碰到 → 该关节 = fixed
- 合并 fixed 关节相连的零件为单个拓扑节点
- 对化简图（边=可动关节）做二部图交替着色 → part0 和 part1

### 渲染模拟阶段
- 逐帧 BVH 碰撞检测防穿模
- 碰到 **passive 关节**零件 → 逐帧更新该关节（让它让路/打开）
- 碰到 **fixed** 零件 → 全部停止（不能穿过固定几何体）

### 静止过滤
- 32 视角 = 16 固定半球（hemi） + 8 轨道（orbit） + 8 扫描（sweep）
- 如果 32 个视角全部静止 → **完全跳过该 animode**（不导出，不渲染）

### 每个 Animode 的 Metadata 必须保存
- 轨迹类型（正弦往复/单向正弦/线性/线性往复/自定义）
- 渲染用的 envmap 路径和文件名
- 关节分类（每个关节的 active/passive/fixed）
- **关节 12 维向量**（每个 active 关节）：`[0:3]` axis_origin（归一化），`[3:6]` axis_direction（单位向量），`[6:9]` 类型 one-hot [revolute, prismatic, continuous]，`[9:11]` 运动范围 [min, max]，`[11]` 存在标志
- 二着色分配（哪些合并零件组 → part0，哪些 → part1）
- Passive 关节的预打开角度

### 参考代码（只读，有 bug 但有用）
- 老版本：`/mnt/data/yurh/infinipart/` — 有 split_precompute.py 等。**注意**：很多 URDF 划分和零件连接有错，但动画逻辑有一些正确
- PartPacker：`/mnt/data/yurh/PartPacker/` — part 划分处理参考
