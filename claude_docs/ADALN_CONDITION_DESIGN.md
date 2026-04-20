# AdaLN Video Condition Injection — Design Document

## Problem

PartPacker's DiT uses AdaLN modulation for **timestep only**. Video condition enters through cross-attention, which is "soft" — the model can learn to ignore it, requiring CFG=11 to amplify.

## Solution

Inject pooled video features into the AdaLN pathway, so every DiT layer's LayerNorm scale/shift/gate is modulated by **both timestep AND video condition**.

## Architecture

### Current DiTLayer (timestep-only AdaLN)

```python
# adaln_linear: dim → dim*6 (produces shift/scale/gate for self-attn + FFN)
t_adaln = self.adaln_linear(F.silu(t_emb))  # [B, 6*C]
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t_adaln.chunk(6)

# Self-attention with AdaLN
h = LayerNorm(x) * (1 + scale_msa) + shift_msa
x = x + gate_msa * self_attn(h)

# Cross-attention (condition enters here, no AdaLN)
h = LayerNorm(x)
x = x + cross_attn(h, c)  # ← only place video condition is used

# FFN with AdaLN
h = LayerNorm(x) * (1 + scale_mlp) + shift_mlp
x = x + gate_mlp * ff(h)
```

### Proposed: Video-Conditioned AdaLN

```python
# NEW: video global embedding (computed once before DiT layers)
video_global = mean_pool(projected_video_features)  # [B, C]
cond_emb = video_cond_mlp(video_global)  # [B, C]

# Combined embedding for AdaLN
combined = t_emb + cond_emb  # additive fusion
t_adaln = self.adaln_linear(F.silu(combined))  # [B, 6*C]

# Rest is identical — but now scale/shift/gate depend on video condition
```

### Why Additive Fusion?

- `t_emb` and `cond_emb` are both [B, C=1536] vectors
- Simple addition preserves the existing AdaLN linear weights
- The pretrained `adaln_linear` weights still work (they were trained with t_emb only; adding cond_emb perturbs the input, which the model adapts to during finetuning)
- Alternative: concat + MLP (more expressive but breaks pretrained weights more)

## Implementation Options

### Option A: Minimal (modify TrainWrapper only, ~30 lines)

Pool video features → MLP → add to timestep embedding before passing to DiT. **Does not modify DiT code at all.**

```python
class TrainWrapper(nn.Module):
    def __init__(self, dit, vjepa_dim, dit_dim):
        self.dit = dit
        self.proj = nn.Linear(vjepa_dim, dit_dim)  # per-token projection
        self.cond_pool_mlp = nn.Sequential(        # NEW: global condition
            nn.Linear(dit_dim, dit_dim),
            nn.GELU(),
            nn.Linear(dit_dim, dit_dim),
        )

    def forward(self, noisy_latent, vjepa_feat, timesteps):
        cond = self.proj(vjepa_feat)                    # [B, N, 1536] for cross-attn
        cond_global = self.cond_pool_mlp(cond.mean(1))  # [B, 1536] for AdaLN
        # Inject into timestep embedding inside DiT
        return self.dit(noisy_latent, cond, timesteps, cond_global=cond_global)
```

Requires a small change to DiT.forward() to accept `cond_global` and add it to `t_emb`.

### Option B: Full (modify DiTLayer, more expressive)

Each DiTLayer gets its own fusion MLP for combining timestep + video condition. More parameters, more expressive.

### Recommendation: Option A

Minimal code change, pretrained weights mostly preserved, easy to ablate (set cond_global=0 to disable). The `cond_pool_mlp` is the only new trainable component (~7M params).

## Training Strategy

1. **Resume DiT from Exp1 step_70000 checkpoint** (trained 70k steps with linear proj + diff JEPA)
2. **New components:** `proj` (2-layer MLP), `cond_pool_mlp` (2-layer MLP, zero-init output)
3. **Warmup 10k steps:** freeze DiT, train proj only via cross-attention. **AdaLN cond path is DISABLED** (`enable_adaln_cond = False`) — cond_pool_mlp has no gradient, stays at zero-init
4. **Phase 2 (10k+):** unfreeze DiT, **enable AdaLN cond path** (`enable_adaln_cond = True`). cond_pool_mlp starts from zero → gradual injection. All parameters train together.

### Why disable AdaLN during warmup?
- Warmup goal: let proj learn cross-attention conditioning while DiT stays pristine
- If AdaLN path is active during warmup, cond_pool_mlp's output (initially ~0) perturbs t_emb through frozen adaln_linear, producing uncontrolled scale/shift changes with no gradient to correct them
- Clean separation: warmup = cross-attention only, Phase 2 = cross-attention + AdaLN

## File Structure

```
train_partpacker_adaln.py      — standalone training script with AdaLN condition injection
                                 (does NOT modify PartPacker repo, imports DiT from it)
encode_diff_jepa_v2.py         — encode [9600, 1408] features (orig_sub + diff_sub)
launch_adaln_train.sh          — one-click: encode + train on 4 GPUs
infer_quick.py                 — inference + Blender render comparison
make_cfg_grid_v2.py            — CFG comparison grid with labels for presentation
```

## Expected Impact

- Video condition directly modulates every layer's activations → impossible for DiT to ignore
- Lower optimal CFG (should work well at CFG=3-5 instead of 11)
- Faster convergence (condition signal stronger from the start)
- ~7M new parameters (negligible vs 1.25B DiT)
