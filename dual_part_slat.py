#!/usr/bin/env python3
"""
DualPartSLatModel: Modified TRELLIS 2 Shape SLat Flow Model with
dual-part cross-attention and VJEPA video conditioning.

Each transformer block: self-attn → VJEPA cross-attn → part cross-attn → MLP
- Self-attn + MLP: frozen from pretrained TRELLIS 2
- VJEPA cross-attn: replaces DinoV3, new projection layer
- Part cross-attn: new module, part0 attends to part1 and vice versa
"""

import sys
import os
from typing import *
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

TRELLIS_ROOT = "/mnt/cpfs/yurh/TRELLIS.2"
sys.path.insert(0, TRELLIS_ROOT)
sys.path.insert(0, os.path.join(TRELLIS_ROOT, "o-voxel"))

from trellis2.modules.sparse import SparseTensor, VarLenTensor
from trellis2.modules.sparse.attention import SparseMultiHeadAttention
from trellis2.modules.norm import LayerNorm32
from trellis2.models.structured_latent_flow import SLatFlowModel
from trellis2.modules.utils import manual_cast


# ================================================================
# VJEPA Feature Projector
# ================================================================

class VJEPAProjector(nn.Module):
    """Project VJEPA2 video features [B, 10240, 1408] → [B, N, 1024].

    Uses self-attention + MLP to compress temporal dimension and project
    to SLat's cond_channels (1024).
    """
    def __init__(self, in_dim=1408, out_dim=1024, num_heads=8, num_layers=2):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, out_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=out_dim, nhead=num_heads,
                dim_feedforward=out_dim * 4, dropout=0.0,
                batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        """x: [B, T, 1408] → [B, T, 1024]"""
        h = self.proj_in(x.float())
        for layer in self.layers:
            h = layer(h)
        h = self.proj_out(h)
        return h


# ================================================================
# Part Cross-Attention
# ================================================================

class PartCrossAttention(nn.Module):
    """Cross-attention between two parts' sparse tokens.

    part0 tokens attend to part1 tokens (and vice versa).
    """
    def __init__(self, channels, num_heads, qk_rms_norm=False):
        super().__init__()
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qk_rms_norm=qk_rms_norm,
        )

    def forward(self, x: SparseTensor, other_part_feats: torch.Tensor) -> SparseTensor:
        """
        Args:
            x: SparseTensor for this part
            other_part_feats: [1, N_other, channels] dense features of the other part
        """
        h = x.replace(self.norm(x.feats))
        h = self.cross_attn(h, other_part_feats)
        return x + h


# ================================================================
# DualPartSLatModel
# ================================================================

class DualPartSLatModel(nn.Module):
    """Modified SLatFlowModel for dual-part generation.

    Wraps the pretrained SLatFlowModel and adds:
    1. PartCrossAttention in each block (new, trainable)
    2. VJEPAProjector replacing DinoV3 conditioning (new, trainable)

    Pretrained self-attention + MLP blocks are frozen.
    """
    def __init__(
        self,
        pretrained_slat: SLatFlowModel,
        vjepa_dim: int = 1408,
        vjepa_proj_layers: int = 2,
        detach_cross_feats: bool = False,
        per_block_exchange: bool = True,
        cross_attn_start_block: int = 0,
    ):
        super().__init__()
        self.detach_cross_feats = detach_cross_feats
        self.per_block_exchange = per_block_exchange

        # Frozen pretrained backbone
        self.slat = pretrained_slat
        for param in self.slat.parameters():
            param.requires_grad = False

        channels = self.slat.model_channels
        num_heads = self.slat.num_heads
        num_blocks = len(self.slat.blocks)
        cond_channels = self.slat.cond_channels

        # New: VJEPA projector (trainable)
        self.vjepa_proj = VJEPAProjector(
            in_dim=vjepa_dim, out_dim=cond_channels,
            num_layers=vjepa_proj_layers,
        )

        # New: Part cross-attention for ALL blocks (allocated upfront for progressive training)
        # cross_attn_start_block controls which blocks are active (can be changed dynamically)
        self.cross_attn_start_block = cross_attn_start_block
        self.part_cross_attns = nn.ModuleList([
            PartCrossAttention(
                channels, num_heads,
                qk_rms_norm=self.slat.qk_rms_norm,
            )
            for _ in range(num_blocks)
        ])

        # Store config
        self.channels = channels
        self.num_blocks = num_blocks

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def in_channels(self):
        return self.slat.in_channels

    @property
    def out_channels(self):
        return self.slat.out_channels

    @property
    def resolution(self):
        return self.slat.resolution

    def forward_single_part(
        self,
        x: SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        other_part_feats: torch.Tensor,
    ) -> SparseTensor:
        """Forward pass for one part.

        Args:
            x: SparseTensor [N, in_channels] — noisy latent for this part
            t: [B] — timestep
            cond: [B, T, cond_channels] — VJEPA features (already projected)
            other_part_feats: [1, N_other, channels] — other part's hidden features
        """
        h = self.slat.input_layer(x)
        h = manual_cast(h, self.slat.dtype)
        t_emb = self.slat.t_embedder(t)
        if self.slat.share_mod:
            t_emb = self.slat.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.slat.dtype)
        cond = manual_cast(cond, self.slat.dtype)

        if self.slat.pe_mode == "ape":
            pe = self.slat.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.slat.dtype)

        for i, block in enumerate(self.slat.blocks):
            # Original block: self-attn + image cross-attn + MLP (frozen)
            h = block(h, t_emb, cond)
            # New: part cross-attention (trainable) — only for active blocks
            if i >= self.cross_attn_start_block:
                h_float = manual_cast(h, torch.float32)
                h_float = self.part_cross_attns[i](h_float, other_part_feats)
                h = manual_cast(h_float, self.slat.dtype)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.slat.out_layer(h)
        return h

    def forward(
        self,
        x_part0: SparseTensor,
        x_part1: SparseTensor,
        t: torch.Tensor,
        vjepa_feats: torch.Tensor,
    ) -> Tuple[SparseTensor, SparseTensor]:
        """Forward pass for both parts.

        If per_block_exchange=True: both parts go through blocks in parallel,
        exchanging features after each block.
        If per_block_exchange=False: extract shallow features once, then
        forward each part independently (faster, less expressive).
        """
        # Project VJEPA features
        cond = self.vjepa_proj(vjepa_feats)  # [B, T, 1024]

        if not self.per_block_exchange:
            # Legacy: shallow feature extraction + independent forward
            with torch.no_grad():
                h0 = self.slat.input_layer(x_part0)
                h0 = manual_cast(h0, self.slat.dtype)
                part0_feats = h0.feats.unsqueeze(0).float()
                h1 = self.slat.input_layer(x_part1)
                h1 = manual_cast(h1, self.slat.dtype)
                part1_feats = h1.feats.unsqueeze(0).float()
            pred0 = self.forward_single_part(x_part0, t, cond, part1_feats)
            pred1 = self.forward_single_part(x_part1, t, cond, part0_feats)
            return pred0, pred1

        # Per-block exchange mode
        h0 = self.slat.input_layer(x_part0)
        h0 = manual_cast(h0, self.slat.dtype)
        h1 = self.slat.input_layer(x_part1)
        h1 = manual_cast(h1, self.slat.dtype)

        t_emb = self.slat.t_embedder(t)
        if self.slat.share_mod:
            t_emb = self.slat.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.slat.dtype)
        cond = manual_cast(cond, self.slat.dtype)

        if self.slat.pe_mode == "ape":
            pe0 = self.slat.pos_embedder(h0.coords[:, 1:])
            h0 = h0 + manual_cast(pe0, self.slat.dtype)
            pe1 = self.slat.pos_embedder(h1.coords[:, 1:])
            h1 = h1 + manual_cast(pe1, self.slat.dtype)

        for i, block in enumerate(self.slat.blocks):
            h0 = block(h0, t_emb, cond)
            h1 = block(h1, t_emb, cond)

            # Part cross-attention only for active blocks
            if i >= self.cross_attn_start_block:
                h0_ctx = h0.feats.unsqueeze(0).float()
                h1_ctx = h1.feats.unsqueeze(0).float()
                if self.detach_cross_feats:
                    h0_ctx = h0_ctx.detach()
                    h1_ctx = h1_ctx.detach()

                h0_float = manual_cast(h0, torch.float32)
                h1_float = manual_cast(h1, torch.float32)
                h0_float = self.part_cross_attns[i](h0_float, h1_ctx)
                h1_float = self.part_cross_attns[i](h1_float, h0_ctx)
                h0 = manual_cast(h0_float, self.slat.dtype)
                h1 = manual_cast(h1_float, self.slat.dtype)

        h0 = manual_cast(h0, x_part0.dtype)
        h0 = h0.replace(F.layer_norm(h0.feats, h0.feats.shape[-1:]))
        h0 = self.slat.out_layer(h0)

        h1 = manual_cast(h1, x_part1.dtype)
        h1 = h1.replace(F.layer_norm(h1.feats, h1.feats.shape[-1:]))
        h1 = self.slat.out_layer(h1)

        return h0, h1

    def trainable_parameters(self):
        """Return only trainable parameters (for optimizer)."""
        params = []
        params.extend(self.vjepa_proj.parameters())
        params.extend(self.part_cross_attns.parameters())
        return params

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def num_total_params(self):
        return sum(p.numel() for p in self.parameters())


# ================================================================
# Loading utilities
# ================================================================

def load_pretrained_slat(ckpt_dir="/mnt/cpfs/yurh/TRELLIS.2-4B", device="cuda:0",
                         resolution="512"):
    """Load pretrained Shape SLat Flow Model from TRELLIS 2 checkpoint.

    Args:
        resolution: "512" or "1024". Only difference is RoPE resolution param (32 vs 64).
    """
    import json
    from safetensors.torch import load_file

    ckpt_name = f"slat_flow_img2shape_dit_1_3B_{resolution}_bf16"

    # Load config from checkpoint json (has exact model args)
    config_path = os.path.join(ckpt_dir, f"ckpts/{ckpt_name}.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(TRELLIS_ROOT, f"configs/gen/{ckpt_name}.json")
    with open(config_path) as f:
        config = json.load(f)

    model_args = config.get("args", config.get("models", {}).get("denoiser", {}).get("args", config))
    slat = SLatFlowModel(**model_args)

    ckpt_path = os.path.join(ckpt_dir, f"ckpts/{ckpt_name}.safetensors")
    if os.path.exists(ckpt_path):
        state_dict = load_file(ckpt_path)
        slat.load_state_dict(state_dict, strict=False)
        print(f"Loaded SLat {resolution} weights from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}, using random init")

    slat = slat.to(device).eval()
    return slat


def load_slat_decoder(ckpt_dir="/mnt/cpfs/yurh/TRELLIS.2-4B", device="cuda:0"):
    """Load pretrained SLat Decoder."""
    import json

    config_path = os.path.join(TRELLIS_ROOT, "configs/scvae/shape_vae_next_dc_f16c32_fp16.json")
    with open(config_path) as f:
        config = json.load(f)

    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
    decoder_args = config["models"]["decoder"]["args"]
    decoder = FlexiDualGridVaeDecoder(**decoder_args)

    ckpt_path = os.path.join(ckpt_dir, "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors")
    if os.path.exists(ckpt_path):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
        decoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded SLat Decoder from {ckpt_path}")

    decoder = decoder.to(device).eval()
    return decoder


def load_slat_encoder(ckpt_dir="/mnt/cpfs/yurh/TRELLIS.2-4B", device="cuda:0"):
    """Load pretrained SLat Encoder (for encoding GT OBJs to GT SLat)."""
    import json

    config_path = os.path.join(TRELLIS_ROOT, "configs/scvae/shape_vae_next_dc_f16c32_fp16.json")
    with open(config_path) as f:
        config = json.load(f)

    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeEncoder
    encoder_args = config["models"]["encoder"]["args"]
    encoder = FlexiDualGridVaeEncoder(**encoder_args)

    # Encoder weights are in the same VAE checkpoint
    ckpt_path = os.path.join(ckpt_dir, "ckpts/shape_enc_next_dc_f16c32_fp16.safetensors")
    if os.path.exists(ckpt_path):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
        encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded SLat Encoder from {ckpt_path}")

    encoder = encoder.to(device).eval()
    return encoder


def build_dual_part_model(ckpt_dir="/mnt/cpfs/yurh/TRELLIS.2-4B", device="cuda:0",
                          detach_cross_feats=False, per_block_exchange=True,
                          resolution="512", cross_attn_start_block=0, vjepa_dim=1408):
    """Build DualPartSLatModel with pretrained backbone.

    Args:
        resolution: "512" or "1024" — which pretrained flow model to wrap.
        cross_attn_start_block: only add part cross-attn from this block onward (0=all, 20=last 10).
        vjepa_dim: conditioning feature dimension (1408=JEPA, 1536=DINOv2).
    """
    slat = load_pretrained_slat(ckpt_dir, device, resolution=resolution)
    model = DualPartSLatModel(slat, vjepa_dim=vjepa_dim,
                              detach_cross_feats=detach_cross_feats,
                              per_block_exchange=per_block_exchange,
                              cross_attn_start_block=cross_attn_start_block).to(device)
    print(f"DualPartSLatModel ({resolution}, cross_attn from block {cross_attn_start_block}): "
          f"{model.num_total_params()/1e6:.0f}M total, {model.num_trainable_params()/1e6:.0f}M trainable")
    return model


if __name__ == "__main__":
    # Quick test
    model_lr = build_dual_part_model(resolution="512")
    model_hr = build_dual_part_model(resolution="1024")
    print("Both models built successfully")
