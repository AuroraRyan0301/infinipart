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
    ):
        super().__init__()

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

        # New: Part cross-attention per block (trainable)
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
        from trellis2.modules.utils import manual_cast
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
            # New: part cross-attention (trainable)
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
        """Forward pass for both parts simultaneously.

        Args:
            x_part0: SparseTensor — noisy latent for part0
            x_part1: SparseTensor — noisy latent for part1
            t: [B] — timestep
            vjepa_feats: [B, 10240, 1408] — raw VJEPA features

        Returns:
            (pred_part0, pred_part1): predicted velocity for each part
        """
        # Project VJEPA features
        cond = self.vjepa_proj(vjepa_feats)  # [B, T, 1024]

        # Get intermediate features of both parts for cross-attention
        # First pass: extract features from both parts (using frozen backbone)
        # We need the hidden states after each block for part cross-attention
        # For simplicity: use detached features from a quick forward pass
        with torch.no_grad():
            # Get part0 hidden features for part1 to attend to
            h0 = self.slat.input_layer(x_part0)
            from trellis2.modules.utils import manual_cast
            h0 = manual_cast(h0, self.slat.dtype)
            part0_feats = h0.feats.unsqueeze(0).float()  # [1, N0, channels]

            h1 = self.slat.input_layer(x_part1)
            h1 = manual_cast(h1, self.slat.dtype)
            part1_feats = h1.feats.unsqueeze(0).float()  # [1, N1, channels]

        # Forward both parts with part cross-attention
        pred_part0 = self.forward_single_part(x_part0, t, cond, part1_feats)
        pred_part1 = self.forward_single_part(x_part1, t, cond, part0_feats)

        return pred_part0, pred_part1

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

def load_pretrained_slat(ckpt_dir="/mnt/data/yurh/TRELLIS.2-4B", device="cuda:0"):
    """Load pretrained Shape SLat Flow Model from TRELLIS 2 checkpoint."""
    import json

    # Load config
    config_path = os.path.join(TRELLIS_ROOT, "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json")
    with open(config_path) as f:
        config = json.load(f)

    model_args = config["models"]["denoiser"]["args"]

    # Build model
    slat = SLatFlowModel(**model_args)

    # Load weights
    ckpt_path = os.path.join(ckpt_dir, "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors")
    if not os.path.exists(ckpt_path):
        # Try alternative paths
        ckpt_path = os.path.join(ckpt_dir, "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_ema.safetensors")
    if os.path.exists(ckpt_path):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
        slat.load_state_dict(state_dict, strict=False)
        print(f"Loaded SLat weights from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}, using random init")

    slat = slat.to(device).eval()
    return slat


def load_slat_decoder(ckpt_dir="/mnt/data/yurh/TRELLIS.2-4B", device="cuda:0"):
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


def load_slat_encoder(ckpt_dir="/mnt/data/yurh/TRELLIS.2-4B", device="cuda:0"):
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


def build_dual_part_model(ckpt_dir="/mnt/data/yurh/TRELLIS.2-4B", device="cuda:0"):
    """Build DualPartSLatModel with pretrained backbone."""
    slat = load_pretrained_slat(ckpt_dir, device)
    model = DualPartSLatModel(slat).to(device)
    print(f"DualPartSLatModel: {model.num_total_params()/1e6:.0f}M total, "
          f"{model.num_trainable_params()/1e6:.0f}M trainable")
    return model


if __name__ == "__main__":
    # Quick test
    model = build_dual_part_model()
    print("Model built successfully")
