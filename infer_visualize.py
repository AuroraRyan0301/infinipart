#!/usr/bin/env python3
"""
Inference + visualization: video -> DiT -> VAE decode -> mesh rendering.

Produces a comparison panel for each sample:
  - Input video frames (from source mp4)
  - GT dual volume meshes (from precompute part0.obj + part1.obj)
  - Predicted dual volume meshes (from DiT -> VAE decode)

Usage:
  CUDA_VISIBLE_DEVICES=3 python infer_visualize.py \
    --ckpt /mnt/data_ssd/infinigen-sim/train_output/step_20000.pt \
    --output_dir /mnt/cpfs/yurh/Infinigen-Sim/output/infer_vis
"""

import argparse
import importlib
import os
import sys

import numpy as np
import torch

PARTPACKER_ROOT = "/mnt/data/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

LATENT_SIZE = 4096  # tokens per part
LATENT_DIM = 64
TOTAL_LATENT = LATENT_SIZE * 2  # 8192
VJEPA_DIM = 1408
DIT_DIM = 1536

# Colors for part0 (blue) and part1 (orange)
PART_COLORS = [
    np.array([0.35, 0.55, 0.85]),   # part0: blue
    np.array([0.95, 0.55, 0.25]),   # part1: orange
]

SAMPLES = [
    {
        "name": "dishwasher/8/senior_3",
        "jepa": "/mnt/data_ssd/infinigen-sim/dishwasher/8_senior_3/views/v05_nobg_jepa.pt",
        "gt_latent": "/mnt/data_ssd/infinigen-sim/dishwasher/8_senior_3/gt_latent.pt",
        "video": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/dishwasher/8/senior_3/hemi_01_nobg.mp4",
        "gt_part0": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/dishwasher/8/senior_3/part0.obj",
        "gt_part1": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/dishwasher/8/senior_3/part1.obj",
        "verify": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/dishwasher/8/senior_3/verify.png",
    },
    {
        "name": "lamp/1/senior_0",
        "jepa": "/mnt/data_ssd/infinigen-sim/lamp/1_senior_0/views/v05_nobg_jepa.pt",
        "gt_latent": "/mnt/data_ssd/infinigen-sim/lamp/1_senior_0/gt_latent.pt",
        "video": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/lamp/1/senior_0/hemi_02_nobg.mp4",
        "gt_part0": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/lamp/1/senior_0/part0.obj",
        "gt_part1": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/lamp/1/senior_0/part1.obj",
        "verify": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/lamp/1/senior_0/verify.png",
    },
    {
        "name": "drawer/0/custom_1_flip",
        "jepa": "/mnt/data_ssd/infinigen-sim/drawer/0_custom_1_flip/views/v05_nobg_jepa.pt",
        "gt_latent": "/mnt/data_ssd/infinigen-sim/drawer/0_custom_1_flip/gt_latent.pt",
        "video": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/drawer/0/custom_1_flip/hemi_05_nobg.mp4",
        "gt_part0": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/drawer/0/custom_1_flip/part0.obj",
        "gt_part1": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/drawer/0/custom_1_flip/part1.obj",
        "verify": "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/drawer/0/custom_1_flip/verify.png",
    },
]


def build_model(device, ckpt_path):
    """Build DiT + projector, load trained weights."""
    from flow.modules.dit import DiT

    dit = DiT(
        hidden_dim=DIT_DIM, num_heads=16, num_layers=24,
        latent_size=LATENT_SIZE, latent_dim=LATENT_DIM,
        qknorm=True, qknorm_type="RMSNorm",
        use_pos_embed=False, use_parts=True, part_embed_mode="part2_only",
    ).to(device, dtype=torch.bfloat16)

    proj = torch.nn.Linear(VJEPA_DIM, DIT_DIM).to(device, dtype=torch.bfloat16)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    dit.load_state_dict(ckpt["dit"])
    if "proj" in ckpt:
        proj.load_state_dict(ckpt["proj"])
    print(f"  Step: {ckpt.get('step', '?')}, Loss: {ckpt.get('loss', '?'):.6f}")

    dit.eval()
    proj.eval()
    return dit, proj


def flow_inference(dit, cond, device, num_steps=50, cfg_scale=5.0):
    """Flow matching sampling: noise -> denoised latent."""
    cond_null = torch.zeros_like(cond)
    cond_input = torch.cat([cond, cond_null], dim=0)
    x = torch.randn(1, TOTAL_LATENT, LATENT_DIM, device=device)

    sigmas = np.linspace(1, 0, num_steps + 1)
    sigmas = 3.0 * sigmas / (1 + (3.0 - 1) * sigmas)

    with torch.inference_mode():
        for i in range(num_steps):
            sigma, sigma_prev = sigmas[i], sigmas[i + 1]
            timesteps = torch.tensor(
                [1000 * sigma] * 2, device=device, dtype=torch.float32)
            x_input = torch.cat([x, x], dim=0).to(dtype=torch.bfloat16)
            pred = dit(x_input, cond_input, timesteps).float()
            cond_v, uncond_v = pred.chunk(2, dim=0)
            pred_v = uncond_v + (cond_v - uncond_v) * cfg_scale
            x = x - (sigma - sigma_prev) * pred_v
    return x


def decode_latent(latent, vae, device, grid_res=256):
    """Decode [1, 8192, 64] -> two trimesh objects."""
    import trimesh

    TRIMESH_ROT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    meshes = []

    for start, end, name in [(0, LATENT_SIZE, "part0"), (LATENT_SIZE, TOTAL_LATENT, "part1")]:
        lat = latent[:, start:end, :]
        data = {"latent": lat.to(device, dtype=torch.bfloat16)}
        with torch.inference_mode():
            results = vae(data, resolution=grid_res)

        if "meshes" in results and len(results["meshes"]) > 0:
            vertices, faces = results["meshes"][0]
            mesh = trimesh.Trimesh(vertices, faces)
            mesh.vertices = mesh.vertices @ TRIMESH_ROT.T
            meshes.append(mesh)
            print(f"    {name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        else:
            meshes.append(None)
            print(f"    {name}: no mesh extracted")

    return meshes


def load_vae(device):
    """Load PartPacker VAE."""
    from vae.model import Model

    vae_ckpt_path = os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt")
    ckpt = torch.load(vae_ckpt_path, weights_only=True, map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    config = importlib.import_module("vae.configs.part_woenc").make_config()
    vae = Model(config).eval().to(device).to(torch.bfloat16)
    vae.load_state_dict(ckpt, strict=True)
    del ckpt
    print("VAE loaded")
    return vae


def extract_video_frames(video_path, n_frames=6):
    """Extract evenly-spaced frames from video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


BLENDER_BIN = "/mnt/data/yurh/blender-4.2.18-linux-x64/blender"
RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_mesh_preview.py")


def render_mesh_blender(obj_path, color, resolution=512, n_views=3, gpu_id="3"):
    """Render mesh views using Blender Cycles (high quality)."""
    import subprocess
    from PIL import Image

    if not os.path.exists(obj_path):
        blank = np.ones((resolution, resolution, 3), dtype=np.uint8) * 30
        return [blank] * n_views

    color_str = f"{color[0]:.3f},{color[1]:.3f},{color[2]:.3f}"
    output_base = obj_path.replace(".obj", "_render.png")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    cmd = [
        BLENDER_BIN, "--background", "--python", RENDER_SCRIPT, "--",
        "--obj", obj_path,
        "--color", color_str,
        "--output", output_base,
        "--views", str(n_views),
        "--resolution", str(resolution),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if result.returncode != 0:
        print(f"    Blender render failed: {result.stderr[-500:]}")
        blank = np.ones((resolution, resolution, 3), dtype=np.uint8) * 30
        return [blank] * n_views

    # Read rendered view images
    views = []
    for vi in range(n_views):
        view_path = output_base.replace(".png", f"_v{vi}.png")
        if os.path.exists(view_path):
            img = np.array(Image.open(view_path).convert('RGB'))
            views.append(img)
        else:
            views.append(np.ones((resolution, resolution, 3), dtype=np.uint8) * 30)

    return views


def build_comparison_panel(sample_name, video_frames, gt_obj_paths, pred_obj_paths,
                           gt_mse, output_dir, verify_path=None, gpu_id="3"):
    """Build a comparison image panel using Blender-rendered mesh views."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_views = 3
    render_res = 512

    print("  Rendering GT meshes (Blender)...")
    gt_views_0 = render_mesh_blender(gt_obj_paths[0], PART_COLORS[0], render_res, n_views, gpu_id)
    gt_views_1 = render_mesh_blender(gt_obj_paths[1], PART_COLORS[1], render_res, n_views, gpu_id)

    print("  Rendering predicted meshes (Blender)...")
    pred_views_0 = render_mesh_blender(pred_obj_paths[0], PART_COLORS[0], render_res, n_views, gpu_id)
    pred_views_1 = render_mesh_blender(pred_obj_paths[1], PART_COLORS[1], render_res, n_views, gpu_id)

    # Layout: 4 rows
    # Row 0: Input video frames (6 frames)
    # Row 1: GT part0 (blue) + GT part1 (orange) — 3 views each
    # Row 2: Pred part0 (blue) + Pred part1 (orange) — 3 views each
    # Row 3: GT verify.png (if exists)

    n_vid = min(len(video_frames), 6)
    has_verify = verify_path and os.path.exists(verify_path)
    n_rows = 4 if has_verify else 3

    fig = plt.figure(figsize=(18, 4.5 * n_rows))
    fig.patch.set_facecolor('#0a0a0f')
    gs = GridSpec(n_rows, 6, figure=fig, hspace=0.15, wspace=0.05)

    def add_title(ax, text, fontsize=11):
        ax.set_title(text, color='white', fontsize=fontsize, pad=5)

    # Row 0: Video frames
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        ax.axis('off')
        if i < n_vid:
            ax.imshow(video_frames[i])
        else:
            ax.imshow(np.ones((100, 100, 3), dtype=np.uint8) * 30)
        if i == 0:
            add_title(ax, f"Input Video: {sample_name}")

    # Row 1: GT meshes
    for i in range(n_views):
        ax = fig.add_subplot(gs[1, i])
        ax.axis('off')
        ax.imshow(gt_views_0[i])
        if i == 0:
            add_title(ax, "GT Part0 (blue)")

    for i in range(n_views):
        ax = fig.add_subplot(gs[1, n_views + i])
        ax.axis('off')
        ax.imshow(gt_views_1[i])
        if i == 0:
            add_title(ax, "GT Part1 (orange)")

    # Row 2: Predicted meshes
    for i in range(n_views):
        ax = fig.add_subplot(gs[2, i])
        ax.axis('off')
        ax.imshow(pred_views_0[i])
        if i == 0:
            add_title(ax, f"Pred Part0 (blue) | MSE: {gt_mse:.4f}")

    for i in range(n_views):
        ax = fig.add_subplot(gs[2, n_views + i])
        ax.axis('off')
        ax.imshow(pred_views_1[i])
        if i == 0:
            add_title(ax, "Pred Part1 (orange)")

    # Row 3: Verify image
    if has_verify:
        ax = fig.add_subplot(gs[3, :])
        ax.axis('off')
        verify_img = plt.imread(verify_path)
        ax.imshow(verify_img)
        add_title(ax, "GT Topology Split (verify.png)")

    safe_name = sample_name.replace("/", "_")
    out_path = os.path.join(output_dir, f"compare_{safe_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="/mnt/data_ssd/infinigen-sim/train_output/step_20000.pt")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/cpfs/yurh/Infinigen-Sim/output/infer_vis")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--grid_res", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu_id", type=str, default="3",
                        help="GPU ID for Blender rendering (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--panel_only", action="store_true",
                        help="Skip inference, only rebuild panels from existing OBJs")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.panel_only:
        # Build DiT model
        dit, proj = build_model(device, args.ckpt)
        # Load VAE
        vae = load_vae(device)

    results_summary = []

    for sample in SAMPLES:
        name = sample["name"]
        safe_name = name.replace("/", "_")
        mesh_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(mesh_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")

        gt_obj_paths = [
            os.path.join(mesh_dir, "gt_part0.obj"),
            os.path.join(mesh_dir, "gt_part1.obj"),
        ]
        pred_obj_paths = [
            os.path.join(mesh_dir, "pred_part0.obj"),
            os.path.join(mesh_dir, "pred_part1.obj"),
        ]

        mse = 0.0

        if not args.panel_only:
            # Load JEPA features
            vj = torch.load(sample["jepa"], weights_only=True, map_location=device)
            if vj.dim() == 2:
                vj = vj.unsqueeze(0)
            vj = vj.to(dtype=torch.bfloat16)

            # Project and run flow inference
            with torch.inference_mode():
                cond = proj(vj)
            print(f"  Running flow inference ({args.num_steps} steps, cfg={args.cfg_scale})...")
            pred_latent = flow_inference(dit, cond, device, args.num_steps, args.cfg_scale)

            # Compare with GT
            gt_latent = torch.load(sample["gt_latent"], weights_only=True, map_location=device)
            if gt_latent.dim() == 2:
                gt_latent = gt_latent.unsqueeze(0)
            mse = torch.nn.functional.mse_loss(pred_latent.float(), gt_latent.float()).item()
            print(f"  GT MSE: {mse:.6f}")

            # Decode predicted latent
            print("  Decoding predicted latent...")
            pred_meshes = decode_latent(pred_latent, vae, device, args.grid_res)

            # Decode GT latent
            print("  Decoding GT latent...")
            gt_meshes = decode_latent(gt_latent, vae, device, args.grid_res)

            # Save meshes
            import trimesh
            for i, (pm, gm) in enumerate(zip(pred_meshes, gt_meshes)):
                if pm is not None:
                    pm.export(pred_obj_paths[i])
                if gm is not None:
                    gm.export(gt_obj_paths[i])
        else:
            print("  Panel-only mode: using existing OBJ files")

        # Extract video frames
        print("  Extracting video frames...")
        frames = extract_video_frames(sample["video"], n_frames=6)

        # Build comparison panel with Blender rendering
        print("  Building comparison panel (Blender Cycles)...")
        panel_path = build_comparison_panel(
            name, frames, gt_obj_paths, pred_obj_paths, mse,
            args.output_dir, sample.get("verify"), gpu_id=args.gpu_id)

        results_summary.append({
            "name": name, "mse": mse, "panel": panel_path,
        })

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results_summary:
        print(f"  {r['name']}: MSE={r['mse']:.4f}")
    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
