#!/usr/bin/env python3
"""
Inference + visualization: compare our video-conditioned model vs original PartPacker.

Mode A (ours): video -> VJEPA2 features -> DiT -> VAE decode -> dual volume
Mode B (original): single image -> DINOv2 -> DiT -> VAE decode -> dual volume (x4 frames)

Produces comparison panels:
  Row 0: Input video frames
  Row 1: GT part0 + GT part1
  Row 2: Our model (video) pred
  Rows 3-6: Original PartPacker per-frame pred (first, mid1, mid2, last)
  Row 7: GT verify.png (if exists)

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python infer_visualize.py \
    --ckpt /mnt/data_ssd/infinigen-sim/train_output/step_22000.pt \
    --output_dir /mnt/cpfs/yurh/Infinigen-Sim/output/infer_vis \
    --device cuda:0 --gpu_id 1
"""

import argparse
import copy
import importlib
import os
import sys

import cv2
import numpy as np
import torch

PARTPACKER_ROOT = "/mnt/data/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

LATENT_SIZE = 4096
LATENT_DIM = 64
TOTAL_LATENT = LATENT_SIZE * 2  # 8192
VJEPA_DIM = 1408
DIT_DIM = 1536

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


# ──────────────────────────────────────────────────────────────
# Our model (video -> VJEPA2 -> proj -> DiT)
# ──────────────────────────────────────────────────────────────

def build_our_model(device, ckpt_path):
    from flow.modules.dit import DiT
    dit = DiT(
        hidden_dim=DIT_DIM, num_heads=16, num_layers=24,
        latent_size=LATENT_SIZE, latent_dim=LATENT_DIM,
        qknorm=True, qknorm_type="RMSNorm",
        use_pos_embed=False, use_parts=True, part_embed_mode="part2_only",
    ).to(device, dtype=torch.bfloat16)
    proj = torch.nn.Linear(VJEPA_DIM, DIT_DIM).to(device, dtype=torch.bfloat16)

    print(f"Loading our checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    dit.load_state_dict(ckpt["dit"])
    if "proj" in ckpt:
        proj.load_state_dict(ckpt["proj"])
    step = ckpt.get('step', '?')
    loss = ckpt.get('loss', 0)
    print(f"  Step: {step}, Loss: {loss:.6f}")
    dit.eval()
    proj.eval()
    return dit, proj, step


def flow_inference(dit, cond, device, num_steps=50, cfg_scale=5.0):
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


# ──────────────────────────────────────────────────────────────
# Original PartPacker (image -> DINOv2 -> DiT)
# ──────────────────────────────────────────────────────────────

def build_original_partpacker(device):
    from flow.model import Model
    print("Loading original PartPacker (pretrained/flow.pt)...")
    # Model config uses relative paths (pretrained/vae.pt), so cd to PartPacker root
    prev_cwd = os.getcwd()
    os.chdir(PARTPACKER_ROOT)
    ckpt_path = os.path.join(PARTPACKER_ROOT, "pretrained", "flow.pt")
    ckpt_dict = torch.load(ckpt_path, weights_only=True)
    if "model" in ckpt_dict:
        ckpt_dict = ckpt_dict["model"]
    config = importlib.import_module("flow.configs.big_parts_strict_pvae").make_config()
    model = Model(config).eval().to(device).to(torch.bfloat16)
    model.load_state_dict(ckpt_dict, strict=True)
    os.chdir(prev_cwd)
    del ckpt_dict
    print("  Original PartPacker loaded (DINOv2-giant + DiT + VAE)")
    return model


def preprocess_image_for_partpacker(image_rgb):
    """Preprocess an RGB numpy image (H,W,3 uint8) for original PartPacker."""
    import rembg
    from flow.utils import recenter_foreground

    # Add alpha channel via rembg
    rgba = rembg.remove(image_rgb)  # -> (H,W,4) uint8
    mask = rgba[..., -1] > 0
    image = recenter_foreground(rgba, mask, border_ratio=0.1)
    image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])  # white bg
    return image


def run_original_partpacker_on_image(model, image_preprocessed, device,
                                      num_steps=50, cfg_scale=7.0, seed=42):
    """Run original PartPacker on a single preprocessed image. Returns latent [1, 8192, 64]."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    image_t = torch.from_numpy(image_preprocessed).permute(2, 0, 1).contiguous()
    image_t = image_t.unsqueeze(0).float().to(device)
    data = {"cond_images": image_t}
    with torch.inference_mode():
        results = model(data, num_steps=num_steps, cfg_scale=cfg_scale)
    return results["latent"]


# ──────────────────────────────────────────────────────────────
# Shared: VAE decode + mesh rendering
# ──────────────────────────────────────────────────────────────

def load_vae(device):
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


def decode_latent(latent, vae, device, grid_res=256):
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


def extract_video_frames(video_path, n_frames=6):
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


def extract_4_key_frames(video_path):
    """Extract first, mid1, mid2, last frames."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    indices = [0, total // 3, 2 * total // 3, total - 1]
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


def render_mesh_blender(obj_path, color, resolution=512, n_views=3, gpu_id="1"):
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
        "--obj", obj_path, "--color", color_str, "--output", output_base,
        "--views", str(n_views), "--resolution", str(resolution),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if result.returncode != 0:
        print(f"    Blender render failed: {result.stderr[-500:]}")
        blank = np.ones((resolution, resolution, 3), dtype=np.uint8) * 30
        return [blank] * n_views

    views = []
    for vi in range(n_views):
        view_path = output_base.replace(".png", f"_v{vi}.png")
        if os.path.exists(view_path):
            img = np.array(Image.open(view_path).convert('RGB'))
            views.append(img)
        else:
            views.append(np.ones((resolution, resolution, 3), dtype=np.uint8) * 30)
    return views


# ──────────────────────────────────────────────────────────────
# Panel building
# ──────────────────────────────────────────────────────────────

def build_comparison_panel(sample_name, video_frames, gt_obj_paths,
                            our_pred_obj_paths, our_mse, our_step,
                            orig_results, output_dir,
                            verify_path=None, gpu_id="1"):
    """
    Build expanded comparison panel.
    orig_results: list of dicts with keys: frame_label, pred_obj_paths, mse, frame_img
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_views = 3
    render_res = 512

    # Render GT
    print("  Rendering GT meshes...")
    gt_v0 = render_mesh_blender(gt_obj_paths[0], PART_COLORS[0], render_res, n_views, gpu_id)
    gt_v1 = render_mesh_blender(gt_obj_paths[1], PART_COLORS[1], render_res, n_views, gpu_id)

    # Render our pred
    print("  Rendering our model predictions...")
    our_v0 = render_mesh_blender(our_pred_obj_paths[0], PART_COLORS[0], render_res, n_views, gpu_id)
    our_v1 = render_mesh_blender(our_pred_obj_paths[1], PART_COLORS[1], render_res, n_views, gpu_id)

    # Render original PartPacker preds
    orig_renders = []
    for oi, orig in enumerate(orig_results):
        print(f"  Rendering original PartPacker [{orig['frame_label']}]...")
        ov0 = render_mesh_blender(orig["pred_obj_paths"][0], PART_COLORS[0], render_res, n_views, gpu_id)
        ov1 = render_mesh_blender(orig["pred_obj_paths"][1], PART_COLORS[1], render_res, n_views, gpu_id)
        orig_renders.append((ov0, ov1))

    # Layout:
    # Row 0: Video frames (6 cols)
    # Row 1: GT part0 (3) + GT part1 (3)
    # Row 2: Our pred (3+3) with MSE
    # Rows 3-6: Original PartPacker per frame (input frame + 2 views part0 + 1 blank + 2 views part1)
    # Row 7: verify.png (optional)

    has_verify = verify_path and os.path.exists(verify_path)
    n_orig = len(orig_results)
    n_rows = 3 + n_orig + (1 if has_verify else 0)

    fig = plt.figure(figsize=(18, 4.0 * n_rows))
    fig.patch.set_facecolor('#0a0a0f')
    gs = GridSpec(n_rows, 6, figure=fig, hspace=0.18, wspace=0.05)

    def add_title(ax, text, fontsize=11):
        ax.set_title(text, color='white', fontsize=fontsize, pad=5)

    # Row 0: Video frames
    n_vid = min(len(video_frames), 6)
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        ax.axis('off')
        if i < n_vid:
            ax.imshow(video_frames[i])
        else:
            ax.imshow(np.ones((100, 100, 3), dtype=np.uint8) * 30)
        if i == 0:
            add_title(ax, f"Input Video: {sample_name}")

    # Row 1: GT
    for i in range(n_views):
        ax = fig.add_subplot(gs[1, i])
        ax.axis('off')
        ax.imshow(gt_v0[i])
        if i == 0:
            add_title(ax, "GT Part0 (blue)")
    for i in range(n_views):
        ax = fig.add_subplot(gs[1, n_views + i])
        ax.axis('off')
        ax.imshow(gt_v1[i])
        if i == 0:
            add_title(ax, "GT Part1 (orange)")

    # Row 2: Our model
    for i in range(n_views):
        ax = fig.add_subplot(gs[2, i])
        ax.axis('off')
        ax.imshow(our_v0[i])
        if i == 0:
            add_title(ax, f"Ours (video, step {our_step}) Part0 | MSE: {our_mse:.4f}")
    for i in range(n_views):
        ax = fig.add_subplot(gs[2, n_views + i])
        ax.axis('off')
        ax.imshow(our_v1[i])
        if i == 0:
            add_title(ax, f"Ours Part1")

    # Rows 3+: Original PartPacker per frame
    for oi, (orig, (ov0, ov1)) in enumerate(zip(orig_results, orig_renders)):
        row = 3 + oi
        # Col 0: input frame
        ax = fig.add_subplot(gs[row, 0])
        ax.axis('off')
        if orig.get("frame_img") is not None:
            ax.imshow(orig["frame_img"])
        add_title(ax, f"PartPacker [{orig['frame_label']}] | MSE: {orig['mse']:.4f}")

        # Cols 1-2: part0 views
        for i in range(min(n_views - 1, 2)):
            ax = fig.add_subplot(gs[row, 1 + i])
            ax.axis('off')
            ax.imshow(ov0[i])
            if i == 0:
                add_title(ax, "Part0")

        # Cols 3-5: part1 views
        for i in range(n_views):
            ax = fig.add_subplot(gs[row, 3 + i])
            ax.axis('off')
            ax.imshow(ov1[i])
            if i == 0:
                add_title(ax, "Part1")

    # Verify row
    if has_verify:
        ax = fig.add_subplot(gs[n_rows - 1, :])
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


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="/mnt/data_ssd/infinigen-sim/train_output/latest.pt")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/cpfs/yurh/Infinigen-Sim/output/infer_vis")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--orig_cfg_scale", type=float, default=7.0,
                        help="CFG scale for original PartPacker")
    parser.add_argument("--grid_res", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu_id", type=str, default="1",
                        help="GPU ID for Blender rendering")
    parser.add_argument("--panel_only", action="store_true")
    parser.add_argument("--skip_original", action="store_true",
                        help="Skip original PartPacker comparison")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Phase 1: Our model (video -> VJEPA -> DiT) ──
    our_step = "?"
    vae = None
    if not args.panel_only:
        dit, proj, our_step = build_our_model(device, args.ckpt)
        vae = load_vae(device)

    our_results = {}  # name -> {pred_obj_paths, mse}

    for sample in SAMPLES:
        name = sample["name"]
        safe_name = name.replace("/", "_")
        mesh_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(mesh_dir, exist_ok=True)

        pred_obj_paths = [
            os.path.join(mesh_dir, "pred_part0.obj"),
            os.path.join(mesh_dir, "pred_part1.obj"),
        ]
        gt_obj_paths = [
            os.path.join(mesh_dir, "gt_part0.obj"),
            os.path.join(mesh_dir, "gt_part1.obj"),
        ]

        mse = 0.0
        if not args.panel_only:
            print(f"\n{'='*60}")
            print(f"[Ours] {name}")
            print(f"{'='*60}")

            vj = torch.load(sample["jepa"], weights_only=True, map_location=device)
            if vj.dim() == 2:
                vj = vj.unsqueeze(0)
            vj = vj.to(dtype=torch.bfloat16)
            with torch.inference_mode():
                cond = proj(vj)

            print(f"  Flow inference ({args.num_steps} steps, cfg={args.cfg_scale})...")
            pred_latent = flow_inference(dit, cond, device, args.num_steps, args.cfg_scale)

            gt_latent = torch.load(sample["gt_latent"], weights_only=True, map_location=device)
            if gt_latent.dim() == 2:
                gt_latent = gt_latent.unsqueeze(0)
            mse = torch.nn.functional.mse_loss(pred_latent.float(), gt_latent.float()).item()
            print(f"  MSE: {mse:.6f}")

            print("  Decoding predicted...")
            pred_meshes = decode_latent(pred_latent, vae, device, args.grid_res)
            print("  Decoding GT...")
            gt_meshes = decode_latent(gt_latent, vae, device, args.grid_res)

            import trimesh
            for i, (pm, gm) in enumerate(zip(pred_meshes, gt_meshes)):
                if pm is not None:
                    pm.export(pred_obj_paths[i])
                if gm is not None:
                    gm.export(gt_obj_paths[i])

        our_results[name] = {"pred_obj_paths": pred_obj_paths,
                              "gt_obj_paths": gt_obj_paths, "mse": mse}

    # Free our model
    if not args.panel_only:
        del dit, proj
        torch.cuda.empty_cache()

    # ── Phase 2: Original PartPacker (image -> DINOv2 -> DiT) ──
    orig_all_results = {}  # name -> list of {frame_label, pred_obj_paths, mse, frame_img}

    if not args.skip_original and not args.panel_only:
        orig_model = build_original_partpacker(device)
        # The original model has its own VAE; we use our standalone VAE for consistency
        # But the original model also does full pipeline (encode+flow+decode)
        # For latent comparison we need to run model forward then decode separately

        for sample in SAMPLES:
            name = sample["name"]
            safe_name = name.replace("/", "_")
            mesh_dir = os.path.join(args.output_dir, safe_name)

            print(f"\n{'='*60}")
            print(f"[Original PartPacker] {name}")
            print(f"{'='*60}")

            # Extract 4 key frames
            key_frames = extract_4_key_frames(sample["video"])
            if len(key_frames) < 4:
                print(f"  WARNING: only got {len(key_frames)} frames")
                orig_all_results[name] = []
                continue

            frame_labels = ["first", "mid1", "mid2", "last"]
            gt_latent = torch.load(sample["gt_latent"], weights_only=True, map_location=device)
            if gt_latent.dim() == 2:
                gt_latent = gt_latent.unsqueeze(0)

            results_list = []
            for fi, (frame, label) in enumerate(zip(key_frames, frame_labels)):
                print(f"\n  Frame: {label}")

                # Preprocess
                print(f"    Preprocessing (rembg + recenter)...")
                preprocessed = preprocess_image_for_partpacker(frame)

                # Run original model
                print(f"    Flow inference ({args.num_steps} steps, cfg={args.orig_cfg_scale})...")
                pred_latent = run_original_partpacker_on_image(
                    orig_model, preprocessed, device,
                    num_steps=args.num_steps, cfg_scale=args.orig_cfg_scale,
                    seed=42 + fi)

                # MSE
                mse = torch.nn.functional.mse_loss(pred_latent.float(), gt_latent.float()).item()
                print(f"    MSE: {mse:.6f}")

                # Decode
                print(f"    Decoding...")
                meshes = decode_latent(pred_latent, vae, device, args.grid_res)

                # Save
                pred_paths = []
                import trimesh
                for pi, mesh in enumerate(meshes):
                    p = os.path.join(mesh_dir, f"orig_{label}_part{pi}.obj")
                    if mesh is not None:
                        mesh.export(p)
                    pred_paths.append(p)

                # Save preprocessed frame for panel
                frame_vis = (preprocessed * 255).astype(np.uint8)

                results_list.append({
                    "frame_label": label,
                    "pred_obj_paths": pred_paths,
                    "mse": mse,
                    "frame_img": frame_vis,
                })

            orig_all_results[name] = results_list

        del orig_model
        torch.cuda.empty_cache()
    else:
        # Load existing orig results for panel_only mode
        for sample in SAMPLES:
            name = sample["name"]
            safe_name = name.replace("/", "_")
            mesh_dir = os.path.join(args.output_dir, safe_name)
            frame_labels = ["first", "mid1", "mid2", "last"]
            results_list = []
            for label in frame_labels:
                p0 = os.path.join(mesh_dir, f"orig_{label}_part0.obj")
                p1 = os.path.join(mesh_dir, f"orig_{label}_part1.obj")
                if os.path.exists(p0):
                    results_list.append({
                        "frame_label": label,
                        "pred_obj_paths": [p0, p1],
                        "mse": 0.0,
                        "frame_img": None,
                    })
            orig_all_results[name] = results_list

    # ── Phase 3: Build panels ──
    if vae is not None:
        del vae
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Building comparison panels...")
    print(f"{'='*60}")

    results_summary = []
    for sample in SAMPLES:
        name = sample["name"]
        our = our_results[name]
        orig_list = orig_all_results.get(name, [])

        frames = extract_video_frames(sample["video"], n_frames=6)

        panel_path = build_comparison_panel(
            name, frames, our["gt_obj_paths"], our["pred_obj_paths"],
            our["mse"], our_step, orig_list,
            args.output_dir, sample.get("verify"), gpu_id=args.gpu_id)

        results_summary.append({
            "name": name, "our_mse": our["mse"],
            "orig_mses": [r["mse"] for r in orig_list],
            "panel": panel_path,
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results_summary:
        orig_str = ", ".join(f"{m:.4f}" for m in r["orig_mses"]) if r["orig_mses"] else "N/A"
        print(f"  {r['name']}:")
        print(f"    Ours (video): MSE={r['our_mse']:.4f}")
        print(f"    PartPacker (first/mid1/mid2/last): MSE={orig_str}")
    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
