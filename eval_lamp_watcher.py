#!/usr/bin/env python3
"""Watch for overfit checkpoints, eval lamp_0_basic_0, render with Blender."""
import os, sys, time, subprocess, shutil
import torch
import numpy as np

TRELLIS_ROOT = "/mnt/cpfs/yurh/TRELLIS.2"
sys.path.insert(0, TRELLIS_ROOT)
sys.path.insert(0, os.path.join(TRELLIS_ROOT, "o-voxel"))
sys.path.insert(0, "/mnt/cpfs/yurh/Infinigen-Sim")

import types
_f = types.ModuleType('cumesh'); _f.remeshing = types.ModuleType('cumesh.remeshing')
sys.modules['cumesh'] = _f; sys.modules['cumesh.remeshing'] = _f.remeshing

from trellis2.modules.sparse import SparseTensor
from dual_part_slat import build_dual_part_model, load_slat_decoder
from infer_slat_progressive import flow_sampling, denormalize_slat, decode_slat_to_mesh

DATA_ROOT = "/mnt/data_ssd/infinigen-sim-data"
CKPT_DIR = os.path.join(DATA_ROOT, "checkpoints/slat_overfit_long")
OUT_DIR = os.path.join(DATA_ROOT, "eval_lamp_overfit")
BLENDER = "/mnt/cpfs/yurh/blender-4.2.18-linux-x64/blender"
SAMPLE_CAT = "lamp"
SAMPLE_MID = "0_basic_0"

RENDER_SCRIPT = '''
import bpy, sys, math, os
argv = sys.argv[sys.argv.index("--") + 1:]
p0, p1, out_png = argv[0], argv[1], argv[2]
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
def imp(path, color, name):
    bpy.ops.wm.obj_import(filepath=path)
    obj = bpy.context.selected_objects[0]; obj.name = name
    mat = bpy.data.materials.new(name=f"m_{name}"); mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.5
    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)
    return obj
import mathutils
objs = []
if os.path.exists(p0): objs.append(imp(p0, (0.15,0.35,0.85,1), "p0"))
if os.path.exists(p1): objs.append(imp(p1, (0.95,0.55,0.1,1), "p1"))
verts = []
for o in objs:
    for v in o.data.vertices: verts.append(o.matrix_world @ v.co)
if verts:
    mn = mathutils.Vector((min(v.x for v in verts),min(v.y for v in verts),min(v.z for v in verts)))
    mx = mathutils.Vector((max(v.x for v in verts),max(v.y for v in verts),max(v.z for v in verts)))
    c = (mn+mx)/2; s = 2/max((mx-mn).x,(mx-mn).y,(mx-mn).z) if max((mx-mn).x,(mx-mn).y,(mx-mn).z) > 0 else 1
    for o in objs: o.location -= c; o.scale *= s
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")); bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam; cam.location=(2.5,-2.5,2); cam.rotation_euler=(math.radians(60),0,math.radians(45))
l = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun",type='SUN')); bpy.context.scene.collection.objects.link(l)
l.location=(3,-3,5); l.data.energy=3
w = bpy.data.worlds.new("W"); bpy.context.scene.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value=(0.95,0.95,0.95,1)
s = bpy.context.scene; s.render.engine='CYCLES'; s.cycles.device='GPU'; s.cycles.samples=64
s.render.resolution_x=800; s.render.resolution_y=800; s.render.filepath=out_png
s.render.image_settings.file_format='PNG'; bpy.ops.render.render(write_still=True)
'''


def eval_checkpoint(ckpt_path, step, device):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cs = ckpt.get("cross_attn_start_block", 15)
    model = build_dual_part_model(device=device, resolution="512", cross_attn_start_block=cs)
    model.vjepa_proj.load_state_dict(ckpt["vjepa_proj"])
    model.part_cross_attns.load_state_dict(ckpt["part_cross_attns"])
    model.eval()

    gt = torch.load(os.path.join(DATA_ROOT, f"slat_gt/{SAMPLE_CAT}/{SAMPLE_MID}.pt"),
                    weights_only=False, map_location=device)
    p0c = gt['p0_lr']['coords'].to(device)
    p1c = gt['p1_lr']['coords'].to(device)

    vdir = os.path.join(DATA_ROOT, f"encoded/{SAMPLE_CAT}/{SAMPLE_MID}/views")
    jf = sorted([f for f in os.listdir(vdir) if f.endswith("_nobg_jepa.pt")])[0]
    jepa = torch.load(os.path.join(vdir, jf), weights_only=False, map_location=device)
    if jepa.dim() == 2: jepa = jepa.unsqueeze(0)

    x0 = SparseTensor(feats=torch.zeros(p0c.shape[0], 32, device=device), coords=p0c)
    x1 = SparseTensor(feats=torch.zeros(p1c.shape[0], 32, device=device), coords=p1c)
    pred0, pred1 = flow_sampling(model, x0, x1, jepa, num_steps=50)
    pred0 = pred0.replace(denormalize_slat(pred0.feats.float(), device))
    pred1 = pred1.replace(denormalize_slat(pred1.feats.float(), device))

    decoder = load_slat_decoder(device=device)
    m0 = decode_slat_to_mesh(decoder, pred0, resolution=512)
    m1 = decode_slat_to_mesh(decoder, pred1, resolution=512)

    tag = f"step_{step}"
    p0_obj = os.path.join(OUT_DIR, f"{tag}_p0.obj")
    p1_obj = os.path.join(OUT_DIR, f"{tag}_p1.obj")
    if m0: m0.export(p0_obj)
    if m1: m1.export(p1_obj)

    # Blender render
    script_path = "/tmp/_eval_lamp_render.py"
    with open(script_path, 'w') as f:
        f.write(RENDER_SCRIPT)
    out_png = os.path.join(OUT_DIR, f"{tag}_pred.png")
    subprocess.run([BLENDER, "--background", "--python", script_path, "--",
                    p0_obj, p1_obj, out_png], capture_output=True, timeout=120)

    del model, decoder
    torch.cuda.empty_cache()
    print(f"  Eval step {step} → {out_png}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    # Also render GT once
    gt_png = os.path.join(OUT_DIR, "gt_orig.png")
    if not os.path.exists(gt_png):
        seed, anim = SAMPLE_MID.split("_", 1)
        gt_p0 = os.path.join(DATA_ROOT, f"precompute/{SAMPLE_CAT}/{seed}/{anim}/part0.obj")
        gt_p1 = os.path.join(DATA_ROOT, f"precompute/{SAMPLE_CAT}/{seed}/{anim}/part1.obj")
        if os.path.exists(gt_p0):
            script_path = "/tmp/_eval_lamp_render.py"
            with open(script_path, 'w') as f:
                f.write(RENDER_SCRIPT)
            subprocess.run([BLENDER, "--background", "--python", script_path, "--",
                            gt_p0, gt_p1, gt_png], capture_output=True, timeout=120)
            print(f"  GT render → {gt_png}", flush=True)

    evaluated = set()
    print(f"Watching {CKPT_DIR} for new checkpoints...", flush=True)

    while True:
        ckpts = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("step_") and f.endswith(".pt")])
        for f in ckpts:
            if f in evaluated: continue
            step = int(f.replace("step_", "").replace(".pt", ""))
            if step < 15000:
                evaluated.add(f)
                continue  # skip old checkpoints
            path = os.path.join(CKPT_DIR, f)
            time.sleep(5)
            try:
                print(f"\nEval checkpoint: {f}", flush=True)
                eval_checkpoint(path, step, device)
                evaluated.add(f)
            except Exception as e:
                print(f"  Error: {e}", flush=True)
                import traceback; traceback.print_exc()

        latest_step = max((int(f.replace("step_","").replace(".pt","")) for f in evaluated if f.startswith("step_")), default=0)
        if latest_step >= 38000:
            print("Training complete, exiting.", flush=True)
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
