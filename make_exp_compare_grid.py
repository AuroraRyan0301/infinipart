#!/usr/bin/env python3
"""Build 3-col labeled grid: Exp1 pred | Exp2 pred | GT, at same step."""
import os, subprocess
from PIL import Image, ImageDraw, ImageFont

BLENDER = "/mnt/cpfs/yurh/blender-4.2.18-linux-x64/blender"
PRECOMPUTE = "/mnt/data/yurh/Infinigen-Sim/data_ssd/precompute_solidified"

import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument("--step", type=int, default=20000)
_args, _ = _p.parse_known_args()
_step = _args.step

EXP1_DIR = f"/mnt/cpfs/yurh/Infinigen-Sim/output/infer_exp1_{_step//1000}k_same/step_{_step}"
EXP2_DIR = f"/mnt/cpfs/yurh/Infinigen-Sim/output/infer_exp2_{_step//1000}k/step_{_step}"
OUT_DIR  = f"/mnt/cpfs/yurh/Infinigen-Sim/output/compare_exp1_vs_exp2_step{_step//1000}k"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = [
    ("faucet_9_senior_0",       "Faucet (9/senior_0)"),
    ("dishwasher_6_senior_2",   "Dishwasher (6/senior_2)"),
    ("cabinet_9_basic_0",       "Cabinet (9/basic_0)"),
    ("plier_9_basic_0",         "Plier (9/basic_0)"),
    ("drawer_7_custom_1_flip",  "Drawer (7/custom_1_flip)"),
    ("window_1_basic_6",        "Window (1/basic_6)"),
    ("lamp_1_basic_1",          "Lamp (1/basic_1)"),
    ("trash_3_senior_0",        "Trash (3/senior_0)"),
]

BLENDER_SCRIPT = r'''
import bpy, sys, os, math, mathutils
argv = sys.argv[sys.argv.index("--") + 1:]
p0_obj, p1_obj, output_png = argv[0], argv[1], argv[2]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
for c in bpy.data.collections: bpy.data.collections.remove(c)
def import_obj(path, color, name):
    bpy.ops.wm.obj_import(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.name = name
    mat = bpy.data.materials.new(name=f"mat_{name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Metallic"].default_value = 0.1
    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)
    return obj
o0 = import_obj(p0_obj, (0.15, 0.35, 0.85, 1.0), "part0")
o1 = import_obj(p1_obj, (0.95, 0.55, 0.1, 1.0), "part1")
all_verts = []
for obj in [o0, o1]:
    for v in obj.data.vertices: all_verts.append(obj.matrix_world @ v.co)
if all_verts:
    bbox_min = mathutils.Vector((min(v.x for v in all_verts), min(v.y for v in all_verts), min(v.z for v in all_verts)))
    bbox_max = mathutils.Vector((max(v.x for v in all_verts), max(v.y for v in all_verts), max(v.z for v in all_verts)))
    center = (bbox_min + bbox_max) / 2
    size = max((bbox_max - bbox_min).x, (bbox_max - bbox_min).y, (bbox_max - bbox_min).z)
    scale = 2.0 / size if size > 0 else 1.0
    for obj in [o0, o1]:
        obj.location -= center
        obj.scale *= scale
cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 50
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam
cam.location = (2.5, -2.5, 2.0)
cam.rotation_euler = (math.radians(60), 0, math.radians(45))
light_data = bpy.data.lights.new("Light", type='SUN')
light_data.energy = 3
light = bpy.data.objects.new("Light", light_data)
bpy.context.scene.collection.objects.link(light)
light.location = (3, -3, 5)
light.rotation_euler = (math.radians(30), math.radians(15), 0)
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.95, 0.95, 0.95, 1.0)
bg.inputs["Strength"].default_value = 0.5
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.samples = 64
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.film_transparent = False
scene.render.filepath = output_png
scene.render.image_settings.file_format = 'PNG'
bpy.ops.render.render(write_still=True)
'''

def render(p0, p1, png):
    if os.path.isfile(png): return True
    if not (os.path.isfile(p0) and os.path.isfile(p1)): return False
    sp = "/tmp/pp_render_cmp.py"
    with open(sp, "w") as f: f.write(BLENDER_SCRIPT)
    subprocess.run([BLENDER, "--background", "--python", sp, "--", p0, p1, png],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    return os.path.isfile(png)


def get_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"]:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()


def label(img, text, size=36):
    font = get_font(size)
    w, h = img.size
    lh = size + 20
    new = Image.new("RGB", (w, h + lh), (255, 255, 255))
    new.paste(img, (0, 0))
    d = ImageDraw.Draw(new)
    bb = d.textbbox((0, 0), text, font=font)
    d.text(((w - (bb[2]-bb[0]))//2, h + 5), text, fill=(0,0,0), font=font)
    return new


def main():
    rows = []
    for tag, display in SAMPLES:
        parts = tag.split("_")
        cat = parts[0]
        seed = parts[1]
        animode = "_".join(parts[2:])
        print(f"  {display}")

        # Exp1 pred
        e1_p0 = os.path.join(EXP1_DIR, f"{tag}_pred_p0.obj")
        e1_p1 = os.path.join(EXP1_DIR, f"{tag}_pred_p1.obj")
        e1_png = os.path.join(OUT_DIR, f"{tag}_exp1.png")
        render(e1_p0, e1_p1, e1_png)

        # Exp2 pred
        e2_p0 = os.path.join(EXP2_DIR, f"{tag}_pred_p0.obj")
        e2_p1 = os.path.join(EXP2_DIR, f"{tag}_pred_p1.obj")
        e2_png = os.path.join(OUT_DIR, f"{tag}_exp2.png")
        render(e2_p0, e2_p1, e2_png)

        # GT
        gt_p0 = os.path.join(PRECOMPUTE, cat, seed, animode, "part0.obj")
        gt_p1 = os.path.join(PRECOMPUTE, cat, seed, animode, "part1.obj")
        gt_png = os.path.join(OUT_DIR, f"{tag}_gt.png")
        render(gt_p0, gt_p1, gt_png)

        imgs = []
        for png, cap in [(e1_png, "Exp1 (Linear, diff-only)"),
                         (e2_png, "Exp2 (MLP+SelfAttn+QKNorm, orig+diff)"),
                         (gt_png, "Ground Truth")]:
            if os.path.isfile(png):
                img = Image.open(png)
                imgs.append(label(img, cap, size=28))
            else:
                imgs.append(label(Image.new("RGB", (512, 512), (200,200,200)), cap+" N/A", size=28))
        rows.append((display, imgs))

    # Assemble grid: rows=samples, cols=[Exp1, Exp2, GT] + row label
    if not rows: return
    cell_w, cell_h = rows[0][1][0].size
    row_lbl_w = 280
    pad = 4
    title_h = 60
    total_w = row_lbl_w + 3 * (cell_w + pad) + pad
    total_h = title_h + len(rows) * (cell_h + pad) + pad

    grid = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    d = ImageDraw.Draw(grid)
    tfont = get_font(40)
    lfont = get_font(32)

    title = f"Exp1 vs Exp2 @ Step {_step//1000}k (same samples, cfg=5.0)"
    bb = d.textbbox((0, 0), title, font=tfont)
    d.text(((total_w - (bb[2]-bb[0]))//2, 8), title, fill=(0,0,0), font=tfont)

    for ri, (display, imgs) in enumerate(rows):
        y = title_h + pad + ri * (cell_h + pad)
        bb = d.textbbox((0, 0), display, font=lfont)
        d.text(((row_lbl_w - (bb[2]-bb[0]))//2, y + (cell_h - (bb[3]-bb[1]))//2),
               display, fill=(0,0,0), font=lfont)
        for ci, img in enumerate(imgs):
            x = row_lbl_w + pad + ci * (cell_w + pad)
            grid.paste(img, (x, y))

    out = os.path.join(OUT_DIR, "compare_grid.png")
    grid.save(out, quality=95)
    print(f"\nSaved: {out} ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
