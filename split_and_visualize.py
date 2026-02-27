#!/usr/bin/env python3
"""
Split Infinite Mobility generated objects into 2 parts (body + moving)
following PartPacker preprocessing conventions.

For each factory type, we know which parts are the "moving" group based on
the dominant articulation joint (largest motion range).

Produces:
  - part0.obj (body/static)
  - part1.obj (moving)
  - split_vis.png (visualization: blue=body, red=moving)

Usage:
  python split_and_visualize.py --output_root outputs
  python split_and_visualize.py --output_root outputs --factory DishwasherFactory
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Factory-specific part grouping rules.
#
# For each factory, "moving_parts" lists part_names that belong to the
# dominant movable joint subtree (part1). Everything else is body (part0).
#
# These are derived from reading each factory's joint definitions:
#   - DishwasherFactory: door=revolute(π/2), handle=fixed-to-door
#   - BeverageFridgeFactory: door=revolute(π/2)
#   - MicrowaveFactory: door=revolute(π/2)
#   - OvenFactory: door=revolute(π/2)
#   - ToiletFactory: toilet_cover=revolute(π/2), toilet_seat=revolute(π/2)
#   - KitchenCabinetFactory: door=revolute or drawer=prismatic
#   - WindowFactory: sash/pane=prismatic
#   - LiteDoorFactory: panel=revolute
#   - OfficeChairFactory: chair_back=revolute(π/10)
#   - BarChairFactory: seat_cap=prismatic (height adjust)
#   - TapFactory: spout=revolute, handle=revolute
#   - LampFactory: shade parts
#   - PotFactory: lid=continuous_prismatic
#   - PanFactory: handle=prismatic (minor)
#   - BottleFactory: cap=continuous_prismatic
#   - TVFactory: legs=revolute_prismatic or fixed
# ---------------------------------------------------------------------------

FACTORY_RULES = {
    # Appliances - door is dominant revolute joint
    "DishwasherFactory": {
        "moving_parts": {"door_part", "handle_part"},
        "description": "door (revolute π/2) + handle (fixed to door)",
    },
    "BeverageFridgeFactory": {
        "moving_parts": {"door_part"},
        "description": "door (revolute π/2)",
    },
    "MicrowaveFactory": {
        "moving_parts": {"door_part"},
        "description": "door (revolute π/2)",
    },
    "OvenFactory": {
        "moving_parts": {"door_part"},
        "description": "door (revolute π/2)",
    },
    # Bathroom
    "ToiletFactory": {
        # Both seat and cover have revolute π/2, pick cover as dominant
        "moving_parts": {"toilet_cover_part", "toilet_seat_part"},
        "description": "toilet cover + seat (revolute π/2)",
    },
    # Shelves
    "KitchenCabinetFactory": {
        "moving_parts": {"door_part", "drawer_part"},
        "description": "door/drawer (revolute or prismatic)",
    },
    # Elements
    "WindowFactory": {
        "moving_parts": {"sash_part", "pane_part"},
        "description": "sash (prismatic)",
    },
    "LiteDoorFactory": {
        "moving_parts": {"door_panel_part", "door_knob_part", "panel_part", "lite_part", "door_part"},
        "description": "door panel + knobs (revolute)",
    },
    # Seating - OfficeChair: seat is on prismatic lift, back tilts
    # leg_wheeled_upper_part = gas lift piston, moves with seat
    # unknown_part = full composite mesh (overlaps everything), must be excluded
    "OfficeChairFactory": {
        "moving_parts": {"seat_part", "chair_back_part", "back_part", "leg_wheeled_upper_part"},
        "exclude_parts": {"unknown_part"},
        "description": "seat + gas lift (prismatic lift)",
    },
    "BarChairFactory": {
        "moving_parts": {"seat_cap_part", "seat_part"},
        "description": "seat (prismatic height adjust)",
    },
    # Decorations / Tableware
    "TapFactory": {
        # actual part names: tap_N_part, handle_N_part
        "moving_parts": {"handle_2_part", "handle_3_part", "handle_part", "spout_part"},
        "description": "handle (revolute)",
    },
    "LampFactory": {
        # actual: bulb, bulb_rack, lamp_head, lamp_N, lamp_leg, lamp_support, pin
        "moving_parts": {"lamp_head_part", "bulb_part", "shade_part"},
        "description": "lamp head / shade",
    },
    "PotFactory": {
        "moving_parts": {"lid_part"},
        "description": "lid (continuous prismatic)",
    },
    "PanFactory": {
        # Pan has minimal articulation; handle is prismatic but small range
        "moving_parts": {"handle_part"},
        "description": "handle (prismatic, minor)",
    },
    "BottleFactory": {
        "moving_parts": {"cap_part"},
        "description": "cap (continuous prismatic)",
    },
    "TVFactory": {
        "moving_parts": {"legs_part", "stand_part"},
        "description": "stand/legs (revolute_prismatic or fixed)",
    },
    # Fixed-only objects (no meaningful articulation)
    "VaseFactory": {
        "moving_parts": set(),
        "description": "no movable joints",
    },
    "TableCocktailFactory": {
        "moving_parts": set(),
        "description": "no movable joints",
    },
    "TableDiningFactory": {
        "moving_parts": set(),
        "description": "no movable joints",
    },
    "PlateOnRackBaseFactory": {
        "moving_parts": set(),
        "description": "no movable joints",
    },
}

# Merge PartNet-Mobility factory rules
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
try:
    from partnet_factory_rules import merge_split_rules
    FACTORY_RULES = merge_split_rules(FACTORY_RULES)
except ImportError:
    pass

# Merge PhysXNet + PhysX_mobility split rules
try:
    from physxnet_factory_rules import merge_split_rules as physxnet_merge_split_rules
    FACTORY_RULES = physxnet_merge_split_rules(FACTORY_RULES)
except ImportError:
    pass


def parse_urdf_moving_indices(urdf_path):
    """Parse URDF to find which part indices are descendants of movable joints.

    Returns a set of integer indices (e.g. {0, 1, 3, 4}) for parts under
    revolute or prismatic joints.
    """
    if not os.path.exists(urdf_path):
        return None

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Build parent→child map and joint types
    # joint: parent_link → child_link, type
    children = {}  # link_name → list of (child_link_name, joint_type)
    for joint in root.findall("joint"):
        jtype = joint.get("type", "fixed")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        children.setdefault(parent, []).append((child, jtype))

    # Find all links that are descendants of a revolute/prismatic joint
    moving_links = set()

    def bfs_descendants(start_link):
        """BFS to find all descendant links."""
        queue = [start_link]
        visited = set()
        while queue:
            link = queue.pop(0)
            if link in visited:
                continue
            visited.add(link)
            for child, _ in children.get(link, []):
                queue.append(child)
        return visited

    # Find revolute/prismatic joints from world
    for parent_link, child_list in children.items():
        for child_link, jtype in child_list:
            if jtype in ("revolute", "prismatic", "continuous"):
                # All descendants of this child are moving
                descendants = bfs_descendants(child_link)
                moving_links.update(descendants)

    # Extract part indices from link names (l_0, l_1, etc.)
    moving_indices = set()
    for link_name in moving_links:
        m = re.match(r"l_(\d+)", link_name)
        if m:
            moving_indices.add(int(m.group(1)))

    return moving_indices if moving_indices else None


def load_obj_mesh(obj_dir, file_name, base_path=None, file_obj_path=None):
    """Load an OBJ mesh from the per-part directory."""
    idx = os.path.splitext(file_name)[0]  # e.g. "0" from "0.obj"

    # Try multiple possible paths
    candidates = [
        os.path.join(obj_dir, idx, file_name),
        os.path.join(obj_dir, file_name),
    ]
    # Also try the file_obj_path relative to base_path
    if file_obj_path and base_path:
        candidates.append(os.path.join(base_path, file_obj_path))
    # Some factories nest outputs inside outputs/Factory/seed/outputs/Factory/seed/objs/
    if obj_dir and base_path:
        # Find any matching OBJ by walking up
        parent = os.path.dirname(obj_dir)  # seed dir
        nested = os.path.join(parent, "outputs")
        if os.path.isdir(nested):
            import glob
            pattern = os.path.join(nested, "**", "objs", idx, file_name)
            matches = glob.glob(pattern, recursive=True)
            candidates.extend(matches)

    obj_path = None
    for c in candidates:
        if os.path.exists(c):
            obj_path = c
            break

    if obj_path is None:
        print(f"  [WARN] OBJ not found: tried {candidates[0]}")
        return None
    try:
        mesh = trimesh.load(obj_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()
        return mesh
    except Exception as e:
        print(f"  [WARN] Failed to load {obj_path}: {e}")
        return None


def box_normalize_parts(part0, part1, bound=0.95):
    """Box normalize both parts jointly to [-bound, bound]."""
    all_verts = np.concatenate([part0.vertices, part1.vertices], axis=0)
    bmin = all_verts.min(axis=0)
    bmax = all_verts.max(axis=0)
    bcenter = (bmax + bmin) / 2
    extent = (bmax - bmin).max()
    if extent < 1e-8:
        return part0, part1
    scale = 2 * bound / extent
    part0.vertices = (part0.vertices - bcenter) * scale
    part1.vertices = (part1.vertices - bcenter) * scale
    return part0, part1


def render_split_visualization(part0, part1, out_path, resolution=(1024, 1024)):
    """Render a multi-view visualization using matplotlib (headless-compatible)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(20, 10))

    # Subsample faces for rendering (full mesh is too slow for mpl)
    max_faces = 5000

    def subsample_mesh(mesh, max_f):
        if mesh.faces.shape[0] <= max_f:
            return mesh.vertices, mesh.faces
        idx = np.random.choice(mesh.faces.shape[0], max_f, replace=False)
        return mesh.vertices, mesh.faces[idx]

    v0, f0 = subsample_mesh(part0, max_faces)
    v1, f1 = subsample_mesh(part1, max_faces)

    angles = [(30, 45), (30, 135), (30, 225), (30, 315)]
    titles = ['Front-Right', 'Front-Left', 'Back-Left', 'Back-Right']

    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 4, idx + 1, projection='3d')

        # Draw part0 (blue)
        polys0 = v0[f0]
        pc0 = Poly3DCollection(polys0, alpha=0.6, linewidths=0.1, edgecolors='navy')
        pc0.set_facecolor((0.3, 0.5, 0.85, 0.6))
        ax.add_collection3d(pc0)

        # Draw part1 (red)
        polys1 = v1[f1]
        pc1 = Poly3DCollection(polys1, alpha=0.6, linewidths=0.1, edgecolors='darkred')
        pc1.set_facecolor((0.85, 0.3, 0.3, 0.6))
        ax.add_collection3d(pc1)

        # Set limits
        all_v = np.concatenate([v0, v1])
        vmin, vmax = all_v.min(0), all_v.max(0)
        ax.set_xlim(vmin[0], vmax[0])
        ax.set_ylim(vmin[1], vmax[1])
        ax.set_zlim(vmin[2], vmax[2])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(titles[idx], fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    fig.suptitle('Blue=Body(part0)  Red=Moving(part1)', fontsize=14, y=0.98)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return True


def render_multiview(part0, part1, out_path, n_views=4, resolution=(512, 512)):
    """Render multiple views (alias for the main vis function)."""
    # Already handled by render_split_visualization which does 4 views
    return False


def process_factory(factory_name, output_root, base_path):
    """Process all instances of a factory type."""
    factory_dir = os.path.join(base_path, output_root, factory_name)
    if not os.path.exists(factory_dir):
        return []

    rules = FACTORY_RULES.get(factory_name)
    if rules is None:
        print(f"[WARN] No rules for {factory_name}, using fallback heuristic")
        rules = {"moving_parts": {"door_part", "lid_part", "cap_part"}, "description": "fallback"}

    moving_set = rules["moving_parts"]
    # unknown_part is ALWAYS a composite full-object mesh (confirmed: ratio≈1.0
    # for all factories). Exclude it globally to avoid doubling geometry.
    exclude_set = {"unknown_part"} | rules.get("exclude_parts", set())
    if not moving_set:
        print(f"[SKIP] {factory_name}: no movable joints defined")
        return []

    results = []

    # Find all data_infos files
    info_files = sorted([f for f in os.listdir(factory_dir) if f.startswith("data_infos_") and f.endswith(".json")])

    for info_file in info_files:
        seed = info_file.replace("data_infos_", "").replace(".json", "")
        info_path = os.path.join(factory_dir, info_file)

        with open(info_path) as f:
            data = json.load(f)

        # Process each instance in the data_infos
        for inst_idx, instance in enumerate(data):
            parts = instance["part"]
            obj_dir = os.path.join(factory_dir, seed, "objs")

            print(f"\n--- {factory_name} seed={seed} instance={inst_idx} ---")
            print(f"  Rule: {rules['description']}")
            print(f"  Moving parts: {moving_set}")

            # Load per-part origins (centroids before centering)
            origins_path = os.path.join(factory_dir, seed, "origins.json")
            origins = {}
            if os.path.exists(origins_path):
                with open(origins_path) as f:
                    origins = json.load(f)
                print(f"  [ORIGINS] Loaded {len(origins)} offsets from origins.json")
            else:
                print(f"  [WARN] No origins.json, parts will overlap at origin!")

            # Group parts
            body_meshes = []
            moving_meshes = []
            body_names = []
            moving_names = []

            # First pass: name-based grouping
            all_loaded = []  # (pname, fname, file_idx, mesh)
            for part in parts:
                pname = part["part_name"]
                fname = part["file_name"]
                fpath = part.get("file_obj_path")
                file_idx = int(os.path.splitext(fname)[0])
                mesh = load_obj_mesh(obj_dir, fname, base_path=base_path, file_obj_path=fpath)
                if mesh is None:
                    continue
                # Apply per-part origin offset to restore world-space positions
                offset = origins.get(str(file_idx))
                if offset is not None:
                    mesh.vertices += np.array(offset)
                all_loaded.append((pname, fname, file_idx, mesh))

            # Try name-based matching
            name_matched_moving = any(pname in moving_set for pname, _, _, _ in all_loaded)

            # URDF fallback: only when name matching finds zero moving parts
            urdf_moving_indices = None
            if not name_matched_moving:
                urdf_path = os.path.join(factory_dir, seed, "scene.urdf")
                urdf_moving_indices = parse_urdf_moving_indices(urdf_path)
                if urdf_moving_indices:
                    print(f"  [URDF fallback] Moving indices from URDF: {sorted(urdf_moving_indices)}")

            excluded_meshes = []
            excluded_names = []
            for pname, fname, file_idx, mesh in all_loaded:
                # Skip excluded parts (e.g. composite meshes)
                if pname in exclude_set:
                    excluded_meshes.append(mesh)
                    excluded_names.append(f"{pname}({fname})")
                    continue

                is_moving = pname in moving_set
                if not is_moving and urdf_moving_indices is not None and file_idx in urdf_moving_indices:
                    is_moving = True

                if is_moving:
                    moving_meshes.append(mesh)
                    moving_names.append(f"{pname}({fname})")
                else:
                    body_meshes.append(mesh)
                    body_names.append(f"{pname}({fname})")

            # If body is empty after exclusion, restore excluded parts as body
            if not body_meshes and excluded_meshes:
                print(f"  [RESTORE] Body empty, restoring excluded parts as body")
                body_meshes = excluded_meshes
                body_names = excluded_names
            elif excluded_names:
                print(f"  [EXCLUDE] {excluded_names}")

            print(f"  part0 (body):   {body_names}")
            print(f"  part1 (moving): {moving_names}")

            if not body_meshes:
                print("  [ERROR] No body meshes")
                continue
            if not moving_meshes:
                print("  [ERROR] No moving meshes")
                continue

            # Merge
            part0 = trimesh.util.concatenate(body_meshes) if len(body_meshes) > 1 else body_meshes[0]
            part1 = trimesh.util.concatenate(moving_meshes) if len(moving_meshes) > 1 else moving_meshes[0]

            print(f"  part0: {part0.vertices.shape[0]} verts, {part0.faces.shape[0]} faces")
            print(f"  part1: {part1.vertices.shape[0]} verts, {part1.faces.shape[0]} faces")

            # Box normalize
            part0, part1 = box_normalize_parts(part0, part1)
            all_v = np.concatenate([part0.vertices, part1.vertices])
            print(f"  Normalized range: [{all_v.min():.3f}, {all_v.max():.3f}]")

            # Save
            out_dir = os.path.join(factory_dir, seed, f"split_{inst_idx}")
            os.makedirs(out_dir, exist_ok=True)

            out0 = os.path.join(out_dir, "part0.obj")
            out1 = os.path.join(out_dir, "part1.obj")

            # Clear visual before export to avoid texture issues
            part0_export = part0.copy()
            part1_export = part1.copy()
            part0_export.visual = trimesh.visual.ColorVisuals()
            part1_export.visual = trimesh.visual.ColorVisuals()
            part0_export.export(out0)
            part1_export.export(out1)
            print(f"  Saved: {out0}")
            print(f"         {out1}")

            # Render visualization
            vis_path = os.path.join(out_dir, "split_vis.png")
            rendered = render_split_visualization(part0, part1, vis_path)
            if rendered:
                print(f"  Vis:   {vis_path}")

            # Render multi-view
            mv_path = os.path.join(out_dir, "split_multiview.png")
            rendered_mv = render_multiview(part0, part1, mv_path, n_views=4)
            if rendered_mv:
                print(f"  Multiview: {mv_path}")

            results.append({
                "factory": factory_name,
                "seed": seed,
                "instance": inst_idx,
                "part0_names": body_names,
                "part1_names": moving_names,
                "part0_verts": part0.vertices.shape[0],
                "part1_verts": part1.vertices.shape[0],
                "out_dir": out_dir,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs",
                        help="Root output directory (relative to base_path)")
    parser.add_argument("--base_path", default="/mnt/data/yurh/Infinite-Mobility",
                        help="Base path of Infinite Mobility repo")
    parser.add_argument("--factory", default=None,
                        help="Process only this factory type")
    args = parser.parse_args()

    base = args.base_path
    root = os.path.join(base, args.output_root)

    if not os.path.exists(root):
        print(f"[ERROR] Output root not found: {root}")
        sys.exit(1)

    # Find all factory directories
    if args.factory:
        factories = [args.factory]
    else:
        factories = sorted([d for d in os.listdir(root)
                           if os.path.isdir(os.path.join(root, d)) and d.endswith("Factory")])

    print(f"=== Infinite Mobility 2-Part Splitting ===")
    print(f"Base: {base}")
    print(f"Factories: {factories}")
    print()

    all_results = []
    for factory in factories:
        results = process_factory(factory, args.output_root, base)
        all_results.extend(results)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total instances processed: {len(all_results)}")
    for r in all_results:
        print(f"  {r['factory']} seed={r['seed']}: "
              f"part0={r['part0_verts']}v part1={r['part1_verts']}v")
        print(f"    body:   {r['part0_names']}")
        print(f"    moving: {r['part1_names']}")

    # Save summary JSON
    summary_path = os.path.join(root, "split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
