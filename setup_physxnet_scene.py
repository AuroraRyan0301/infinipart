#!/usr/bin/env python3
"""
Set up PhysXNet/PhysX_mobility objects for render_articulation.py.

Creates the directory structure expected by render_articulation.py:
  outputs/{Factory}/{id}/scene.urdf
  outputs/{Factory}/{id}/origins.json
  outputs/{Factory}/{id}/outputs/{Factory}/{id}/objs/{group_idx}/{group_idx}.obj

IMPORTANT: The OBJ files are indexed by GROUP index (matching link names l_0, l_1, ...),
NOT by raw part labels. For multi-part groups the OBJ files are merged.

Usage:
  python setup_physxnet_scene.py --id 10005 --factory ElectronicsPhysXNetFactory
  python setup_physxnet_scene.py --id 10005 --factory ElectronicsPhysXNetFactory --source physx_mobility
"""
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(_SCRIPT_DIR))

PHYSXNET_BASE = os.environ.get("PHYSXNET_BASE", os.path.join(_DATA_DIR, "PhysXNet/version_1"))
PHYSX_MOBILITY_BASE = os.environ.get("PHYSXMOB_BASE", os.path.join(_DATA_DIR, "PhysX_mobility"))
INFINIGEN_SIM_BASE = _SCRIPT_DIR


def merge_obj_files(src_paths, dst_path):
    """Merge multiple OBJ files into one, adjusting face vertex indices."""
    vertex_offset = 0
    vt_offset = 0
    vn_offset = 0
    lines_out = []

    for src in src_paths:
        if not os.path.exists(src):
            continue
        v_count = 0
        vt_count = 0
        vn_count = 0
        face_lines = []
        other_lines = []

        with open(src) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("v "):
                    v_count += 1
                    other_lines.append(line)
                elif stripped.startswith("vt "):
                    vt_count += 1
                    other_lines.append(line)
                elif stripped.startswith("vn "):
                    vn_count += 1
                    other_lines.append(line)
                elif stripped.startswith("f "):
                    face_lines.append(stripped)
                elif stripped.startswith("mtllib") or stripped.startswith("usemtl") or stripped.startswith("o ") or stripped.startswith("g "):
                    other_lines.append(line)
                else:
                    other_lines.append(line)

        # Write vertices/normals/texcoords
        lines_out.extend(other_lines)

        # Write faces with adjusted indices
        for fline in face_lines:
            parts = fline.split()
            new_parts = [parts[0]]  # "f"
            for vert in parts[1:]:
                indices = vert.split("/")
                new_indices = []
                for i, idx_str in enumerate(indices):
                    if idx_str == "":
                        new_indices.append("")
                    else:
                        try:
                            idx_val = int(idx_str)
                            if i == 0:
                                new_indices.append(str(idx_val + vertex_offset))
                            elif i == 1:
                                new_indices.append(str(idx_val + vt_offset))
                            elif i == 2:
                                new_indices.append(str(idx_val + vn_offset))
                            else:
                                new_indices.append(idx_str)
                        except ValueError:
                            new_indices.append(idx_str)
                new_parts.append("/".join(new_indices))
            lines_out.append(" ".join(new_parts) + "\n")

        vertex_offset += v_count
        vt_offset += vt_count
        vn_offset += vn_count

    with open(dst_path, "w") as f:
        f.writelines(lines_out)


def setup_scene(object_id, factory_name, source="physxnet", base_dir=INFINIGEN_SIM_BASE):
    if source == "physxnet":
        src_base = PHYSXNET_BASE
        urdf_path = os.path.join(src_base, "urdf", f"{object_id}.urdf")
        partseg_dir = os.path.join(src_base, "partseg", str(object_id), "objs")
    else:
        src_base = PHYSX_MOBILITY_BASE
        urdf_path = os.path.join(src_base, "urdf", f"{object_id}.urdf")
        partseg_dir = os.path.join(src_base, "partseg", str(object_id), "objs")

    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found: {urdf_path}")
        return False

    scene_dir = os.path.join(base_dir, "outputs", factory_name, str(object_id))
    objs_dir = os.path.join(scene_dir, "outputs", factory_name, str(object_id), "objs")
    os.makedirs(scene_dir, exist_ok=True)
    os.makedirs(objs_dir, exist_ok=True)

    # Parse URDF and fix mesh paths to absolute
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Build link_name -> [abs_mesh_paths] mapping (BEFORE renaming)
    # Also collect group indices from link names like l_gN -> group N
    link_meshes = {}  # link_name (original) -> [abs_mesh_paths]
    for link_el in root.findall("link"):
        name = link_el.get("name")
        meshes = []
        for visual in link_el.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh_el = geom.find("mesh")
            if mesh_el is None:
                continue
            fn = mesh_el.get("filename")
            if not fn:
                continue

            # Resolve relative path to absolute
            if not os.path.isabs(fn):
                abs_fn = os.path.normpath(os.path.join(os.path.dirname(urdf_path), fn))
            else:
                abs_fn = fn
            mesh_el.set("filename", abs_fn)
            meshes.append(abs_fn)

        if meshes:
            link_meshes[name] = meshes

        # Rewrite link name: strip _gN -> _N uniformly
        # e.g. l_g0 -> l_0, abstract_g0_g1 -> abstract_0_1
        if name:
            new_name = re.sub(r'_g(\d)', r'_\1', name)
            if new_name != name:
                link_el.set("name", new_name)

    # Also fix joint parent/child references and joint names
    for joint_el in root.findall("joint"):
        for tag in ["parent", "child"]:
            el = joint_el.find(tag)
            if el is not None:
                lnk = el.get("link", "")
                new_lnk = re.sub(r'_g(\d)', r'_\1', lnk)
                if new_lnk != lnk:
                    el.set("link", new_lnk)
        # Fix joint names
        name = joint_el.get("name", "")
        new_name = re.sub(r'_g(\d)', r'_\1', name)
        if new_name != name:
            joint_el.set("name", new_name)

    # Write fixed URDF
    out_urdf = os.path.join(scene_dir, "scene.urdf")
    tree.write(out_urdf, xml_declaration=True, encoding='utf-8')

    # Build group_index -> [mesh_files] from link names
    # l_gN or l_N links map to group index N
    group_meshes = {}  # group_idx (int) -> [abs_mesh_paths]
    for orig_name, meshes in link_meshes.items():
        # Extract group index from original link name: l_g0 -> 0, l_g1 -> 1
        m = re.match(r'^l_g?(\d+)$', orig_name)
        if m:
            gidx = int(m.group(1))
            group_meshes[gidx] = meshes

    # Create origins.json with GROUP indices (not raw part labels)
    origins = {}
    for gidx in sorted(group_meshes.keys()):
        origins[str(gidx)] = [0.0, 0.0, 0.0]
    with open(os.path.join(scene_dir, "origins.json"), "w") as f:
        json.dump(origins, f, indent=2)

    # Create OBJ files indexed by group: objs/{gidx}/{gidx}.obj
    # For PhysX_mobility: also copy MTL files so Blender can read materials
    obj_count = 0
    for gidx, meshes in sorted(group_meshes.items()):
        gidx_dir = os.path.join(objs_dir, str(gidx))
        os.makedirs(gidx_dir, exist_ok=True)
        dst = os.path.join(gidx_dir, f"{gidx}.obj")

        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)

        if len(meshes) == 1:
            # Single mesh: symlink
            os.symlink(meshes[0], dst)
            # Also symlink MTL file if it exists (PhysX_mobility has MTLs)
            mtl_src = meshes[0].replace('.obj', '.mtl')
            if os.path.exists(mtl_src):
                # Read OBJ to find the mtllib name Blender will look for
                mtl_name = None
                with open(meshes[0]) as f:
                    for line in f:
                        if line.strip().startswith('mtllib '):
                            mtl_name = line.strip().split(None, 1)[1]
                            break
                if mtl_name:
                    mtl_dst = os.path.join(gidx_dir, mtl_name)
                    if os.path.exists(mtl_dst) or os.path.islink(mtl_dst):
                        os.remove(mtl_dst)
                    os.symlink(mtl_src, mtl_dst)
        else:
            # Multiple meshes: merge OBJ files
            merge_obj_files(meshes, dst)
            # Also merge MTL files if they exist (PhysX_mobility)
            mtl_srcs = [m.replace('.obj', '.mtl') for m in meshes if os.path.exists(m.replace('.obj', '.mtl'))]
            if mtl_srcs:
                # Collect all mtllib names referenced in merged OBJ
                mtl_names = set()
                for m in meshes:
                    if os.path.exists(m):
                        with open(m) as f:
                            for line in f:
                                if line.strip().startswith('mtllib '):
                                    mtl_names.add(line.strip().split(None, 1)[1])
                # Merge all MTL content into each referenced name
                merged_mtl = []
                for ms in mtl_srcs:
                    with open(ms) as f:
                        merged_mtl.append(f.read())
                merged_content = "\n".join(merged_mtl)
                for mn in mtl_names:
                    mtl_dst = os.path.join(gidx_dir, mn)
                    with open(mtl_dst, 'w') as f:
                        f.write(merged_content)

        obj_count += 1

    print(f"OK: {factory_name}/{object_id} -> {scene_dir}")
    print(f"    URDF: {out_urdf}")
    print(f"    Groups: {len(origins)} (origins.json)")
    print(f"    OBJs: {obj_count} files")
    for gidx, meshes in sorted(group_meshes.items()):
        mesh_names = [os.path.basename(m) for m in meshes]
        print(f"      l_{gidx} -> {mesh_names}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--factory", type=str, required=True)
    parser.add_argument("--source", choices=["physxnet", "physx_mobility"], default="physxnet")
    parser.add_argument("--base_dir", default=INFINIGEN_SIM_BASE)
    args = parser.parse_args()

    setup_scene(args.id, args.factory, source=args.source, base_dir=args.base_dir)
