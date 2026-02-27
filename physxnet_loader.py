"""
PhysXNet and PhysX_mobility data loader for Blender rendering pipeline.

Loads objects from PhysXNet/PhysX_mobility datasets, generating
the necessary data structures (origins.json, scene.urdf, OBJ paths)
that render_articulation.py expects.

Key differences from PartNet-Mobility / Infinite-Mobility:
  - PhysXNet OBJs: partseg/{id}/objs/{label}.obj (integer labels)
  - PhysX_mobility OBJs: partseg/{id}/objs/original-{N}.obj (with MTLs)
  - PhysX_mobility URDFs: urdf/{id}.urdf (ready to use)
  - PhysXNet: no URDFs yet; we generate inline from group_info JSON

Data is referenced by absolute path. Nothing is copied.

Material variant logic (seed > N):
  For a factory with N objects, seeds >= N trigger material variants.
  The base object is objects[seed % N]. For each part, material properties
  (Blender metallic/roughness/color) are randomly swapped from another
  object of the same category, using the same-name part pool.

Usage from render_articulation.py:
  Call prepare_physxnet_scene(factory_name, seed) to get paths/data needed
  by the existing rendering pipeline.
"""

import json
import math
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict

from physxnet_factory_rules import (
    PHYSXNET_BASE, PHYSXMOB_BASE,
    PHYSXNET_JSON_DIR, PHYSXNET_PARTSEG_DIR,
    PHYSXMOB_JSON_DIR, PHYSXMOB_PARTSEG_DIR, PHYSXMOB_URDF_DIR,
    factory_dataset, seed_to_object_id,
    ALL_MATERIAL_DEFAULTS,
)


# ======================================================================
# JSON loading
# ======================================================================

def load_physxnet_json(obj_id):
    """Load PhysXNet finaljson for an object."""
    path = os.path.join(PHYSXNET_JSON_DIR, f"{obj_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_physxmob_json(obj_id):
    """Load PhysX_mobility finaljson for an object."""
    path = os.path.join(PHYSXMOB_JSON_DIR, f"{obj_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ======================================================================
# OBJ file listing
# ======================================================================

def get_physxnet_obj_paths(obj_id):
    """Get {label_int: abs_path} for PhysXNet object.

    PhysXNet uses integer-named OBJs: 0.obj, 1.obj, ...
    """
    objs_dir = os.path.join(PHYSXNET_PARTSEG_DIR, str(obj_id), "objs")
    if not os.path.isdir(objs_dir):
        return {}
    result = {}
    for fn in os.listdir(objs_dir):
        if fn.endswith('.obj'):
            name = fn[:-4]
            try:
                label = int(name)
                result[label] = os.path.join(objs_dir, fn)
            except ValueError:
                continue
    return result


def get_physxmob_obj_paths(obj_id):
    """Get {original_name: abs_path} for PhysX_mobility object.

    PhysX_mobility uses original-N.obj naming.
    Returns {str: str} where key is like "original-31".
    """
    objs_dir = os.path.join(PHYSXMOB_PARTSEG_DIR, str(obj_id), "objs")
    if not os.path.isdir(objs_dir):
        return {}
    result = {}
    for fn in os.listdir(objs_dir):
        if fn.endswith('.obj'):
            name = fn[:-4]
            result[name] = os.path.join(objs_dir, fn)
    return result


# ======================================================================
# URDF handling for PhysX_mobility
# ======================================================================

def get_physxmob_urdf_path(obj_id):
    """Get absolute URDF path for a PhysX_mobility object."""
    path = os.path.join(PHYSXMOB_URDF_DIR, f"{obj_id}.urdf")
    if os.path.exists(path):
        return path
    return None


def rewrite_physxmob_urdf(obj_id, output_path):
    """Rewrite PhysX_mobility URDF with absolute mesh paths.

    The original URDF uses relative paths like ./../partseg/{id}/objs/original-N.obj.
    We rewrite them as absolute paths for Blender to find.
    Also converts link names to IM convention (l_N).
    """
    src_path = get_physxmob_urdf_path(obj_id)
    if src_path is None:
        return None

    tree = ET.parse(src_path)
    root = tree.getroot()

    # Rewrite mesh filenames to absolute paths
    objs_dir = os.path.join(PHYSXMOB_PARTSEG_DIR, str(obj_id), "objs")
    for mesh_el in root.iter("mesh"):
        filename = mesh_el.get("filename")
        if filename:
            # Extract just the filename part (e.g., original-31.obj)
            basename = os.path.basename(filename)
            abs_path = os.path.join(objs_dir, basename)
            mesh_el.set("filename", abs_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path, xml_declaration=True, encoding='utf-8')
    return output_path


# ======================================================================
# URDF generation from PhysXNet group_info
# ======================================================================

# PhysXNet group_info joint type mapping
JOINT_TYPE_MAP = {
    'A': 'revolute',
    'B': 'prismatic',
    'C': 'continuous',
}


def generate_urdf_from_group_info(obj_id, group_info, parts_info, output_path):
    """Generate a URDF file from PhysXNet group_info.

    group_info format:
      Static group:  "gid": [list_of_part_labels] (last element is just a label too)
      Joint group:   "gid": [[child_part_labels], "parent_gid",
                              [ax,ay,az, ox,oy,oz, lower,upper], "joint_type"]
      joint_type: 'A'=revolute, 'B'=prismatic, 'C'=continuous

    Each part label gets its own link (l_{label}) for compatibility with the
    render pipeline's load_scene_parts() which loads objs/{i}/{i}.obj per part.
    Parts in the same group share the same joint motion (via fixed joints
    from a common abstract link).
    """
    # Classify groups
    static_groups = {}   # gid -> [part_labels]
    joint_groups = {}    # gid -> (child_labels, parent_gid, params, joint_type)

    for gid_str, val in group_info.items():
        gid = int(gid_str)
        if (isinstance(val, list) and len(val) >= 4
                and isinstance(val[-1], str) and val[-1] in JOINT_TYPE_MAP):
            child_labels = val[0] if isinstance(val[0], list) else [val[0]]
            parent_gid = int(val[1])
            params = val[2]  # [ax, ay, az, ox, oy, oz, lower, upper]
            jtype = JOINT_TYPE_MAP[val[-1]]
            joint_groups[gid] = (child_labels, parent_gid, params, jtype)
        else:
            # Static group: all elements are part labels
            if isinstance(val, list):
                static_groups[gid] = [int(x) for x in val if isinstance(x, (int, float))]
            else:
                static_groups[gid] = [int(val)]

    # Get OBJ paths
    obj_paths = get_physxnet_obj_paths(obj_id)

    # Build URDF XML
    robot = ET.Element("robot", name="scene")

    # Create world link (base, no visual)
    world_link = ET.SubElement(robot, "link", name="l_world")
    inertial = ET.SubElement(world_link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="1.0")
    ET.SubElement(inertial, "inertia", ixx="1.0", ixy="0.0", ixz="0.0",
                  iyy="1.0", iyz="0.0", izz="1.0")

    def make_part_link(label):
        """Create a link for a single part label with its OBJ visual."""
        link_name = f"l_{label}"
        link_el = ET.SubElement(robot, "link", name=link_name)
        link_inertial = ET.SubElement(link_el, "inertial")
        ET.SubElement(link_inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(link_inertial, "mass", value="1.0")
        ET.SubElement(link_inertial, "inertia", ixx="1.0", ixy="0.0", ixz="0.0",
                      iyy="1.0", iyz="0.0", izz="1.0")
        if label in obj_paths:
            visual = ET.SubElement(link_el, "visual")
            geom = ET.SubElement(visual, "geometry")
            ET.SubElement(geom, "mesh", filename=obj_paths[label],
                          scale="1 1 1")
            ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        return link_name

    # Find root static group
    root_gid = None
    parent_gids_of_joints = {jg[1] for jg in joint_groups.values()}
    for gid in static_groups:
        if gid in parent_gids_of_joints or root_gid is None:
            root_gid = gid

    # Create per-part links for static groups, fixed to world
    for gid, labels in static_groups.items():
        for label in labels:
            link_name = make_part_link(label)
            joint_el = ET.SubElement(robot, "joint",
                                     name=f"joint_fixed_world_{label}",
                                     type="fixed")
            ET.SubElement(joint_el, "parent", link="l_world")
            ET.SubElement(joint_el, "child", link=link_name)
            ET.SubElement(joint_el, "origin", xyz="0 0 0", rpy="0 0 0")

    # Track group-level abstract link names for parent referencing
    group_abstract_link = {}  # gid -> abstract_link_name

    # Create links for joint groups
    # Each joint group creates:
    #   1. An abstract link (the joint endpoint)
    #   2. Per-part links fixed to the abstract link
    for gid, (child_labels, parent_gid, params, jtype) in joint_groups.items():
        abstract_name = f"abstract_{parent_gid}_{gid}"
        group_abstract_link[gid] = abstract_name

        # Determine parent link name
        if parent_gid in static_groups:
            parent_link_name = "l_world"
        elif parent_gid in group_abstract_link:
            parent_link_name = group_abstract_link[parent_gid]
        elif parent_gid in joint_groups:
            # Parent joint group processed later; use its abstract name
            parent_link_name = f"abstract_{joint_groups[parent_gid][1]}_{parent_gid}"
            group_abstract_link[parent_gid] = parent_link_name
        else:
            parent_link_name = "l_world"

        # Abstract link (no visual, just a joint target)
        ET.SubElement(robot, "link", name=abstract_name)

        # Extract joint parameters
        ax, ay, az = params[0], params[1], params[2]
        ox, oy, oz = params[3], params[4], params[5]
        lower, upper = params[6], params[7]

        # Movable joint: parent -> abstract
        joint_el = ET.SubElement(robot, "joint",
                                 name=f"joint_{jtype}_{parent_link_name}_{abstract_name}",
                                 type=jtype)
        ET.SubElement(joint_el, "parent", link=parent_link_name)
        ET.SubElement(joint_el, "child", link=abstract_name)
        ET.SubElement(joint_el, "origin", xyz=f"{ox} {oy} {oz}", rpy="0 0 0")
        ET.SubElement(joint_el, "axis", xyz=f"{ax} {ay} {az}")
        if jtype in ("revolute", "prismatic"):
            ET.SubElement(joint_el, "limit",
                          lower=str(lower), upper=str(upper),
                          effort="100", velocity="1.0")
        elif jtype == "continuous":
            if abs(lower) > 1e-6 or abs(upper) > 1e-6:
                ET.SubElement(joint_el, "limit",
                              lower=str(lower), upper=str(upper),
                              effort="100", velocity="1.0")

        # Per-part links, fixed to abstract link
        for label in child_labels:
            link_name = make_part_link(label)
            fixed_el = ET.SubElement(robot, "joint",
                                     name=f"joint_fixed_{abstract_name}_{label}",
                                     type="fixed")
            ET.SubElement(fixed_el, "parent", link=abstract_name)
            ET.SubElement(fixed_el, "child", link=link_name)
            ET.SubElement(fixed_el, "origin", xyz="0 0 0", rpy="0 0 0")

    # Write URDF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    tree.write(output_path, xml_declaration=True, encoding='utf-8')
    return output_path


# ======================================================================
# Compute origins (centroids) for each part
# ======================================================================

def compute_obj_centroid(obj_path):
    """Compute centroid of an OBJ file by averaging vertex positions."""
    sx, sy, sz = 0.0, 0.0, 0.0
    count = 0
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                sx += float(parts[1])
                sy += float(parts[2])
                sz += float(parts[3])
                count += 1
    if count == 0:
        return [0.0, 0.0, 0.0]
    return [sx / count, sy / count, sz / count]


def compute_physxnet_origins(obj_id):
    """Compute origins.json style dict for a PhysXNet object.

    PhysXNet OBJs are already in a shared coordinate system (not centroid-subtracted),
    so the origin for each part is [0, 0, 0]. The centroid is only needed if
    we want to match the IM pipeline where OBJs are centroid-subtracted.

    For PhysXNet, we set origins to [0,0,0] for all parts since the OBJs
    share a global coordinate frame.
    """
    obj_paths = get_physxnet_obj_paths(obj_id)
    origins = {"world": [0.0, 0.0, 0.0]}
    for label in sorted(obj_paths.keys()):
        origins[str(label)] = [0.0, 0.0, 0.0]
    return origins


def compute_physxmob_origins(obj_id):
    """Compute origins.json style dict for a PhysX_mobility object.

    PhysX_mobility OBJs also share a global coordinate frame.
    We map each link to its part index based on the URDF.
    """
    # Parse URDF to find link->OBJ mapping
    urdf_path = get_physxmob_urdf_path(obj_id)
    if urdf_path is None:
        return {"world": [0.0, 0.0, 0.0]}

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    origins = {"world": [0.0, 0.0, 0.0]}

    for link_el in root.findall("link"):
        name = link_el.get("name")
        # Extract part index from link name like "l_39"
        if name.startswith("l_") and name[2:].isdigit():
            part_idx = name[2:]
            origins[part_idx] = [0.0, 0.0, 0.0]

    return origins


# ======================================================================
# Scene preparation (main interface for render_articulation.py)
# ======================================================================

def prepare_physxnet_scene(factory_name, seed, output_base="/mnt/data/yurh/Infinigen-Sim/outputs"):
    """Prepare all files needed for render_articulation.py to render a PhysXNet/PhysX_mobility object.

    Creates:
      - outputs/{factory_name}/{seed}/scene.urdf (generated or rewritten)
      - outputs/{factory_name}/{seed}/origins.json
      - (PhysXNet only) OBJ symlinks at outputs/{factory_name}/{seed}/outputs/{factory_name}/{seed}/objs/{i}/{i}.obj

    Returns dict with:
      - 'scene_dir': path to the scene directory
      - 'urdf_path': path to URDF
      - 'origins_path': path to origins.json
      - 'objs_dir': path to OBJs directory
      - 'obj_id': actual dataset object ID
      - 'is_variant': whether this is a material variant seed
      - 'dataset': 'physxnet' or 'physxmob'
      - 'json_data': the loaded JSON data for the object
    """
    ds = factory_dataset(factory_name)
    if ds is None:
        raise ValueError(f"Unknown factory: {factory_name}")

    obj_id, is_variant = seed_to_object_id(factory_name, seed, dataset=ds)
    if obj_id is None:
        raise ValueError(f"No objects for factory {factory_name} seed {seed}")

    scene_dir = os.path.join(output_base, factory_name, str(seed))
    os.makedirs(scene_dir, exist_ok=True)

    # Nested objs path to match IM convention expected by render_articulation.py
    objs_dir = os.path.join(scene_dir, "outputs", factory_name, str(seed), "objs")
    urdf_path = os.path.join(scene_dir, "scene.urdf")
    origins_path = os.path.join(scene_dir, "origins.json")

    if ds == "physxmob":
        # PhysX_mobility: rewrite URDF with absolute paths
        json_data = load_physxmob_json(obj_id)
        rewrite_physxmob_urdf(obj_id, urdf_path)
        origins = compute_physxmob_origins(obj_id)

        # Create OBJ symlinks matching the URDF link structure
        # The URDF references absolute paths to original OBJs, so we also need
        # the objs_dir structure for the render pipeline
        _create_physxmob_objs_links(obj_id, objs_dir, urdf_path)

    else:
        # PhysXNet: generate URDF from group_info
        json_data = load_physxnet_json(obj_id)
        if json_data is None:
            raise ValueError(f"JSON not found for PhysXNet object {obj_id}")

        group_info = json_data.get("group_info", {})
        parts_info = json_data.get("parts", [])
        generate_urdf_from_group_info(obj_id, group_info, parts_info, urdf_path)
        origins = compute_physxnet_origins(obj_id)

        # Create OBJ symlinks in IM directory structure
        _create_physxnet_objs_links(obj_id, objs_dir)

    # Write origins.json
    with open(origins_path, 'w') as f:
        json.dump(origins, f, indent=2)

    return {
        'scene_dir': scene_dir,
        'urdf_path': urdf_path,
        'origins_path': origins_path,
        'objs_dir': objs_dir,
        'obj_id': obj_id,
        'is_variant': is_variant,
        'dataset': ds,
        'json_data': json_data,
        'factory_name': factory_name,
        'seed': seed,
    }


def _create_physxnet_objs_links(obj_id, objs_dir):
    """Create symlinks for PhysXNet OBJs in IM directory structure.

    IM expects: objs/{i}/{i}.obj
    PhysXNet has: partseg/{id}/objs/{i}.obj
    """
    src_paths = get_physxnet_obj_paths(obj_id)
    for label, src_path in src_paths.items():
        dst_dir = os.path.join(objs_dir, str(label))
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, f"{label}.obj")
        if os.path.exists(dst_path) or os.path.islink(dst_path):
            os.remove(dst_path)
        os.symlink(src_path, dst_path)

        # Also symlink MTL if exists
        mtl_src = src_path.replace('.obj', '.mtl')
        if os.path.exists(mtl_src):
            mtl_dst = os.path.join(dst_dir, f"{label}.mtl")
            if os.path.exists(mtl_dst) or os.path.islink(mtl_dst):
                os.remove(mtl_dst)
            os.symlink(mtl_src, mtl_dst)


def _create_physxmob_objs_links(obj_id, objs_dir, urdf_path):
    """Create OBJ symlinks for PhysX_mobility in IM structure.

    Parses URDF to find link_name -> OBJ mapping, then creates:
      objs/{part_idx}/{part_idx}.obj -> original OBJ

    For links with multiple visuals, we create the first OBJ as the main
    and additional ones as {part_idx}_extra_{N}.obj (they get joined in Blender).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    src_objs_dir = os.path.join(PHYSXMOB_PARTSEG_DIR, str(obj_id), "objs")

    for link_el in root.findall("link"):
        name = link_el.get("name")
        if not (name.startswith("l_") and name[2:].isdigit()):
            continue

        part_idx = int(name[2:])
        dst_dir = os.path.join(objs_dir, str(part_idx))
        os.makedirs(dst_dir, exist_ok=True)

        # Collect all visual mesh files for this link
        visuals = link_el.findall("visual")
        obj_files = []
        for v in visuals:
            geom = v.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None:
                    fn = mesh.get("filename")
                    if fn:
                        # The URDF has been rewritten with absolute paths
                        obj_files.append(fn)

        if not obj_files:
            continue

        # Create symlink for the first (main) OBJ
        dst_path = os.path.join(dst_dir, f"{part_idx}.obj")
        if os.path.exists(dst_path) or os.path.islink(dst_path):
            os.remove(dst_path)

        if len(obj_files) == 1:
            os.symlink(obj_files[0], dst_path)
            # Symlink MTL too
            mtl_src = obj_files[0].replace('.obj', '.mtl')
            if os.path.exists(mtl_src):
                mtl_dst = os.path.join(dst_dir, f"{part_idx}.mtl")
                if os.path.exists(mtl_dst) or os.path.islink(mtl_dst):
                    os.remove(mtl_dst)
                os.symlink(mtl_src, mtl_dst)
        else:
            # Multiple OBJs per link: symlink each one
            # The main one gets {part_idx}.obj, extras get extra_{N}.obj
            os.symlink(obj_files[0], dst_path)
            mtl_src = obj_files[0].replace('.obj', '.mtl')
            if os.path.exists(mtl_src):
                mtl_dst = os.path.join(dst_dir, f"{part_idx}.mtl")
                if os.path.exists(mtl_dst) or os.path.islink(mtl_dst):
                    os.remove(mtl_dst)
                os.symlink(mtl_src, mtl_dst)

            for i, extra_obj in enumerate(obj_files[1:]):
                extra_dst = os.path.join(dst_dir, f"{part_idx}_extra_{i}.obj")
                if os.path.exists(extra_dst) or os.path.islink(extra_dst):
                    os.remove(extra_dst)
                os.symlink(extra_obj, extra_dst)
                extra_mtl = extra_obj.replace('.obj', '.mtl')
                if os.path.exists(extra_mtl):
                    extra_mtl_dst = os.path.join(dst_dir, f"{part_idx}_extra_{i}.mtl")
                    if os.path.exists(extra_mtl_dst) or os.path.islink(extra_mtl_dst):
                        os.remove(extra_mtl_dst)
                    os.symlink(extra_mtl, extra_mtl_dst)


# ======================================================================
# Material variant logic
# ======================================================================

def get_material_pool(factory_name, dataset="physxnet"):
    """Build a per-part-name material pool for a factory's category.

    Returns: {part_name: [list of material_dicts]}
    Each material_dict has keys like 'material', 'density', etc. from the JSON.
    """
    from physxnet_factory_rules import get_physxnet_factory_ids, get_physxmob_factory_ids

    if dataset == "physxnet":
        ids = get_physxnet_factory_ids(factory_name)
        json_dir = PHYSXNET_JSON_DIR
    else:
        ids = get_physxmob_factory_ids(factory_name)
        json_dir = PHYSXMOB_JSON_DIR

    pool = defaultdict(list)
    for obj_id in ids:
        json_path = os.path.join(json_dir, f"{obj_id}.json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        for part in data.get("parts", []):
            pname = part.get("name", "Unknown")
            mat_info = {
                "material": part.get("material", "Unknown"),
                "density": part.get("density", "1.0 g/cm^3"),
                "youngs_modulus": part.get("Young's Modulus (GPa)", 1.0),
                "poisson_ratio": part.get("Poisson's Ratio", 0.3),
            }
            pool[pname].append(mat_info)

    return dict(pool)


# ======================================================================
# Material name -> Blender PBR properties mapping
# ======================================================================

MATERIAL_PBR = {
    # Metals
    "Steel":            {"metallic": 0.95, "roughness": 0.25, "color": (0.56, 0.57, 0.58)},
    "Stainless Steel":  {"metallic": 0.95, "roughness": 0.20, "color": (0.63, 0.64, 0.65)},
    "Iron":             {"metallic": 0.90, "roughness": 0.35, "color": (0.50, 0.50, 0.50)},
    "Cast Iron":        {"metallic": 0.85, "roughness": 0.45, "color": (0.35, 0.35, 0.35)},
    "Aluminum":         {"metallic": 0.90, "roughness": 0.20, "color": (0.77, 0.78, 0.78)},
    "Copper":           {"metallic": 0.95, "roughness": 0.25, "color": (0.72, 0.45, 0.20)},
    "Brass":            {"metallic": 0.90, "roughness": 0.25, "color": (0.78, 0.67, 0.35)},
    "Bronze":           {"metallic": 0.85, "roughness": 0.30, "color": (0.55, 0.35, 0.17)},
    "Chrome":           {"metallic": 1.00, "roughness": 0.05, "color": (0.55, 0.56, 0.56)},
    "Titanium":         {"metallic": 0.90, "roughness": 0.25, "color": (0.54, 0.50, 0.50)},
    "Gold":             {"metallic": 1.00, "roughness": 0.10, "color": (1.00, 0.77, 0.34)},
    "Silver":           {"metallic": 1.00, "roughness": 0.10, "color": (0.95, 0.93, 0.88)},
    "Zinc":             {"metallic": 0.85, "roughness": 0.30, "color": (0.65, 0.65, 0.67)},
    "Nickel":           {"metallic": 0.90, "roughness": 0.20, "color": (0.66, 0.66, 0.66)},
    "Metal":            {"metallic": 0.85, "roughness": 0.30, "color": (0.60, 0.60, 0.60)},
    "Metal Alloy":      {"metallic": 0.85, "roughness": 0.30, "color": (0.58, 0.58, 0.60)},

    # Plastics
    "Plastic":          {"metallic": 0.00, "roughness": 0.40, "color": (0.60, 0.60, 0.60)},
    "ABS Plastic":      {"metallic": 0.00, "roughness": 0.35, "color": (0.15, 0.15, 0.15)},
    "Polycarbonate":    {"metallic": 0.00, "roughness": 0.20, "color": (0.85, 0.85, 0.85)},
    "Polypropylene":    {"metallic": 0.00, "roughness": 0.50, "color": (0.75, 0.75, 0.75)},
    "PVC":              {"metallic": 0.00, "roughness": 0.45, "color": (0.70, 0.70, 0.70)},
    "Nylon":            {"metallic": 0.00, "roughness": 0.55, "color": (0.80, 0.78, 0.75)},
    "Acrylic":          {"metallic": 0.00, "roughness": 0.10, "color": (0.90, 0.90, 0.90)},
    "Polyethylene":     {"metallic": 0.00, "roughness": 0.50, "color": (0.80, 0.80, 0.80)},
    "Silicone":         {"metallic": 0.00, "roughness": 0.70, "color": (0.70, 0.68, 0.65)},
    "Silicone Rubber":  {"metallic": 0.00, "roughness": 0.75, "color": (0.65, 0.63, 0.60)},
    "Resin":            {"metallic": 0.05, "roughness": 0.30, "color": (0.80, 0.75, 0.65)},

    # Glass/Ceramic
    "Glass":            {"metallic": 0.00, "roughness": 0.05, "color": (0.90, 0.92, 0.95)},
    "Tempered Glass":   {"metallic": 0.00, "roughness": 0.05, "color": (0.88, 0.90, 0.93)},
    "Ceramic":          {"metallic": 0.00, "roughness": 0.40, "color": (0.85, 0.82, 0.78)},
    "Porcelain":        {"metallic": 0.00, "roughness": 0.15, "color": (0.95, 0.93, 0.90)},

    # Wood
    "Wood":             {"metallic": 0.00, "roughness": 0.55, "color": (0.55, 0.35, 0.20)},
    "Bamboo":           {"metallic": 0.00, "roughness": 0.50, "color": (0.70, 0.60, 0.35)},
    "Plywood":          {"metallic": 0.00, "roughness": 0.60, "color": (0.65, 0.50, 0.30)},
    "MDF":              {"metallic": 0.00, "roughness": 0.65, "color": (0.60, 0.50, 0.35)},
    "Engineered Wood":  {"metallic": 0.00, "roughness": 0.55, "color": (0.58, 0.45, 0.28)},
    "Oak":              {"metallic": 0.00, "roughness": 0.50, "color": (0.60, 0.40, 0.22)},

    # Fabric/Leather
    "Fabric":           {"metallic": 0.00, "roughness": 0.80, "color": (0.55, 0.50, 0.45)},
    "Leather":          {"metallic": 0.00, "roughness": 0.60, "color": (0.30, 0.18, 0.10)},
    "Synthetic Leather":{"metallic": 0.00, "roughness": 0.55, "color": (0.25, 0.15, 0.10)},
    "Cotton":           {"metallic": 0.00, "roughness": 0.85, "color": (0.80, 0.78, 0.75)},
    "Polyester":        {"metallic": 0.00, "roughness": 0.70, "color": (0.60, 0.58, 0.55)},
    "Felt":             {"metallic": 0.00, "roughness": 0.90, "color": (0.55, 0.52, 0.48)},
    "Foam":             {"metallic": 0.00, "roughness": 0.85, "color": (0.75, 0.72, 0.68)},
    "Foam and Leather": {"metallic": 0.00, "roughness": 0.65, "color": (0.45, 0.30, 0.18)},

    # Rubber
    "Rubber":           {"metallic": 0.00, "roughness": 0.80, "color": (0.15, 0.15, 0.15)},
    "Synthetic Rubber": {"metallic": 0.00, "roughness": 0.80, "color": (0.20, 0.20, 0.20)},
    "TPU":              {"metallic": 0.00, "roughness": 0.65, "color": (0.30, 0.30, 0.30)},

    # Stone/Concrete
    "Stone":            {"metallic": 0.00, "roughness": 0.60, "color": (0.55, 0.52, 0.48)},
    "Marble":           {"metallic": 0.00, "roughness": 0.20, "color": (0.90, 0.88, 0.85)},
    "Granite":          {"metallic": 0.00, "roughness": 0.50, "color": (0.45, 0.42, 0.40)},
    "Concrete":         {"metallic": 0.00, "roughness": 0.70, "color": (0.60, 0.58, 0.55)},

    # Special
    "Carbon Fiber":     {"metallic": 0.30, "roughness": 0.20, "color": (0.15, 0.15, 0.15)},
    "Fiberglass":       {"metallic": 0.05, "roughness": 0.40, "color": (0.80, 0.80, 0.80)},
    "Paper":            {"metallic": 0.00, "roughness": 0.85, "color": (0.90, 0.88, 0.83)},
    "Cardboard":        {"metallic": 0.00, "roughness": 0.80, "color": (0.65, 0.55, 0.40)},
}


def get_pbr_for_material(material_name):
    """Look up PBR properties for a material name.

    Tries exact match first, then substring match, then falls back to defaults.
    Returns (metallic, roughness, color_rgb_tuple).
    """
    # Exact match
    if material_name in MATERIAL_PBR:
        props = MATERIAL_PBR[material_name]
        return props["metallic"], props["roughness"], props["color"]

    # Substring match (case-insensitive)
    mat_lower = material_name.lower()
    for key, props in MATERIAL_PBR.items():
        if key.lower() in mat_lower:
            return props["metallic"], props["roughness"], props["color"]

    # Default
    return 0.10, 0.45, (0.60, 0.60, 0.60)


def apply_physxnet_materials_blender(scene_info, part_objects):
    """Apply material properties to Blender objects based on PhysXNet JSON data.

    This is the Blender-side function (called inside Blender context).
    Uses the JSON part data to set metallic/roughness/base color on each part.

    Args:
        scene_info: dict from prepare_physxnet_scene()
        part_objects: dict of {part_idx: bpy.types.Object} from load_scene_parts()
    """
    import bpy

    json_data = scene_info['json_data']
    if json_data is None:
        return

    factory_name = scene_info['factory_name']
    seed = scene_info['seed']
    is_variant = scene_info['is_variant']
    dataset = scene_info['dataset']

    # Build label -> part info mapping
    parts = json_data.get("parts", [])
    label_to_info = {}
    for p in parts:
        label = p.get("label")
        if label is not None:
            label_to_info[label] = p

    # Build group_idx -> [part_labels] from group_info
    # part_objects keys are GROUP indices (from URDF l_N), not raw part labels
    group_info_data = json_data.get("group_info", {})
    group_to_labels = {}
    for gid_str, val in group_info_data.items():
        gid = int(gid_str)
        if isinstance(val, list) and len(val) >= 4 and isinstance(val[-1], str) and val[-1] in ('A', 'B', 'C', 'D', 'CB'):
            labels = val[0] if isinstance(val[0], list) else [val[0]]
            group_to_labels[gid] = labels
        elif isinstance(val, list):
            group_to_labels[gid] = [x for x in val if isinstance(x, int)]

    # For material variants, build the material pool and select swaps
    variant_materials = {}
    if is_variant:
        pool = get_material_pool(factory_name, dataset=dataset)
        rng = random.Random(seed)
        for label, info in label_to_info.items():
            pname = info.get("name", "Unknown")
            if pname in pool and len(pool[pname]) > 1:
                variant_materials[label] = rng.choice(pool[pname])

    # Get factory-level defaults
    factory_defaults = ALL_MATERIAL_DEFAULTS.get(factory_name, {"default": (0.10, 0.45)})
    default_metallic, default_roughness = factory_defaults.get("default", (0.10, 0.45))

    for part_idx, obj in part_objects.items():
        # part_idx is GROUP index; map to part labels for material lookup
        part_labels = group_to_labels.get(part_idx, [part_idx])
        info = {}
        for lbl in part_labels:
            info = label_to_info.get(lbl, {})
            if info:
                break
        material_name = info.get("material", "Unknown")

        # If variant mode, swap material using first label
        first_label = part_labels[0] if part_labels else part_idx
        if first_label in variant_materials:
            material_name = variant_materials[first_label].get("material", material_name)

        # Look up PBR properties
        metallic, roughness, color = get_pbr_for_material(material_name)

        # Add seed-based slight variation
        rng_mat = random.Random(hash((factory_name, seed, part_idx)))
        metallic = max(0, min(1, metallic + rng_mat.uniform(-0.05, 0.05)))
        roughness = max(0, min(1, roughness + rng_mat.uniform(-0.08, 0.08)))

        # Apply to all material slots on the Blender object
        for mat_slot in obj.material_slots:
            mat = mat_slot.material
            if mat is None or not mat.use_nodes or mat.node_tree is None:
                continue
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    node.inputs['Metallic'].default_value = metallic
                    node.inputs['Roughness'].default_value = roughness
                    node.inputs['Base Color'].default_value = (*color, 1.0)
                    if metallic > 0.5:
                        node.inputs['Specular'].default_value = 0.5 + metallic * 0.3
                    break

        # If no material slots, create one
        if not obj.material_slots:
            mat = bpy.data.materials.new(name=f"physxnet_mat_{part_idx}")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs['Metallic'].default_value = metallic
                bsdf.inputs['Roughness'].default_value = roughness
                bsdf.inputs['Base Color'].default_value = (*color, 1.0)
                if metallic > 0.5:
                    bsdf.inputs['Specular'].default_value = 0.5 + metallic * 0.3
            obj.data.materials.append(mat)


# ======================================================================
# Utility: check if factory is PhysXNet/PhysX_mobility
# ======================================================================

def is_physxnet_factory(factory_name):
    """Return True if factory_name belongs to PhysXNet or PhysX_mobility."""
    return factory_dataset(factory_name) is not None


# ======================================================================
# Main: test/demo
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PhysXNet loader test")
    parser.add_argument("--factory", default="ElectronicsPhysXMobilityFactory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_base", default="/mnt/data/yurh/Infinigen-Sim/outputs")
    args = parser.parse_args()

    print(f"Preparing scene: {args.factory} seed={args.seed}")
    info = prepare_physxnet_scene(args.factory, args.seed, args.output_base)

    print(f"\n  Object ID:  {info['obj_id']}")
    print(f"  Dataset:    {info['dataset']}")
    print(f"  Variant:    {info['is_variant']}")
    print(f"  Scene dir:  {info['scene_dir']}")
    print(f"  URDF:       {info['urdf_path']}")
    print(f"  Origins:    {info['origins_path']}")
    print(f"  OBJs dir:   {info['objs_dir']}")

    if info['json_data']:
        d = info['json_data']
        print(f"  Object:     {d.get('object_name', '?')} / {d.get('category', '?')}")
        print(f"  Parts:      {len(d.get('parts', []))}")
        gi = d.get('group_info', {})
        n_joints = sum(1 for v in gi.values()
                       if isinstance(v, list) and len(v) >= 4
                       and isinstance(v[-1], str) and v[-1] in ('A', 'B', 'C'))
        print(f"  Joints:     {n_joints}")

    # Verify URDF
    if os.path.exists(info['urdf_path']):
        tree = ET.parse(info['urdf_path'])
        root = tree.getroot()
        n_links = len(root.findall("link"))
        n_joints = len(root.findall("joint"))
        movable = sum(1 for j in root.findall("joint") if j.get("type") != "fixed")
        print(f"  URDF links: {n_links}, joints: {n_joints} ({movable} movable)")
    else:
        print("  URDF: NOT FOUND")

    # Verify OBJ symlinks
    if os.path.isdir(info['objs_dir']):
        import glob
        objs = glob.glob(os.path.join(info['objs_dir'], "*", "*.obj"))
        print(f"  OBJ files:  {len(objs)}")
    else:
        print("  OBJs dir: NOT FOUND")
