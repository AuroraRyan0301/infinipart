#!/usr/bin/env python3
"""
Create synthetic test data (URDF + OBJ) for testing split_precompute.py.

Creates two test cases:
1. LampFactory-like: chain structure (base -> arm_joint -> arm -> head_joint -> head)
2. DishwasherFactory-like: tree structure (body with door, upper_rack, lower_rack siblings)

These match the examples in CLAUDE.md exactly, so we can verify 2-coloring correctness.
"""

import json
import os
import numpy as np
import trimesh


def create_box_mesh(center, half_extents):
    """Create a simple box mesh centered at given position."""
    box = trimesh.creation.box(extents=half_extents * 2)
    box.vertices += center
    return box


def create_test_lamp(base_dir):
    """Create LampFactory-like test data.

    Chain: base -> arm_joint(revolute) -> arm -> head_joint(revolute) -> head

    Expected results from CLAUDE.md:
    - basic_0 (active: arm_joint): head_joint=fixed, merge arm+head -> B
      Graph: base -[arm_joint]- B(arm+head)
      2-coloring: base=part0, B=part1

    - basic_1 (active: head_joint): arm_joint=fixed, merge base+arm -> A
      Graph: A(base+arm) -[head_joint]- head
      2-coloring: A=part0, head=part1

    - senior_0 (active: arm_joint + head_joint): no fixed joints
      Graph: base -[arm_joint]- arm -[head_joint]- head
      2-coloring: base=part0, arm=part1, head=part0
    """
    factory = "TestLampFactory"
    seed = "0"
    scene_dir = os.path.join(base_dir, factory, seed)
    objs_dir = os.path.join(scene_dir, "objs")

    os.makedirs(scene_dir, exist_ok=True)

    # Create meshes
    # Part 0: base (bottom cube)
    base_mesh = create_box_mesh(
        center=np.array([0.0, 0.0, 0.1]),
        half_extents=np.array([0.15, 0.15, 0.1])
    )

    # Part 1: arm (vertical cylinder-like)
    arm_mesh = create_box_mesh(
        center=np.array([0.0, 0.0, 0.5]),
        half_extents=np.array([0.03, 0.03, 0.3])
    )

    # Part 2: head (top piece)
    head_mesh = create_box_mesh(
        center=np.array([0.0, 0.0, 0.85]),
        half_extents=np.array([0.12, 0.12, 0.05])
    )

    # Save OBJs (centroid-subtracted, origin in origins.json)
    origins = {}
    for idx, (mesh, name) in enumerate([(base_mesh, "base"), (arm_mesh, "arm"), (head_mesh, "head")]):
        centroid = mesh.vertices.mean(axis=0)
        centered_mesh = mesh.copy()
        centered_mesh.vertices -= centroid
        origins[str(idx)] = centroid.tolist()

        part_dir = os.path.join(objs_dir, str(idx))
        os.makedirs(part_dir, exist_ok=True)
        centered_mesh.export(os.path.join(part_dir, f"{idx}.obj"))

    with open(os.path.join(scene_dir, "origins.json"), "w") as f:
        json.dump(origins, f, indent=2)

    # Create URDF
    urdf = """<?xml version="1.0" ?>
<robot name="test_lamp">
  <link name="l_0"/>
  <link name="l_1"/>
  <link name="l_2"/>

  <joint name="arm_joint" type="revolute">
    <parent link="l_0"/>
    <child link="l_1"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5"/>
  </joint>

  <joint name="head_joint" type="revolute">
    <parent link="l_1"/>
    <child link="l_2"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.8" upper="0.8"/>
  </joint>
</robot>
"""
    with open(os.path.join(scene_dir, "scene.urdf"), "w") as f:
        f.write(urdf)

    print(f"Created TestLampFactory at {scene_dir}")
    return scene_dir


def create_test_dishwasher(base_dir):
    """Create DishwasherFactory-like test data.

    Tree: body(l_0) -> door_joint(revolute) -> door(l_1)
          body(l_0) -> upper_rack_joint(prismatic) -> upper_rack(l_2)
          body(l_0) -> lower_rack_joint(prismatic) -> lower_rack(l_3)

    Expected results from CLAUDE.md:
    - basic_1 (active: upper_rack_joint):
      BVH: rack slides out, hits door -> door_joint=passive; lower_rack not hit -> fixed
      Merge fixed: body+lower_rack -> A
      Graph: upper_rack -[rack_joint]- A(body+lower_rack) -[door_joint]- door
      2-coloring: upper_rack=part0, A=part1, door=part0
      NOTE: upper_rack and door get SAME color (part0)!
    """
    factory = "TestDishwasherFactory"
    seed = "0"
    scene_dir = os.path.join(base_dir, factory, seed)
    objs_dir = os.path.join(scene_dir, "objs")

    os.makedirs(scene_dir, exist_ok=True)

    # Part 0: body (large box, open front)
    body_mesh = create_box_mesh(
        center=np.array([0.0, 0.0, 0.3]),
        half_extents=np.array([0.3, 0.3, 0.3])
    )

    # Part 1: door (thin panel at front)
    door_mesh = create_box_mesh(
        center=np.array([0.0, -0.32, 0.3]),
        half_extents=np.array([0.28, 0.02, 0.28])
    )

    # Part 2: upper rack (slides out -Y direction, close to door)
    upper_rack_mesh = create_box_mesh(
        center=np.array([0.0, -0.05, 0.45]),
        half_extents=np.array([0.25, 0.2, 0.02])
    )

    # Part 3: lower rack (stays inside, far from upper rack trajectory)
    lower_rack_mesh = create_box_mesh(
        center=np.array([0.0, 0.0, 0.15]),
        half_extents=np.array([0.25, 0.2, 0.02])
    )

    origins = {}
    for idx, mesh in enumerate([body_mesh, door_mesh, upper_rack_mesh, lower_rack_mesh]):
        centroid = mesh.vertices.mean(axis=0)
        centered_mesh = mesh.copy()
        centered_mesh.vertices -= centroid
        origins[str(idx)] = centroid.tolist()

        part_dir = os.path.join(objs_dir, str(idx))
        os.makedirs(part_dir, exist_ok=True)
        centered_mesh.export(os.path.join(part_dir, f"{idx}.obj"))

    with open(os.path.join(scene_dir, "origins.json"), "w") as f:
        json.dump(origins, f, indent=2)

    # URDF: body is root, 3 children connected by joints
    urdf = """<?xml version="1.0" ?>
<robot name="test_dishwasher">
  <link name="l_0"/>
  <link name="l_1"/>
  <link name="l_2"/>
  <link name="l_3"/>

  <joint name="door_joint" type="revolute">
    <parent link="l_0"/>
    <child link="l_1"/>
    <origin xyz="0 -0.3 0.0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="1.57"/>
  </joint>

  <joint name="upper_rack_joint" type="prismatic">
    <parent link="l_0"/>
    <child link="l_2"/>
    <origin xyz="0 0 0.45" rpy="0 0 0"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.4"/>
  </joint>

  <joint name="lower_rack_joint" type="prismatic">
    <parent link="l_0"/>
    <child link="l_3"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.4"/>
  </joint>
</robot>
"""
    with open(os.path.join(scene_dir, "scene.urdf"), "w") as f:
        f.write(urdf)

    print(f"Created TestDishwasherFactory at {scene_dir}")
    return scene_dir


def create_test_simple_door(base_dir):
    """Create a simple door test case.

    Chain: frame(l_0) -> hinge_joint(revolute) -> door(l_1)

    Expected:
    - basic_0 (active: hinge_joint):
      No other joints -> no passive/fixed classification needed
      Graph: frame -[hinge_joint]- door
      2-coloring: frame=part0, door=part1
    """
    factory = "TestSimpleDoorFactory"
    seed = "0"
    scene_dir = os.path.join(base_dir, factory, seed)
    objs_dir = os.path.join(scene_dir, "objs")

    os.makedirs(scene_dir, exist_ok=True)

    # Part 0: frame
    frame_mesh = create_box_mesh(
        center=np.array([0.0, 0.0, 0.5]),
        half_extents=np.array([0.5, 0.05, 0.5])
    )

    # Part 1: door panel
    door_mesh = create_box_mesh(
        center=np.array([0.0, -0.1, 0.5]),
        half_extents=np.array([0.4, 0.02, 0.45])
    )

    origins = {}
    for idx, mesh in enumerate([frame_mesh, door_mesh]):
        centroid = mesh.vertices.mean(axis=0)
        centered_mesh = mesh.copy()
        centered_mesh.vertices -= centroid
        origins[str(idx)] = centroid.tolist()

        part_dir = os.path.join(objs_dir, str(idx))
        os.makedirs(part_dir, exist_ok=True)
        centered_mesh.export(os.path.join(part_dir, f"{idx}.obj"))

    with open(os.path.join(scene_dir, "origins.json"), "w") as f:
        json.dump(origins, f, indent=2)

    urdf = """<?xml version="1.0" ?>
<robot name="test_simple_door">
  <link name="l_0"/>
  <link name="l_1"/>

  <joint name="hinge_joint" type="revolute">
    <parent link="l_0"/>
    <child link="l_1"/>
    <origin xyz="-0.4 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="1.57"/>
  </joint>
</robot>
"""
    with open(os.path.join(scene_dir, "scene.urdf"), "w") as f:
        f.write(urdf)

    print(f"Created TestSimpleDoorFactory at {scene_dir}")
    return scene_dir


if __name__ == "__main__":
    base_dir = "/mnt/data/yurh/Infinigen-Sim/test_data"
    os.makedirs(base_dir, exist_ok=True)

    create_test_lamp(base_dir)
    create_test_dishwasher(base_dir)
    create_test_simple_door(base_dir)

    print(f"\nAll test data created in {base_dir}")
    print("Run: python split_precompute.py --factory TestLampFactory --seed 0 "
          f"--base {base_dir} --output_dir ./precompute_output")
