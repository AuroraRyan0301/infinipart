"""
PhysXNet and PhysX_mobility factory rules for render_articulation.py.

Maps PhysXNet/PhysX_mobility categories to:
  - PHYSXNET_RENDER_RULES: render animation rules (moving_parts, animode_joints)
  - PHYSXNET_SPLIT_RULES: 2-part split rules (moving_parts, description)
  - PHYSXNET_ANIMODES: max animode index per factory
  - PHYSXNET_FACTORY_LIST: ordered list of all factory names
  - PHYSXMOB_RENDER_RULES, etc.: same for PhysX_mobility

Data paths (absolute, no copying):
  - PhysXNet: /mnt/data/fulian/dataset/PhysXNet/version_1/
  - PhysX_mobility: /mnt/data/fulian/dataset/PhysX_mobility/

PhysXNet group_info joint type mapping:
  'A' = revolute (hinge with limits)
  'B' = prismatic (slider with limits)
  'C' = continuous (unlimited rotation)

PhysXNet group_info format:
  Static group:  "gid": [list_of_part_label_ints]
  Joint group:   "gid": [[child_part_labels], "parent_gid",
                          [ax, ay, az, ox, oy, oz, lower, upper], "joint_type"]

PhysX_mobility: ALL 2024 objects are PartNet-Mobility subsets with URDFs.
"""

import json
import os
from collections import defaultdict

# ======================================================================
# Dataset paths
# ======================================================================
PHYSXNET_BASE = "/mnt/data/fulian/dataset/PhysXNet/version_1"
PHYSXMOB_BASE = "/mnt/data/fulian/dataset/PhysX_mobility"

PHYSXNET_JSON_DIR = os.path.join(PHYSXNET_BASE, "finaljson")
PHYSXNET_PARTSEG_DIR = os.path.join(PHYSXNET_BASE, "partseg")
PHYSXMOB_JSON_DIR = os.path.join(PHYSXMOB_BASE, "finaljson")
PHYSXMOB_PARTSEG_DIR = os.path.join(PHYSXMOB_BASE, "partseg")
PHYSXMOB_URDF_DIR = os.path.join(PHYSXMOB_BASE, "urdf")


# ======================================================================
# PhysXNet category grouping (450 raw categories -> ~15 factory groups)
# ======================================================================

# Each key is the factory group name, value is a set of raw category strings
# that map to this factory. Categories not listed go to "MiscPhysXNetFactory".
PHYSXNET_CATEGORY_GROUPS = {
    "FurniturePhysXNetFactory": {
        "Furniture", "Furniture Component", "FurnitureComponent",
        "Outdoor Furniture", "OutdoorFurniture", "OfficeFurniture",
        "Office Furniture", "Seating Furniture", "Adjustable Chair",
        "OfficeChair", "Medical Furniture", "MedicalFurniture",
        "Furniture/Storage", "Furniture/Clock", "Furniture/Decor",
        "Furniture/Display Fixture", "Furniture/Building Component",
        "BathroomFurniture", "Bathroom Cabinet", "StorageFurniture",
        "Storage Furniture", "Cabinet", "Workbench", "UrbanFurniture",
        "Urban Furniture", "StreetFurniture", "Street Furniture",
        "GameTable",
    },
    "ChairPhysXNetFactory": {
        "Chair",
    },
    "LightingPhysXNetFactory": {
        "Lighting", "Lighting Fixture", "Lighting Appliance",
        "Lighting Device", "LightingFixture", "LightingDevice",
        "Ceiling Light Fixture", "CeilingLightFixture", "CeilingLamp",
        "Ceiling Light", "Lighting Equipment", "Decorative Lighting",
        "Decorative Lamp", "Lamp", "OutdoorLighting",
        "Outdoor Lighting Fixture", "Lighting Fixture Component",
        "Pendulum Clock",  # often has lighting substructure
    },
    "ElectronicsPhysXNetFactory": {
        "Electronics", "ElectronicDevice", "ElectronicDisplay",
        "ElectronicDisplayDevice", "DisplayDevice", "DisplayStand",
        "ElectronicDeviceComponent", "ElectronicDeviceInterface",
        "ElectronicComponent", "Electronic Component", "Electronic Interface Device",
        "Interactive Display Device", "DisplayEquipment",
        "Electronics Accessory", "Electronic Controller",
        "ControlInterface", "ControlDevice", "ControlBox",
        "ElectronicController", "Electronic Control Device",
        "ElectronicControlDevice",
    },
    "AudioPhysXNetFactory": {
        "AudioDevice", "Headphones",
    },
    "ContainerPhysXNetFactory": {
        "Container", "StorageContainer", "StorageBox", "StorageObject",
        "Household Container", "Gardening Container", "GardeningContainer",
        "PlantContainer", "Planter", "Decorative Container",
        "FoodContainer", "Laboratory Container", "Luggage/Container",
        "Decorative Plant", "Packaging", "Packaging Container",
        "CeramicContainer", "LiquidContainer", "Storage Vessel",
    },
    "KitchenPhysXNetFactory": {
        "Kitchen Appliance", "KitchenAppliance", "Kitchenware",
        "KitchenTool", "Kitchen Utensil", "Tableware",
        "Drinkware", "Cup",
    },
    "PlumbingPhysXNetFactory": {
        "Plumbing Fixture", "Plumbing Component",
        "Plumbing Fixture Component", "Bathroom Fixture",
        "Bathroom Appliance", "Bathroom Accessory",
        "WaterDispenser", "Manual Water Pump",
        "Liquid Dispensing Device", "Liquid Dispenser",
        "Liquid Transfer Tool", "Dispenser",
        "Liquid Container / Pump Bottle", "Liquid Dispenser / Pump Bottle",
        "Liquid Container / Pump Dispenser",
        "Liquid Dispenser (e.g., Soap or Lotion Dispenser)",
        "Liquid Dispenser / Soap Pump", "Spray Bottle",
        "Container/Dispenser", "Liquid Container",
        "Aerosol Can",
    },
    "ToolPhysXNetFactory": {
        "Cutting Tool", "CuttingTool", "HandTool", "Hand Tool",
        "Tool", "Tool/Weapon", "Tool/Knife", "CleaningTool",
        "Cleaning Tool", "Gardening Tool", "Mechanical Device",
        "MechanicalComponent", "Mechanical Interface Component",
        "Mechanical Component",
    },
    "WeaponPhysXNetFactory": {
        "Weapon", "MeleeWeapon", "Knife", "Cutlery",
    },
    "ClockPhysXNetFactory": {
        "Clock", "Timekeeping Device", "PendulumClock", "WallClock",
    },
    "DoorPhysXNetFactory": {
        "Door", "Architectural Component", "Architectural Element",
        "Architectural Structure", "ArchitecturalStructure",
        "ArchitecturalComponent", "Architectural Fixture",
        "DoorAssembly", "Building Fixture",
    },
    "BagPhysXNetFactory": {
        "Bag", "Luggage",
    },
    "WastePhysXNetFactory": {
        "Waste Container", "WasteContainer", "Waste Management Container",
        "Public Waste Container", "Public Waste Bin",
        "Trash Can", "Trash Bin", "Sanitation Facility",
    },
    "AppliancePhysXNetFactory": {
        "Appliance", "Home Appliance", "HomeAppliance",
        "Home Appliance Component", "Electrical Appliance",
        "Electrical Component", "ElectricalComponent",
        "ElectricalEnclosure", "Electrical Switch",
    },
    "ClothingPhysXNetFactory": {
        "Clothing Accessory", "ClothingAccessory", "WearableAccessory",
        "Wearable", "WearableDevice", "Wearable Device",
        "Headwear", "Protective Headgear", "Protective Gear",
        "ProtectiveGear", "Safety Equipment", "Clothing", "Apparel",
    },
    "SportsPhysXNetFactory": {
        "Sports Equipment", "SportsEquipment", "Fitness Equipment",
        "SportingGoods", "Board Game Piece",
    },
    "DecorPhysXNetFactory": {
        "Decorative Object", "Home Decor", "HomeDecor",
        "Household Item", "HouseholdObject",
    },
    "MobilityAidPhysXNetFactory": {
        "MobilityAid",
    },
}

# Build reverse map: raw_category -> factory_name
_PHYSXNET_CAT_TO_FACTORY = {}
for factory_name, cats in PHYSXNET_CATEGORY_GROUPS.items():
    for cat in cats:
        _PHYSXNET_CAT_TO_FACTORY[cat] = factory_name


# ======================================================================
# PhysX_mobility category grouping (132 raw -> ~12 factory groups)
# ======================================================================

PHYSXMOB_CATEGORY_GROUPS = {
    "FurniturePhysXMobilityFactory": {
        "Storage Furniture", "Furniture",
        "Furniture/Building Component",
    },
    "ElectronicsPhysXMobilityFactory": {
        "ElectronicDevice", "Electronic Device", "Electronic Storage Device",
        "Electrical Appliance", "Electrical Control Device",
        "ComputerPeripheral", "Electromechanical Device",
        "Electrical Component", "Electronic Control Device",
        "ElectronicControlDevice", "Electronic Device Accessory",
        "Electronic Component", "Electronic Controller",
        "ElectronicController", "DigitalCamera",
        "Input Device", "Communication Device",
        "Computer Component (CPU Cooler)", "Computer Cooling Device",
        "Electronic Cooling Device", "Cooling Device", "Ventilation Equipment",
        "Electrical Device", "Electrical Appliance Component",
        "OfficeEquipment", "OfficeTool",
    },
    "KitchenPhysXMobilityFactory": {
        "Kitchen Appliance", "KitchenAppliance", "Kitchenware",
        "Cookware", "Drinkware",
    },
    "AppliancePhysXMobilityFactory": {
        "Home Appliance", "Household Appliance", "Household Utility Object",
        "Household Container", "Household Object", "Household Tool",
        "Household Waste Container",
    },
    "PlumbingPhysXMobilityFactory": {
        "Plumbing Fixture", "Sanitary Ware", "SanitaryWare",
        "Bathroom Accessory", "Hygiene Container", "Hygiene Accessory",
        "Liquid Container / Pump Bottle", "Liquid Dispenser",
        "Liquid Container / Pump Dispenser", "Liquid Container",
        "Liquid Dispenser / Pump Bottle", "Liquid Dispenser / Soap Pump",
        "Liquid Dispenser (e.g., Soap or Lotion Dispenser)",
        "Dispenser", "Spray Bottle",
        "LiquidContainer", "CeramicContainer",
    },
    "ToolPhysXMobilityFactory": {
        "Cutting Tool", "Hand Tool", "HandTool", "Office Tool",
        "Office Equipment", "CuttingTool", "Multi-tool Knife",
        "Folding Knife", "Folding Tool", "Utility Knife",
    },
    "ContainerPhysXMobilityFactory": {
        "Container", "Packaging Container", "StorageBox",
        "SecurityStorageDevice", "SecurityStorage", "Security Storage Device",
    },
    "OpticalPhysXMobilityFactory": {
        "OpticalAccessory", "VisionAidDevice", "VisionAid", "Optical Device",
    },
    "WritingPhysXMobilityFactory": {
        "Writing Instrument", "WritingInstrument",
    },
    "EducationPhysXMobilityFactory": {
        "Educational Object", "Educational Tool", "Educational Instrument",
        "Educational Model", "Educational/Decorative Object",
        "Decorative Object / Educational Instrument",
    },
    "WastePhysXMobilityFactory": {
        "Waste Container", "WasteContainer",
    },
    "DoorPhysXMobilityFactory": {
        "BuildingComponent", "Architectural Component",
        "Architectural Element", "Architectural Structure",
        "Building Component", "Building Fixture",
    },
    "ClockPhysXMobilityFactory": {
        "Timekeeping Device", "WallClock", "Clock",
    },
    "TransportPhysXMobilityFactory": {
        "Transport Equipment", "Transport/Utility Vehicle",
        "Transport Vehicle", "TransportVehicle",
        "Utility Cart", "Cart", "Shopping Cart",
        "Utility Vehicle / Shopping Cart",
        "Utility Vehicle / Transport Cart",
        "Utility Vehicle / Handcart",
        "Hand Truck / Trolley", "Wheelbarrow",
        "Handcart/Wheelbarrow", "Mobile Display Cart",
        "Mobile Vending Cart", "Material Handling Equipment",
        "MaterialHandlingEquipment", "Manual Transport Equipment",
        "Utility Transport Equipment", "Agricultural Equipment",
        "Transport/Utility Object", "Vehicle",
    },
    "LuggagePhysXMobilityFactory": {
        "Luggage", "TravelAccessory", "TravelBag", "Travel Equipment",
    },
    "LightingPhysXMobilityFactory": {
        "Lighting Fixture", "Lighting Device",
    },
    "IgnitionPhysXMobilityFactory": {
        "Handheld Fire Ignition Device", "Portable Fire Ignition Device",
        "Handheld Ignition Device", "Everyday Object / Fire-starting Tool",
    },
}

_PHYSXMOB_CAT_TO_FACTORY = {}
for factory_name, cats in PHYSXMOB_CATEGORY_GROUPS.items():
    for cat in cats:
        _PHYSXMOB_CAT_TO_FACTORY[cat] = factory_name


# ======================================================================
# Build ID lists per factory (lazy-loaded singleton)
# ======================================================================

_physxnet_factory_ids = None
_physxmob_factory_ids = None

# Disk cache paths (avoid re-scanning 34k+ JSONs every time)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".factory_cache")
_PHYSXNET_CACHE = os.path.join(_CACHE_DIR, "physxnet_ids.json")
_PHYSXMOB_CACHE = os.path.join(_CACHE_DIR, "physxmob_ids.json")


def _scan_and_cache(json_dir, cat_to_factory_map, cache_path, fallback_factory):
    """Scan JSON dir, group by factory, save to disk cache."""
    factory_ids = defaultdict(list)
    if not os.path.isdir(json_dir):
        return dict(factory_ids)

    for f in sorted(os.listdir(json_dir)):
        if not f.endswith('.json'):
            continue
        obj_id = f[:-5]
        fpath = os.path.join(json_dir, f)
        try:
            with open(fpath) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, IOError):
            continue
        cat = d.get('category', 'Unknown')
        factory = cat_to_factory_map.get(cat, fallback_factory)
        factory_ids[factory].append(obj_id)

    result = dict(factory_ids)
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cache_path, 'w') as fh:
            json.dump(result, fh)
    except IOError:
        pass
    return result


def _load_from_cache(cache_path, json_dir):
    """Load from disk cache if newer than json_dir."""
    if not os.path.isfile(cache_path):
        return None
    try:
        cache_mtime = os.path.getmtime(cache_path)
        dir_mtime = os.path.getmtime(json_dir) if os.path.isdir(json_dir) else 0
        if cache_mtime > dir_mtime:
            with open(cache_path) as fh:
                return json.load(fh)
    except (IOError, json.JSONDecodeError):
        pass
    return None


def _load_physxnet_ids():
    """Load PhysXNet object IDs grouped by factory. Disk-cached."""
    global _physxnet_factory_ids
    if _physxnet_factory_ids is not None:
        return _physxnet_factory_ids

    cached = _load_from_cache(_PHYSXNET_CACHE, PHYSXNET_JSON_DIR)
    if cached is not None:
        _physxnet_factory_ids = cached
        return _physxnet_factory_ids

    _physxnet_factory_ids = _scan_and_cache(
        PHYSXNET_JSON_DIR, _PHYSXNET_CAT_TO_FACTORY,
        _PHYSXNET_CACHE, "MiscPhysXNetFactory")
    return _physxnet_factory_ids


def _load_physxmob_ids():
    """Load PhysX_mobility object IDs grouped by factory. Disk-cached."""
    global _physxmob_factory_ids
    if _physxmob_factory_ids is not None:
        return _physxmob_factory_ids

    cached = _load_from_cache(_PHYSXMOB_CACHE, PHYSXMOB_JSON_DIR)
    if cached is not None:
        _physxmob_factory_ids = cached
        return _physxmob_factory_ids

    _physxmob_factory_ids = _scan_and_cache(
        PHYSXMOB_JSON_DIR, _PHYSXMOB_CAT_TO_FACTORY,
        _PHYSXMOB_CACHE, "MiscPhysXMobilityFactory")
    return _physxmob_factory_ids


def get_physxnet_factory_ids(factory_name):
    """Get list of PhysXNet object IDs for a given factory name."""
    ids = _load_physxnet_ids()
    return ids.get(factory_name, [])


def get_physxmob_factory_ids(factory_name):
    """Get list of PhysX_mobility object IDs for a given factory name."""
    ids = _load_physxmob_ids()
    return ids.get(factory_name, [])


def get_all_physxnet_factories():
    """Get dict: factory_name -> [obj_ids]."""
    return _load_physxnet_ids()


def get_all_physxmob_factories():
    """Get dict: factory_name -> [obj_ids]."""
    return _load_physxmob_ids()


# ======================================================================
# Seed mapping
# ======================================================================
#
# For each factory:
#   seed 0..N-1  -> retrieval mode: seed i maps to objects[i]
#   seed N..M    -> material variant mode:
#                   base_object = objects[seed % N]
#                   material swaps seeded by (seed) for reproducibility

def seed_to_object_id(factory_name, seed, dataset="physxnet"):
    """Map a seed to an object ID.

    Returns (obj_id, is_variant) where is_variant=True means material swap.
    """
    if dataset == "physxnet":
        ids = get_physxnet_factory_ids(factory_name)
    else:
        ids = get_physxmob_factory_ids(factory_name)

    if not ids:
        return None, False

    n = len(ids)
    if seed < n:
        return ids[seed], False  # retrieval mode
    else:
        # variant mode: pick base object, flag for material swap
        base_idx = seed % n
        return ids[base_idx], True


# ======================================================================
# Render rules (for render_articulation.py)
# ======================================================================
#
# PhysXNet joint types from group_info:
#   A = revolute, B = prismatic, C = continuous
#
# For render rules, we define animode_joints per factory group.
# "moving_parts" is less relevant for PhysXNet (no semantic part names
# like PartNet). We use a special sentinel set {"__all__"} to indicate
# "animate all jointed parts".

PHYSXNET_RENDER_RULES = {
    "FurniturePhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # doors open (hinges)
            1: [("prismatic",)],                   # drawers slide
            2: [("continuous",)],                   # wheels spin
            3: [("revolute",), ("prismatic",)],    # doors + drawers
            4: [("revolute",), ("prismatic",), ("continuous",)],  # all
        },
    },
    "ChairPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("prismatic",)],                   # height adjust
            1: [("continuous",)],                   # seat spin / wheels
            2: [("revolute",)],                    # back tilt
            3: [("prismatic",), ("continuous",), ("revolute",)],  # all
        },
    },
    "LightingPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("continuous",)],                   # arm/head rotation
            1: [("revolute",)],                    # arm tilt
            2: [("prismatic",)],                   # height/extension
            3: [("continuous",), ("revolute",), ("prismatic",)],  # all
        },
    },
    "ElectronicsPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # lids/screens open
            1: [("prismatic",)],                   # buttons/sliders
            2: [("continuous",)],                   # dials/knobs
            3: [("revolute",), ("prismatic",), ("continuous",)],  # all
        },
    },
    "AudioPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("continuous",)],                   # ear cups rotate
            1: [("revolute",)],                    # fold
            2: [("continuous",), ("revolute",)],   # all
        },
    },
    "ContainerPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # lid opens
            1: [("prismatic",)],                   # slider
            2: [("continuous",)],                   # cap twist
            3: [("revolute",), ("prismatic",), ("continuous",)],  # all
        },
    },
    "KitchenPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # lid/door opens
            1: [("prismatic",)],                   # drawer/slider
            2: [("continuous",)],                   # knob turn
            3: [("revolute",), ("prismatic",), ("continuous",)],  # all
        },
    },
    "PlumbingPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("continuous",)],                   # faucet handle rotation
            1: [("revolute",)],                    # spout swing
            2: [("prismatic",)],                   # pump press
            3: [("continuous",), ("revolute",), ("prismatic",)],  # all
        },
    },
    "ToolPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("continuous",)],                   # scissor/plier pivot
            1: [("prismatic",)],                   # extensible parts
            2: [("revolute",)],                    # folding
            3: [("continuous",), ("prismatic",), ("revolute",)],  # all
        },
    },
    "WeaponPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # blade fold
            1: [("prismatic",)],                   # blade slide
            2: [("revolute",), ("prismatic",)],    # all
        },
    },
    "ClockPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("continuous",)],                   # hands rotate
            1: [("prismatic",)],                   # pendulum/slider
            2: [("continuous",), ("prismatic",)],  # all
        },
    },
    "DoorPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("continuous",)],                   # door swing
            1: [("revolute",)],                    # hinge with limits
            2: [("prismatic",)],                   # sliding door
            3: [("continuous",), ("revolute",), ("prismatic",)],  # all
        },
    },
    "BagPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("continuous",)],                   # zipper/flap
            1: [("prismatic",)],                   # handle extend
            2: [("continuous",), ("prismatic",)],  # all
        },
    },
    "WastePhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # lid flip
            1: [("prismatic",)],                   # foot pedal
            2: [("continuous",)],                   # wheel
            3: [("revolute",), ("prismatic",), ("continuous",)],  # all
        },
    },
    "AppliancePhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # door opens
            1: [("prismatic",)],                   # buttons/drawer
            2: [("continuous",)],                   # knobs/dials
            3: [("revolute",), ("prismatic",), ("continuous",)],  # all
        },
    },
    "ClothingPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # visor/strap fold
            1: [("continuous",)],                   # strap rotation
            2: [("revolute",), ("continuous",)],   # all
        },
    },
    "SportsPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # joints fold
            1: [("continuous",)],                   # wheels spin
            2: [("revolute",), ("continuous",)],   # all
        },
    },
    "DecorPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # hinges
            1: [("continuous",)],                   # rotatable parts
            2: [("revolute",), ("continuous",)],   # all
        },
    },
    "MobilityAidPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # fold
            1: [("continuous",)],                   # wheels
            2: [("prismatic",)],                   # height adjust
            3: [("revolute",), ("continuous",), ("prismatic",)],  # all
        },
    },
    "MiscPhysXNetFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],
            1: [("prismatic",)],
            2: [("continuous",)],
            3: [("revolute",), ("prismatic",), ("continuous",)],
        },
    },
}


# PhysX_mobility uses the same URDF-based joint types as PartNet-Mobility
PHYSXMOB_RENDER_RULES = {
    "FurniturePhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # doors
            1: [("prismatic",)],                   # drawers
            2: [("revolute",), ("prismatic",)],    # all
        },
    },
    "ElectronicsPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # lids/covers
            1: [("prismatic",)],                   # buttons/sliders
            2: [("continuous",)],                   # dials
            3: [("revolute",), ("prismatic",), ("continuous",)],
        },
    },
    "KitchenPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # door/lid
            1: [("prismatic",)],                   # drawer
            2: [("continuous",)],                   # knob
            3: [("revolute",), ("prismatic",), ("continuous",)],
        },
    },
    "AppliancePhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],
            1: [("prismatic",)],
            2: [("revolute",), ("prismatic",)],
        },
    },
    "PlumbingPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("continuous",)],                   # handles
            1: [("revolute",)],                    # spout
            2: [("prismatic",)],                   # pump
            3: [("continuous",), ("revolute",), ("prismatic",)],
        },
    },
    "ToolPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # fold/pivot
            1: [("prismatic",)],                   # slide
            2: [("revolute",), ("prismatic",)],
        },
    },
    "ContainerPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # lid
            1: [("prismatic",)],                   # slider
            2: [("revolute",), ("prismatic",)],
        },
    },
    "OpticalPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # arms fold
        },
    },
    "WritingPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("prismatic",)],                   # cap slide
            1: [("revolute",)],                    # cap flip
            2: [("prismatic",), ("revolute",)],
        },
    },
    "EducationPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("continuous",)],                   # globe rotation
            1: [("revolute",)],                    # tilt
            2: [("continuous",), ("revolute",)],
        },
    },
    "WastePhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # lid
            1: [("prismatic",)],                   # pedal
            2: [("revolute",), ("prismatic",)],
        },
    },
    "DoorPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # swing
            1: [("prismatic",)],                   # slide
            2: [("revolute",), ("prismatic",)],
        },
    },
    "ClockPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("continuous",)],                   # hands
            1: [("revolute",)],                    # door
            2: [("continuous",), ("revolute",)],
        },
    },
    "TransportPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("continuous",)],                   # wheels
            1: [("revolute",)],                    # fold
            2: [("prismatic",)],                   # extend
            3: [("continuous",), ("revolute",), ("prismatic",)],
        },
    },
    "LuggagePhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("prismatic",)],                   # handle extend
            1: [("revolute",)],                    # lid
            2: [("prismatic",), ("revolute",)],
        },
    },
    "LightingPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # head tilt
            1: [("continuous",)],                   # arm swing
            2: [("revolute",), ("continuous",)],
        },
    },
    "IgnitionPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],                    # lid flip
            1: [("prismatic",)],                   # slide
            2: [("revolute",), ("prismatic",)],
        },
    },
    "MiscPhysXMobilityFactory": {
        "moving_parts": {"__all__"},
        "animode_joints": {
            0: [("revolute",)],
            1: [("prismatic",)],
            2: [("continuous",)],
            3: [("revolute",), ("prismatic",), ("continuous",)],
        },
    },
}


# ======================================================================
# Split rules (for split_and_visualize.py)
# ======================================================================

PHYSXNET_SPLIT_RULES = {name: {"moving_parts": {"__all__"}, "description": f"PhysXNet {name.replace('PhysXNetFactory','').replace('PhysXMobilityFactory','')} objects"}
                        for name in PHYSXNET_RENDER_RULES}

PHYSXMOB_SPLIT_RULES = {name: {"moving_parts": {"__all__"}, "description": f"PhysX_mobility {name.replace('PhysXMobilityFactory','').replace('PhysXNetFactory','')} objects"}
                        for name in PHYSXMOB_RENDER_RULES}


# ======================================================================
# Derived constants
# ======================================================================

PHYSXNET_ANIMODES = {name: max(rules["animode_joints"].keys())
                     for name, rules in PHYSXNET_RENDER_RULES.items()}

PHYSXMOB_ANIMODES = {name: max(rules["animode_joints"].keys())
                     for name, rules in PHYSXMOB_RENDER_RULES.items()}

PHYSXNET_FACTORY_LIST = sorted(PHYSXNET_RENDER_RULES.keys())
PHYSXMOB_FACTORY_LIST = sorted(PHYSXMOB_RENDER_RULES.keys())

ALL_FACTORY_LIST = sorted(set(PHYSXNET_FACTORY_LIST + PHYSXMOB_FACTORY_LIST))
ALL_ANIMODES = {**PHYSXNET_ANIMODES, **PHYSXMOB_ANIMODES}
ALL_RENDER_RULES = {**PHYSXNET_RENDER_RULES, **PHYSXMOB_RENDER_RULES}
ALL_SPLIT_RULES = {**PHYSXNET_SPLIT_RULES, **PHYSXMOB_SPLIT_RULES}


# ======================================================================
# Material defaults (PBR: metallic, roughness)
# ======================================================================

PHYSXNET_MATERIAL_DEFAULTS = {
    "FurniturePhysXNetFactory":   {"default": (0.0, 0.55)},
    "ChairPhysXNetFactory":       {"default": (0.0, 0.50)},
    "LightingPhysXNetFactory":    {"default": (0.30, 0.40)},
    "ElectronicsPhysXNetFactory": {"default": (0.20, 0.40)},
    "AudioPhysXNetFactory":       {"default": (0.20, 0.45)},
    "ContainerPhysXNetFactory":   {"default": (0.0, 0.50)},
    "KitchenPhysXNetFactory":     {"default": (0.70, 0.30)},
    "PlumbingPhysXNetFactory":    {"default": (0.85, 0.25)},
    "ToolPhysXNetFactory":        {"default": (0.80, 0.30)},
    "WeaponPhysXNetFactory":      {"default": (0.85, 0.25)},
    "ClockPhysXNetFactory":       {"default": (0.30, 0.40)},
    "DoorPhysXNetFactory":        {"default": (0.0, 0.50)},
    "BagPhysXNetFactory":         {"default": (0.0, 0.60)},
    "WastePhysXNetFactory":       {"default": (0.50, 0.40)},
    "AppliancePhysXNetFactory":   {"default": (0.60, 0.35)},
    "ClothingPhysXNetFactory":    {"default": (0.0, 0.65)},
    "SportsPhysXNetFactory":      {"default": (0.10, 0.50)},
    "DecorPhysXNetFactory":       {"default": (0.0, 0.45)},
    "MobilityAidPhysXNetFactory": {"default": (0.50, 0.40)},
    "MiscPhysXNetFactory":        {"default": (0.10, 0.45)},
}

PHYSXMOB_MATERIAL_DEFAULTS = {
    "FurniturePhysXMobilityFactory":  {"default": (0.0, 0.55)},
    "ElectronicsPhysXMobilityFactory":{"default": (0.20, 0.40)},
    "KitchenPhysXMobilityFactory":    {"default": (0.70, 0.30)},
    "AppliancePhysXMobilityFactory":  {"default": (0.50, 0.40)},
    "PlumbingPhysXMobilityFactory":   {"default": (0.85, 0.25)},
    "ToolPhysXMobilityFactory":       {"default": (0.80, 0.30)},
    "ContainerPhysXMobilityFactory":  {"default": (0.0, 0.50)},
    "OpticalPhysXMobilityFactory":    {"default": (0.60, 0.30)},
    "WritingPhysXMobilityFactory":    {"default": (0.40, 0.35)},
    "EducationPhysXMobilityFactory":  {"default": (0.10, 0.45)},
    "WastePhysXMobilityFactory":      {"default": (0.50, 0.40)},
    "DoorPhysXMobilityFactory":       {"default": (0.0, 0.50)},
    "ClockPhysXMobilityFactory":      {"default": (0.30, 0.40)},
    "TransportPhysXMobilityFactory":  {"default": (0.50, 0.40)},
    "LuggagePhysXMobilityFactory":    {"default": (0.0, 0.55)},
    "LightingPhysXMobilityFactory":   {"default": (0.30, 0.40)},
    "IgnitionPhysXMobilityFactory":   {"default": (0.60, 0.30)},
    "MiscPhysXMobilityFactory":       {"default": (0.10, 0.45)},
}

ALL_MATERIAL_DEFAULTS = {**PHYSXNET_MATERIAL_DEFAULTS, **PHYSXMOB_MATERIAL_DEFAULTS}


# ======================================================================
# Merge functions (for render_articulation.py integration)
# ======================================================================

def merge_render_rules(existing_rules: dict) -> dict:
    """Merge PhysXNet+PhysX_mobility render rules into existing FACTORY_RULES."""
    merged = dict(existing_rules)
    for name, rules in ALL_RENDER_RULES.items():
        if name not in merged:
            merged[name] = rules
    return merged


def merge_split_rules(existing_rules: dict) -> dict:
    """Merge PhysXNet+PhysX_mobility split rules into existing FACTORY_RULES."""
    merged = dict(existing_rules)
    for name, rules in ALL_SPLIT_RULES.items():
        if name not in merged:
            merged[name] = rules
    return merged


# ======================================================================
# Determine dataset type from factory name
# ======================================================================

def factory_dataset(factory_name):
    """Return 'physxnet', 'physxmob', or None."""
    if factory_name in PHYSXNET_RENDER_RULES:
        return "physxnet"
    elif factory_name in PHYSXMOB_RENDER_RULES:
        return "physxmob"
    return None


# ======================================================================
# Summary (when run directly)
# ======================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("PhysXNet + PhysX_mobility Factory Rules Summary")
    print("=" * 72)

    # Load IDs
    pxn_ids = _load_physxnet_ids()
    pxm_ids = _load_physxmob_ids()

    total_pxn = sum(len(v) for v in pxn_ids.values())
    total_pxm = sum(len(v) for v in pxm_ids.values())

    print(f"\n-- PhysXNet: {total_pxn} objects in {len(pxn_ids)} factories --")
    for name in sorted(pxn_ids.keys()):
        ids = pxn_ids[name]
        n_animodes = PHYSXNET_ANIMODES.get(name, 0)
        print(f"  {name:<35} {len(ids):>6} objects  animodes=0..{n_animodes}")

    print(f"\n-- PhysX_mobility: {total_pxm} objects in {len(pxm_ids)} factories --")
    for name in sorted(pxm_ids.keys()):
        ids = pxm_ids[name]
        n_animodes = PHYSXMOB_ANIMODES.get(name, 0)
        print(f"  {name:<40} {len(ids):>5} objects  animodes=0..{n_animodes}")

    print(f"\n-- Totals --")
    print(f"  PhysXNet factories:       {len(pxn_ids)}")
    print(f"  PhysX_mobility factories: {len(pxm_ids)}")
    print(f"  Total unique factories:   {len(ALL_FACTORY_LIST)}")
    print(f"  PhysXNet objects:         {total_pxn}")
    print(f"  PhysX_mobility objects:   {total_pxm}")
