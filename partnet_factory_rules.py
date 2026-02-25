"""
PartNet-Mobility factory rules for render_articulation.py and split_and_visualize.py.

Maps PartNet-Mobility categories to:
  - PARTNET_RENDER_RULES: render animation rules (moving_parts, animode_joints)
  - PARTNET_SPLIT_RULES: 2-part split rules (moving_parts, description)
  - PARTNET_ANIMODES: max animode index per factory
  - PARTNET_FACTORY_LIST: ordered list of all factory names

Naming convention:
  - If a PartNet category overlaps with an existing Infinite Mobility factory,
    the factory name gets a "SapienFactory" suffix (e.g. BottleSapienFactory, DishwasherSapienFactory).
  - Part names use snake_case ending with "_part" (e.g. lid_part, door_part).

Overlapping categories (PartNet <-> IM factory):
  Bottle       <-> BottleFactory
  Dishwasher   <-> DishwasherFactory
  Lamp         <-> LampFactory
  Microwave    <-> MicrowaveFactory
  Oven         <-> OvenFactory
  Toilet       <-> ToiletFactory
  Window       <-> WindowFactory
  Door         <-> LiteDoorFactory
  Faucet       <-> TapFactory
  StorageFurn. <-> KitchenCabinetFactory
  Table        <-> TableCocktailFactory / TableDiningFactory
  Refrigerator <-> BeverageFridgeFactory
  KitchenPot   <-> PotFactory
  Chair        <-> OfficeChairFactory / BarChairFactory
"""

# ──────────────────────────────────────────────────────────────────────────────
# Overlapping IM factories (PartNet category -> existing IM factory name)
# ──────────────────────────────────────────────────────────────────────────────
OVERLAP_MAP = {
    "Bottle":            "BottleFactory",
    "Dishwasher":        "DishwasherFactory",
    "Lamp":              "LampFactory",
    "Microwave":         "MicrowaveFactory",
    "Oven":              "OvenFactory",
    "Toilet":            "ToiletFactory",
    "Window":            "WindowFactory",
    "Door":              "LiteDoorFactory",
    "Faucet":            "TapFactory",
    "StorageFurniture":  "KitchenCabinetFactory",
    "Table":             "TableDiningFactory",
    "Refrigerator":      "BeverageFridgeFactory",
    "KitchenPot":        "PotFactory",
    "Chair":             "OfficeChairFactory",
}

# Categories that do NOT overlap with any existing IM factory
NEW_CATEGORIES = [
    "Scissors", "Fan", "Globe", "Safe", "TrashCan", "Laptop", "Box",
    "Bucket", "Display", "Eyeglasses", "Kettle", "Pliers",
    "Stapler", "Suitcase", "Cart",
    "CoffeeMachine", "FoldingChair", "Mouse", "Pen", "USB",
    "Switch", "WashingMachine",
    "Keyboard", "Phone", "Printer", "Remote",
]


# ──────────────────────────────────────────────────────────────────────────────
# Render rules  (for render_articulation.py)
#
# Format per entry:
#   "FactoryName": {
#       "moving_parts": {set of part_name strings},
#       "animode_joints": {
#           0: [("joint_type",)],              # animode 0
#           1: [("joint_type", index)],         # animode 1 (optional)
#           ...
#       },
#       "exclude_parts": {set},                 # optional
#   }
#
# Joint types used by PartNet-Mobility URDFs:
#   "revolute"   - hinge with limits
#   "prismatic"  - slider with limits
#   "continuous"  - unlimited rotation
# ──────────────────────────────────────────────────────────────────────────────

PARTNET_RENDER_RULES = {
    # ══════════════════════════════════════════════════════════════════════════
    # Overlapping categories (Sapien suffix)
    # ══════════════════════════════════════════════════════════════════════════

    # Bottle: rotation/translation lid
    "BottleSapienFactory": {
        "moving_parts": {"rotation_lid_part", "translation_lid_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # lid slides up
            1: [("continuous",)],                      # lid rotates (unscrew)
            2: [("prismatic",), ("continuous",)],      # all
        },
    },

    # Dishwasher: rotation_door, button, knob, shelf
    "DishwasherSapienFactory": {
        "moving_parts": {"rotation_door_part", "button_part", "knob_part", "shelf_part"},
        "animode_joints": {
            0: [("revolute",)],                        # door opens
            1: [("prismatic",)],                       # shelf slides
            2: [("revolute",), ("prismatic",)],        # all
        },
    },

    # Lamp: head, bars, button, fastener, lever, toggle_button
    "LampSapienFactory": {
        "moving_parts": {"head_part", "rotation_bar_part", "translation_bar_part",
                         "button_part", "fastener_part", "fastener_connector_part",
                         "lever_part", "toggle_button_part"},
        "animode_joints": {
            0: [("revolute", 0)],                      # head tilts
            1: [("revolute", -1)],                     # body arm rotates
            2: [("revolute",)],                        # all hinges
        },
    },

    # Microwave: door, button, knob, rotation_tray
    "MicrowaveSapienFactory": {
        "moving_parts": {"door_part", "button_part", "knob_part", "rotation_tray_part"},
        "animode_joints": {
            0: [("revolute",)],                        # door opens
            1: [("prismatic",)],                       # buttons press
            2: [("revolute",), ("prismatic",)],        # all
        },
    },

    # Oven: door, button, knob, rotation_tray, translation_tray
    "OvenSapienFactory": {
        "moving_parts": {"door_part", "button_part", "knob_part",
                         "rotation_tray_part", "translation_tray_part"},
        "animode_joints": {
            0: [("revolute", 0)],                      # door opens (first revolute)
            1: [("revolute", -1)],                     # knobs turn
            2: [("revolute",)],                        # all revolute
        },
    },

    # Toilet: lid, seat, button, knob, lever, pump_lid
    "ToiletSapienFactory": {
        "moving_parts": {"lid_part", "seat_part", "button_part", "knob_part",
                         "lever_part", "pump_lid_part"},
        "animode_joints": {
            0: [("revolute", -1)],                     # lid only (last revolute)
            1: [("revolute", 0)],                      # seat only (first revolute)
            2: [("revolute",)],                        # both lid+seat
        },
    },

    # Window: rotation_window, translation_window
    "WindowSapienFactory": {
        "moving_parts": {"rotation_window_part", "translation_window_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # sliding pane
            1: [("revolute",)],                        # hinged pane (if exists)
            2: [("prismatic",), ("revolute",)],        # all
        },
    },

    # Door: rotation_door, translation_door
    "DoorSapienFactory": {
        "moving_parts": {"rotation_door_part", "translation_door_part"},
        "animode_joints": {
            0: [("revolute",)],                        # door swings
        },
    },

    # Faucet: spout, stem, switch
    "FaucetSapienFactory": {
        "moving_parts": {"spout_part", "stem_part", "switch_part"},
        "animode_joints": {
            0: [("revolute", 0)],                      # spout swivels
            1: [("revolute", -1)],                     # handle turns
            2: [("continuous",)],                      # continuous rotation
            3: [("revolute",), ("continuous",)],       # all
        },
    },

    # StorageFurniture: rotation_door, translation_door, drawer, board, caster, wheel
    "StorageFurnitureSapienFactory": {
        "moving_parts": {"rotation_door_part", "translation_door_part", "drawer_part",
                         "board_part", "caster_part", "wheel_part"},
        "animode_joints": {
            0: [("revolute",)],                        # doors open
            1: [("prismatic",)],                       # drawers slide
            2: [("revolute",), ("prismatic",)],        # all
        },
    },

    # Table: drawer, door, board, handle, caster, wheel
    "TableSapienFactory": {
        "moving_parts": {"drawer_part", "door_part", "board_part", "handle_part",
                         "caster_part", "wheel_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # drawer slides
            1: [("revolute",)],                        # door/flap (if any)
            2: [("prismatic",), ("revolute",)],        # all
        },
    },

    # Refrigerator: door
    "RefrigeratorSapienFactory": {
        "moving_parts": {"door_part"},
        "animode_joints": {
            0: [("revolute", 0)],                      # top door
            1: [("revolute", -1)],                     # bottom door
            2: [("revolute",)],                        # both doors
        },
    },

    # KitchenPot: lid, button, slider
    "KitchenPotSapienFactory": {
        "moving_parts": {"lid_part", "button_part", "slider_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # lid lifts
            1: [("continuous",)],                      # lid rotates
            2: [("prismatic",), ("continuous",)],      # all
        },
    },

    # Chair: seat, caster, knob, lever, rotation_body, wheel
    "ChairSapienFactory": {
        "moving_parts": {"seat_part", "caster_part", "knob_part", "lever_part",
                         "rotation_body_part", "wheel_part"},
        "exclude_parts": {"unknown_part"},
        "animode_joints": {
            0: [("revolute",)],                        # back tilts
            1: [("prismatic",)],                       # seat height
            2: [("continuous",)],                      # wheel rotation
            3: [("revolute",), ("prismatic",), ("continuous",)],  # all
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # New categories (no overlap with existing IM factories)
    # ══════════════════════════════════════════════════════════════════════════

    # Scissors: leg_part (both legs share name, one is base one is moving)
    "ScissorsFactory": {
        "moving_parts": {"leg_part"},
        "animode_joints": {
            0: [("revolute",)],                        # blade pivots (scissor open/close)
        },
    },

    # Fan: rotor, button, slider
    "FanFactory": {
        "moving_parts": {"rotor_part", "button_part", "slider_part"},
        "animode_joints": {
            0: [("continuous",)],                      # blade rotation
            1: [("revolute",)],                        # head tilt/oscillation
            2: [("continuous",), ("revolute",)],       # all
        },
    },

    # Globe: sphere, circle
    "GlobeFactory": {
        "moving_parts": {"sphere_part", "circle_part"},
        "animode_joints": {
            0: [("continuous",)],                      # sphere rotation
            1: [("revolute",)],                        # tilt axis
            2: [("continuous",), ("revolute",)],       # all
        },
    },

    # Safe: door, button, handle, knob
    "SafeFactory": {
        "moving_parts": {"door_part", "button_part", "handle_part", "knob_part"},
        "animode_joints": {
            0: [("revolute",)],                        # door opens
            1: [("continuous",)],                      # switch/dial rotates
            2: [("revolute",), ("continuous",)],       # all
        },
    },

    # TrashCan: cover_lid, lid, door, drawer, foot_pad, handle, wheel
    "TrashCanFactory": {
        "moving_parts": {"cover_lid_part", "lid_part", "door_part", "drawer_part",
                         "foot_pad_part", "handle_part", "wheel_part"},
        "animode_joints": {
            0: [("revolute",)],                        # cover flips open
            1: [("prismatic",)],                       # foot pedal / drawer
            2: [("revolute",), ("prismatic",)],        # all
        },
    },

    # Laptop: screen
    "LaptopFactory": {
        "moving_parts": {"screen_part"},
        "animode_joints": {
            0: [("revolute",)],                        # screen opens
        },
    },

    # Box: rotation_lid, drawer, handle, lock
    "BoxFactory": {
        "moving_parts": {"rotation_lid_part", "drawer_part", "handle_part", "lock_part"},
        "animode_joints": {
            0: [("revolute",)],                        # lid hinges open
            1: [("prismatic",)],                       # drawer slides
            2: [("revolute",), ("prismatic",)],        # all
        },
    },

    # Bucket: handle
    "BucketFactory": {
        "moving_parts": {"handle_part"},
        "animode_joints": {
            0: [("revolute",)],                        # handle swings
        },
    },

    # Display: rotation_screen, translation_screen, button
    "DisplayFactory": {
        "moving_parts": {"rotation_screen_part", "translation_screen_part", "button_part"},
        "animode_joints": {
            0: [("revolute",)],                        # screen tilts
            1: [("prismatic",)],                       # height adjust
            2: [("revolute",), ("prismatic",)],        # all
        },
    },

    # Eyeglasses: leg
    "EyeglassesFactory": {
        "moving_parts": {"leg_part"},
        "animode_joints": {
            0: [("revolute", 0)],                      # left leg folds
            1: [("revolute", -1)],                     # right leg folds
            2: [("revolute",)],                        # both legs fold
        },
    },

    # Kettle: lid, rotation_lid, button, handle, lever
    "KettleFactory": {
        "moving_parts": {"lid_part", "rotation_lid_part", "button_part",
                         "handle_part", "lever_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # lid lifts
            1: [("revolute",)],                        # lid flips (hinged kettle)
            2: [("prismatic",), ("revolute",)],        # all
        },
    },

    # Pliers: leg, rotation_blade
    "PliersFactory": {
        "moving_parts": {"leg_part", "rotation_blade_part"},
        "animode_joints": {
            0: [("revolute",)],                        # jaw opens/closes
        },
    },

    # Stapler: lid, stapler_body
    "StaplerFactory": {
        "moving_parts": {"lid_part", "stapler_body_part"},
        "animode_joints": {
            0: [("revolute",)],                        # top lid hinges open
        },
    },

    # Suitcase: translation_handle, rotation_handle, button, lid, lock, caster, wheel
    "SuitcaseFactory": {
        "moving_parts": {"translation_handle_part", "rotation_handle_part", "button_part",
                         "lid_part", "lock_part", "caster_part", "wheel_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # handle extends
            1: [("revolute",)],                        # lid opens
            2: [("prismatic",), ("revolute",)],        # all
        },
    },

    # Cart: wheel, caster, steering_wheel
    "CartFactory": {
        "moving_parts": {"wheel_part", "caster_part", "steering_wheel_part"},
        "animode_joints": {
            0: [("continuous",)],                      # wheels spin
        },
    },

    # CoffeeMachine: button, knob, lever, lid, container, rotation variants, slider, rotor, wheel, toggle, portafilter
    "CoffeeMachineFactory": {
        "moving_parts": {"button_part", "knob_part", "lever_part", "lid_part",
                         "container_part", "rotation_container_part", "rotation_lid_part",
                         "rotation_slider_part", "rotor_part", "slider_part",
                         "toggle_button_part", "wheel_part", "portafilter_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # buttons/levers press
            1: [("revolute",)],                        # lid opens (if exists)
            2: [("prismatic",), ("revolute",)],        # all
        },
    },

    # FoldingChair: leg, seat
    "FoldingChairFactory": {
        "moving_parts": {"leg_part", "seat_part"},
        "animode_joints": {
            0: [("revolute",)],                        # folds (all revolute)
        },
    },

    # Mouse: button, wheel, ball
    "MouseFactory": {
        "moving_parts": {"button_part", "wheel_part", "ball_part"},
        "animode_joints": {
            0: [("continuous",)],                      # scroll wheel rotates
            1: [("revolute",)],                        # button click
            2: [("continuous",), ("revolute",)],       # all
        },
    },

    # Pen: cap, button
    "PenFactory": {
        "moving_parts": {"cap_part", "button_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # cap slides off
        },
    },

    # USB: lid, handle, usb_rotation
    "USBFactory": {
        "moving_parts": {"lid_part", "handle_part", "usb_rotation_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # cap slides off
            1: [("revolute",)],                        # cap flips (if hinged)
            2: [("prismatic",), ("revolute",)],        # all
        },
    },

    # Switch: (no data_infos found — keep generic)
    "SwitchFactory": {
        "moving_parts": {"switch_part", "button_part"},
        "animode_joints": {
            0: [("revolute", 0)],                      # first switch toggles
            1: [("revolute", -1)],                     # last switch toggles
            2: [("revolute",)],                        # all switches
        },
    },

    # WashingMachine: door, button, knob
    "WashingMachineFactory": {
        "moving_parts": {"door_part", "button_part", "knob_part"},
        "animode_joints": {
            0: [("revolute",)],                        # door opens
            1: [("prismatic",)],                       # buttons press
            2: [("revolute",), ("prismatic",)],        # all
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # Additional categories found in data (not in original NEW_CATEGORIES)
    # ══════════════════════════════════════════════════════════════════════════

    # Keyboard: key, tilt_leg
    "KeyboardFactory": {
        "moving_parts": {"key_part", "tilt_leg_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # key presses
            1: [("revolute",)],                        # tilt leg flips
            2: [("prismatic",), ("revolute",)],        # all
        },
    },

    # Phone: button, rotation_button, rotation_lid, slider, translation_lid
    "PhoneFactory": {
        "moving_parts": {"button_part", "rotation_button_part", "rotation_lid_part",
                         "slider_part", "translation_lid_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # buttons press / slider
            1: [("revolute",)],                        # lid flips
            2: [("prismatic",), ("revolute",)],        # all
        },
    },

    # Printer: button, drawer, knob, slider, toggle_button
    "PrinterFactory": {
        "moving_parts": {"button_part", "drawer_part", "knob_part",
                         "slider_part", "toggle_button_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # drawer / slider
            1: [("revolute",)],                        # knob turns
            2: [("prismatic",), ("revolute",)],        # all
        },
    },

    # Remote: button, knob, rotation_button
    "RemoteFactory": {
        "moving_parts": {"button_part", "knob_part", "rotation_button_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # buttons press
            1: [("revolute",)],                        # rotation button
            2: [("prismatic",), ("revolute",)],        # all
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Split rules  (for split_and_visualize.py)
#
# Format per entry:
#   "FactoryName": {
#       "moving_parts": {set of part_name strings},
#       "description": "brief description of articulation",
#   }
# ──────────────────────────────────────────────────────────────────────────────

PARTNET_SPLIT_RULES = {
    # ══════════════════════════════════════════════════════════════════════════
    # Overlapping categories (Sapien suffix)
    # ══════════════════════════════════════════════════════════════════════════

    "BottleSapienFactory": {
        "moving_parts": {"rotation_lid_part", "translation_lid_part"},
        "description": "lid (slider + rotation)",
    },
    "DishwasherSapienFactory": {
        "moving_parts": {"rotation_door_part", "button_part", "knob_part", "shelf_part"},
        "description": "door (hinge) + shelf (slider)",
    },
    "LampSapienFactory": {
        "moving_parts": {"head_part", "rotation_bar_part", "translation_bar_part",
                         "button_part", "fastener_part", "fastener_connector_part",
                         "lever_part", "toggle_button_part"},
        "description": "lamp head + body arm (hinge)",
    },
    "MicrowaveSapienFactory": {
        "moving_parts": {"door_part", "button_part", "knob_part", "rotation_tray_part"},
        "description": "door (hinge) + buttons (slider) + tray",
    },
    "OvenSapienFactory": {
        "moving_parts": {"door_part", "button_part", "knob_part",
                         "rotation_tray_part", "translation_tray_part"},
        "description": "door (hinge) + knobs (hinge) + trays",
    },
    "ToiletSapienFactory": {
        "moving_parts": {"lid_part", "seat_part", "button_part", "knob_part",
                         "lever_part", "pump_lid_part"},
        "description": "lid + seat (hinge) + flush controls",
    },
    "WindowSapienFactory": {
        "moving_parts": {"rotation_window_part", "translation_window_part"},
        "description": "sliding/hinged window pane",
    },
    "DoorSapienFactory": {
        "moving_parts": {"rotation_door_part", "translation_door_part"},
        "description": "door (hinge/slider)",
    },
    "FaucetSapienFactory": {
        "moving_parts": {"spout_part", "stem_part", "switch_part"},
        "description": "spout (hinge) + stem/switch (rotation)",
    },
    "StorageFurnitureSapienFactory": {
        "moving_parts": {"rotation_door_part", "translation_door_part", "drawer_part",
                         "board_part", "caster_part", "wheel_part"},
        "description": "door (hinge) + drawer (slider)",
    },
    "TableSapienFactory": {
        "moving_parts": {"drawer_part", "door_part", "board_part", "handle_part",
                         "caster_part", "wheel_part"},
        "description": "drawer (slider) + door (hinge)",
    },
    "RefrigeratorSapienFactory": {
        "moving_parts": {"door_part"},
        "description": "door x2 (hinge)",
    },
    "KitchenPotSapienFactory": {
        "moving_parts": {"lid_part", "button_part", "slider_part"},
        "description": "lid (slider + rotation) + controls",
    },
    "ChairSapienFactory": {
        "moving_parts": {"seat_part", "caster_part", "knob_part", "lever_part",
                         "rotation_body_part", "wheel_part"},
        "description": "seat + wheels + controls (revolute/prismatic/continuous)",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # New categories (no overlap)
    # ══════════════════════════════════════════════════════════════════════════

    "ScissorsFactory": {
        "moving_parts": {"leg_part"},
        "description": "blade/leg (hinge pivot)",
    },
    "FanFactory": {
        "moving_parts": {"rotor_part", "button_part", "slider_part"},
        "description": "rotor (rotation) + controls",
    },
    "GlobeFactory": {
        "moving_parts": {"sphere_part", "circle_part"},
        "description": "sphere (rotation on axis)",
    },
    "SafeFactory": {
        "moving_parts": {"door_part", "button_part", "handle_part", "knob_part"},
        "description": "door (hinge) + handle/knob (rotation)",
    },
    "TrashCanFactory": {
        "moving_parts": {"cover_lid_part", "lid_part", "door_part", "drawer_part",
                         "foot_pad_part", "handle_part", "wheel_part"},
        "description": "cover/lid (hinge) + foot pedal + drawer",
    },
    "LaptopFactory": {
        "moving_parts": {"screen_part"},
        "description": "screen (hinge opens)",
    },
    "BoxFactory": {
        "moving_parts": {"rotation_lid_part", "drawer_part", "handle_part", "lock_part"},
        "description": "lid (hinge) + drawer (slider)",
    },
    "BucketFactory": {
        "moving_parts": {"handle_part"},
        "description": "handle (hinge swings)",
    },
    "DisplayFactory": {
        "moving_parts": {"rotation_screen_part", "translation_screen_part", "button_part"},
        "description": "screen (tilt + height adjust)",
    },
    "EyeglassesFactory": {
        "moving_parts": {"leg_part"},
        "description": "legs fold (hinge x2)",
    },
    "KettleFactory": {
        "moving_parts": {"lid_part", "rotation_lid_part", "button_part",
                         "handle_part", "lever_part"},
        "description": "lid (slider/hinge) + controls",
    },
    "PliersFactory": {
        "moving_parts": {"leg_part", "rotation_blade_part"},
        "description": "jaw/leg pivots (hinge)",
    },
    "StaplerFactory": {
        "moving_parts": {"lid_part", "stapler_body_part"},
        "description": "lid + body hinges open",
    },
    "SuitcaseFactory": {
        "moving_parts": {"translation_handle_part", "rotation_handle_part", "button_part",
                         "lid_part", "lock_part", "caster_part", "wheel_part"},
        "description": "handle extends (slider) + lid opens",
    },
    "CartFactory": {
        "moving_parts": {"wheel_part", "caster_part", "steering_wheel_part"},
        "description": "wheels spin (continuous rotation)",
    },
    "CoffeeMachineFactory": {
        "moving_parts": {"button_part", "knob_part", "lever_part", "lid_part",
                         "container_part", "rotation_container_part", "rotation_lid_part",
                         "rotation_slider_part", "rotor_part", "slider_part",
                         "toggle_button_part", "wheel_part", "portafilter_part"},
        "description": "buttons/levers (slider) + lid (hinge) + containers",
    },
    "FoldingChairFactory": {
        "moving_parts": {"leg_part", "seat_part"},
        "description": "seat + legs fold (hinge)",
    },
    "MouseFactory": {
        "moving_parts": {"button_part", "wheel_part", "ball_part"},
        "description": "scroll wheel (rotation) + buttons (click)",
    },
    "PenFactory": {
        "moving_parts": {"cap_part", "button_part"},
        "description": "cap slides off (slider)",
    },
    "USBFactory": {
        "moving_parts": {"lid_part", "handle_part", "usb_rotation_part"},
        "description": "lid/cap slides or flips off",
    },
    "SwitchFactory": {
        "moving_parts": {"switch_part", "button_part"},
        "description": "switch toggles (hinge x1-3)",
    },
    "WashingMachineFactory": {
        "moving_parts": {"door_part", "button_part", "knob_part"},
        "description": "door (hinge) + buttons/knobs",
    },

    # Additional categories
    "KeyboardFactory": {
        "moving_parts": {"key_part", "tilt_leg_part"},
        "description": "keys press + tilt legs flip",
    },
    "PhoneFactory": {
        "moving_parts": {"button_part", "rotation_button_part", "rotation_lid_part",
                         "slider_part", "translation_lid_part"},
        "description": "buttons + lid (flip/slide)",
    },
    "PrinterFactory": {
        "moving_parts": {"button_part", "drawer_part", "knob_part",
                         "slider_part", "toggle_button_part"},
        "description": "drawer (slider) + buttons/knobs",
    },
    "RemoteFactory": {
        "moving_parts": {"button_part", "knob_part", "rotation_button_part"},
        "description": "buttons press + knob turns",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Max animode index per factory
# (Derived from the number of entries in animode_joints minus 1)
# ──────────────────────────────────────────────────────────────────────────────

PARTNET_ANIMODES = {name: max(rules["animode_joints"].keys())
                    for name, rules in PARTNET_RENDER_RULES.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Ordered list of all PartNet factory names
# ──────────────────────────────────────────────────────────────────────────────

PARTNET_FACTORY_LIST = sorted(PARTNET_RENDER_RULES.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Reverse mapping: PartNet category name -> factory name used here
# ──────────────────────────────────────────────────────────────────────────────

PARTNET_CATEGORY_TO_FACTORY = {}
# Overlapping categories get Sapien suffix
for cat, im_factory in OVERLAP_MAP.items():
    PARTNET_CATEGORY_TO_FACTORY[cat] = cat + "SapienFactory"
# New categories keep their original name
for cat in NEW_CATEGORIES:
    PARTNET_CATEGORY_TO_FACTORY[cat] = cat + "Factory"


# ──────────────────────────────────────────────────────────────────────────────
# Utility: merge PartNet rules into existing FACTORY_RULES dicts
# ──────────────────────────────────────────────────────────────────────────────

def merge_render_rules(existing_rules: dict) -> dict:
    """Merge PARTNET_RENDER_RULES into an existing FACTORY_RULES dict
    (from render_articulation.py). Returns a new dict with all entries."""
    merged = dict(existing_rules)
    for name, rules in PARTNET_RENDER_RULES.items():
        if name not in merged:
            merged[name] = rules
    return merged


def merge_split_rules(existing_rules: dict) -> dict:
    """Merge PARTNET_SPLIT_RULES into an existing FACTORY_RULES dict
    (from split_and_visualize.py). Returns a new dict with all entries."""
    merged = dict(existing_rules)
    for name, rules in PARTNET_SPLIT_RULES.items():
        if name not in merged:
            merged[name] = rules
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Summary when run directly
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("PartNet Factory Rules Summary")
    print("=" * 72)

    print(f"\nTotal factories: {len(PARTNET_FACTORY_LIST)}")
    print(f"  Overlapping (Sapien suffix): {len(OVERLAP_MAP)}")
    print(f"  New (no overlap):            {len(NEW_CATEGORIES)}")

    print("\n── Overlapping categories ──")
    print(f"  {'PartNet Category':<22} {'IM Factory':<28} {'PartNet Factory':<24}")
    print(f"  {'─'*22} {'─'*28} {'─'*24}")
    for cat, im in sorted(OVERLAP_MAP.items()):
        pn = PARTNET_CATEGORY_TO_FACTORY[cat]
        print(f"  {cat:<22} {im:<28} {pn:<24}")

    print("\n── New categories ──")
    for cat in sorted(NEW_CATEGORIES):
        factory = PARTNET_CATEGORY_TO_FACTORY[cat]
        rules = PARTNET_RENDER_RULES[factory]
        n_animodes = PARTNET_ANIMODES[factory]
        moving = ", ".join(sorted(rules["moving_parts"]))
        print(f"  {factory:<22} animodes=0..{n_animodes}  moving=[{moving}]")

    print("\n── All factories (alphabetical) ──")
    for i, name in enumerate(PARTNET_FACTORY_LIST):
        split_desc = PARTNET_SPLIT_RULES[name]["description"]
        n_animodes = PARTNET_ANIMODES[name]
        print(f"  {i+1:2d}. {name:<24} animodes=0..{n_animodes:<2d} | {split_desc}")
