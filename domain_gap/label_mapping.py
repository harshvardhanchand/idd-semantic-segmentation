"""Simple Cityscapes trainId → IDD Level1 ID mapping."""

# IDD-Lite Level 1 class names (7 classes)
IDD_LITE_L1_CLASSES = [
    "drivable",
    "non-drivable",
    "living-things",
    "vehicles",
    "road-side-objs",
    "far-objects",
    "sky",
]

# Direct mapping: Cityscapes trainId (0-18) → IDD Level1 ID (0-6)
CITYSCAPES_TO_IDD_MAPPING = {
    0: 0,  # road → drivable
    1: 1,  # sidewalk → non-drivable
    2: 5,  # building → far-objects
    3: 4,  # wall → road-side-objs
    4: 4,  # fence → road-side-objs
    5: 4,  # pole → road-side-objs
    6: 4,  # traffic_light → road-side-objs
    7: 4,  # traffic_sign → road-side-objs
    8: 5,  # vegetation → far-objects
    9: 1,  # terrain → non-drivable
    10: 6,  # sky → sky
    11: 2,  # person → living-things
    12: 2,  # rider → living-things
    13: 3,  # car → vehicles
    14: 3,  # truck → vehicles
    15: 3,  # bus → vehicles
    16: 3,  # train → vehicles
    17: 3,  # motorcycle → vehicles
    18: 3,  # bicycle → vehicles
    255: 255,  # ignore → ignore
}


def print_mapping_summary():
    """Print a summary of the Cityscapes to IDD mapping."""
    print("Cityscapes → IDD Level 1 Mapping:")
    cityscapes_classes = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic_light",
        "traffic_sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    for cs_id, cs_name in enumerate(cityscapes_classes):
        idd_id = CITYSCAPES_TO_IDD_MAPPING.get(cs_id, 255)
        idd_name = (
            IDD_LITE_L1_CLASSES[idd_id]
            if idd_id < len(IDD_LITE_L1_CLASSES)
            else "ignore"
        )
        print(f"  {cs_id:2d} {cs_name:13s} → {idd_id} {idd_name}")
