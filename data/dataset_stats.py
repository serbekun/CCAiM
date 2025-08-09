# dataset_stats.py
import json

LABELS_JSON = "clouds_1/labels.json"

# WMO
ALL_CLASSES = [
    "Cirrus",
    "Cirrocumulus",
    "Cirrostratus",
    "Altocumulus",
    "Altostratus",
    "Nimbostratus",
    "Stratocumulus",
    "Stratus",
    "Cumulus",
    "Cumulonimbus"
]

def main():
    # load json
    try:
        with open(LABELS_JSON, "r") as f:
            labels = json.load(f)
    except FileNotFoundError:
        print(f"[Error] file {LABELS_JSON} not found!")
        return

    total_images = len(labels) - 1
    print(f"\nData Set statistic ({LABELS_JSON})")
    print(f"Images total: {total_images}\n")

    # read all classes
    for cloud_class in ALL_CLASSES:
        count = sum(1 for v in labels.values() if v == cloud_class)
        percent = (count / total_images * 100) if total_images > 0 else 0
        print(f"{cloud_class:15s} — {count:3d} col. ({percent:5.2f}%)")

if __name__ == "__main__":
    main()