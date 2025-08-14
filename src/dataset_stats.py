# dataset_stats.py
import json
import os

LABELS_JSON = "clouds_1/labels.json"

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

    try:
        with open(LABELS_JSON, "r") as f:
            labels = json.load(f)
    except FileNotFoundError:
        print(f"[Error] file {LABELS_JSON} not found!")
        return

    total_images = len(labels) - 1

    report_lines = []
    report_lines.append(f"statistic dataset ({LABELS_JSON})")
    report_lines.append(f"images col: {total_images}\n")

    for cloud_class in ALL_CLASSES:
        count = sum(1 for v in labels.values() if v == cloud_class)
        percent = (count / total_images * 100) if total_images > 0 else 0
        report_lines.append(f"{cloud_class:15s} — {count:3d} col. ({percent:5.2f}%)")

    report_text = "\n".join(report_lines)

    print(report_text)

    json_dir = os.path.dirname(os.path.abspath(LABELS_JSON))
    output_path = os.path.join(json_dir, "dataset_stats.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

if __name__ == "__main__":
    main()