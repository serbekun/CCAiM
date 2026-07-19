# evaluate_ccaim.py
# validation metrics (confusion matrix, per-class recall, macro-F1) for an
# already saved model, without training. Uses the same deterministic val
# split as both training lines (see common.py), so scratch and ResNet18
# models evaluated here are directly comparable.
import torch
import json
import sys
import os

from torch.utils.data import DataLoader

from model import CCAiMModel  # model architecture (scratch line)
from common import (pick_device, TransformedSubset, val_transform, load_split,
                    build_resnet18, confusion_matrix, print_val_report)

# setting
MODEL_PATH = "CCAiM_V0_0_5.pth"
MODEL_PATH = "../models/" + MODEL_PATH
LABELS_JSON = "labels.json"
BATCH_SIZE = 16

# argument check (model path is optional: default is set above)
if len(sys.argv) not in (1, 2):
    print(f"Use: python {sys.argv[0]} [path/to/model.pth]")
    sys.exit(1)

if len(sys.argv) == 2:
    MODEL_PATH = sys.argv[1]
    if not os.path.exists(MODEL_PATH):
        # bare filename: look it up in the models directory
        MODEL_PATH = "../models/" + sys.argv[1]

if not os.path.exists(MODEL_PATH):
    print(f"[Error] model file {MODEL_PATH} not found.")
    sys.exit(1)

DEVICE = pick_device()
print(f"[INFO] using device: {DEVICE}")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# class names must come from the checkpoint: rebuilding them any other way
# (e.g. sorting labels.json values) can mismatch the training index order
if isinstance(checkpoint, dict) and "classes" in checkpoint:
    classes = checkpoint["classes"]
    state_dict = checkpoint["model_state_dict"]
else:
    # old checkpoint without class names: fall back to labels.json,
    # ordered by its index keys ("0", "1", ...), not alphabetically
    print(f"[WARN] checkpoint has no class list, falling back to {LABELS_JSON}")
    with open(LABELS_JSON, "r") as f:
        labels_data = json.load(f)
    classes = [labels_data[str(i)] for i in range(len(labels_data))]
    state_dict = checkpoint

# model: pick the architecture by the checkpoint "arch" field (R18 line saves
# it) or by the R18 filename prefix; everything else is the scratch line
is_resnet = (isinstance(checkpoint, dict) and checkpoint.get("arch") == "resnet18") \
    or "R18" in os.path.basename(MODEL_PATH)
if is_resnet:
    model = build_resnet18(len(classes), pretrained=False).to(DEVICE)
else:
    model = CCAiMModel(num_classes=len(classes)).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

# same deterministic val split both training lines use
hf_split, dataset_classes, train_subset, val_subset = load_split()
if list(dataset_classes) != list(classes):
    print("[WARN] class list in checkpoint differs from the dataset class list; "
          "metrics may map to wrong names")

val_dataset = TransformedSubset(val_subset, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

val_true = []
val_pred = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        val_true.extend(labels.cpu().tolist())
        val_pred.extend(preds.cpu().tolist())

arch_name = "resnet18" if is_resnet else "scratch (CCAiMModel)"
print(f"\nevaluation of {MODEL_PATH} [{arch_name}] on {len(val_true)} val images")
cm = confusion_matrix(val_true, val_pred, len(classes))
print_val_report(cm, classes)
