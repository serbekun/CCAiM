# predict_ccaim.py
import torch
from PIL import Image
import json
import sys
import os

from model import CCAiMModel  # model architecture (scratch line)
from common import build_resnet18, val_transform

# setting
MODEL_PATH = "CCAiM_R18_V0_0_5.pth"
MODEL_PATH = "../models/" + MODEL_PATH
LABELS_JSON = "labels.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argument check (model path is optional: default is the scratch line)
if len(sys.argv) not in (2, 3):
    print(f"Use: python {sys.argv[0]} <path/to/image> [path/to/model.pth]")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"[Error] fail {image_path} not found.")
    sys.exit(1)

if len(sys.argv) == 3:
    MODEL_PATH = sys.argv[2]
    if not os.path.exists(MODEL_PATH):
        # bare filename: look it up in the models directory
        MODEL_PATH = "../models/" + sys.argv[2]

if not os.path.exists(MODEL_PATH):
    print(f"[Error] model file {MODEL_PATH} not found.")
    sys.exit(1)

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

# transform image: val_transform from common.py, the exact transform both
# lines use for validation (ImageNet normalization is shared)
transform = val_transform

# load and transform
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# predict
with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)[0]

# output
sorted_indices = torch.argsort(probs, descending=True)

print(f"\nprediction result for image: {image_path}\n")
for idx in sorted_indices:
    cls_name = classes[idx]
    prob = probs[idx].item() * 100
    print(f"{cls_name:20s} — {prob:.2f}%")
