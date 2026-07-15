# predict_ccaim.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import sys
import os

from model import CCAiMModel  # model architecture

# setting
MODEL_PATH = "CCAiM_V0_0_5.pth"
MODEL_PATH = "../models/" + MODEL_PATH
LABELS_JSON = "labels.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argument check
if len(sys.argv) != 2:
    print(f"Use: python {sys.argv[0]} <path/to/image>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"[Error] fail {image_path} not found.")
    sys.exit(1)

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

# model
model = CCAiMModel(num_classes=len(classes)).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

# transform image (must match val_transform used during training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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