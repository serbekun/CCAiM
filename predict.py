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
MODEL_PATH = "ccaim_model.pth"
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

# load label
with open(LABELS_JSON, "r") as f:
    labels_data = json.load(f)

# collect special class
classes = sorted(set(labels_data.values()))

# model
model = CCAiMModel(num_classes=len(classes)).to(DEVICE)

if not os.path.exists(MODEL_PATH):
    print(f"[Error] model file {MODEL_PATH} not found.")
    sys.exit(1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# transform image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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