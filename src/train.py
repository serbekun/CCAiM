# train_ccaim.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os

from model import CCAiMModel  # file with architecture

# setting
DATA_DIR = "../data/CCAiM-CloudsDataset/clouds_1"
LABELS_JSON = "labels.json"
MODEL_PATH = "CCAiM_V0_0_2.pth"
MODEL_PATH = "../models/" + MODEL_PATH
NUM_CLASSES = 10
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.00001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data set class
class CloudDataset(Dataset):
    def __init__(self, data_dir, labels_json, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        with open(labels_json, "r") as f:
            self.labels = json.load(f)
        self.files = list(self.labels.keys())

        # mapping class index
        classes = sorted(set(self.labels.values()))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = f"{self.files[idx]}.jpg"
        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert("RGB")

        label_str = self.labels[self.files[idx]]
        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# data set and loader
dataset = CloudDataset(DATA_DIR, LABELS_JSON, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# model
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = CCAiMModel(num_classes=NUM_CLASSES).to(DEVICE)

if os.path.exists(MODEL_PATH):
    print(f"[INFO] loading model from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("[INFO] creating new model")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[INFO] model saved (Val Loss became best: {best_val_loss:.4f})")