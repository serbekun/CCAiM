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
DATA_DIR = "../data/clouds_1"
LABELS_JSON = "labels.json"
MODEL_PATH = "CCAiM_V0_0_4.pth"
MODEL_PATH = "../models/" + MODEL_PATH
NUM_CLASSES = 10
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.001
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

# wrapper so each split can use its own transform after random_split
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# transforms (separate for train and val)
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    NORMALIZE,
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])


# data set and loader (base dataset returns raw PIL images; subsets add transforms)
dataset = CloudDataset(DATA_DIR, LABELS_JSON, transform=None)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataset = TransformedSubset(train_subset, transform=train_transform)
val_dataset = TransformedSubset(val_subset, transform=val_transform)

# loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = CCAiMModel(num_classes=NUM_CLASSES).to(DEVICE)

if os.path.exists(MODEL_PATH):
    print(f"[INFO] loading model from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("[INFO] creating new model")

# class imbalance: inverse-frequency weights computed from the dataset
class_counts = [0] * NUM_CLASSES
for f in dataset.files:
    class_counts[dataset.class_to_idx[dataset.labels[f]]] += 1

class_weights = torch.tensor(
    [1.0 / c if c > 0 else 0.0 for c in class_counts],
    dtype=torch.float,
)
# normalize so weights average to 1 (keeps loss scale comparable)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)


best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[INFO] model saved (Val Loss became best: {best_val_loss:.4f})")