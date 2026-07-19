# train_ccaim.py
# scratch line: CCAiMModel trained from scratch. This line is the project
# baseline — it measures the contribution of dataset growth, so its training
# logic must stay untouched (shared code lives in common.py).
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from model import CCAiMModel  # file with architecture
from common import (SEED, pick_device, TransformedSubset, train_transform,
                    val_transform, load_split, compute_class_weights,
                    confusion_matrix, macro_f1, print_val_report)

# setting
MODEL_PATH = "CCAiM_V0_0_5.pth"
MODEL_PATH = "../models/" + MODEL_PATH
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.00001
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 15  # stop if val loss hasn't improved for this many epochs

torch.manual_seed(SEED)

DEVICE = pick_device()
print(f"[INFO] using device: {DEVICE}")

# load the Hugging Face dataset and the deterministic train/val split
# (shared with the ResNet18 line so both lines are comparable)
hf_split, CLASSES, train_subset, val_subset = load_split()
NUM_CLASSES = len(CLASSES)

train_dataset = TransformedSubset(train_subset, transform=train_transform)
val_dataset = TransformedSubset(val_subset, transform=val_transform)

# loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = CCAiMModel(num_classes=NUM_CLASSES).to(DEVICE)

if os.path.exists(MODEL_PATH):
    print(f"[INFO] loading model from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:  # old checkpoints are a bare state_dict
        model.load_state_dict(checkpoint)
else:
    print("[INFO] creating new model")

class_weights = compute_class_weights(hf_split, train_subset, NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


best_val_loss = float("inf")
best_cm = None  # confusion matrix of the best (saved) epoch
epochs_without_improvement = 0

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
    val_true = []
    val_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_true.extend(labels.cpu().tolist())
            val_pred.extend(preds.cpu().tolist())
    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / val_total
    cm = confusion_matrix(val_true, val_pred, NUM_CLASSES)
    val_f1 = macro_f1(cm)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_f1:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_cm = cm
        epochs_without_improvement = 0
        # store the class list with the weights so predict.py maps output
        # neurons to the exact names/order used in training
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
        }, MODEL_PATH)
        print(f"[INFO] model saved (Val Loss became best: {best_val_loss:.4f})")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"[INFO] early stopping: no val loss improvement for {EARLY_STOP_PATIENCE} epochs")
            break

# full report for the best (saved) epoch
if best_cm is not None:
    print_val_report(best_cm, CLASSES)
