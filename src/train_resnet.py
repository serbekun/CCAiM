# train_ccaim_resnet.py
# ResNet18 line: transfer learning from ImageNet-pretrained ResNet18.
# This is the practical line (max accuracy for a future API / web demo);
# the scratch line in train.py stays the project baseline. Same dataset,
# split, transforms and loss weighting (see common.py) so val metrics of
# the two lines are directly comparable.
#
# Two-phase training:
#   phase 1 (head): backbone frozen, only the new fc head is trained
#   phase 2 (finetune, optional): layer4 unfrozen, tiny LR
# Pass --no-finetune to stop after phase 1.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

from common import (SEED, pick_device, TransformedSubset, train_transform,
                    val_transform, load_split, compute_class_weights,
                    build_resnet18, confusion_matrix, macro_f1,
                    print_val_report)

# setting
MODEL_PATH = "CCAiM_R18_V0_0_5.pth"  # R18 prefix keeps this line separate from scratch weights
MODEL_PATH = "../models/" + MODEL_PATH
BATCH_SIZE = 16
HEAD_EPOCHS = 15         # phase 1: train only the new fc head
FINE_TUNE_EPOCHS = 30    # phase 2: also train the top block (layer4)
HEAD_LR = 0.001
FINE_TUNE_LR = 0.00001   # very small: large steps destroy pretrained features
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 10  # stop a phase if val loss hasn't improved for this many epochs
FINE_TUNE = "--no-finetune" not in sys.argv

torch.manual_seed(SEED)

DEVICE = pick_device()
print(f"[INFO] using device: {DEVICE}")

# load the Hugging Face dataset and the deterministic train/val split
# (shared with the scratch line so both lines are comparable)
hf_split, CLASSES, train_subset, val_subset = load_split()
NUM_CLASSES = len(CLASSES)

train_dataset = TransformedSubset(train_subset, transform=train_transform)
val_dataset = TransformedSubset(val_subset, transform=val_transform)

# loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = build_resnet18(NUM_CLASSES, pretrained=True).to(DEVICE)

if os.path.exists(MODEL_PATH):
    print(f"[INFO] loading model from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    print("[INFO] starting from ImageNet-pretrained weights")

class_weights = compute_class_weights(hf_split, train_subset, NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)


best_val_loss = float("inf")
best_cm = None  # confusion matrix of the best (saved) epoch


def train_phase(phase_name, epochs, optimizer):
    # best_val_loss is shared across phases: phase 2 only overwrites the
    # checkpoint if it actually beats the best result of phase 1
    global best_val_loss, best_cm
    epochs_without_improvement = 0

    for epoch in range(epochs):
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

        print(f"[{phase_name}] Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_f1:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_cm = cm
            epochs_without_improvement = 0
            # store the class list with the weights so predict.py maps output
            # neurons to the exact names/order used in training; "arch" lets
            # predict.py rebuild the right architecture
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "arch": "resnet18",
            }, MODEL_PATH)
            print(f"[INFO] model saved (Val Loss became best: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"[INFO] early stopping: no val loss improvement for {EARLY_STOP_PATIENCE} epochs")
                break


# phase 1: freeze the whole backbone, train only the new head
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.fc.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
train_phase("head", HEAD_EPOCHS, optimizer)

# phase 2 (optional): unfreeze the top block, fine-tune with a tiny LR
if FINE_TUNE:
    for param in model.layer4.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=FINE_TUNE_LR, weight_decay=WEIGHT_DECAY,
    )
    train_phase("finetune", FINE_TUNE_EPOCHS, optimizer)

# full report for the best (saved) epoch
if best_cm is not None:
    print_val_report(best_cm, CLASSES)
