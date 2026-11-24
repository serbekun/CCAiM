# train_efficientnet.py
# EfficientNet-B0 line: transfer learning from ImageNet-pretrained EfficientNet-B0.
# Same idea as train_resnet.py: same dataset, split, transforms and loss
# weighting (see common.py), so val metrics of all lines are directly comparable.
#
# B0 is used on purpose: its native resolution is 224x224, exactly what
# common.py feeds to every line (B1+ expect 240+ and would need new transforms).
#
# Two-phase training:
#   phase 1 (head): backbone frozen, only the new classifier head is trained
#   phase 2 (finetune, optional): top blocks (features[FINE_TUNE_FROM:]) unfrozen, tiny LR
# Pass --no-finetune to stop after phase 1.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

from common import (SEED, TransformedSubset, train_transform,
                    val_transform, load_split, compute_class_weights,
                    build_efficientnet_b0, confusion_matrix, macro_f1,
                    print_val_report, print_startup_banner, format_lr)

# setting
MODEL_PATH = "CCAiM_EB0_V0_0_7.pth"  # EB0 prefix keeps this line separate from scratch/R18 weights
MODEL_PATH = "../models/" + MODEL_PATH
BATCH_SIZE = 16
HEAD_EPOCHS = 15         # phase 1: train only the new classifier head
FINE_TUNE_EPOCHS = 30    # phase 2: also train the top blocks of the backbone
HEAD_LR = 0.001
FINE_TUNE_LR = 0.00001   # very small: large steps destroy pretrained features
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 10  # stop a phase if val loss hasn't improved for this many epochs
LR_DECAY_FACTOR = 0.5     # multiply LR by this when val loss plateaus
LR_DECAY_PATIENCE = 3     # epochs without val loss improvement before decaying LR
MIN_LR = 1e-7             # don't decay below this
FINE_TUNE = "--no-finetune" not in sys.argv

# torchvision EfficientNet-B0: model.features is a Sequential of 9 blocks
#   [0] stem, [1]-[7] MBConv stages, [8] final 1x1 conv (-> 1280 channels)
# FINE_TUNE_FROM=6 unfreezes stage 6 (192ch x4), stage 7 (320ch) and the final
# conv, which is roughly the analogue of layer4 in ResNet18. Lower = more
# layers get tuned (7 = only the last stage + final conv, 5 = even more).
FINE_TUNE_FROM = 6

torch.manual_seed(SEED)

# load the Hugging Face dataset and the deterministic train/val split
# (shared with the other lines so all of them are comparable)
hf_split, CLASSES, train_subset, val_subset = load_split()
NUM_CLASSES = len(CLASSES)

# banner reports both phases' budgets: total epochs and both learning rates
TOTAL_EPOCHS = HEAD_EPOCHS + (FINE_TUNE_EPOCHS if FINE_TUNE else 0)
IF_FINE_TUNE = f" / {format_lr(FINE_TUNE_LR)} (finetune)" if FINE_TUNE else ""
LR_LABEL = f"{format_lr(HEAD_LR)} (head){IF_FINE_TUNE}"
DEVICE = print_startup_banner("train_efficientnet", NUM_CLASSES, BATCH_SIZE,
                              TOTAL_EPOCHS, LR_LABEL)

train_dataset = TransformedSubset(train_subset, transform=train_transform)
val_dataset = TransformedSubset(val_subset, transform=val_transform)

# loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = build_efficientnet_b0(NUM_CLASSES, pretrained=True).to(DEVICE)

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


def train_phase(phase_name, epochs, optimizer, frozen_modules):
    # best_val_loss is shared across phases: phase 2 only overwrites the
    # checkpoint if it actually beats the best result of phase 1
    global best_val_loss, best_cm
    epochs_without_improvement = 0
    # gradually lower LR when val loss plateaus (patience is shorter than
    # EARLY_STOP_PATIENCE so LR gets reduced before the phase is stopped);
    # each phase gets its own scheduler since it gets its own optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_DECAY_FACTOR,
        patience=LR_DECAY_PATIENCE, min_lr=MIN_LR,
    )

    for epoch in range(epochs):
        model.train()
        # frozen parts stay in eval mode: otherwise their BatchNorm running
        # stats keep drifting on small batches (16) and stochastic depth keeps
        # randomly dropping blocks of a backbone that isn't being trained
        for module in frozen_modules:
            module.eval()

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

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[{phase_name}] Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_f1:.3f} | LR: {current_lr:.2e}")

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
                "arch": "efficientnet_b0",
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
for param in model.classifier.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.classifier.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
train_phase("head", HEAD_EPOCHS, optimizer, frozen_modules=[model.features])

# phase 2 (optional): unfreeze the top blocks, fine-tune with a tiny LR
if FINE_TUNE:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    for param in model.features[FINE_TUNE_FROM:].parameters():
        param.requires_grad = True
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=FINE_TUNE_LR, weight_decay=WEIGHT_DECAY,
    )
    train_phase("finetune", FINE_TUNE_EPOCHS, optimizer,
                frozen_modules=[model.features[:FINE_TUNE_FROM]])

# full report for the best (saved) epoch
if best_cm is not None:
    print_val_report(best_cm, CLASSES)