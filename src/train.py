# train_ccaim.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import os

from model import CCAiMModel  # file with architecture

# setting
HF_DATASET = "serbekun/CCAiM-CloudsDataset"
MODEL_PATH = "CCAiM_V0_0_5.pth"
MODEL_PATH = "../models/" + MODEL_PATH
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 15  # stop if val loss hasn't improved for this many epochs
SEED = 42

torch.manual_seed(SEED)
def pick_device():
    # Use CUDA only if the installed torch build has a compatible cubin for this
    # GPU; otherwise CUDA ops fail at runtime, so fall back to CPU. A cubin built
    # for sm_{major}{m} runs on a device sm_{major}{minor} as long as m <= minor
    # (same major, equal-or-lower minor), so we don't require an exact match.
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()

        def parse(arch):  # "sm_86" -> (8, 6); "sm_100" -> (10, 0)
            n = arch[len("sm_"):]
            return int(n[:-1]), int(n[-1])

        for arch in torch.cuda.get_arch_list():
            a_major, a_minor = parse(arch)
            if a_major == major and a_minor <= minor:
                return torch.device("cuda")
        print(f"[WARN] GPU (sm_{major}{minor}) not supported by this PyTorch build; using CPU")
    return torch.device("cpu")

DEVICE = pick_device()
print(f"[INFO] using device: {DEVICE}")

# data set class (wraps a Hugging Face split with image/label columns)
class CloudDataset(Dataset):
    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        image = row["image"].convert("RGB")  # PIL image from HF
        label = row["label"]                 # already an int (ClassLabel)

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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    NORMALIZE,
])


# load the Hugging Face dataset (cached locally by data/download.py)
hf_split = load_dataset(HF_DATASET)["train"]
CLASSES = hf_split.features["label"].names
NUM_CLASSES = len(CLASSES)

# data set and loader (base dataset returns raw PIL images; subsets add transforms)
# the split must be deterministic: on resume, a reshuffled split would leak
# already-trained images into the validation set
dataset = CloudDataset(hf_split, transform=None)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

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

# class imbalance: inverse-frequency weights computed from the train split only,
# so no information about the validation set leaks into the loss
all_labels = hf_split["label"]
class_counts = [0] * NUM_CLASSES
for idx in train_subset.indices:
    class_counts[all_labels[idx]] += 1

class_weights = torch.tensor(
    [1.0 / c if c > 0 else 0.0 for c in class_counts],
    dtype=torch.float,
)
# normalize so weights average to 1 (keeps loss scale comparable)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


best_val_loss = float("inf")
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