# common.py
# code shared by both training lines (scratch train.py and train_resnet.py)
# so the dataset, train/val split, transforms and metrics stay identical
# and the two lines are directly comparable
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# setting shared by both lines
HF_DATASET = "serbekun/CCAiM-CloudsDataset"
SEED = 42  # fixed for both lines: same split -> comparable val metrics


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


# transforms (separate for train and val); ImageNet stats, which is also
# exactly what the pretrained ResNet18 line expects
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


def load_split():
    """Load the HF dataset and return (hf_split, classes, train_subset, val_subset).

    The split must be deterministic: it uses its own generator seeded with SEED,
    so both training lines (and any resume) get the exact same train/val split
    and no already-trained images leak into the validation set.
    """
    from datasets import load_dataset  # lazy: predict.py doesn't need it

    hf_split = load_dataset(HF_DATASET)["train"]
    classes = hf_split.features["label"].names

    dataset = CloudDataset(hf_split, transform=None)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    return hf_split, classes, train_subset, val_subset


def compute_class_weights(hf_split, train_subset, num_classes):
    """Inverse-frequency class weights computed from the train split only,
    so no information about the validation set leaks into the loss."""
    all_labels = hf_split["label"]
    class_counts = [0] * num_classes
    for idx in train_subset.indices:
        class_counts[all_labels[idx]] += 1

    class_weights = torch.tensor(
        [1.0 / c if c > 0 else 0.0 for c in class_counts],
        dtype=torch.float,
    )
    # normalize so weights average to 1 (keeps loss scale comparable)
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights


def build_resnet18(num_classes, pretrained):
    """ResNet18 with the final fc replaced for our classes.

    pretrained=True loads ImageNet weights (training); pretrained=False builds
    the bare architecture (inference, weights come from a checkpoint).
    """
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def confusion_matrix(true_labels, pred_labels, num_classes):
    """rows = true class, cols = predicted class"""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    return cm


def per_class_stats(cm):
    """Per-class (recall, f1, support) lists from a confusion matrix.

    Classes absent from the val split get recall/f1 = 0 with support 0;
    they still count toward macro-F1, which keeps the metric honest about
    classes the model was never tested on.
    """
    recalls, f1s, supports = [], [], []
    for i in range(cm.size(0)):
        tp = cm[i, i].item()
        support = cm[i].sum().item()
        predicted = cm[:, i].sum().item()
        recall = tp / support if support > 0 else 0.0
        precision = tp / predicted if predicted > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    return recalls, f1s, supports


def macro_f1(cm):
    _, f1s, _ = per_class_stats(cm)
    return sum(f1s) / len(f1s)


def print_val_report(cm, classes):
    """Confusion matrix + per-class recall + macro-F1: overall accuracy alone
    is misleading on this dataset because of the heavy class imbalance."""
    recalls, f1s, supports = per_class_stats(cm)

    print("\nvalidation confusion matrix (rows: true, cols: predicted)")
    header = " " * 15 + "".join(f"{cls[:4]:>5s}" for cls in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = "".join(f"{cm[i, j].item():5d}" for j in range(len(classes)))
        print(f"{cls:15s}{row}")

    print("\nper-class metrics")
    for cls, recall, f1, support in zip(classes, recalls, f1s, supports):
        print(f"{cls:15s} recall {recall * 100:6.2f}% | F1 {f1:.3f} | support {support}")

    accuracy = 100.0 * cm.trace().item() / cm.sum().item()
    print(f"\nval accuracy: {accuracy:.2f}% | macro-F1: {macro_f1(cm):.3f}")
