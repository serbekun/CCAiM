# common.py
# code shared by both training lines (scratch train.py and train_resnet.py)
# so the dataset, train/val split, transforms and metrics stay identical
# and the two lines are directly comparable
import os
import sys
import time
import atexit
import datetime

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# setting shared by both lines
HF_DATASET = "serbekun/CCAiM-CloudsDataset"
SEED = 42  # fixed for both lines: same split -> comparable val metrics
MODEL_LINES = "scratch (CCAiMModel) + resnet18 (ImageNet head start)"

# training runs are saved here (repo-root/logs); anchored to this file so the
# path doesn't depend on the working directory the script is launched from
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# best-effort memory type per GPU name (torch doesn't expose it); unknown
# cards simply omit the memory type instead of guessing
GPU_MEMORY_TYPE = {
    "Quadro P2200": "GDDR5X",
    "Quadro P2000": "GDDR5",
    "Tesla T4": "GDDR6",
    "Tesla V100": "HBM2",
    "GeForce GTX 1080 Ti": "GDDR5X",
}

# compute capability major -> architecture family name
GPU_ARCH = {
    5: "Maxwell", 6: "Pascal", 7: "Volta/Turing/Ampere",
    8: "Ampere/Ada", 9: "Hopper", 10: "Blackwell", 12: "Blackwell",
}


class _Tee:
    """Write to several streams at once (terminal + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


def format_lr(lr):
    # 0.00001 -> "1e-5", 0.001 -> "1e-3" (drop the zero-padded exponent)
    mantissa, exp = f"{lr:.0e}".split("e")
    return f"{mantissa}e{int(exp)}"


def format_duration(seconds):
    # 3725 -> "1h 02m 05s"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def print_startup_banner(script_name, num_classes, batch_size, epochs, lr,
                         image_size=224):
    """Print the environment/config banner and tee the whole run into a log file.

    Must be called after load_split() so the class count is known; returns the
    picked device so the caller doesn't call pick_device()/print it again.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{script_name}_{stamp}.log")
    log_file = open(log_path, "a", encoding="utf-8")
    # keep a handle on the original stdout; everything printed from here on
    # goes to the terminal and to the log file
    sys.stdout = _Tee(sys.__stdout__, log_file)

    device = pick_device()

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        name = torch.cuda.get_device_name(0)
        mem_gb = props.total_memory / (1024 ** 3)
        mem_type = GPU_MEMORY_TYPE.get(name)
        mem_part = f"{mem_gb:.0f} GB" + (f" {mem_type}" if mem_type else "")
        arch = GPU_ARCH.get(props.major, "unknown")
        gpu = f"{name} — {mem_part}, {arch} (sm_{props.major}{props.minor})"
    else:
        gpu = "no CUDA GPU (running on CPU)"

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    torch_ver = torch.__version__.split("+")[0]
    cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
    build = f"cu{cuda_ver.replace('.', '')}" if cuda_ver else "cpu"
    lr_label = format_lr(lr) if isinstance(lr, (int, float)) else lr

    print(f"[INFO] using device: {device}")
    print(f"[INFO] gpu   : {gpu}")
    print(f"[INFO] env   : python {py_ver}, torch {torch_ver} ({build}), cuda {cuda_ver or 'n/a'}")
    print(f"[INFO] data  : {HF_DATASET} — {num_classes} classes")
    print(f"[INFO] cfg   : {image_size}x{image_size}, batch {batch_size}, "
          f"epochs {epochs}, lr {lr_label}, seed {SEED}")
    print(f"[INFO] lines : {MODEL_LINES}")
    print(f"[INFO] log   : {os.path.relpath(log_path)}")

    # report total time at interpreter shutdown: this runs both on normal
    # completion and on Ctrl+C (KeyboardInterrupt), and the tee above makes
    # sure the line lands in the log file too
    start = time.time()
    atexit.register(lambda: print(
        f"[INFO] training time: {format_duration(time.time() - start)}"))

    return device


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
