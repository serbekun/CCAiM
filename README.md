# CCAiM – Cloud Classification AI Model

<div align="center">
  <img src="assets/logo.png" alt="CCAiM logo" width="300">
</div>

**Live project card: [ccaim.serbekun.com](https://ccaim.serbekun.com)**
**Model + datasets: [Hugging Face collection](https://hf.co/collections/serbekun/ccaim)**

## Project Goal

CCAiM aims to develop an AI-powered model for classifying clouds based on ground-level photographs. The project uses image recognition techniques to identify cloud types according to the [WMO International Cloud Atlas](https://en.wikipedia.org/wiki/International_Cloud_Atlas) classification — 10 classes, photos taken from the ground with whatever camera is at hand.

## Project Status

Dataset collection + training, both model lines alive. The current published model is **V0.0.6** (ResNet18 line): **50.34%** validation accuracy and **macro-F1 0.390** on 916 labelled photos.

Not impressive numbers on their own — cloud classes are visually ambiguous and the dataset is small and unbalanced. The point of the project is an honest, reproducible baseline and a dataset that keeps growing, not a leaderboard score.

## Results

Validation metrics on the fixed seed-42 split (same split and metrics for both lines, so they are directly comparable):

| Model | Val accuracy | Macro-F1 |
| --- | --- | --- |
| ResNet18 V0.0.6 | 50.34% | 0.390 |
| ResNet18 V0.0.5 | 42.95% | 0.314 |
| scratch V0.0.6 | 35.57% | 0.237 |
| scratch V0.0.5 | 27.52% | 0.123 |

## Two Model Lines

The project trains two separate model lines on the same data, split and metrics:

- **scratch line** (`src/train.py`, weights `CCAiM_V0_0_X.pth`) — the compact `CCAiMModel` CNN trained from scratch (~9.9M parameters). This is the project baseline: it measures the contribution of dataset growth, so its results are only compared against earlier scratch versions.
- **ResNet18 line** (`src/train_resnet.py`, weights `CCAiM_R18_V0_0_X.pth`) — transfer learning from ImageNet-pretrained ResNet18, trained in two phases: the new head first (backbone frozen), then an optional fine-tune of `layer4` with a tiny LR. This is the practical line for maximum accuracy and the future API / web demo.

> ⚠️ Note: the accuracy gain of the ResNet18 line comes from ImageNet pretraining — it is a one-time head start, **not** a result of dataset growth. Project progress is still measured by the scratch line and by dataset expansion.

Both lines share `src/common.py`: same fixed seed, same 80/20 train/val split, same class-weighted loss (inverse frequency, computed from the train split only), same transforms (224×224, random crop / flips / color jitter for training, resize + center crop for validation), and the same report — confusion matrix, per-class recall and macro-F1. Each run prints a startup banner and tees everything into a timestamped log under `logs/`, with a total training time at the end.

## Dataset

All data lives on Hugging Face: **[serbekun/CCAiM-CloudsDataset](https://huggingface.co/datasets/serbekun/CCAiM-CloudsDataset)** — and a raw timelapse: [serbekun/CCAiM-CloudsVideo](https://huggingface.co/datasets/serbekun/CCAiM-CloudsVideo).

| Class | Abbr | Images | What it looks like |
| --- | --- | --- | --- |
| Cumulus | Cu | 398 | fluffy white clouds with flat bases |
| Cirrus | Ci | 102 | thin, wispy clouds high in the sky |
| Altocumulus | Ac | 96 | white/gray layered clouds with shading |
| Stratocumulus | Sc | 94 | low lumpy clouds with blue sky gaps |
| Altostratus | As | 63 | gray/blue layer clouds preceding storms |
| Cirrostratus | Cs | 42 | transparent, whitish veil clouds |
| Stratus | St | 38 | uniform gray cloud blanket |
| Cirrocumulus | Cc | 33 | small, white patchy clouds |
| Cumulonimbus | Cb | 33 | towering thunderstorm clouds |
| Nimbostratus | Ns | 17 | dark precipitation clouds |

916 images total, heavily unbalanced — that is why the loss is class-weighted and why macro-F1 is reported next to accuracy.

## Training Hardware

Everything runs locally, no Colab and no rented GPU — on the home server **node3** (Arch Linux):

- **GPU** NVIDIA Quadro P2200 — 5 GB GDDR5X, Pascal (`sm_61`), 1280 CUDA cores, 75 W, no tensor cores
- **CPU** Intel Core i7-7700K — 4 cores / 8 threads @ 4.2 (4.5) GHz
- **RAM** 16 GB
- **Runtime** Python 3.14 + PyTorch (CUDA 12.6), `device: cuda`

Pascal is slow for this, so the models stay small and the runs are short enough to finish overnight.

## Quickstart

```bash
pip install -r requirements.txt

# data: downloads the Hugging Face dataset into the local HF cache
python3 data/download.py

# scratch line (project baseline)
cd src && python3 train.py

# ResNet18 line (phase 1: head, then phase 2: fine-tune layer4)
cd src && python3 train_resnet.py
# head-only training, backbone stays frozen
cd src && python3 train_resnet.py --no-finetune
```

Inference works with both lines — the architecture is detected from the checkpoint:

```bash
cd src

# scratch model (default)
python3 predict.py path/to/image.jpg

# any specific model, e.g. the ResNet18 line
python3 predict.py path/to/image.jpg CCAiM_R18_V0_0_6.pth
```

Validation report (confusion matrix, per-class recall, macro-F1) for any saved model, on the same val split both lines train against:

```bash
cd src
python3 evaluate.py CCAiM_V0_0_6.pth
python3 evaluate.py CCAiM_R18_V0_0_6.pth
```

Model weights are not stored in this repository (`*.pth` is gitignored) — all of them live on [Hugging Face](https://huggingface.co/serbekun/CCAiM).

## Repository Layout

```
src/          training, evaluation, inference, dataset stats
  common.py       shared: dataset, split, transforms, metrics, banner/log/report
  model.py        CCAiMModel architecture (scratch line)
  train.py        scratch line training
  train_resnet.py ResNet18 line training (two phases)
  evaluate.py     validation metrics for a checkpoint
  predict.py      single-image inference
  dataset_stats.py
  labels.json
data/         download helper for the Hugging Face dataset
models/       local checkpoints (gitignored, published on Hugging Face)
assets/       logo
logs/         per-run training logs (gitignored)
```

## Old Models

Trained back when I understood neural networks much less than now — they are kept for history, not for use.

- v_0.0.1 — first model, learned on 23 photos
- v_0.0.2 — 42 photos
- v_0.0.3 — 88 photos
- v_0.0.4 — 165 photos

## Roadmap

- Dataset expansion — bigger, better labelled, more contributors
- V1 — first stable model on a minimal viable dataset
- Image classification API for integrations
- Interactive web demo
- Model evaluation tools (`evaluate.py` is the start)

## How to Contribute

The most helpful contribution right now: if you find a discrepancy between the cloud class in the label and what the image actually shows, fix it — label noise hurts more than a small dataset. New photographs (CC0, from the ground, sky in frame) are welcome too.

## License

- All data lives on [Hugging Face](https://huggingface.co/datasets/serbekun/CCAiM-CloudsDataset).
- Code in this repository is licensed under the MIT License (see [LICENSE](LICENSE)).
- Photographs located in folders named `clouds_<dataset_number>` are licensed under CC0 1.0 Universal (Public Domain): use them freely for any purpose, including commercial, without attribution.
- If other datasets are added in the future, their license terms will be specified in a separate license file inside their respective folder.
