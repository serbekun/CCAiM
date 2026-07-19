# CCAiM – Cloud Classification AI Model

<div align="center">
  <img src="assets/logo.png" alt="CCAiM logo" width="300">
</div> 

## project links
CCAiM hugging face [collection](https://hf.co/collections/serbekun/ccaim) 🤗

## Project Status

Currently in the dataset collection phase.

## Project Goal

CCAiM aims to develop an AI-powered model for classifying clouds based on ground-level photographs. The project uses image recognition techniques to identify cloud types according to the [WMO International Cloud Atlas classification](https://en.wikipedia.org/wiki/International_Cloud_Atlas).

## Target Cloud Classes

The model will be trained to recognize the following cloud types:
 1. Cirrus (Ci) – Thin, wispy clouds high in the sky
 2. Cirrostratus (Cs) – Transparent, whitish veil clouds
 3. Cirrocumulus (Cc) – Small, white patchy clouds
 4. Altostratus (As) – Gray/blue layer clouds preceding storms
 5. Altocumulus (Ac) – White/gray layered clouds with shading
 6. Stratus (St) – Uniform gray cloud blanket
 7. Stratocumulus (Sc) – Low lumpy clouds with blue sky gaps
 8. Nimbostratus (Ns) – Dark precipitation clouds
 9. Cumulus (Cu) – Fluffy white clouds with flat bases
 10. Cumulonimbus (Cb) – Towering thunderstorm clouds

## 🛠 Planned Features

dreams
 - Dataset expansion – Using contributor photographs
 - Multiple model versions:
 - V1: Initial first stable model (minimal viable dataset)
 - Image classification API for integrations
 - Interactive web demo
 - Model evaluation tools

now can
 - collect dataset
 - create first model V0.0.1

## License Information

 - All data exist in [hugging face](https://huggingface.co/datasets/serbekun/CCAiM-CloudsDataset)
 - Code in this repository is licensed under the MIT License (see LICENSE).
 - Photographs located in folders named clouds_<dataset_number> are licensed under CC0 1.0 Universal (Public Domain), meaning they can be used freely for any purpose, including commercial use, without attribution.
 - If other datasets are added in the future, their license terms will be specified in a separate license file inside their respective folder.

## models

### All models exist in [hugging face](https://huggingface.co/serbekun/CCAiM)

### Two model lines

The project trains two separate model lines on the same data, split and metrics:

- **scratch line** (`src/train.py`, weights `CCAiM_V0_0_X.pth`) — the compact CCAiMModel CNN trained from scratch. This is the project baseline: it measures the contribution of dataset growth, so its results are only compared against earlier scratch versions.
- **ResNet18 line** (`src/train_resnet.py`, weights `CCAiM_R18_V0_0_X.pth`) — transfer learning from ImageNet-pretrained ResNet18 (frozen backbone first, then optional fine-tuning of the top block). This is the practical line for maximum accuracy and the future API / web demo.

> ⚠️ Note: the accuracy gain of the ResNet18 line comes from ImageNet pretraining — it is a one-time head start, **not** a result of dataset growth. Project progress is still measured by the scratch line and by dataset expansion.

Both lines use the same fixed random seed and train/val split (`src/common.py`), the same class-weighted loss, and report confusion matrix, per-class recall and macro-F1 on validation, so they are directly comparable.

Run the lines (from `src/`):

```bash
# scratch line
python3 train.py

# ResNet18 line (two phases: head, then fine-tuning of layer4)
python3 train_resnet.py
# head-only training, backbone stays frozen
python3 train_resnet.py --no-finetune
```

Inference works with both lines — the architecture is detected from the checkpoint / filename:

```bash
# scratch model (default, unchanged)
python3 predict.py path/to/image.jpg

# any specific model, e.g. the ResNet18 line
python3 predict.py path/to/image.jpg CCAiM_R18_V0_0_1.pth
```

Validation metrics (confusion matrix, per-class recall, macro-F1) for any saved model, on the same val split both lines train against:

```bash
python3 evaluate.py CCAiM_V0_0_5.pth
python3 evaluate.py CCAiM_R18_V0_0_5.pth
```

### models list
- v_0.0.1   # first model learned by 23 photo.
- v_0.0.2   # second model learned by 42 photo
- v_0.0.3   # model learned by 88 photo
- v_0.0.4   # model learned by 165 photo

## How to Contribute

- The most helpful contribution at the moment:
If you find a discrepancy between the cloud class specified in the JSON label and the actual image content, correcting it will greatly improve dataset quality.

