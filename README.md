# CCAiM – Cloud Classification AI Model

<div align="center">
  <img src="assets/logo.jpg" alt="CCAiM logo" width="300">
</div> 

## 📌 Project Status

Currently in the dataset collection phase.

## 🎯 Project Goal

CCAiM aims to develop an AI-powered model for classifying clouds based on ground-level photographs. The project uses image recognition techniques to identify cloud types according to the [WMO International Cloud Atlas classification](https://en.wikipedia.org/wiki/International_Cloud_Atlas).

## ☁️ Target Cloud Classes

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
 - 📷 Dataset expansion – Using contributor photographs
 - 🧠 Multiple model versions:
 - V1: Initial first stable model (minimal viable dataset)
 - 🔍 Image classification API for integrations
 - 🌐 Interactive web demo
 - 📊 Model evaluation tools

now can
 - collect dataset
 - create first model V0.0.1

## 📄 License Information

 - All data exist in [hugging face](https://huggingface.co/datasets/serbekun/CCAiM-CloudsDataset)
 - Code in this repository is licensed under the MIT License (see LICENSE).
 - Photographs located in folders named clouds_<dataset_number> are licensed under CC0 1.0 Universal (Public Domain), meaning they can be used freely for any purpose, including commercial use, without attribution.
 - If other datasets are added in the future, their license terms will be specified in a separate license file inside their respective folder.

## models

### All models exist in [hugging face](https://huggingface.co/serbekun/CCAiM)

### models list
- v_0.0.1   # first model learned by 23 photo.
- v_0.0.2   # second model learned by 42 photo
- v_0.0.3   # model learned by 88 photo

## 🤝 How to Contribute

- The most helpful contribution at the moment:
If you find a discrepancy between the cloud class specified in the JSON label and the actual image content, correcting it will greatly improve dataset quality.
