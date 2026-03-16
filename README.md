<div align="center">

#  ML4SCI DeepLense — GSoC 2026

### Strong Gravitational Lensing with Deep Learning

**Rafiqus Salehin** · [@rsalehin](https://github.com/rsalehin)

[![ML4SCI](https://img.shields.io/badge/ML4SCI-DeepLense-blue?style=flat-square)](https://ml4sci.org)
[![GSoC](https://img.shields.io/badge/GSoC-2026-4285F4?style=flat-square&logo=google&logoColor=white)](https://summerofcode.withgoogle.com)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br>

*This repository contains solutions to the ML4SCI DeepLense evaluation tests for GSoC 2026.
Each test is self-contained with its own notebook, assets, and requirements.*

</div>

---

##  Tests Overview

| Test | Topic | Task Summary | Evaluation Metrics | Status |
|:----:|:------|:-------------|:-------------------|:------:|
| [**I**](Test_I_Classification/) | Multi-Class Classification | Classify strong lensing images into three dark matter substructure classes | ROC · AUC | ✅ |
| [**V**](Test_V_Lens_Finding/) | Lens Finding & Data Pipelines | Binary classification of lensed vs non-lensed galaxies from observational data | ROC · AUC | 🔄 |
| [**VI**](Test_VI_Super_Resolution/) | Image Super-Resolution | Upscale low-resolution lensing images; adapt to real HSC/HST data | MSE · SSIM · PSNR | 🔄 |
| [**VII**](Test_VII_Physics_Guided_ML/) | Physics-Guided ML | PINN-based lens classifier incorporating the gravitational lensing equation | ROC · AUC | 🔄 |
| [**IX**](Test_IX_Foundation_Model/) | Foundation Model | MAE pre-training on lensing images + fine-tuning for classification and super-resolution | ROC · AUC · MSE · SSIM · PSNR | 🔄 |

---

## 🗂 Repository Structure

```
DeepLense-GSoC-2026/
│
├── README.md
│
├── Test_I_Classification/
│   ├── README.md
│   ├── Test_I_Classification.ipynb
│   ├── requirements.txt
│   └── assets/
│       └── figures/
│
├── Test_V_Lens_Finding/
│   ├── README.md
│   ├── Test_V_Lens_Finding.ipynb
│   ├── requirements.txt
│   └── assets/
│       └── figures/
│
├── Test_VI_Super_Resolution/
│   ├── README.md
│   ├── Test_VI_A_Simulated.ipynb
│   ├── Test_VI_B_RealData.ipynb
│   ├── requirements.txt
│   └── assets/
│       └── figures/
│
├── Test_VII_Physics_Guided_ML/
│   ├── README.md
│   ├── Test_VII_Physics_Guided_ML.ipynb
│   ├── requirements.txt
│   └── assets/
│       └── figures/
│
└── Test_IX_Foundation_Model/
    ├── README.md
    ├── Test_IX_A_MAE_Classification.ipynb
    ├── Test_IX_B_MAE_SuperResolution.ipynb
    ├── requirements.txt
    └── assets/
        └── figures/
```

---

##  Test Summaries

### Test I — Multi-Class Classification
> **Dataset:** 30,000 simulated strong lensing images (150×150, single-channel) across three balanced classes — No Substructure, CDM Subhalo (Sphere), and Axion Vortex.

Nine architectures evaluated: ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B3, AlexNet, VGG-16, ViT-Base/16, Equivariant-D4, and E-ResNet. Includes comprehensive interpretability analysis (Grad-CAM, Attention Rollout, deep ensemble uncertainty), failure mode analysis, ablation study, equivariance verification, and a residual image approach via convolutional autoencoder.

**Best result:** DenseNet-121 — Macro AUC **0.9962**, Sphere Recall **0.9364**

→ [Full details](Test_I_Classification/README.md)

---

### Test V — Lens Finding & Data Pipelines
> **Dataset:** Observational data of strong lenses and non-lensed galaxies. 3-channel (64×64) images. Highly imbalanced — non-lenses significantly outnumber lenses.

Binary classification under class imbalance. Strategy includes imbalance-aware training (focal loss / weighted sampling), multi-channel feature exploitation, and threshold optimisation for deployment-relevant precision-recall trade-offs.

→ [Full details](Test_V_Lens_Finding/README.md)

---

### Test VI — Image Super-Resolution
> **Task VI.A:** Simulated HR/LR lensing image pairs. **Task VI.B:** 300 real HSC/HST HR/LR pairs — few-shot / transfer learning from VI.A.

Deep learning super-resolution pipeline (e.g. EDSR, SRGAN, or SwinIR backbone). Task VI.B explores domain adaptation from simulated to real telescope data with limited labelled pairs.

→ [Full details](Test_VI_Super_Resolution/README.md)

---

### Test VII — Physics-Guided ML
> **Dataset:** Same three-class lensing dataset as Test I.

Physics-Informed Neural Network (PINN) incorporating the gravitational lensing equation as an explicit architectural or loss constraint. The lens equation acts as a differentiable physical prior, penalising predictions inconsistent with smooth macro-lens deflection. Performance benchmarked against the Test I baseline.

→ [Full details](Test_VII_Physics_Guided_ML/README.md)

---

### Test IX — Foundation Model
> **Task IX.A:** MAE pre-training on No Substructure images → fine-tune for 3-class classification. **Task IX.B:** Fine-tune the same MAE backbone for super-resolution.

Masked Autoencoder pre-training builds a general-purpose feature representation of strong lensing images from unlabelled data. Task IX.B investigates whether the same representation transfers to the pixel-reconstruction objective of super-resolution — probing the universality of lensing image features learned via self-supervision.

→ [Full details](Test_IX_Foundation_Model/README.md)

---

## 🛠 Common Dependencies

All tests use Python 3.10 and PyTorch 2.x. Test-specific requirements are listed in each folder's `requirements.txt`. Shared core dependencies:

```bash
pip install torch torchvision
pip install numpy matplotlib scikit-learn
pip install timm einops
```

All experiments were run on **Google Colab (NVIDIA A100 GPU, high-memory runtime)**.

---

##  Submission

Submitted via the [ML4SCI Google Form](https://forms.gle/) · Deadline: **April 1, 2026**

For project enquiries: [ml4-sci@cern.ch](mailto:ml4-sci@cern.ch) — *please include Project Title in subject*

---

## 📚 References

- Alexander, S., Gleyzer, S., McDonough, E., Toomey, M. W., & Usai, E. (2020). "Deep Learning the Morphology of Dark Matter Substructure." *The Astrophysical Journal*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
- Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). "Densely Connected Convolutional Networks." *CVPR*.
- Dosovitskiy, A., et al. (2020). "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*.
- Weiler, M., & Cesa, G. (2019). "General E(2)-Equivariant Steerable CNNs." *NeurIPS*.
- Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." *ICCV*.
- Abnar, S., & Zuidema, W. (2020). "Quantifying Attention Flow in Transformers." *ACL*.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). "Masked Autoencoders Are Scalable Vision Learners." *CVPR*.

---

<div align="center">
<sub>ML4SCI DeepLense · GSoC 2026 · Rafiqus Salehin</sub>
</div>
