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
| [**V**](Test_V_Lens_Finding/) | Lens Finding & Data Pipelines | Binary classification of lensed vs non-lensed galaxies from observational data | ROC · AUC | ✅ |
| [**VI**](Test_VI_Super_Resolution/) | Image Super-Resolution | Upscale low-resolution lensing images; adapt to real HSC/HST data | MSE · SSIM · PSNR | 🔄 |

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

```

---

##  Test Summaries

### Test I — Multi-Class Dark Matter Substructure Classification

> **Dataset:** 30,000 simulated strong lensing images (150×150,
> single-channel) across three balanced classes — No Substructure,
> CDM Subhalo (Sphere), and Axion Vortex.

Eleven architectures benchmarked: AlexNet, VGG-16, ResNet-18,
ResNet-50, DenseNet-121, EfficientNet-B3, ViT-Base/16,
Equivariant-D4, E-ResNet D₄, EqDenseNet-C8, and a Top-6
soft-vote ensemble. Includes GradCAM and Attention Rollout
interpretability, deep ensemble uncertainty decomposition,
failure mode analysis, ablation study, equivariance verification
via `in_type.transform`, and a residual-image approach via
convolutional autoencoder.

| Rank | Model | Params | Macro AUC | Sphere AUC | Sphere PR-AUC |
|:----:|:------|-------:|:---------:|:----------:|:-------------:|
| 1 | EqDenseNet-C8 | 0.10M | **0.9973** | 0.9955 | 0.9932 |
| 2 | Ensemble-Top6 | — | 0.9970 | 0.9948 | 0.9923 |
| 3 | DenseNet-121 | 7.0M | 0.9962 | 0.9937 | 0.9903 |
| 4 | E-ResNet D₄ | 0.4M | 0.9952 | 0.9918 | 0.9871 |
| 5 | ResNet-50 | 23.5M | 0.9946 | 0.9909 | 0.9862 |

**Best individual result:** EqDenseNet-C8 — Macro AUC **0.9973**,
Sphere PR-AUC **0.9932**, at only 0.10M parameters.

→ [Full details](Test_I_Classification/README.md)

---

### Test V — Gravitational Lens Finding on Real HSC Observational Data

> **Dataset:** Real Hyper Suprime-Cam (HSC) survey cutouts.
> 3-channel (g, r, i), 64×64 pixels. Test imbalance: 99.8:1
> (195 lenses / 19,455 non-lenses).

Binary lens/non-lens classification under severe class imbalance.
Pipeline includes Focal Loss, WeightedRandomSampler,
PR-AUC-first evaluation, Youden's J threshold selection,
post-hoc Platt scaling calibration, GradCAM interpretability,
and algebraic equivariance verification. Three silent `escnn`
implementation errors identified and corrected.

| Rank | Model | Params | AUC-ROC | AUC-PR | Sens@τ* |
|:----:|:------|-------:|:-------:|:------:|:-------:|
| 1 | Soft-Ensemble | — | **0.9905** | **0.8233** | 0.9487 |
| 2 | ResNet-34 | 21.3M | 0.9881 | 0.7851 | 0.9026 |
| 3 | EqDenseNet-C8 | 0.18M | 0.9872 | 0.7728 | **0.9641** |
| 4 | DenseNet-121 | 6.95M | 0.9844 | 0.7632 | 0.9282 |
| 5 | E-ResNet D₄ | 0.51M | 0.9840 | 0.6963 | 0.9333 |
| 6 | EfficientNet-B2 | 7.70M | 0.9790 | 0.7087 | 0.9179 |

**Best ensemble:** AUC-ROC **0.9905**, AUC-PR **0.8233**
(83× above the random PR baseline of 0.010).
**Best individual by sensitivity:** EqDenseNet-C8 — FN = 7
(missed only 7 out of 195 lenses), at 0.18M parameters
trained entirely from scratch.
All five individual models exceed the best prior DeepLense
HSC baseline (AUC-ROC 0.816) by more than 0.16 AUC points.

→ [Full details](Test_V_Lens_Finding/README.md)

---

### Test VI — Image Super-Resolution
> **Task VI.A:** Simulated HR/LR lensing image pairs. **Task VI.B:** 300 real HSC/HST HR/LR pairs — few-shot / transfer learning from VI.A.

Deep learning super-resolution pipeline (e.g. EDSR, SRGAN, or SwinIR backbone). Task VI.B explores domain adaptation from simulated to real telescope data with limited labelled pairs.

→ [Full details](Test_VI_Super_Resolution/README.md)


---

##  Common Dependencies

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

##  References

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
