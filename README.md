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
| [**VI**](Test_VI_Image_Super-resolution/) | Image Super-Resolution | Upscale low-resolution lensing images; adapt to real HSC/HST data | MSE · SSIM · PSNR | ✅ |

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

| Rank | Architecture | Pretrained | Params (M) | Macro AUC | No Sub AUC | Sphere AUC | Vortex AUC | Sphere PR-AUC | Test Acc. | Sphere Recall |
|:----:|:-------------|:----------:|:----------:|:---------:|:----------:|:----------:|:----------:|:-------------:|:---------:|:-------------:|
| 1 | **EqDenseNet-C8** | ❌ | **0.1** | **0.9973** | **0.9974** | **0.9955** | **0.9990** | **0.9932** | **97.35%** | **0.9448** |
| 2 | Ensemble-Top6 | Mixed | N/A | 0.9970 | 0.9970 | 0.9948 | 0.9990 | 0.9923 | 97.25% | 0.9360 |
| 3 | DenseNet-121 | ✅ | 7.0 | 0.9962 | 0.9963 | 0.9937 | 0.9985 | 0.9903 | 96.88% | 0.9364 |
| 4 | E-ResNet | ❌ | 0.4 | 0.9952 | 0.9960 | 0.9918 | 0.9977 | 0.9871 | 95.96% | 0.9224 |
| 5 | ResNet-50 | ✅ | 23.5 | 0.9946 | 0.9951 | 0.9909 | 0.9974 | 0.9862 | 96.08% | 0.9180 |
| 6 | ResNet-18 | ✅ | 11.2 | 0.9927 | 0.9939 | 0.9872 | 0.9966 | 0.9813 | 95.24% | 0.9064 |
| 7 | EfficientNet-B3 | ✅ | 10.7 | 0.9898 | 0.9915 | 0.9830 | 0.9947 | 0.9749 | 93.87% | 0.8848 |
| 8 | Equivariant-D4 | ❌ | 0.2 | 0.9784 | 0.9829 | 0.9700 | 0.9818 | 0.9574 | 89.48% | 0.8016 |
| 9 | ViT-Base | ✅ | 85.4 | 0.9761 | 0.9861 | 0.9612 | 0.9805 | 0.9413 | 88.81% | 0.7800 |
| 10 | VGG-16 | ✅ | 134.3 | 0.8944 | 0.9385 | 0.8659 | 0.8783 | 0.8065 | 72.43% | 0.5044 |
| 11 | AlexNet | ✅ | 57.0 | 0.6589 | 0.7358 | 0.6053 | 0.6351 | 0.4380 | 43.65% | 0.1240 |

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

| Rank | Architecture | Pretrained | Params (M) | AUC-ROC ↑ | AUC-PR ↑ | Sens@τ* | Spec@τ* | Prec@τ* | FP | FN | FP:TP |
|:----:|:-------------|:----------:|:----------:|:---------:|:--------:|:-------:|:-------:|:-------:|---:|---:|------:|
| 1 | **Soft-Ensemble** | Mixed | N/A | **0.9905** | **0.8233** | 0.9487 | 0.9805 | **0.328** | 379 | 10 | **2.0** |
| 2 | ResNet-34 | ✅ | 21.29 | 0.9881 | 0.7851 | 0.9026 | **0.9808** | 0.320 | **374** | 19 | 2.1 |
| 3 | EqDenseNet-C8 | ❌ | **0.183** | 0.9872 | 0.7728 | **0.9641** | 0.9430 | 0.145 | 1,109 | **7** | 5.9 |
| 4 | DenseNet-121 | ✅ | 6.95 | 0.9844 | 0.7632 | 0.9282 | 0.9670 | 0.220 | 642 | 14 | 3.5 |
| 5 | E-ResNet D₄ | ❌ | 0.513 | 0.9840 | 0.6963 | 0.9333 | 0.9451 | 0.146 | 1,068 | 13 | 5.9 |
| 6 | EfficientNet-B2 | ✅ | 7.70 | 0.9790 | 0.7087 | 0.9179 | 0.9600 | 0.187 | 778 | 16 | 4.3 |

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
> **Task VI.A:** Super-resolution on simulated HR/LR lensing pairs.  
> **Task VI.B:** Sim-to-real adaptation on 300 real HSC/HST pairs.

This test studies gravitational lens image super-resolution across two regimes: supervised reconstruction on simulated data and few-shot transfer to real telescope observations. Test VI.A benchmarks multiple SR architectures on simulated paired data; Test VI.B measures the sim-to-real domain gap and evaluates fine-tuning strategies on real HSC/HST pairs.

| Part | Best Model | Setting | MSE ↓ | PSNR ↑ | SSIM ↑ | Key Result |
|:----:|:-----------|:--------|------:|-------:|-------:|:-----------|
| **VI.A** | **EDSR-FullImg** | Simulated paired data | **0.67 × 10⁻⁴** | **41.840 dB** | **0.97678** | Best overall result on simulated lensing SR; full-image training outperformed all patch-trained variants |
| **VI.B** | **Strategy C (Progressive Unfreezing)** | Real HSC/HST pairs | **0.00034** | **35.959 dB** | **0.8991** | Best test-set point estimate on real data; sim-to-real fine-tuning recovers most of the zero-shot performance drop |

**Domain-gap summary:** simulated reference **SSIM 0.9768** → zero-shot real-data **0.4712** → fine-tuned real-data **0.8991**.

→ [Full details](Test_VI_Image_Super-resolution/)


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

Submitted via the [ML4SCI Google Form](https://forms.gle/) 

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
