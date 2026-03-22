<div align="center">

# ML4SCI DeepLense — GSoC 2026
## Common Test V: Gravitational Lens Finding on Real Observational Data

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0+cu128-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![escnn](https://img.shields.io/badge/escnn-1.0.11-blueviolet?style=flat-square)](https://github.com/QUVA-Lab/escnn)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA_A100_80GB-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://www.nvidia.com)
[![Dataset](https://img.shields.io/badge/Data-HSC_Real_Survey-blue?style=flat-square)](https://hsc.mtk.nao.ac.jp)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Notebook](https://img.shields.io/badge/Notebook-Open_in_Colab-F9AB00?style=flat-square&logo=google-colab&logoColor=white)](https://colab.research.google.com)

<br>

**Binary classification of gravitational lenses in real multi-band survey images (HSC)**
using five architectures — three ImageNet-pretrained CNNs and two physics-motivated
equivariant networks encoding rotational symmetry directly into their weights —
followed by a soft ensemble combining all five models.

Unlike Test I (simulated single-channel data), all images here are **real Hyper
Suprime-Cam (HSC) observational data** across three photometric bands (g, r, i),
presenting real sky backgrounds, PSF effects, and a severe 99.8:1 class imbalance
at test time that challenges all models and evaluation metrics.

<br>

| | |
|:-:|:-:|
| **Best AUC-ROC** | **0.9905** — Soft-Ensemble (5 models) |
| **Best AUC-PR** | **0.8233** — Soft-Ensemble (83× above random baseline of 0.010) |
| **Best Precision@τ*** | **0.328** — Soft-Ensemble (FP:TP = 2.0:1) |
| **Best Individual AUC-ROC** | 0.9881 — ResNet-34 (21.29M params, ImageNet) |
| **Best Efficiency (AUC-ROC)** | **0.9872** — EqDenseNet-C8 (**0.183M params, from scratch**) |
| **Fewest Missed Lenses** | **FN = 7** — EqDenseNet-C8 (96.4% sensitivity at τ*) |
| **Task** | Binary classification (lens / non-lens) |
| **Input** | 64 × 64 × 3 (g, r, i bands — real HSC imaging) |
| **Test imbalance** | 195 lenses / 19,455 non-lenses (99.8:1) |

</div>

---

> **Note on the Real-Data Setting**
>
> Test V is structurally harder than Test I. All images are real HSC observational
> data — not simulations — introducing genuine sky backgrounds, variable PSF, and
> photometric noise. The test-set class ratio is 99.8:1: AUC-ROC alone is insufficient
> to characterise operational usefulness. PR-AUC is the primary diagnostic metric;
> a random classifier achieves precision ≈ 0.010 at all recall levels.
>
> All equivariant models are adapted from Test I via a 3-channel input representation
> (`3 × trivial_repr`) and trained entirely from scratch. Three escnn bugs that broke
> D₄ invariance in the original Test I implementation were identified and corrected
> during this work via the `in_type.transform` group action verification protocol —
> documented in §5 and applied to both notebooks.

---

## Table of Contents

1. [Scientific Background](#1-scientific-background)
2. [Dataset](#2-dataset)
   - [2.1 Data Source and Structure](#21-data-source-and-structure)
   - [2.2 Class Statistics and Imbalance Regimes](#22-class-statistics-and-imbalance-regimes)
   - [2.3 Key Observational Properties](#23-key-observational-properties)
3. [Environment Setup](#3-environment-setup)
4. [Training & Evaluation Framework](#4-training--evaluation-framework)
   - [4.1 Data Pipeline](#41-data-pipeline)
   - [4.2 Key Design Decisions](#42-key-design-decisions)
   - [4.3 Evaluation Protocol](#43-evaluation-protocol)
5. [Architecture Evaluations](#5-architecture-evaluations)
   - [5.1 EfficientNet-B2](#51-efficientnet-b2)
   - [5.2 ResNet-34](#52-resnet-34)
   - [5.3 DenseNet-121](#53-densenet-121)
   - [5.4 E-ResNet D₄ (3-channel)](#54-e-resnet-d-3-channel)
   - [5.5 EqDenseNet-C8 (3-channel)](#55-eqdensenet-c8-3-channel)
   - [5.6 Soft Ensemble](#56-soft-ensemble)
6. [Comprehensive Results Summary](#6-comprehensive-results-summary)
   - [6.1 Full Benchmark Table](#61-full-benchmark-table)
   - [6.2 ROC and Precision-Recall Curves](#62-roc-and-precision-recall-curves)
   - [6.3 Parameter Efficiency](#63-parameter-efficiency)
   - [6.4 Key Takeaways](#64-key-takeaways)
7. [Interpretability Analysis](#7-interpretability-analysis)
   - [7.1 Grad-CAM: EfficientNet-B2 and DenseNet-121](#71-grad-cam-efficientnet-b2-and-densenet-121)
   - [7.2 Per-Channel Attribution — Integrated Gradients](#72-per-channel-attribution--integrated-gradients)
8. [Failure Mode Analysis](#8-failure-mode-analysis)
   - [8.1 Per-model Scores on Ensemble False Negatives](#81-per-model-scores-on-ensemble-false-negatives)
   - [8.2 Cross-architecture Disagreement](#82-cross-architecture-disagreement)
   - [8.3 Statistical Characterisation of Missed vs Caught Lenses](#83-statistical-characterisation-of-missed-vs-caught-lenses)
   - [8.4 False Positive Characterisation](#84-false-positive-characterisation)
9. [Discussion](#9-discussion)
10. [Limitations & Future Work](#10-limitations--future-work)
11. [GSoC 2026 Research Directions](#11-gsoc-2026-research-directions)
12. [Repository Structure](#12-repository-structure)
13. [Citation](#13-citation)

---

## 1. Scientific Background

Strong gravitational lensing — the deflection of light from a distant source galaxy by
a massive foreground lens — produces characteristic arc and Einstein ring morphologies.
Detecting these systems at survey scale is a prerequisite for constraining dark matter
substructure models, measuring cosmological parameters through lens statistics, and
identifying targets for high-resolution follow-up.

The challenge is the extreme scarcity of genuine gravitational lenses relative to
the non-lens population in ground-based imaging surveys:

```
          HYPER SUPRIME-CAM FIELD
                    │
         ┌──────────┴──────────────┐
         │                         │
   ~0.01% lenses        ~99.99% non-lenses
   (Einstein rings,      (galaxies, stars,
    arcs, knots)          galaxy pairs,
                          ring galaxies, ...)
                    │
         At 64×64 + PSF smearing:
         Arc geometry is rarely resolved.
         Models must classify on subtle
         radial flux profiles, not shapes.
```

At the pixel resolution of this dataset (64×64), PSF convolution suppresses resolved
arc geometry in most cases. The discriminative signal is statistical — a radial
brightness profile difference between lens and non-lens populations — rather than
explicit morphology. This makes the problem harder than it appears from physical
intuition, and motivates both the use of deep models and the investigation of
physically-motivated architectural priors.

---

## 2. Dataset

### 2.1 Data Source and Structure

All images are real Hyper Suprime-Cam (HSC) survey cutouts from the Wide layer,
stored as `.npy` arrays of shape `(3, 64, 64)` in three photometric bands (g, r, i).
This is real observational data — not simulations — with genuine sky backgrounds,
PSF variation, and photometric noise.

```
lens_finding_data/
├── train_lenses/      1,730  .npy files    ← confirmed gravitational lenses
├── train_nonlenses/  28,675  .npy files    ← galaxies, stars, other objects
├── test_lenses/         195  .npy files    ← held out until final evaluation
└── test_nonlenses/   19,455  .npy files    ← held out until final evaluation
```

The train set is further split 90:10 by stratified sampling to produce a validation
set for early stopping and threshold selection. The test set is untouched until
final evaluation.

### 2.2 Class Statistics and Imbalance Regimes

| Split | Lenses | Non-Lenses | Total | Ratio (neg:pos) |
|:------|-------:|-----------:|------:|----------------:|
| **Train** | 1,730 | 28,675 | 30,405 | 16.6 : 1 |
| **Val** (from train) | 173 | 2,868 | 3,041 | 16.6 : 1 |
| **Test** | 195 | 19,455 | 19,650 | **99.8 : 1** |

Two distinct imbalance regimes are relevant throughout. The training imbalance
(16.6:1) is severe and requires explicit handling. The test imbalance (99.8:1)
is extreme and fundamentally changes what metrics are informative. Under 99.8:1,
a random score-based classifier achieves precision ≈ 0.010 at all recall levels —
this is the PR-AUC random baseline throughout.

### 2.3 Key Observational Properties

Three properties of the data drive all modelling choices:

**1. PSF suppresses arc morphology at 64×64.** The defining lensing signature —
resolved arcs or Einstein rings — is structurally suppressed by PSF convolution and
pixel sampling. What remains is a radial brightness gradient around the central
source. Explicit arc detection is not viable at this image scale; models must exploit
statistical flux distributions.

**2. The classification signal is isotropic in expectation.** Mean difference maps
(lens − non-lens) across 300 samples show no preferred orientation after averaging.
Orientation is not a systematically useful discriminative feature at the population
level. This motivates D₄ augmentation for pretrained models and equivariant
architectures.

**3. Severe and asymmetric imbalance across splits.** Train: 16.6:1, Test: 99.8:1.
PR-AUC is more diagnostic than AUC-ROC under the test imbalance. Model rankings on
the two metrics partially disagree — both are reported throughout.

---

## 3. Environment Setup

All experiments were conducted on **Google Colab with a single NVIDIA A100-SXM4-80GB GPU**.

### Requirements

```bash
pip install escnn          # equivariant neural networks (Sections 5.4–5.5)
pip install grad-cam       # Grad-CAM saliency maps (Section 7.1)
pip install captum         # Integrated Gradients (Section 7.2)
pip install "numpy==1.26.4"  # pin AFTER other installs — escnn requires numpy 1.x
# ⚠️  Restart runtime after numpy pin before importing model code
```

> **Note:** `escnn` pulls `lie-learn` which requires `numpy < 2`. Pin `numpy==1.26.4`
> and restart the kernel before any model imports.

### Verified Version Matrix

| Package | Version | Note |
|:--------|:-------:|:-----|
| Python | 3.12.12 | Colab default |
| PyTorch | 2.10.0+cu128 | cu128 build |
| torchvision | 0.25.0+cu128 | — |
| numpy | 1.26.4 | pinned — see above |
| escnn | 1.0.11 | D₄ and C8 equivariant layers |
| grad-cam | 1.5.5 | Grad-CAM visualisation |
| captum | latest | Integrated Gradients |
| scikit-learn | 1.6.1 | AUC, calibration, PR curves |
| CUDA | 12.8 | A100 |

---

## 4. Training & Evaluation Framework

### 4.1 Data Pipeline

```
Raw .npy files (3, 64, 64) — g, r, i bands
              │
              ▼
Per-sample global min-max normalisation:
x̂ = (x − min(x)) / (max(x) − min(x))
applied across all three channels jointly
→ preserves relative cross-band flux structure
              │
              ├─── [TRAIN] D₄ augmentation:
              │    random {torch.flip (hflip/vflip), torch.rot90 (90°/180°/270°)}
              │    exact pixel permutation — no interpolation
              │
              ├─── [VAL]   No augmentation
              ├─── [TEST]  No augmentation
              │
              ▼
(3, 64, 64) float32 tensor → model input
```

**Normalisation rationale:** Per-sample global min-max preserves relative
cross-band flux structure. Per-channel normalisation was rejected because it
destroys inter-band intensity relationships. Absolute photometric calibration
is not preserved; this is an acceptable tradeoff for classification.

**Augmentation rationale:** The population-level difference maps show no preferred
orientation. D₄ augmentation uses exact `torch.rot90` and `torch.flip` operations —
lossless pixel permutation with no interpolation. `T.RandomRotation` was rejected
because it introduces bilinear interpolation artifacts and is not genuine D₄.

### 4.2 Key Design Decisions

| Decision | Choice | Rationale |
|:---------|:------:|:----------|
| Optimiser | AdamW (wd=1e-3) | Standard; weight decay regularises fine-tuned models |
| Scheduler | CosineAnnealingLR | Smooth decay without manual step tuning |
| Loss | Focal Loss (γ=2, α=0.5) | Focusing term down-weights easy negatives; α=0.5 avoids double-compensation with sampler |
| Gradient clipping | max_norm=1.0 | Prevents gradient explosions |
| Batch balancing | WeightedRandomSampler (~50:50) | Ensures stable positive-class gradient signal with only 1,557 train lenses |
| Checkpoint criterion | Highest validation AUC | Most relevant metric; min_delta=1e-4 guard against noise |
| Threshold selection | Youden's J on **val set** → applied fixed to test set | Prevents threshold leakage |

**On Focal Loss alpha:** Using `WeightedRandomSampler` produces ~50:50 batches.
Setting `alpha=0.943` (inverse class frequency) on top of already-balanced batches
would amplify positive-class gradients twice — a double-compensation flaw.
`alpha=0.5` (neutral) is used here; only the focusing term `gamma` addresses imbalance.

### 4.3 Evaluation Protocol

Five metrics are computed for every architecture:

| Metric | What it measures | Why it matters |
|:-------|:----------------|:---------------|
| **AUC-ROC** | Threshold-free rank discrimination | Primary required metric; robust to class ratio |
| **AUC-PR** | Precision across recall range | Most diagnostic under 99.8:1 imbalance; random baseline ≈ 0.010 |
| **Brier score** | Mean squared probability error | Reported for completeness only — dominated by majority class under severe imbalance |
| **Youden τ*** | argmax(TPR − FPR) from **val set** | Deployment-clean operating point |
| **FP / FN counts @ τ*** | Absolute operational cost | Rate metrics mislead under 99.8:1; absolute counts determine follow-up cost |

> **Note on Brier score:** Under 99.8:1 imbalance, a trivial classifier that always
> predicts 0 achieves Brier ≈ 0.0099 — comparable to or better than trained models.
> Do not use as a primary metric here.

---

## 5. Architecture Evaluations

Five architectures are evaluated: three ImageNet-pretrained CNNs (§5.1–5.3), two
physics-motivated equivariant networks trained from scratch (§5.4–5.5), and a soft
ensemble combining all five (§5.6).

All equivariant models use `3 × trivial_repr` input — each photometric band treated
as a scalar field that transforms trivially under the group. Photometric flux has no
directional component, making the trivial representation physically correct.

### Architecture Overview

```
ImageNet pretrained (fine-tuned)         Equivariant from scratch
───────────────────────────────          ─────────────────────────────────
EfficientNet-B2  (7.70M, compound)       E-ResNet D₄   (0.513M, residual)
ResNet-34       (21.29M, residual)       EqDenseNet-C8 (0.183M, dense)
DenseNet-121     (6.95M, dense)
                    │
                    └──── Soft Ensemble (mean of all 5)
```

---

### 5.1 EfficientNet-B2

**Role:** Compound-scaled pretrained baseline — tests NAS-derived scaling with
ImageNet initialisation.

**Architecture:** EfficientNet-B2 with classifier replaced:
`Dropout(0.3) → Linear(1408, 1)`. All 7.70M parameters fine-tuned.

**Training:** Early stopping at epoch 23, best val AUC = 0.9886.

| Metric | Value |
|:-------|------:|
| **AUC-ROC** | **0.9790** |
| **AUC-PR** | **0.7087** |
| PR-AUC / random baseline | 72× |
| Youden τ* (val set) | 0.2751 |
| Sensitivity @ τ* | 0.9179 |
| Specificity @ τ* | 0.9600 |
| Precision @ τ* | 0.187 (179/957) |
| FP count | 778 |
| FN count | 16 |
| FP : TP ratio | 4.3 : 1 |
| Parameters | 7.70M |
| Pretrained | ✅ ImageNet |

**Confusion matrix at τ* = 0.275:**

| | Predicted Non-Lens | Predicted Lens |
|:---|---:|---:|
| **True Non-Lens** | 18,677 | 778 |
| **True Lens** | 16 | 179 |

EfficientNet-B2 ranks last among individual models on AUC-ROC (0.9790). Its
lower cross-model correlation (0.737–0.765) contributes prediction diversity to
the ensemble despite its weaker individual performance.

---

### 5.2 ResNet-34

**Role:** Deeper residual pretrained baseline — best individual model on both
AUC-ROC and AUC-PR.

**Architecture:** ResNet-34 with classifier replaced:
`Linear(512, 1000)` → `Linear(512, 1)`. All 21.29M parameters fine-tuned.

**Training:** Early stopping at epoch 26, best val AUC = 0.9954.

| Metric | Value |
|:-------|------:|
| **AUC-ROC** | **0.9881** |
| **AUC-PR** | **0.7851** |
| PR-AUC / random baseline | 79× |
| Youden τ* (val set) | 0.2792 |
| Sensitivity @ τ* | 0.9026 |
| Specificity @ τ* | 0.9808 |
| Precision @ τ* | 0.320 (176/550) |
| **FP count** | **374** |
| FN count | 19 |
| FP : TP ratio | 2.1 : 1 |
| Parameters | 21.29M |
| Pretrained | ✅ ImageNet |

**Confusion matrix at τ* = 0.279:**

| | Predicted Non-Lens | Predicted Lens |
|:---|---:|---:|
| **True Non-Lens** | 19,081 | 374 |
| **True Lens** | 19 | 176 |

ResNet-34 leads all individual models on AUC-ROC (0.9881), AUC-PR (0.7851),
and fewest false positives (374). Its Brier score (0.0089) is the only model
to fall below the trivial 0.0099 baseline — noted but not interpreted as good
calibration under this imbalance.

---

### 5.3 DenseNet-121

**Role:** Dense-connectivity pretrained baseline — strongest parameter efficiency
among pretrained models.

**Architecture:** DenseNet-121 with classifier replaced:
`Linear(1024, 1000)` → `Linear(1024, 1)`. All 6.95M parameters fine-tuned.

**Training:** Early stopping at epoch 36, best val AUC = 0.9943.

| Metric | Value |
|:-------|------:|
| **AUC-ROC** | **0.9844** |
| **AUC-PR** | **0.7632** |
| PR-AUC / random baseline | 77× |
| Youden τ* (val set) | **0.1359** |
| Sensitivity @ τ* | 0.9282 |
| Specificity @ τ* | 0.9670 |
| Precision @ τ* | 0.220 (181/823) |
| FP count | 642 |
| FN count | 14 |
| FP : TP ratio | 3.5 : 1 |
| Parameters | **6.95M** |
| Pretrained | ✅ ImageNet |

**Confusion matrix at τ* = 0.136:**

| | Predicted Non-Lens | Predicted Lens |
|:---|---:|---:|
| **True Non-Lens** | 18,813 | 642 |
| **True Lens** | 14 | 181 |

DenseNet-121's τ* of 0.136 is the lowest among pretrained models, indicating
its score distribution is compressed toward low values. The calibration curve
should be consulted directly rather than inferred from the threshold value.

---

### 5.4 E-ResNet D₄ (3-channel)

**Role:** Physics-motivated D₄-equivariant ResNet — tests rotational symmetry as
an architectural prior, trained entirely from scratch.

**Architecture:** D₄-equivariant ResNet (`flipRot2dOnR2(N=4)`, order 8) with
`in_type = 3 × trivial_repr`. Key constraint: all downsampling uses
`enn.PointwiseAvgPool + stride-1 conv` — strided `R2Conv` breaks equivariance.

**D₄ Invariance Verification (untrained, all 8 elements):**

| Element | |Δlogit| | Element | |Δlogit| |
|:--------|--------:|:--------|--------:|
| identity | 0.00e+00 ✓ | flip | 1.40e-06 ✓ |
| rot 90° | 2.38e-06 ✓ | flip∘rot90 | 1.64e-06 ✓ |
| rot 180° | 7.84e-06 ✓ | flip∘rot180 | 1.43e-06 ✓ |
| rot 270° | 4.11e-06 ✓ | flip∘rot270 | 2.38e-06 ✓ |
| **Max: 7.84e-06 ✓** | | | |

**Training:** Early stopping at epoch 17, best val AUC = 0.9932 (reached at epoch 2).

| Metric | Value |
|:-------|------:|
| **AUC-ROC** | **0.9840** |
| **AUC-PR** | **0.6963** |
| PR-AUC / random baseline | 70× |
| Youden τ* (val set) | 0.3064 |
| Sensitivity @ τ* | 0.9333 |
| Specificity @ τ* | 0.9451 |
| Precision @ τ* | 0.146 (182/1250) |
| FP count | 1,068 |
| FN count | 13 |
| FP : TP ratio | 5.9 : 1 |
| Parameters | **0.513M** |
| Pretrained | ❌ Scratch |

**Confusion matrix at τ* = 0.306:**

| | Predicted Non-Lens | Predicted Lens |
|:---|---:|---:|
| **True Non-Lens** | 18,387 | 1,068 |
| **True Lens** | 13 | 182 |

E-ResNet D₄ achieves AUC-ROC 0.9840 at 0.513M parameters — close to DenseNet-121
(0.9844) at 13.5× fewer parameters. On AUC-PR (0.6963), it has the lowest score
of any model in this benchmark. Best checkpoint was selected at epoch 2 of 17,
which may reflect a lucky initialisation rather than stable convergence.

---

### 5.5 EqDenseNet-C8 (3-channel)

**Role:** Novel combination of C8 cyclic equivariance with DenseNet-style dense
connectivity — most parameter-efficient model in the benchmark.

**Architecture:** C8-equivariant DenseNet (`rot2dOnR2(N=8)`, order 8) with
`in_type = 3 × trivial_repr`. Spatial trace: 64→32→16→8→1×1 (via
`PointwiseAvgPool` + `GroupPooling`). 0.183M parameters, from scratch.

**C8 Invariance Verification (untrained):**

| Rotation | |Δlogit| | Note |
|:---------|--------:|:-----|
| 0° | 0.00e+00 ✓ | — |
| 45° | 2.50e-03 † | Discrete grid interpolation artifact |
| 90° | 2.98e-08 ✓ | — |
| 135° | 2.51e-03 † | Discrete grid interpolation artifact |
| 180° | 1.34e-06 ✓ | — |
| 225° | 2.50e-03 † | Discrete grid interpolation artifact |
| 270° | 5.74e-07 ✓ | — |
| 315° | 2.51e-03 † | Discrete grid interpolation artifact |

† C4 subgroup (90° rotations) verified at machine precision. Odd elements show
~2.5e-03 from `in_type.transform` bilinear resampling of the discrete pixel grid —
a grid artifact, not an architecture flaw. C8 equivariance holds exactly in the
continuous domain via escnn weight-sharing.

**Training:** Early stopping at epoch 41, best val AUC = 0.9958.

| Metric | Value |
|:-------|------:|
| **AUC-ROC** | **0.9872** |
| **AUC-PR** | **0.7728** |
| PR-AUC / random baseline | 78× |
| Youden τ* (val set) | **0.0548** |
| **Sensitivity @ τ*** | **0.9641** |
| Specificity @ τ* | 0.9430 |
| Precision @ τ* | 0.145 (188/1297) |
| FP count | 1,109 |
| **FN count** | **7** |
| FP : TP ratio | 5.9 : 1 |
| Parameters | **0.183M** |
| Pretrained | ❌ Scratch |

**Confusion matrix at τ* = 0.055:**

| | Predicted Non-Lens | Predicted Lens |
|:---|---:|---:|
| **True Non-Lens** | 18,346 | 1,109 |
| **True Lens** | 7 | 188 |

EqDenseNet-C8 achieves the highest sensitivity of any model (96.4% — fewest
missed lenses: FN=7) and the strongest AUC-ROC among from-scratch models (0.9872),
exceeding DenseNet-121 (0.9844) at 37.9× fewer parameters. On AUC-PR (0.7728) it
exceeds both DenseNet-121 (0.7632) and EfficientNet-B2 (0.7087), but not ResNet-34.

The τ* of 0.055 — the lowest in the benchmark — indicates strong score compression.
High recall and high FP count are the direct consequence of this threshold.

---

### 5.6 Soft Ensemble

**Members:** EfficientNet-B2, ResNet-34, DenseNet-121, E-ResNet D₄, EqDenseNet-C8

**Method:** Mean of per-model sigmoid probabilities — no retraining. Youden τ*
computed on mean of all five models' val-set probabilities (deployment-clean).

**Per-model probability correlation matrix:**

| | EffNet-B2 | ResNet-34 | DenseNet | EResNet | EqDenseNet |
|:---|:---:|:---:|:---:|:---:|:---:|
| EfficientNet-B2 | 1.000 | 0.765 | 0.761 | 0.748 | 0.737 |
| ResNet-34 | 0.765 | 1.000 | 0.817 | 0.750 | 0.819 |
| DenseNet-121 | 0.761 | 0.817 | 1.000 | 0.708 | 0.804 |
| E-ResNet D₄ | 0.748 | 0.750 | 0.708 | 1.000 | 0.761 |
| EqDenseNet-C8 | 0.737 | 0.819 | 0.804 | 0.761 | 1.000 |

Range: 0.708–0.819. Lowest pair: DenseNet-121 / E-ResNet D₄ (0.708). Highest pair:
ResNet-34 / EqDenseNet-C8 (0.819). There is no clean pretrained vs equivariant
separation in the correlation structure.

| Metric | Value | vs ResNet-34 |
|:-------|------:|-------------:|
| **AUC-ROC** | **0.9905** | +0.0024 ↑ |
| **AUC-PR** | **0.8233** | +0.038 ↑ |
| PR-AUC / random baseline | **83×** | — |
| Youden τ* (val set) | 0.3143 | — |
| Sensitivity @ τ* | 0.9487 | — |
| Specificity @ τ* | 0.9805 | — |
| **Precision @ τ*** | **0.328** (185/564) | — |
| FP count | 379 | +5 |
| FN count | 10 | −9 |
| FP : TP ratio | **2.0 : 1** | — |

**Confusion matrix at τ* = 0.314:**

| | Predicted Non-Lens | Predicted Lens |
|:---|---:|---:|
| **True Non-Lens** | 19,076 | 379 |
| **True Lens** | 10 | 185 |

The ensemble improves on AUC-ROC (+0.0024) and AUC-PR (+0.038) over ResNet-34 —
threshold-free improvements holding across the full operating range. Note: ResNet-34
achieves 374 FP at its Youden threshold vs ensemble's 379 — the ensemble does not
uniformly reduce false positives at all operating points.

---

## 6. Comprehensive Results Summary

### 6.1 Full Benchmark Table

All architectures evaluated on the held-out test set (19,650 images).
Sorted by AUC-ROC.

| Rank | Architecture | Pretrained | Params (M) | AUC-ROC ↑ | AUC-PR ↑ | Sens@τ* | Spec@τ* | Prec@τ* | FP | FN | FP:TP |
|:----:|:-------------|:----------:|:----------:|:---------:|:--------:|:-------:|:-------:|:-------:|---:|---:|------:|
| 1 | **Soft-Ensemble** | Mixed | 36.64 | **0.9905** | **0.8233** | 0.9487 | 0.9805 | **0.328** | 379 | 10 | **2.0** |
| 2 | ResNet-34 | ✅ | 21.29 | 0.9881 | 0.7851 | 0.9026 | **0.9808** | 0.320 | **374** | 19 | 2.1 |
| 3 | EqDenseNet-C8 | ❌ | **0.183** | 0.9872 | 0.7728 | **0.9641** | 0.9430 | 0.145 | 1,109 | **7** | 5.9 |
| 4 | DenseNet-121 | ✅ | 6.95 | 0.9844 | 0.7632 | 0.9282 | 0.9670 | 0.220 | 642 | 14 | 3.5 |
| 5 | E-ResNet D₄ | ❌ | 0.513 | 0.9840 | 0.6963 | 0.9333 | 0.9451 | 0.146 | 1,068 | 13 | 5.9 |
| 6 | EfficientNet-B2 | ✅ | 7.70 | 0.9790 | 0.7087 | 0.9179 | 0.9600 | 0.187 | 778 | 16 | 4.3 |

*Random PR baseline: 0.0099. All six models exceed the Mriganka 2022 HSC baseline
(AUC 0.816) by more than 0.16 points. Sensitivity, specificity, precision, FP, FN,
and FP:TP are at the val-set Youden threshold for each model — thresholds differ
across models and direct CM comparisons are indicative, not controlled.*

### 6.2 ROC and Precision-Recall Curves

**ROC curves:** All models achieve AUC-ROC > 0.979. Individual model span: 0.009
(0.9790–0.9881). Ordering: Ensemble > ResNet-34 > EqDenseNet-C8 > DenseNet-121 >
E-ResNet D₄ > EfficientNet-B2. Curves are tightly clustered; a zoomed inset is
required to distinguish ordering.

**PR curves:** Rankings differ from ROC-AUC. PR-AUC span across individual models:
0.089 (0.6963–0.7851) — substantially wider. E-ResNet D₄ shows the largest
precision collapse at high recall (lowest PR-AUC: 0.6963). EqDenseNet-C8 outperforms
E-ResNet D₄ substantially on PR-AUC (0.7728 vs 0.6963) despite similar parameter
counts. PR-AUC is the more diagnostic metric under this imbalance.

### 6.3 Parameter Efficiency

```
AUC-ROC parameter efficiency:
──────────────────────────────────────────────────────────────────
Rank  Architecture      AUC-ROC  Params  Pretrained
────  ──────────────── ──────── ─────── ──────────
  1   Soft-Ensemble     0.9905   36.64M  mixed
  2   ResNet-34         0.9881   21.29M  ✅
  3   EqDenseNet-C8     0.9872    0.183M ❌  ← 37.9× smaller than DenseNet-121
  4   DenseNet-121      0.9844    6.95M  ✅
  5   E-ResNet D₄       0.9840    0.513M ❌
  6   EfficientNet-B2   0.9790    7.70M  ✅
──────────────────────────────────────────────────────────────────
```

**AUC-ROC:** EqDenseNet-C8 (0.183M, from scratch) exceeds DenseNet-121 (6.95M,
pretrained) and EfficientNet-B2 (7.70M, pretrained) at 37.9× and 42× fewer
parameters. E-ResNet D₄ (0.513M) also exceeds EfficientNet-B2 at 15× fewer parameters.

**AUC-PR:** EqDenseNet-C8 (0.7728) exceeds DenseNet-121 (0.7632) and EfficientNet-B2
(0.7087) on PR-AUC. E-ResNet D₄ (0.6963) has the lowest PR-AUC of any model —
below EfficientNet-B2 despite fewer parameters. The equivariant efficiency advantage
is real for EqDenseNet-C8 on both metrics but does not generalise to E-ResNet D₄ on
PR-AUC.

### 6.4 Key Takeaways

**1. Ensemble leads on threshold-free metrics.**
AUC-ROC 0.9905 and AUC-PR 0.8233 both exceed every individual model. The PR-AUC
gain of +0.038 over ResNet-34 indicates maintained precision further into the recall
range. At the respective Youden thresholds, ResNet-34 individually produces fewer FP
(374 vs 379) — the FP advantage depends on threshold.

**2. EqDenseNet-C8 achieves competitive AUC at 0.183M parameters.**
AUC-ROC 0.9872 exceeds DenseNet-121 (0.9844) and EfficientNet-B2 (0.9790) at 37.9×
and 42× fewer parameters. Does not exceed ResNet-34 (0.9881) on either metric.
Efficiency result holds for two of three pretrained comparators.

**3. PR-AUC separates models more than ROC-AUC.**
ROC-AUC span: 0.009. PR-AUC span: 0.089. Rankings partially disagree between metrics.
E-ResNet D₄ ranks 5th on ROC-AUC but last on PR-AUC — a significant divergence.

**4. FN and FP tradeoffs differ at operating points.**
EqDenseNet-C8 has the fewest FN (7) but the most FP (1,109) at its τ*.
ResNet-34 has the fewest FP (374) but the most FN (19). These comparisons are at
different thresholds and describe operating point choices, not absolute model properties.

**5. EfficientNet-B2 underperforms relative to its parameter count.**
7.70M pretrained parameters, yet ranks last on AUC-ROC (0.9790) and fourth on AUC-PR
(0.7087) among individual models — below DenseNet-121 on both despite similar size.

---

## 7. Interpretability Analysis

### 7.1 Grad-CAM: EfficientNet-B2 and DenseNet-121

Grad-CAM heatmaps are computed at the final convolutional feature map of each model
and upsampled to 64×64. At this resolution the maps are spatially coarse — they
reflect gradient accumulation over large receptive fields. Descriptions are limited
to what is directly visible; no claims are made about model reasoning.

**EfficientNet-B2 — Correct Lenses (TP, p=1.000):**
High activation (red) occupies the upper portion of the image in all 6 panels —
a dark, structurally featureless region. The bright central source falls in
low-activation territory. The pattern is visually nearly identical across all 6
panels despite diverse source morphologies. The maps are not consistent with
physically motivated attention on the lens or arc region.

**EfficientNet-B2 — Missed Lenses (FN, p=0.001–0.035):**
Activation is near-uniform blue for 4 of 6 panels — essentially no gradient signal.
Two panels show activation at the lower image boundary, away from the source. No
consistent pattern characterises the missed lenses.

**EfficientNet-B2 — False Positives (FP, p=0.948–0.995):**
Grad-CAM activation patterns are visually indistinguishable from the TP maps.
The heatmaps provide no basis for distinguishing TP from FP cases.

**DenseNet-121 — Correct Lenses (TP, p=0.999–1.000):**
High activation in the lower image portion in most panels, source in moderate-to-low
activation territory. More inter-panel variation than EfficientNet-B2, but no
consistent physically motivated pattern.

**DenseNet-121 — Missed Lenses (FN, p=0.000–0.014):**
All 6 panels show near-uniform blue — more uniformly low than EfficientNet-B2.

**DenseNet-121 — False Positives (FP, p=0.964–0.985):**
Variable activation patterns across panels — vertical gradients, edge responses,
arc-like structure. No consistent pattern.

**Overall assessment:** Neither Grad-CAM nor spatial attention analysis, at 64×64
resolution, reveals physically interpretable model reasoning. EfficientNet-B2
produces a near-uniform upper-image template identical across TP and FP cases.
DenseNet-121 shows more inter-sample variation but no consistent physically
motivated pattern. The most honest summary is that these tools are insufficient
to characterise model reasoning at this resolution.

### 7.2 Per-Channel Attribution — Integrated Gradients

Channel-masking Grad-CAM (zeroing non-target channels) creates out-of-distribution
inputs — after global min-max normalisation, zero is not "absence of signal" but the
minimum observed pixel value. **Integrated Gradients (IG)** with a physically
motivated baseline replaces this approach.

**Baseline:** Per-channel training set mean (g=0.270, r=0.188, i=0.105), computed
over 500 randomly sampled training images. All IG interpolation steps remain within
the data distribution.

**IG Results — EfficientNet-B2, TP lenses (n=8) and FN lenses (n=6):**

| | g band | r band | i band |
|:---|:------:|:------:|:------:|
| Lens 1 | 0.369 | 0.507 | 0.123 |
| Lens 2 | 0.314 | 0.571 | 0.115 |
| Lens 3 | 0.381 | 0.513 | 0.106 |
| Lens 4 | 0.319 | 0.545 | 0.136 |
| Lens 5 | 0.389 | 0.458 | 0.152 |
| Lens 6 | 0.341 | 0.526 | 0.133 |
| Lens 7 | 0.309 | 0.525 | 0.166 |
| Lens 8 | 0.315 | 0.558 | 0.127 |
| **TP Mean** | **0.342** | **0.525** | **0.132** |
| **FN Mean** | **0.332** | **0.564** | **0.104** |

The r band accounts for the largest share of attribution in both TP (~52%) and FN
(~56%) cases. The g band is second (~33%). The i band is lowest (~10–13%). Attribution
is consistent across the 8 individual TP lenses (r: 0.458–0.571).

> **Interpretation caveat:** Higher r-band attribution means the model's output changed
> more when r-band input was moved from the training mean to the actual value — not
> that r-band is inherently more informative for lens detection. This reflects
> EfficientNet-B2's sensitivity under this specific baseline and sample.

---

## 8. Failure Mode Analysis

Analysis at ensemble operating point τ* = 0.3143. **FN = 10, FP = 379.**

### 8.1 Per-model Scores on Ensemble False Negatives

| | EffNet-B2 | ResNet-34 | DenseNet | EResNet | EqDenseNet | Ensemble |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| FN 1 | 0.003 | 0.014 | 0.000 | 0.091 | 0.005 | 0.023 |
| FN 2 | 0.001 | 0.005 | 0.000 | 0.174 | 0.017 | 0.039 |
| FN 3 | 0.006 | 0.052 | 0.001 | 0.141 | 0.003 | 0.041 |
| FN 4 | 0.063 | 0.026 | 0.023 | 0.120 | 0.011 | 0.049 |
| FN 5 | 0.055 | 0.034 | 0.014 | 0.404 | 0.023 | 0.106 |
| FN 6 | 0.288 | 0.204 | 0.194 | 0.094 | 0.024 | 0.161 |
| FN 7 | 0.606 | 0.067 | 0.096 | 0.293 | 0.094 | 0.231 |
| FN 8 | 0.348 | 0.636 | 0.011 | 0.250 | 0.125 | 0.274 |
| FN 9 | 0.591 | 0.286 | 0.084 | 0.492 | 0.098 | 0.310 |
| FN 10 | 0.340 | 0.333 | 0.154 | 0.629 | 0.108 | 0.313 |

**FN 1–4** (ensemble scores 0.023–0.049): missed by all models with very low
confidence — the hardest cases in the test set.

**FN 9 and FN 10** (scores 0.310 and 0.313): borderline cases just below τ* = 0.3143.
FN 10 reaches 0.629 from E-ResNet D₄ and FN 9 reaches 0.591 from EfficientNet-B2,
but both are averaged down by the majority of members. This illustrates the primary
cost of soft averaging: confident minority votes are suppressed.

### 8.2 Cross-architecture Disagreement

| | Count | Fraction |
|:---|:---:|:---:|
| Equivariant models score higher than pretrained | **6** | 6/10 |
| Pretrained models score higher than equivariant | 4 | 4/10 |

Equivariant models assign higher maximum confidence on FN 1–5 and FN 10 — in all 6
cases driven by E-ResNet D₄. Pretrained models lead on FN 6–9. Whether this reflects
the rotational symmetry prior or different training dynamics cannot be determined from
a single training run.

### 8.3 Statistical Characterisation of Missed vs Caught Lenses

| Property | Missed (FN=10) | Caught (TP=185) |
|:---------|:--------------:|:---------------:|
| Mean max intensity | 1.0000 | 1.0000 |
| **Mean pixel intensity** | **0.1333** | **0.0885** |
| Mean i-band peak | 1.0000 | 1.0000 |
| Mean ensemble score | 0.155 | 0.837 |
| Max ensemble score among FN | 0.313 | — |

Peak intensity is uninformative — both groups saturate at 1.0 after normalisation
by construction. Mean pixel intensity is higher for missed lenses (0.133 vs 0.089),
consistent with more spatially distributed flux. FN 9–10 are potentially recoverable
at a lower threshold; the PR curve slope in the 0.31–0.32 score range should be
consulted to estimate the FP cost.

### 8.4 False Positive Characterisation

Top-18 false positives by ensemble confidence (p=0.898–0.964) include compact
isolated sources similar to lenses at this resolution, close galaxy pairs, sources
with faint companions, and sources with extended asymmetric emission. The highest-
confidence false positive (p=0.964) shows visible arc-like peripheral emission — a
genuine ambiguity at 64×64 that cannot be excluded from the image alone. All 18
would require higher-resolution imaging or spectroscopic follow-up to classify
definitively.

---

## 9. Discussion

### The Real-Data Challenge

Test V uses real HSC observational images rather than simulations, introducing genuine
sky backgrounds, PSF variation, and a 99.8:1 test imbalance. The discrimination task
is fundamentally statistical at 64×64. PSF convolution suppresses resolved arc geometry
in most cases; models classify on radial flux profiles and multi-band intensity
relationships rather than explicit arc detection.

### The Equivariance Hypothesis: Partial Confirmation

EqDenseNet-C8 (0.183M, from scratch) achieves AUC-ROC 0.9872 and AUC-PR 0.7728,
both exceeding DenseNet-121 (0.9844/0.7632, 6.95M, ImageNet) at 37.9× fewer
parameters. The parameter efficiency result is real for EqDenseNet-C8 against two
of three pretrained comparators on both metrics.

E-ResNet D₄ (0.513M) achieves AUC-ROC 0.9840 close to DenseNet-121 (0.9844), but
has the lowest AUC-PR of any model (0.6963). The equivariance advantage does not
generalise uniformly across architectures or metrics. Whether it reflects the symmetry
prior, architecture family, or training dynamics is not isolable from this experiment.

### Three Documented escnn Implementation Bugs

Three bugs breaking D₄ invariance were identified and corrected via the
`in_type.transform` verification protocol:

1. **Strided `R2Conv`** produces |Δlogit| ~ 3.4.
   Fix: `enn.PointwiseAvgPool + stride-1 conv` in main path and shortcut.

2. **`nn.Sequential` wrapping equivariant modules** does not handle `GeometricTensor`.
   Fix: Use `enn.SequentialModule` throughout.

3. **`nn.AdaptiveAvgPool2d` after `GroupPooling`** on non-1×1 maps breaks invariance.
   Fix: `enn.PointwiseAvgPool(kernel_size=H)` inside the equivariant chain.

These corrections apply to both Test V and the companion Test I notebook.

### Comparison to Prior Work

| Reference | Method | AUC-ROC |
|:----------|:-------|:-------:|
| Mriganka (2022) | CNN baseline on HSC | 0.816 |
| Sreehari (2024) | SSL ViT on HSC-like data | not directly comparable |
| **This work — best individual** | ResNet-34 (fine-tuned) | **0.9881** |
| **This work — ensemble** | Soft-Ensemble (5 models) | **0.9905** |

All five individual models exceed the 2022 baseline by >0.16 AUC points. A direct
comparison with Sreehari 2024 is not possible due to differing test splits and metrics.

---

## 10. Limitations & Future Work

**Image resolution:** 64×64 suppresses arc morphology. Higher-resolution inputs
would expose morphological information with the largest expected benefit for partial
arcs and low-flux lenses.

**Small positive count:** 1,557 training lenses constrains all from-scratch models.
Both equivariant models exhibit overfitting — train loss reaches near-zero before
early stopping. E-ResNet D₄ near-zero around epoch 8 (stopped epoch 17); EqDenseNet-C8
around epoch 20 (stopped epoch 41).

**Calibration:** All models are poorly calibrated on the 99.8:1 test distribution —
trained on ~50:50 batches, probability scales are anchored to the training distribution.
Post-hoc calibration (Platt scaling or isotonic regression) is required before using
raw scores for probabilistic survey triage.

**Test set size:** 195 test lenses yields AUC estimates with non-trivial variance.
The individual model AUC-ROC range is 0.009 (0.9790–0.9881). Rankings at this
scale should be interpreted with this uncertainty in mind.

**Single dataset:** All results are on HSC data. Generalisation to KiDS, DES, or
Euclid has not been evaluated. Domain shift from PSF differences and pipeline
processing can be substantial.

---

## 11. GSoC 2026 Research Directions

Three directions are prioritised based on where current models fail and where
physics suggests the most leverage.

---

**Direction 1 — Equivariant Lens Finding on Multi-Band Real Data**

Test V establishes the 3-channel `trivial_repr` adaptation and the `in_type.transform`
verification protocol as reusable infrastructure. EqDenseNet-C8 provides the strongest
from-scratch efficiency result (0.183M, AUC-ROC 0.9872, AUC-PR 0.7728). Whether
higher-order symmetry groups (C16, SO(2)-approximate) improve over C8 on this task
remains an open empirical question.

Proposed work:
- Systematic ablation: group order × parameter count × AUC-ROC / AUC-PR using
  `flipRot2dOnR2(N)` and `rot2dOnR2(N)` families on full-resolution HSC cutouts
- Test whether higher-order groups improve performance on partial arcs at higher
  resolution where arc morphology is directly accessible
- Establish `in_type.transform` verification as standard practice for all future
  equivariant DeepLense architectures

---

**Direction 2 — Self-Supervised Equivariant Pretraining for Lens Finding**

The gap between from-scratch equivariant models and ImageNet-pretrained CNNs is
quantified here: EqDenseNet-C8 (0.7728 AUC-PR) marginally exceeds DenseNet-121
(0.7632); E-ResNet D₄ (0.6963) does not. Whether SSL pretraining on unlabelled HSC
cutouts (DINO, iBOT, MAE) would provide a domain-appropriate initialisation that
closes the remaining gap is the central open question from this benchmark.

Proposed work:
- Equivariant CNN or ViT backbone SSL pretraining on unlabelled HSC cutouts
- Fine-tune on the labelled task with the training infrastructure developed here
- Compare SSL-equivariant vs SSL-standard vs supervised-equivariant on both
  AUC-ROC and AUC-PR — directly extending Sreehari 2024 with equivariance as prior

---

**Direction 3 — Uncertainty-Aware Candidate Ranking for Survey Deployment**

The ensemble produces poorly calibrated probabilities and no uncertainty
decomposition. FN 9 and FN 10 (scores 0.310 and 0.313, just below τ* = 0.3143)
are borderline cases that an uncertainty-aware ranking system could flag for
secondary screening rather than hard-rejecting.

Proposed work:
- MC-dropout and deep ensemble uncertainty decomposition (aleatoric vs epistemic)
  on the Test V model stack
- Risk-stratified ranking combining predicted probability and uncertainty for
  survey triage allocation
- Evaluate whether uncertainty-aware ranking recovers FN 9–10 while maintaining
  specificity on the Test V test set
- Connect to Rubin Observatory LSST DESC lens-finding requirements, where
  follow-up resource allocation is explicitly budget-constrained

---

**Research arc:**

```
Test V (this work)
    ↓
Direction 1: verify + extend equivariant architectures to real multi-band data
    ↓
Direction 2: close the pretraining gap via equivariant SSL on unlabelled HSC
    ↓
Direction 3: deploy with uncertainty quantification at LSST survey scale
```

---

## 12. Repository Structure

```
DeepLense-GSoC-2026/
│
└── Test_V_Lens_Finding/
    │
    ├── README.md                              ← This file
    ├── Test_V_Lens_Finding_Final.ipynb        ← Main notebook (all experiments)
    ├── requirements.txt
    │
    └── figures/                               ← All figures generated by the notebook
        │
        ├── ── Section 1: EDA ────────────────────────────────────────────
        ├── eda_sample_grid.png
        ├── eda_intensity_distributions.png
        ├── eda_spatial_structure.png
        │
        ├── ── Section 4–5: Training Curves & Architecture Evaluation ────
        ├── EfficientNet-B2_curves.png
        ├── EfficientNet-B2_results.png
        ├── ResNet-34_curves.png
        ├── ResNet-34_results.png
        ├── DenseNet-121_curves.png
        ├── DenseNet-121_results.png
        ├── EResNet-D4_curves.png
        ├── EResNet-D4_results.png
        ├── EqDenseNet-C8_curves.png
        ├── EqDenseNet-C8_results.png
        ├── Soft-Ensemble_results.png
        │
        ├── ── Section 6: Results Summary ────────────────────────────────
        ├── roc_comparison.png
        ├── pr_comparison.png
        ├── param_efficiency_auc_roc.png
        ├── param_efficiency_auc_pr.png
        │
        ├── ── Section 7: Correlation & Ensemble ─────────────────────────
        ├── ensemble_correlation_matrix.png
        │
        ├── ── Section 8: Interpretability ───────────────────────────────
        ├── gradcam_EfficientNet_B2_Co.png
        ├── gradcam_EfficientNet_B2_Mi.png
        ├── gradcam_EfficientNet_B2_Fa.png
        ├── gradcam_DenseNet_121_Co.png
        ├── gradcam_DenseNet_121_Mi.png
        ├── gradcam_DenseNet_121_Fa.png
        ├── ig_channel_attribution.png          ← Integrated Gradients per-channel
        │
        └── ── Section 9: Failure Mode Analysis ──────────────────────────
            ├── failure_false_negatives.png
            └── failure_false_positives_top18.png
```

### Model Weights

All weights saved as PyTorch state dicts.
Load with `model.load_state_dict(torch.load(path, map_location=device))`.

| Model | AUC-ROC | AUC-PR | Params | Weight File |
|:------|:-------:|:------:|:------:|:------------|
| ResNet-34 | 0.9881 | 0.7851 | 21.29M | `ResNet-34_testV_best.pth` |
| EqDenseNet-C8 | 0.9872 | 0.7728 | 0.183M | `EqDenseNet-C8_testV_best.pth` |
| DenseNet-121 | 0.9844 | 0.7632 | 6.95M | `DenseNet-121_testV_best.pth` |
| E-ResNet D₄ | 0.9840 | 0.6963 | 0.513M | `EResNet-D4_testV_best.pth` |
| EfficientNet-B2 | 0.9790 | 0.7087 | 7.70M | `EfficientNet-B2_testV_best.pth` |

---

## 13. Citation

```bibtex
@misc{deeplense_gsoc2026_testv,
  author       = {Rafiqus Salehin},
  title        = {ML4SCI DeepLense GSoC 2026 — Common Test V:
                  Gravitational Lens Finding on Real HSC Observational Data},
  year         = {2026},
  url          = {https://github.com/rsalehin/DeepLense-GSoC-2026}
}
```

**Related work:**

- Mriganka (2022) — CNN baseline for gravitational lens finding on HSC
- Sreehari (2024) — Self-supervised ViT for lens finding on HSC-like data
- Weiler & Cesa (2019) — General E(2)-Equivariant Steerable CNNs (`escnn`)
- Lin et al. (2017) — Focal Loss for Dense Object Detection
- He et al. (2016) — Deep Residual Learning for Image Recognition
- Huang et al. (2017) — Densely Connected Convolutional Networks
- Tan & Le (2019) — EfficientNet: Rethinking Model Scaling

---

<div align="center">

**ML4SCI DeepLense — GSoC 2026**
*Finding gravitational lenses in real survey data, one Einstein ring at a time.*

</div>
