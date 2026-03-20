# Test V — Gravitational Lens Finding on Real HSC Data

<!-- BADGES -->
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![escnn](https://img.shields.io/badge/escnn-0.1.9-purple)
![GPU](https://img.shields.io/badge/GPU-T4%20%2F%20A100-green?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

> **Binary classification of strong gravitational lenses on real Hyper Suprime-Cam (HSC) survey images using pretrained CNNs, equivariant networks, and a soft-vote ensemble — with explicit handling of class imbalance via Focal Loss and stratified sampling.**

---

## Summary

| | |
|---|---|
| **Task** | Binary lens finding (lens / non-lens) |
| **Data** | Real HSC observational images, 3-channel (g, r, i), 64×64 px |
| **Imbalance ratio** | `[PLACEHOLDER: ~N:1 non-lens:lens]` |
| **Best AUC** | `[PLACEHOLDER: 0.XXX — ModelName]` |
| **Best PR-AUC** | `[PLACEHOLDER: 0.XXX — ModelName]` |
| **Best efficiency** | `[PLACEHOLDER: 0.XXX AUC / XM params — ModelName]` |
| **Ensemble AUC** | `[PLACEHOLDER: 0.XXX]` |

---

## Table of Contents

1. [Scientific Background](#1-scientific-background)
2. [Dataset](#2-dataset)
3. [Environment Setup](#3-environment-setup)
4. [Training & Evaluation Framework](#4-training--evaluation-framework)
5. [Architecture Evaluations](#5-architecture-evaluations)
6. [Comprehensive Results Summary](#6-comprehensive-results-summary)
7. [Threshold Calibration](#7-threshold-calibration)
8. [Interpretability Analysis](#8-interpretability-analysis)
9. [Failure Mode Analysis](#9-failure-mode-analysis)
10. [Discussion](#10-discussion)
11. [Limitations & Future Work](#11-limitations--future-work)
12. [GSoC Research Directions](#12-gsoc-research-directions)
13. [Repository Structure](#13-repository-structure)
14. [Citation & References](#14-citation--references)

---

## 1. Scientific Background

### 1.1 Gravitational lens finding at survey scale

Strong gravitational lenses are rare alignments in which a massive foreground galaxy deflects and distorts light from a background source into characteristic arcs or Einstein rings. They are powerful cosmological probes — enabling independent H₀ measurements, dark matter substructure mapping, and tests of general relativity at galactic scales.

The Rubin Observatory Legacy Survey of Space and Time (LSST) is projected to image ~10⁹ galaxies, from which O(10⁵) strong lenses are expected. Manual inspection is infeasible at this scale, making automated lens-finding a first-order scientific problem. The Hyper Suprime-Cam (HSC) survey provides an ideal precursor testbed: real multi-band imaging with photometric depth comparable to early LSST observations.

### 1.2 Why real data is harder than simulation

Simulated lensing datasets (as in Test I) are generated under controlled assumptions: clean PSFs, idealised source morphologies, and uniform noise. Real HSC data introduces:

- **Diverse contaminants**: ring galaxies, mergers, and spiral arms mimic lensing arcs.
- **Spatially variable PSF**: HSC PSF varies across the focal plane and between observing epochs.
- **Survey depth heterogeneity**: sky background and noise vary by pointing.
- **Selection bias in training labels**: human-labelled positives systematically bias toward high signal-to-noise lenses (compact Einstein rings); partial arcs and low-flux lenses are underrepresented.

Models pretrained on simulated data exhibit domain shift when applied to HSC, motivating training directly on real labelled data.

### 1.3 The class imbalance problem in lens surveys

In realistic imaging surveys, strong lenses constitute `[PLACEHOLDER: ~0.X%]` of all galaxy-scale objects. The HSC training set provided for this task reflects this imbalance:

- **Training positives (lenses)**: `[PLACEHOLDER: N_lenses_train]`
- **Training negatives (non-lenses)**: `[PLACEHOLDER: N_nonlenses_train]`
- **Imbalance ratio**: `[PLACEHOLDER: ~N:1]`

Naive cross-entropy optimisation on such data collapses to near-trivial solutions that predict the majority class. This motivates the use of Focal Loss and stratified sampling as primary imbalance mitigation strategies (see §4.2–4.3).

---

## 2. Dataset

### 2.1 Data source and structure

The dataset consists of real HSC survey cutouts, provided as `.npy` arrays of shape `(3, 64, 64)` representing three photometric bands: **g**, **r**, and **i**.

```
data/
├── train/
│   ├── lenses/         # [PLACEHOLDER: N] .npy files
│   └── nonlenses/      # [PLACEHOLDER: N] .npy files
└── test/
    ├── lenses/         # [PLACEHOLDER: N] .npy files
    └── nonlenses/      # [PLACEHOLDER: N] .npy files
```

### 2.2 Class statistics and imbalance ratio

| Split | Lenses | Non-lenses | Total | Ratio (neg:pos) |
|-------|--------|------------|-------|-----------------|
| Train | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Test  | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

After 90:10 stratified train-val split from the training set:

| Split | Lenses | Non-lenses |
|-------|--------|------------|
| Train (90%) | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Val (10%)   | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

### 2.3 Per-channel (g/r/i) properties

| Channel | Mean (lenses) | Std (lenses) | Mean (non-lenses) | Std (non-lenses) |
|---------|--------------|--------------|-------------------|------------------|
| g       | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| r       | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| i       | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

### 2.4 Key observational properties

`[PLACEHOLDER — EDA findings: describe distinguishing pixel-level features observed in §1.5 of the notebook: flux concentration, arc morphology, central brightness excess, per-channel intensity offsets, etc.]`

---

## 3. Environment Setup

### Requirements

```bash
pip install torch torchvision
pip install escnn
pip install numpy matplotlib scikit-learn tqdm
pip install grad-cam          # pytorch-grad-cam
```

### Verified version matrix

| Package | Version |
|---------|---------|
| Python  | 3.10    |
| PyTorch | 2.x     |
| escnn   | 0.1.9   |
| torchvision | `[PLACEHOLDER]` |
| scikit-learn | `[PLACEHOLDER]` |
| numpy   | `[PLACEHOLDER]` |
| pytorch-grad-cam | `[PLACEHOLDER]` |

> All experiments run on `[PLACEHOLDER: GPU model, e.g. NVIDIA A100 40GB / T4]`. Expected wall-clock: `[PLACEHOLDER: ~Xh total for all 5 models]`.

---

## 4. Training & Evaluation Framework

### 4.1 Data pipeline

**Loading**: Each `.npy` file is loaded as a float32 tensor of shape `(3, 64, 64)`.

**Normalisation**: Per-sample, per-channel min-max normalisation:
```
x_norm[c] = (x[c] − min(x[c])) / (max(x[c]) − min(x[c]) + ε)
```
This choice preserves inter-channel flux ratios (physically informative for colour-based lens discrimination) while preventing inter-sample scale variance from dominating training gradients.

**Train augmentation** (D₄-symmetric):
- Random horizontal flip
- Random vertical flip
- Random 90° rotation (k ∈ {0, 1, 2, 3})

Physical motivation: gravitational lenses have no preferred orientation on the sky. D₄ augmentation enforces this symmetry at the data level for pretrained models; for equivariant architectures, it further expands effective coverage within the symmetry group.

**Validation augmentation**: None. Deterministic evaluation.

### 4.2 Imbalance handling strategy

Two complementary mechanisms are applied:

1. **`WeightedRandomSampler`** (training DataLoader): each sample is weighted inversely to its class frequency, producing approximately balanced batches at the sampler level. Expected batch composition: `[PLACEHOLDER: ~50:50 lens:non-lens per batch]`.

2. **Focal Loss `α` weighting**: `α` is set proportional to the inverse class frequency, providing an additional gradient-level correction for the minority class.

These two mechanisms are applied jointly and are not redundant: the sampler controls *which* samples enter each batch; the loss function controls *how much* each sample contributes to the gradient given the model's current confidence.

### 4.3 Focal Loss rationale

```
FL(p_t) = −α_t (1 − p_t)^γ log(p_t)
```

| Hyperparameter | Value | Rationale |
|---|---|---|
| γ (focusing) | 2 | Standard; down-weights easy negatives quadratically |
| α (class weight) | `[PLACEHOLDER: 0.XX]` | Inverse frequency of lens class |

**Why Focal Loss over weighted BCE?** Weighted BCE scales gradient magnitude uniformly by class weight — all non-lens examples contribute identically regardless of model confidence. Focal Loss additionally suppresses the gradient from *confidently correct* non-lens predictions (p → 0), concentrating learning capacity on the hard examples near the decision boundary. On a severely imbalanced dataset with many easy-to-classify non-lenses, this matters.

### 4.4 Evaluation protocol

All models are evaluated on the held-out **test set** using:

| Metric | Rationale |
|--------|-----------|
| **AUC-ROC** | Primary metric; threshold-free ranking quality |
| **AUC-PR** | More sensitive than ROC under heavy class imbalance |
| Confusion matrix at τ = 0.5 | Absolute TP/FP/TN/FN counts |
| Youden's J optimal threshold | τ* = argmax(TPR − FPR); operationally motivated |
| Sensitivity / Specificity at τ* | Survey-relevant operating point |
| Calibration curve | Reliability of probability outputs |

---

## 5. Architecture Evaluations

All models share:
- Input: 3-channel, 64×64 images
- Output: scalar sigmoid probability (lens score ∈ [0, 1])
- Training: `[PLACEHOLDER: N]` max epochs with EarlyStopping (patience = `[PLACEHOLDER]`, monitor: val AUC)
- Optimiser: `[PLACEHOLDER: Adam / AdamW]`, lr = `[PLACEHOLDER]`
- Scheduler: `[PLACEHOLDER: CosineAnnealingLR / ReduceLROnPlateau]`

---

### 5.1 EfficientNet-B2

**Architecture overview**: EfficientNet-B2 scales width, depth, and resolution via compound scaling from a MobileNet-style MBConv backbone. The final classifier head is replaced:

```
AdaptiveAvgPool2d → Dropout(p=0.3) → Linear(in_features, 1)
```

**Weight initialisation**: ImageNet pretrained (torchvision).

**Training results**:

| Metric | Val | Test |
|--------|-----|------|
| AUC-ROC | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| AUC-PR  | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Params  | `[PLACEHOLDER]` | — |

`[PLACEHOLDER — embed 4-panel figure: ROC + PR + CM + Calibration]`

---

### 5.2 ResNet-34

**Architecture overview**: 34-layer residual network with basic (non-bottleneck) blocks. Head replacement identical to §5.1.

**Weight initialisation**: ImageNet pretrained (torchvision).

**Training results**:

| Metric | Val | Test |
|--------|-----|------|
| AUC-ROC | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| AUC-PR  | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Params  | `[PLACEHOLDER]` | — |

`[PLACEHOLDER — embed 4-panel figure]`

---

### 5.3 DenseNet-121

**Architecture overview**: 121-layer densely connected network (growth rate = 32, 4 dense blocks). Each layer receives feature maps from all preceding layers. Head replacement: final `Linear` replaced with `Linear(in_features, 1)`.

**Weight initialisation**: ImageNet pretrained (torchvision).

**Training results**:

| Metric | Val | Test |
|--------|-----|------|
| AUC-ROC | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| AUC-PR  | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Params  | `[PLACEHOLDER]` | — |

`[PLACEHOLDER — embed 4-panel figure]`

---

### 5.4 E-ResNet D₄ (3-channel adaptation)

**Architecture overview**: ResNet-style network with layers equivariant to the D₄ symmetry group (4 rotations + 4 reflections) using `escnn`. Adapted from Test I for 3-channel (multi-band) input.

**Key adaptation from Test I**: The input representation is changed from `1 × trivial_repr` (single-channel simulated) to `3 × trivial_repr` (three scalar fields — one per photometric band). Each band transforms trivially under D₄ (scalar fields are rotation- and reflection-invariant in physical units), so the stem correctly initialises.

**Classifier head**: `GroupPool → AdaptiveAvgPool2d → Linear(feat_dim, 1) → Sigmoid`

**Weight initialisation**: Random (no pretrained weights available for equivariant architectures).

**Architecture summary**:

| Property | Value |
|---|---|
| Symmetry group | D₄ (order 8) |
| Parameters | `[PLACEHOLDER]` |
| Spatial dims (64×64 input) | `[PLACEHOLDER: 64→32→16→8→...]` |
| Feature field type | `[PLACEHOLDER: regular repr of D₄]` |

**Training results**:

| Metric | Val | Test |
|--------|-----|------|
| AUC-ROC | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| AUC-PR  | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Params  | `[PLACEHOLDER]` | — |

`[PLACEHOLDER — embed 4-panel figure]`

---

### 5.5 EqDenseNet-C8 (3-channel adaptation)

**Architecture overview**: DenseNet-style network with layers equivariant to C₈ (cyclic group of 8 rotations) using `escnn`. Adapted from Test I for 3-channel input via the same stem adaptation as §5.4.

**Spatial dimension trace** (verify no spatial collapse):

```
Input: 64×64
After stem: [PLACEHOLDER]
After dense block 1: [PLACEHOLDER]
After transition 1: [PLACEHOLDER]
After dense block 2: [PLACEHOLDER]
After transition 2: [PLACEHOLDER]
After dense block 3: [PLACEHOLDER]
GroupPool output: [PLACEHOLDER]
```

**Architecture summary**:

| Property | Value |
|---|---|
| Symmetry group | C₈ (order 8) |
| Growth rate | `[PLACEHOLDER]` |
| Dense block config | `[PLACEHOLDER: e.g. (6, 12, 24)]` |
| Parameters | `[PLACEHOLDER]` |

**Training results**:

| Metric | Val | Test |
|--------|-----|------|
| AUC-ROC | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| AUC-PR  | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Params  | `[PLACEHOLDER]` | — |

`[PLACEHOLDER — embed 4-panel figure]`

**Equivariance verification**:

| Model | L₂ prob. divergence (untrained) | L₂ prob. divergence (trained) |
|---|---|---|
| E-ResNet D₄ | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| EqDenseNet-C8 | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| EfficientNet-B2 (baseline) | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

---

### 5.6 Soft Ensemble

**Members**: Top-`[PLACEHOLDER: N]` models by validation AUC: `[PLACEHOLDER: e.g. EfficientNet-B2, DenseNet-121, EqDenseNet-C8]`.

**Method**: Mean of per-model sigmoid probability vectors:
```
p_ensemble(x) = (1/N) Σ_i σ(f_i(x))
```

**Test results**:

| Metric | Value |
|--------|-------|
| AUC-ROC | `[PLACEHOLDER]` |
| AUC-PR  | `[PLACEHOLDER]` |

`[PLACEHOLDER — embed 4-panel figure]`

`[PLACEHOLDER — 1-paragraph discussion: does ensemble improve over best individual model? variance reduction vs. bias amplification on real imbalanced data]`

---

## 6. Comprehensive Results Summary

### 6.1 Full benchmark table

| Model | AUC-ROC ↑ | AUC-PR ↑ | Params | Pretrained | Lens Recall @ τ* |
|-------|-----------|----------|--------|------------|------------------|
| EfficientNet-B2 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | ✓ | `[PLACEHOLDER]` |
| ResNet-34 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | ✓ | `[PLACEHOLDER]` |
| DenseNet-121 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | ✓ | `[PLACEHOLDER]` |
| E-ResNet D₄ | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | ✗ | `[PLACEHOLDER]` |
| EqDenseNet-C8 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | ✗ | `[PLACEHOLDER]` |
| **Soft Ensemble** | `[PLACEHOLDER]` | `[PLACEHOLDER]` | — | — | `[PLACEHOLDER]` |

### 6.2 Performance tiers

`[PLACEHOLDER — 2–3 sentences grouping models: e.g. pretrained tier (AUC ~0.9X) vs equivariant tier (AUC ~0.8X), or any reversal]`

### 6.3 Parameter efficiency

`[PLACEHOLDER — AUC vs params scatter plot: embed figure. Discuss which model achieves the best AUC per parameter — likely an equivariant model if pretrained models overfit or under-adapt]`

### 6.4 Key takeaways

`[PLACEHOLDER — 3–5 bullet points once results are filled in. Example skeleton:]`

- **Pretrained models**: `[PLACEHOLDER: e.g. DenseNet-121 achieves the highest AUC among individual models at 0.XXX, benefiting from ImageNet feature reuse despite domain shift]`
- **Equivariant models**: `[PLACEHOLDER: e.g. EqDenseNet-C8 reaches 0.XXX AUC with Xk fewer parameters, demonstrating competitive efficiency under rotational symmetry]`
- **Ensemble**: `[PLACEHOLDER: e.g. soft ensemble improves AUC by +0.0XX over best individual, consistent with variance reduction from model diversity]`
- **Focal Loss effectiveness**: `[PLACEHOLDER: evidence from PR-AUC vs AUC gap]`

---

## 7. Threshold Calibration

### 7.1 Youden's J optimal threshold

τ* = argmax(TPR − FPR) selects the threshold that maximises the geometric separation between sensitivity and specificity. This is the operationally preferred threshold for survey deployment when false negatives and false positives carry asymmetric costs.

| Model | Youden τ* | TPR at τ* | FPR at τ* |
|-------|-----------|-----------|-----------|
| EfficientNet-B2 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| ResNet-34 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| DenseNet-121 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| E-ResNet D₄ | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| EqDenseNet-C8 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Soft Ensemble | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

### 7.2 Sensitivity / specificity at operating point

| Model | Sensitivity | Specificity | F1 @ τ* |
|-------|-------------|-------------|---------|
| EfficientNet-B2 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| ResNet-34 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| DenseNet-121 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| E-ResNet D₄ | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| EqDenseNet-C8 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Soft Ensemble | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

### 7.3 Implications for survey deployment

In LSST-scale lens-finding pipelines, the cost asymmetry is typically:

- **False negative (missed lens)**: High cost — lost science target, no recovery path.
- **False positive (flagged non-lens)**: Moderate cost — expert follow-up required, but manageable at ~few×10⁴ scale.

This motivates operating at higher sensitivity (lower τ*), accepting increased FPR, and triaging flagged candidates with a lightweight second-stage classifier. The Youden threshold provides a principled starting point; τ* = `[PLACEHOLDER: model-specific]` achieves sensitivity `[PLACEHOLDER]` at specificity `[PLACEHOLDER]` for the best-performing model.

---

## 8. Interpretability Analysis

### 8.1 Grad-CAM: EfficientNet-B2 vs EqDenseNet-C8

Grad-CAM heatmaps are computed on the final convolutional feature map for EfficientNet-B2, and on the last equivariant layer's feature maps (projected to spatial domain via group-pooling) for EqDenseNet-C8.

**Correct lens predictions — where does each model attend?**

`[PLACEHOLDER — embed Grad-CAM figure: 2×N grid (EfficientNet-B2 row, EqDenseNet-C8 row) for N=6 correctly classified lenses. Caption: does EqDenseNet-C8 concentrate more symmetrically around the Einstein ring?]`

**False negatives — missed lenses:**

`[PLACEHOLDER — embed figure: Grad-CAM on 6 missed lenses. Caption: does attention degrade spatially for hard/faint lenses?]`

**False positives — flagged non-lenses:**

`[PLACEHOLDER — embed figure: Grad-CAM on 6 false positives. Caption: what arc-like or ring-like structure confused the model?]`

### 8.2 Per-channel saliency analysis

`[PLACEHOLDER — embed per-channel saliency figure: for each of the 3 bands (g, r, i), which drives the binary decision? Expected: r/i bands more discriminative due to better S/N for red lens galaxies; g band may capture blue arc emission from background source]`

### 8.3 Ring concentration: equivariant vs pretrained

`[PLACEHOLDER — quantitative or qualitative analysis: does EqDenseNet-C8 show tighter, more rotationally symmetric attention around the arc/ring compared to EfficientNet-B2? Embed comparison figure]`

---

## 9. Failure Mode Analysis

### 9.1 False negatives — missed lenses

`[PLACEHOLDER — embed gallery: 12 lenses with lowest model confidence (sorted ascending by p_ensemble). Below each image: true label, ensemble score, per-model score]`

**Statistical characterisation of hard lenses**:

`[PLACEHOLDER — fill in after EDA: e.g. mean flux, arc extent, half-light radius, S/N. Expected: missed lenses are fainter, have smaller Einstein radii or partial arcs]`

### 9.2 False positives — flagged non-lenses

`[PLACEHOLDER — embed gallery: 12 non-lenses with highest model confidence. Below each image: true label, ensemble score]`

**Characterisation**: `[PLACEHOLDER: e.g. ring galaxies, edge-on spirals with dust lanes, galaxy mergers with tidal streams]`

### 9.3 Cross-architecture disagreement

**Cases where equivariant models catch what pretrained models miss**:

`[PLACEHOLDER — embed gallery: lenses where E-ResNet D₄ or EqDenseNet-C8 score > 0.5 but all pretrained models score < 0.5. How many? What morphological feature do they share?]`

**Cases where pretrained models catch what equivariant models miss**:

`[PLACEHOLDER — embed gallery: opposite disagreement direction]`

### 9.4 Physical interpretation

`[PLACEHOLDER — 3–5 paragraph discussion. Seed topics:]`

- **Low-flux lenses**: `[PLACEHOLDER: faint Einstein rings near sky noise floor — missed by all models]`
- **Partial arcs**: `[PLACEHOLDER: incomplete ring geometry — may confuse equivariant models that implicitly encode full-ring priors]`
- **Non-standard morphologies**: `[PLACEHOLDER: double-source-plane lenses, point-source lensing — different from typical ring-arc training examples]`
- **PSF artefacts**: `[PLACEHOLDER: HSC PSF diffraction spikes can mimic arc structure]`

---

## 10. Discussion

### 10.1 Strategy justification

**Focal Loss over weighted BCE**:

`[PLACEHOLDER — summarise §4.3 rationale with empirical evidence: e.g. "Replacing weighted BCE with Focal Loss (γ=2) improved val AUC by +0.0XX on EfficientNet-B2 in ablation, consistent with the expected suppression of easy-negative gradients on a ~N:1 imbalanced dataset."]`

**Stratified sampling over random sampling**:

`[PLACEHOLDER — expected: stratified split ensures val set contains sufficient positive examples (~10% of N_lenses) to make AUC estimates reliable; random split risks val sets with 0 lenses on very small datasets]`

**D₄ augmentation as physical prior**:

`[PLACEHOLDER — gravitational lensing geometry is rotationally symmetric on the sky. D₄ augmentation enforces this prior at the data level, improving generalisation across sky orientations. Note: for equivariant models, this is reinforcing a property already encoded in the architecture; for pretrained CNNs, it is the primary invariance mechanism]`

### 10.2 Equivariant architecture generalisation from Test I

Test I established that D₄-CNN and E-ResNet improve on standard CNNs for 3-class lensing substructure classification on **simulated** single-channel images. Test V extends this to:

- **Binary classification** (vs multi-class)
- **Real HSC data** (vs simulation)
- **3-channel input** (vs 1-channel)

The key architectural modification is the stem input representation: `1 × trivial_repr` → `3 × trivial_repr`. This is the minimal change that correctly handles multi-band input while preserving the group-equivariance constraints throughout the network.

`[PLACEHOLDER — compare AUC gap between equivariant and pretrained models in Test I vs Test V. Does domain shift (real data) disproportionately hurt equivariant models (trained from scratch) vs pretrained models (ImageNet features)?]`

### 10.3 Comparison to prior DeepLense lens-finding work

| Reference | Method | Dataset | Reported AUC |
|-----------|--------|---------|--------------|
| Mriganka (2022) | `[PLACEHOLDER: CNN baseline]` | HSC | 0.816 |
| Saranga (2023) | `[PLACEHOLDER: architecture]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Sreehari (2024) | Self-supervised (SSL) | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| **This work** | Equivariant + pretrained ensemble | HSC | **`[PLACEHOLDER]`** |

`[PLACEHOLDER — 2–3 sentence contextualisation: does this work exceed the Mriganka 2022 baseline? How does it compare to SSL approaches?]`

---

## 11. Limitations & Future Work

**Current limitations**:

- **Image resolution**: 64×64 px images discard sub-arc-second morphological detail available in full HSC cutouts. Ring radius estimation and partial arc detection are limited at this resolution.
- **Domain gap**: Models are trained on labelled HSC data; generalisation to other surveys (KiDS, DES, Euclid) is not evaluated.
- **No uncertainty quantification**: Models output point-estimate probabilities; predictive uncertainty (aleatoric + epistemic) is not characterised. Bayesian or Monte Carlo dropout extensions would improve survey deployment reliability.
- **Real galaxy overlap**: The non-lens class contains varied galaxy morphologies but may not fully represent the complete contaminant population in a live survey stream.
- **Label quality**: Human-labelled positives are biased toward high-confidence, visually obvious lenses; the model may inherit this selection bias.

**Future directions**:

- **Self-supervised pretraining on unlabelled HSC data** (following Sreehari 2024): reduces dependence on scarce labelled lenses by learning galaxy morphology representations from the large unlabelled pool.
- **Higher-resolution inputs** (128×128, 256×256): expected to improve arc detection for partial and low-flux lenses.
- **C₁₆ or SO(2) equivariant models**: stronger rotational equivariance than D₄/C₈ for point-symmetric Einstein rings.
- **Cross-survey generalisation**: train on HSC, evaluate on KiDS/DES — relevant for LSST readiness.
- **Two-stage pipeline**: this model as stage-1 ranker; lightweight stage-2 classifier on top-K candidates to suppress false positives.

---

## 12. GSoC Research Directions

Test V directly informs three proposed GSoC 2026 directions within ML4SCI DeepLense:

1. **Equivariant lens finding on multi-band real data**: This test establishes the 3-channel stem adaptation as the critical interface between single-channel simulated benchmarks (Test I) and operational multi-band survey data. GSoC extension: systematic evaluation of higher-order symmetry groups (C₁₆, SO(2)-approximate) on full-resolution HSC cutouts.

2. **Domain-adaptive pretraining for lens finding**: The performance gap between pretrained CNNs (ImageNet initialisation) and equivariant networks (random initialisation) is attributable in part to the absence of equivariant pretraining corpora. GSoC extension: self-supervised equivariant pretraining on unlabelled HSC, enabling equivariant networks to close the parameter-efficiency gap.

3. **Uncertainty-aware lens candidates for follow-up prioritisation**: Current models output calibrated probabilities but not decomposed uncertainties. GSoC extension: deep ensemble or MC-dropout uncertainty quantification to produce risk-stratified candidate lists for telescope follow-up allocation.

`[PLACEHOLDER — 1 paragraph connecting specific Test V results (e.g. equivariant efficiency, ensemble gain, failure mode characterisation) to the concrete GSoC proposal]`

---

## 13. Repository Structure

```
test_v/
├── README.md                    # This file
├── test_v_deeplense.ipynb       # Main notebook (all sections)
├── requirements.txt
├── models/
│   ├── efficientnet_b2.pt       # Best val checkpoint
│   ├── resnet34.pt
│   ├── densenet121.pt
│   ├── eresnet_d4.pt
│   └── eqdensenet_c8.pt
├── figures/
│   ├── eda/
│   │   ├── class_samples.png
│   │   ├── pixel_distributions.png
│   │   ├── mean_std_images.png
│   │   └── difference_map.png
│   ├── training/
│   │   ├── efficientnet_b2_4panel.png
│   │   ├── resnet34_4panel.png
│   │   ├── densenet121_4panel.png
│   │   ├── eresnet_d4_4panel.png
│   │   ├── eqdensenet_c8_4panel.png
│   │   └── ensemble_4panel.png
│   ├── results/
│   │   ├── roc_overlay.png
│   │   ├── pr_overlay.png
│   │   └── auc_vs_params.png
│   ├── interpretability/
│   │   ├── gradcam_correct.png
│   │   ├── gradcam_fn.png
│   │   ├── gradcam_fp.png
│   │   └── channel_saliency.png
│   └── failure_modes/
│       ├── false_negatives.png
│       ├── false_positives.png
│       └── disagreement.png
└── utils/
    ├── dataset.py               # HSCDataset, WeightedSampler setup
    ├── focal_loss.py            # FocalLoss implementation
    ├── train.py                 # train_one_epoch, evaluate, EarlyStopping
    ├── equivariant_models.py    # E-ResNet D₄, EqDenseNet-C8 (3-channel)
    └── gradcam.py               # Grad-CAM wrapper
```

---

## 14. Citation & References

If you use this work, please cite:

```bibtex
@misc{deeplense_testv_2026,
  author       = {[PLACEHOLDER: Your Name]},
  title        = {Test V: Gravitational Lens Finding on Real HSC Data — ML4SCI DeepLense GSoC 2026},
  year         = {2026},
  howpublished = {\url{[PLACEHOLDER: GitHub repo URL]}},
}
```

**References**:

- Focal Loss: Lin et al. (2017). *Focal Loss for Dense Object Detection*. arXiv:1708.02002
- escnn: Cesa et al. (2022). *A Program to Build E(N)-Equivariant Steerable CNNs*. ICLR 2022
- EfficientNet: Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML 2019
- DenseNet: Huang et al. (2017). *Densely Connected Convolutional Networks*. CVPR 2017
- ResNet: He et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016
- DeepLense Mriganka (2022): `[PLACEHOLDER: full citation]`
- DeepLense Saranga (2023): `[PLACEHOLDER: full citation]`
- DeepLense Sreehari (2024): `[PLACEHOLDER: full citation]`
- HSC Survey: Aihara et al. (2018). *The Hyper Suprime-Cam SSP Survey: Overview and survey design*. PASJ 70, S4
- Grad-CAM: Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV 2017
- Youden (1950). *Index for rating diagnostic tests*. Cancer 3, 32–35

---

*ML4SCI DeepLense — GSoC 2026 Evaluation Test V*
*Generated from `test_v_deeplense.ipynb` — placeholders updated inline as each notebook section is completed.*
