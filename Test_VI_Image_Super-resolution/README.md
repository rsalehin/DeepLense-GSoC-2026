# Test VI.A — Deep Learning-Based Super-Resolution of Simulated Strong Lensing Images

**GSoC 2026 | ML4SCI DeepLense Evaluation Test**  
**Subtask:** Task VI.A — Supervised Super-Resolution on Simulated HR/LR Pairs  
**Companion:** Task VI.B — Domain-Adapted SR on Real HSC/HST Pairs (see `README_TestVIB.md`)

---

## Table of Contents

1. [Scientific Motivation](#1-scientific-motivation)
2. [Task Definition and Dataset](#2-task-definition-and-dataset)
3. [Architecture Selection and Justification](#3-architecture-selection-and-justification)
4. [Loss Function Strategy](#4-loss-function-strategy)
5. [Training Setup and Augmentation](#5-training-setup-and-augmentation)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Results](#7-results)
8. [Ablation Study: Loss Function](#8-ablation-study-loss-function)
9. [Qualitative Analysis](#9-qualitative-analysis)
10. [Discussion](#10-discussion)
11. [Limitations and Future Work](#11-limitations-and-future-work)
12. [Dependencies and Reproducibility](#12-dependencies-and-reproducibility)
13. [References](#13-references)

---

## 1. Scientific Motivation

Strong gravitational lensing occurs when a massive foreground object (galaxy or cluster) bends the light of a background source, producing distorted arcs and Einstein rings. The angular radius of the Einstein ring, θ_E, encodes the projected mass of the lens along the line of sight. At sub-galactic scales, perturbations to the ring morphology — small breaks, asymmetries, flux anomalies — are direct signatures of dark matter substructure: CDM subhalos, axion vortex structures, or other beyond-ΛCDM candidates.

The observational challenge is resolution. Ground-based surveys such as HSC-SSP and the forthcoming Rubin/LSST will yield on the order of 10⁴–10⁵ lens candidates (Oguri & Marshall 2010), but at image quality that is fundamentally limited by atmospheric seeing and pixel scale. Space-based follow-up (HST, JWST) is expensive and cannot be applied at survey scale. Super-resolution — recovering high-frequency spatial detail from low-resolution inputs using learned priors — offers a computationally efficient path to improving image quality for the full survey population prior to downstream classification or regression tasks.

This test establishes a **systematic multi-architecture benchmark** for deep learning-based SR on simulated strong lensing images. The central scientific question is not merely which model achieves highest PSNR, but which model best preserves the physically critical structure: the Einstein ring morphology and arc-level substructure perturbations. These are the features that downstream dark matter analyses depend on.

This work connects directly to and extends several prior SR projects within DeepLense:
- **Pranath Reddy (GSoC 2023):** Residual SR + Conditional Diffusion on Model I — the first SR work in DeepLense.
- **Atal Gupta (GSoC 2024):** DDPM/SR3/SRDiff/ResShift on real galaxy data (Model IV) — diffusion-only focus.
- **Anirudh Shankar (GSoC 2024):** Physics-informed unsupervised SR (LensSR) — no HR ground truth required.

The present work fills the gap: a **rigorous supervised discriminative SR benchmark** with domain-justified architecture selection, loss function ablation, and lensing-specific qualitative analysis. No prior DeepLense SR project has compared SRCNN, EDSR, RCAN, and SwinIR under a unified experimental protocol.

---

## 2. Task Definition and Dataset

### 2.1 Task

Given a low-resolution (LR) strong lensing image `x ∈ R^(H/s × W/s × C)`, learn a mapping `f_θ: x → ŷ` such that `ŷ` approximates the high-resolution (HR) ground truth `y ∈ R^(H × W × C)` under the metrics MSE, SSIM, and PSNR.

Scale factor `s` is determined empirically from dataset inspection (see notebook Section 1; expected: **s = 4**).

### 2.2 Dataset

| Property | Value |
|---|---|
| Source | DeepLense simulated dataset (no substructure class) |
| Download | [Google Drive](https://drive.google.com/file/d/1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw/view?usp=sharing) |
| Lens model | Singular Isothermal Ellipsoid (SIE) |
| Light profile | Sérsic |
| Noise model | Gaussian + Poissonian; SNR ~ 20 |
| PSF | Gaussian |
| Channels | 1 (grayscale) |
| HR resolution | [TBD — confirmed in EDA] × [TBD] |
| LR resolution | [TBD — confirmed in EDA] × [TBD] |
| Upscale factor | [TBD — confirmed in EDA; expected 4×] |
| Train / Val / Test split | 80 / 10 / 10 % |

> **Note on class selection:** The dataset contains only the "no substructure" class. This is appropriate for SR pretraining: the task is learning the lensing image manifold, not substructure-specific features. The SR model trained here is intended as a universal preprocessing step applicable to all three classes.

### 2.3 Normalization Strategy

All images are normalized to `[0, 1]` per-image using min-max scaling:

```
x_norm = (x - x.min()) / (x.max() - x.min() + ε)
```

with `ε = 1e-8` for numerical stability. Global dataset-wide normalization is **not used** because the dynamic range of lensing images varies significantly with Einstein ring brightness. Per-image normalization preserves relative ring-to-background contrast, which is the physically critical quantity.

### 2.4 Patch-Based Training

Full-image training is memory-prohibitive for high-resolution SR and produces artificially few training samples. Following standard SR practice (Dong et al. 2014; Lim et al. 2017), we train on randomly sampled **HR patches of size 64×64** (LR patches: 16×16 for 4× SR, or [TBD]×[TBD] once scale factor is confirmed). Patch extraction is performed on-the-fly during data loading.

Benefits for lensing SR specifically:
- Patches containing arc/ring segments are sampled many times per image, increasing supervision density on physically informative regions.
- Patches from the featureless sky background act as implicit regularization on noise reconstruction.
- Effective dataset size scales as `N_images × (patches_per_image)`, dramatically expanding training data.

At inference, full images are reconstructed by tiling with overlap and averaging (no seam artifacts for all tested architectures except SRCNN, which has no explicit receptive field boundary effects).

---

## 3. Architecture Selection and Justification

We benchmark four architectures spanning the modern SR design space. Selection is domain-driven: each architecture is included because of a specific, articulable property relevant to strong lensing images — not generic CV performance rankings.

### 3.1 Bicubic Interpolation (Baseline)

Classic polynomial upsampling. No learned parameters. Included as the **metric floor**: all deep learning models must improve on this to justify their added complexity. Sets the PSNR/SSIM lower bound.

Expected failure mode: blurs high-frequency ring edges. The Einstein ring appears as a smooth, low-contrast arc rather than a sharp-edged annulus.

### 3.2 SRCNN (Dong et al. 2014)

The foundational deep SR network. Three convolutional layers: patch extraction → nonlinear mapping → reconstruction. Applied to bicubic-upsampled input.

**Why include it:** Historical baseline. Establishes the minimum contribution of learned SR over classical upsampling. Low parameter count (~57k) makes it a useful data-efficiency reference. Not expected to be competitive with EDSR or SwinIR.

**Lensing limitation:** Receptive field of ~13×13 pixels. Entirely local — cannot capture the ring as a coherent global structure.

### 3.3 EDSR (Lim et al. 2017) — Primary CNN Baseline

Enhanced Deep Super-Resolution. Key architectural decision: **batch normalization is removed from all residual blocks**.

**Domain justification (critical):** BN normalizes feature statistics within a batch, effectively discarding the absolute intensity scale of each image. In natural image SR this is acceptable. In lensing SR it is not: the ratio of Einstein ring peak intensity to background sky is a physically meaningful quantity encoding the lensing signal strength. BN would normalize this ratio away during training. EDSR's BN-free design preserves this physical information throughout the network. This is the same reason Saranga Mahanta (GSoC 2022) found that fully training from random weights outperformed frozen pretrained backbones on lensing data: ImageNet statistics are simply irrelevant to the lensing image distribution.

Architecture: 32 residual blocks, 256 feature channels, pixel-shuffle upsampling. Pretrained weights from DIV2K used for initialization, then fully fine-tuned.

### 3.4 RCAN (Zhang et al. 2018) — Channel Attention CNN

Residual Channel Attention Network. Adds a channel attention (squeeze-and-excitation) mechanism on top of EDSR-style residual blocks.

**Domain justification:** The channel attention mechanism learns to weight feature channels by their global average activation. In practice, this steers the network's representational capacity toward feature maps that activate on the arc/ring region and suppresses feature maps that activate primarily on featureless sky background. This is analogous in motivation to the CBAM attention modules used by Saranga Mahanta (2022) for classification: attention as a learned spatial relevance filter. For SR, the benefit is that reconstruction fidelity is disproportionately allocated to physically informative regions. We visualize channel attention weight distributions (see Section 9) to verify this interpretation.

Architecture: 10 residual groups × 20 residual channel attention blocks, 64 channels. Total ~15M parameters.

### 3.5 SwinIR (Liang et al. 2021) — Transformer Baseline

Swin Transformer-based image restoration. Replaces convolutional feature extraction with shifted-window self-attention (SWSA) blocks, preserving the patch-based processing efficiency of Swin Transformers.

**Domain justification:** The Einstein ring is a global structure spanning the full image. Its angular radius θ_E is determined by the lens mass at cosmological distances — it is a non-local property of the image. CNNs capture this global context only after many stacked layers whose receptive fields gradually expand. SwinIR's shifted-window attention, even in shallow configurations, captures dependencies across the full image from the first transformer block. This is the SR analogue of the motivation for using ViT-B/16 in Test V and Test I: global structure in lensing images rewards global receptive field architectures.

The shifted-window mechanism also provides an implicit positional bias that aligns with the near-circular symmetry of the lensing configuration: features at the same radial distance from center are related by the same lensing geometry, and the window overlap introduced by the shifting mechanism encourages the model to learn these radially consistent representations.

Architecture: 6 RSTB blocks × 6 SwinTransformer layers, window size 8×8, 180 channels. Total ~11.8M parameters.

### Architecture Summary

| Model | Type | Params | Key property | Lensing justification |
|---|---|---|---|---|
| Bicubic | Classical | 0 | — | Metric floor |
| SRCNN | CNN | ~57k | Learned local mapping | Historical baseline |
| EDSR | Deep CNN | ~43M | BN-free residuals | Preserves ring intensity ratio |
| RCAN | Attention CNN | ~15M | Channel attention | Concentrates capacity on arc |
| SwinIR | Transformer | ~11.8M | Global self-attention | Captures full-ring structure |

---

## 4. Loss Function Strategy

### 4.1 Rationale

The choice of loss function in SR is not a training detail — it determines what property of the image the network is optimized to reconstruct. For lensing SR, the relationship between loss function and scientific validity is direct and non-trivial.

**L2 (MSE):** Minimizing pixel-wise squared error maximizes PSNR by construction. However, L2 regression to the mean: when the model is uncertain about the exact position of a high-frequency ring edge, it places probability mass on the mean of plausible reconstructions — a blurred edge. A model optimized purely for PSNR can produce a reconstruction with strong global metrics but a systematically blurred Einstein ring whose inferred radius is biased. This is scientifically unacceptable.

**L1 (MAE):** Less aggressive regression-to-mean than L2. Produces sharper edges because L1 optimal estimator is the median (not mean) of the distribution. Used as the default training loss in EDSR (Lim et al. 2017).

**SSIM Loss:** Directly optimizes for structural similarity — a perceptual metric that measures luminance, contrast, and structural consistency jointly. Penalizes loss of ring edge structure more than background blurring. However, optimizing SSIM alone can produce over-sharpened artifacts.

**L1 + λ·(1 − SSIM):** Hybrid loss combining absolute reconstruction fidelity with structural quality. This is the primary training objective for EDSR, RCAN, and SwinIR in this work. The weight λ = [TBD — determined by validation SSIM] balances the two terms.

### 4.2 Loss Ablation

An explicit ablation is conducted on the best-performing architecture (identified post-Phase 3) with three loss configurations:

| Config | Loss | Optimizes for |
|---|---|---|
| L2-only | MSE | PSNR; risks ring blurring |
| L1-only | MAE | Sharpness; stable training |
| L1 + SSIM | L1 + λ·(1−SSIM) | Fidelity + structural quality |

Results reported in Section 8.

---

## 5. Training Setup and Augmentation

### 5.1 Augmentation Strategy

All augmentations are **physically motivated** by the symmetry properties of strong gravitational lensing. This is not generic regularization.

Gravitational lensing has approximate **azimuthal symmetry** around the lens center: the Einstein ring is (ideally) circular, and the lensing geometry is invariant under rotations about the optical axis. The D4 dihedral group — comprising 90°, 180°, 270° rotations and horizontal/vertical reflections — therefore generates physically equivalent lensing configurations. All eight D4 transforms of a lensing image are valid training examples.

This argument is identical in structure to the rationale for equivariant architectures (Apoorva Singh, GSoC 2021) and directly informed the augmentation design in Test I and Test V. For SR, D4 augmentation is applied to HR/LR pairs jointly (same transform applied to both), ensuring the upsampling target is consistent.

| Augmentation | Physical basis | Applied to |
|---|---|---|
| 90°/180°/270° rotation | Azimuthal lensing symmetry | HR + LR pair jointly |
| Horizontal flip | Reflection symmetry | HR + LR pair jointly |
| Vertical flip | Reflection symmetry | HR + LR pair jointly |
| Random crop (64×64 HR) | Patch-based training efficiency | HR + LR pair jointly |

Random arbitrary-angle rotations (non-D4) are **not used**: they require bicubic interpolation of the training target, which introduces interpolation artifacts into the HR ground truth and corrupts the supervision signal.

### 5.2 Optimizer and Schedule

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Initial learning rate | 1e-4 |
| LR scheduler | CosineAnnealingLR (T_max = total epochs) |
| Batch size | 16 (patch-based) |
| Epochs | [TBD — EDSR: ~100; SwinIR: ~200] |
| Hardware | [TBD] |
| Mixed precision | FP16 via torch.cuda.amp |

Early stopping: monitored on validation SSIM with patience = 20 epochs. Best checkpoint saved by validation SSIM (not PSNR — see Section 4.1 rationale).

---

## 6. Evaluation Protocol

### 6.1 Quantitative Metrics

All metrics computed on the **test split** (held out from training and validation). For each model, we report:

| Metric | Formula | Notes |
|---|---|---|
| MSE | (1/N) Σ (ŷᵢ − yᵢ)² | Lower is better |
| PSNR | 10·log₁₀(MAX²/MSE) | Higher is better; in dB |
| SSIM | Structural similarity index (Wang et al. 2004) | Higher is better; range [0, 1] |

Metrics computed on full reconstructed images (not patches). Images rescaled to [0, 1] before metric computation.

### 6.2 Ring Reconstruction Analysis

Beyond aggregate metrics, we analyze **radial intensity profiles**: for each test image, we extract the azimuthally-averaged intensity as a function of radius from the image center (r). The Einstein ring manifests as a peak in this profile at r = θ_E (in pixel units). We compare:

- HR ground truth radial profile
- LR bicubic upsampled profile
- SR reconstructed profile (per model)

The key diagnostic: **does the SR model preserve the ring peak position (θ_E)?** A model that maximizes PSNR by blurring will shift the peak inward and reduce its amplitude. A model that preserves ring sharpness will match the HR peak position and amplitude within measurement error. This analysis directly bridges the SR metric to the downstream dark matter inference goal.

### 6.3 Attention Visualization (RCAN)

For RCAN, channel attention weights are extracted from the final residual group's attention module and visualized as a spatial heatmap overlaid on the input LR image. The hypothesis: high-attention channels correspond to features that activate on the lensing arc/ring, not on the featureless sky background. This provides interpretability evidence connecting to Ojha (2024)'s GradCAM analysis on the Lensiformer classification model.

---

## 7. Results

> **[PLACEHOLDER — to be completed after training]**

### 7.1 Quantitative Benchmark

| Model | MSE ↓ | PSNR ↑ (dB) | SSIM ↑ | Params | Train time |
|---|---|---|---|---|---|
| Bicubic | TBD | TBD | TBD | — | — |
| SRCNN | TBD | TBD | TBD | ~57k | TBD |
| EDSR | TBD | TBD | TBD | ~43M | TBD |
| RCAN | TBD | TBD | TBD | ~15M | TBD |
| SwinIR | TBD | TBD | TBD | ~11.8M | TBD |

**Best model:** TBD  
**Key finding:** TBD

### 7.2 Per-Metric Rankings

> [TBD — expected: SwinIR ≥ RCAN ≥ EDSR >> SRCNN > Bicubic on SSIM; EDSR may exceed RCAN on PSNR due to parameter count]

---

## 8. Ablation Study: Loss Function

> **[PLACEHOLDER — to be completed after training]**

Ablation conducted on [TBD — best architecture from Section 7].

| Loss | MSE ↓ | PSNR ↑ | SSIM ↑ | Ring peak error (px) |
|---|---|---|---|---|
| L2 only | TBD | TBD | TBD | TBD |
| L1 only | TBD | TBD | TBD | TBD |
| L1 + SSIM (λ=TBD) | TBD | TBD | TBD | TBD |

**Key finding:** TBD  
**Scientific interpretation:** TBD — expected finding: L2 loss produces highest PSNR but worst ring peak preservation; L1+SSIM achieves best SSIM and best ring peak fidelity at the cost of ~[X] dB PSNR.

---

## 9. Qualitative Analysis

> **[PLACEHOLDER — figures to be added after training]**

### 9.1 Side-by-Side Reconstruction Panels

For 6 representative test images (selected to cover varied ring radii and arc configurations):

```
[Figure TBD]
Row layout per image: HR Ground Truth | LR Input | Bicubic | EDSR | RCAN | SwinIR
Colourmap: viridis (single-channel)
Scale bar: [TBD] arcsec
```

### 9.2 Radial Intensity Profile Comparison

```
[Figure TBD]
X-axis: Radius from image center (pixels)
Y-axis: Azimuthally-averaged intensity (normalized)
Lines: HR (black), LR bicubic (grey dashed), EDSR (blue), RCAN (orange), SwinIR (red)
Annotation: θ_E position marked with vertical dashed line
```

Expected observation: Bicubic and SRCNN profiles show broadened, shifted ring peak relative to HR ground truth. EDSR/RCAN/SwinIR profiles recover peak position within [TBD] pixels.

### 9.3 RCAN Channel Attention Visualization

```
[Figure TBD]
Left: Input LR image
Right: Channel attention heatmap (averaged across top-k attention channels) overlaid on LR
Hypothesis: High attention weight concentrates on arc/ring region, not background sky
```

### 9.4 Failure Cases

```
[Figure TBD — to be identified after training]
Cases where SR model underperforms bicubic or introduces artifacts.
Expected: Low-SNR images where noise is amplified; edge-of-frame rings where receptive field is incomplete.
```

---

## 10. Discussion

> **[PLACEHOLDER — to be expanded with actual results]**

### 10.1 Why the Winner Wins in Lensing Terms

**[TBD — pre-written template for three scenarios:]**

*If SwinIR wins:* The shifted-window self-attention mechanism captures the full Einstein ring as a coherent global structure from early transformer blocks. The ring's angular extent spans [TBD] pixels across; at this scale, CNNs require [TBD] stacked residual blocks to accumulate equivalent receptive field coverage. SwinIR's global attention bypasses this depth requirement, explaining its SSIM advantage over EDSR despite comparable parameter count.

*If RCAN wins:* Channel attention provides the critical advantage over EDSR: by concentrating representational capacity on arc-bearing feature maps and suppressing sky-background feature maps, RCAN allocates its reconstruction budget where the physical signal is. The attention visualization (Section 9.3) confirms this: [TBD — high-weight channels localize to ring region / alternative finding].

*If EDSR wins:* The BN-free residual design and scale of the network ([TBD]M parameters) dominates. The attention mechanism of RCAN and global attention of SwinIR do not provide sufficient benefit over raw depth and width for this dataset scale and upsampling factor.

### 10.2 Loss Function: PSNR vs. Physical Fidelity

The loss ablation (Section 8) reveals a fundamental tension between maximizing standard SR metrics and preserving physically meaningful structure. [TBD — fill with actual result]. This finding has direct implications for how DeepLense SR models should be evaluated and trained: PSNR optimality is not scientific optimality. We recommend SSIM as the primary metric for SR model selection when the downstream task is ring morphology analysis or substructure detection, rather than PSNR.

### 10.3 Connection to Prior SR Work in DeepLense

This benchmark establishes a deterministic SR baseline that is directly complementary to:
- **Pranath Reddy (2023):** That work explored conditional diffusion SR — a generative approach that models the posterior distribution over HR images. Diffusion SR produces diverse, sharp samples but requires many forward passes per image and is sensitive to guidance scale. The present discriminative models (EDSR/RCAN/SwinIR) are deterministic, fast at inference, and directly optimizable for SSIM. Together they bracket the SR design space.
- **Atal Gupta (2024):** Focused on real galaxy data (Model IV). The present work establishes what SR performance is achievable on clean simulated data, providing the upper bound against which Model IV performance can be compared in Test VI.B.
- **Anirudh Shankar (2024):** Physics-informed unsupervised SR (LensSR) requires no HR ground truth — a harder problem. The supervised performance documented here is the empirical target that unsupervised SR should aspire to close the gap toward.

### 10.4 Downstream Utility: SR as Preprocessing for Classification

A natural question is whether SR preprocessing improves downstream dark matter substructure classification accuracy. A preliminary experiment using the Test I best classifier applied to LR vs. SR-enhanced inputs would provide direct evidence for the value of SR in the DeepLense pipeline. This is deferred to future work but is flagged as the highest-priority scientific follow-on: it transforms SR from an image quality exercise into a demonstrated component of the dark matter inference pipeline.

---

## 11. Limitations and Future Work

**Known limitations:**

- SR models trained on simulated Gaussian PSF images (VI.A) are expected to degrade on real HST/HSC images with spatially varying, non-Gaussian PSFs. This degradation is quantified in Test VI.B.
- Training on the no-substructure class only may introduce a bias: the SR model is trained on relatively uniform ring morphologies and has not seen the arc perturbations characteristic of CDM or axion substructure during training. The impact on SR quality for substructure classes is not evaluated here.
- PSNR/SSIM metrics are agnostic to the physical significance of different image regions. A metric that down-weights background sky and up-weights arc/ring pixels would be more scientifically appropriate and is not explored.
- SwinIR's window-based attention operates on fixed 8×8 windows. Rings smaller than one window are captured within a single attention context; rings spanning multiple windows rely on the shifted-window mechanism for cross-window coherence. The effect on ring reconstruction fidelity as a function of θ_E is not systematically characterized.

**Future work:**

1. **Downstream classification loop:** Apply SR outputs as inputs to the Test I classifier and quantify accuracy improvement on the full three-class task.
2. **Substructure-class SR quality:** Evaluate SR models on CDM and axion images (held out from SR training) to assess generalization.
3. **Physics-constrained loss:** Incorporate a lensing-equation-based consistency term (as in Anirudh Shankar's LensSR) into the supervised training objective — hybrid of supervised and physics-informed.
4. **Real data adaptation:** Test VI.B addresses this directly for 300 HSC/HST pairs.
5. **Equivariant SR:** Replace convolutional blocks in EDSR/RCAN with e2cnn equivariant layers (D4 group). The D4 augmentation used here is a weak form of enforcing azimuthal symmetry; an equivariant SR architecture would enforce it exactly. Connection to Apoorva Singh (2021) and Geo Jolly (2023).

---

## 12. Dependencies and Reproducibility

```
Python           3.10+
PyTorch          2.0+
torchvision      0.15+
numpy            1.24+
scikit-image     0.21+       # SSIM computation
matplotlib       3.7+
tqdm             4.65+
einops           0.6+        # SwinIR tensor rearrangements
timm             0.9+        # Pretrained weight loading
wandb            (optional)  # Experiment tracking
```

### Reproducibility

All experiments use fixed random seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

**Checkpoint availability:** Trained model weights to be uploaded to [TBD — Google Drive / HuggingFace Hub].

**Dataset access:** [Google Drive link](https://drive.google.com/file/d/1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw/view?usp=sharing)

```
dataset/
├── HR/
│   ├── image_0001.npy
│   ├── ...
└── LR/
    ├── image_0001.npy
    ├── ...
```

> **Directory structure confirmed in notebook EDA cell — update above if actual layout differs.**

---

## 13. References

**Super-Resolution architectures:**

- Dong, C., Loy, C. C., He, K., & Tang, X. (2014). Learning a deep convolutional network for image super-resolution. *ECCV*. [SRCNN]
- Lim, B., Son, S., Kim, H., Nah, S., & Lee, K. M. (2017). Enhanced deep residual networks for single image super-resolution. *CVPR Workshops*. [EDSR]
- Zhang, Y., Li, K., Li, K., Wang, L., Zheng, B., & Fu, Y. (2018). Image super-resolution using very deep residual channel attention networks. *ECCV*. [RCAN]
- Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., & Timofte, R. (2021). SwinIR: Image restoration using Swin Transformer. *ICCV Workshops*. [SwinIR]
- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. *IEEE TIP*. [SSIM]

**DeepLense prior SR work:**

- Reddy, P. (2023). Super-Resolution for Strong Lensing Images. *GSoC 2023, ML4SCI*. [GitHub folder: `Super_Resolution_Pranath_Reddy`]
- Gupta, A. (2024). Super-Resolution with Diffusion Models. *GSoC 2024, ML4SCI*. [GitHub folder: `Super_Resolution_Atal_Gupta`]
- Shankar, A. (2024). Physics-Informed Unsupervised Super-Resolution (LensSR). *GSoC 2024, ML4SCI*. [GitHub folder: `DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar`]

**DeepLense foundational and related work:**

- Alexander, S., et al. (2020). Deep learning the morphology of dark matter substructure. *ApJ 893(1):15*. arXiv:1909.07346.
- Singh, A. V. (2021). Equivariant Neural Networks for DeepLense. *GSoC 2021, ML4SCI*.
- Mahanta, S. K. (2022, 2023). Updating the DeepLense Pipeline. *GSoC 2022–2023, ML4SCI*.
- Ojha, A. (2024). Physics-Informed Neural Network for Dark Matter Morphology. *GSoC 2024, ML4SCI*.

**Astrophysics context:**

- Oguri, M. & Marshall, P. J. (2010). Gravitationally lensed quasars and supernovae in future wide-field optical imaging surveys. *MNRAS 405(4):2579–2593*.
- Weiler, M. & Cesa, G. (2019). General E(2)-Equivariant Steerable CNNs. *NeurIPS*. arXiv:1911.08251.

---

*Prepared for ML4SCI DeepLense GSoC 2026 evaluation. All result cells marked [TBD] to be filled upon experiment completion.*

