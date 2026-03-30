<div align="center">

# Test VI.A — Gravitational Lens Image Super-Resolution

### ML4SCI DeepLense · GSoC 2026

**Rafiqus Salehin** · [@rsalehin](https://github.com/rsalehin)

[![ML4SCI](https://img.shields.io/badge/ML4SCI-DeepLense-blue?style=flat-square)](https://ml4sci.org)
[![GSoC](https://img.shields.io/badge/GSoC-2026-4285F4?style=flat-square&logo=google&logoColor=white)](https://summerofcode.withgoogle.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](../LICENSE)

<br>

*Multi-architecture deep learning super-resolution benchmark on simulated strong gravitational lensing images. Five architectures evaluated across pixel-level, structural, and physically-motivated metrics, with controlled ablations on training procedure, loss function, and patch strategy.*

</div>

---

## Table of Contents

- [Task Description](#task-description)
- [Dataset](#dataset)
- [Scientific Motivation](#scientific-motivation)
- [Pipeline Overview](#pipeline-overview)
- [Architectures](#architectures)
- [Results](#results)
  - [Main Benchmark Table](#main-benchmark-table)
  - [Training Dynamics](#training-dynamics)
  - [Loss Function Ablation](#loss-function-ablation)
  - [Patch vs Full-Image Ablation](#patch-vs-full-image-ablation)
- [Physical Evaluation — Einstein Ring Metrics](#physical-evaluation--einstein-ring-metrics)
  - [Radial Profile Comparison](#radial-profile-comparison)
  - [Ring Peak Shift Distribution](#ring-peak-shift-distribution)
  - [Radial Profile MSE](#radial-profile-mse)
- [Qualitative Analysis](#qualitative-analysis)
  - [Full Comparison Grid](#full-comparison-grid)
  - [Difference Maps](#difference-maps)
  - [Failure Cases](#failure-cases)
- [Scientific Discussion](#scientific-discussion)
- [Limitations](#limitations)
- [Bridge to Test VI.B](#bridge-to-test-vib)
- [Reproducibility](#reproducibility)
- [References](#references)

---

## Task Description

Train a deep learning super-resolution model to upscale low-resolution (LR) strong lensing images to high-resolution (HR) using simulated paired samples as ground truth. The SR model must not only achieve strong pixel-level metrics (MSE, PSNR, SSIM) but also preserve the physically relevant Einstein ring structure — specifically the ring peak radius θ_E — since systematic bias in θ_E propagates directly to errors in inferred lens mass.

**Evaluation metrics:** MSE ↓ · PSNR ↑ · SSIM ↑ · Ring peak shift ↓ · Radial profile MSE ↓

---

## Dataset

| Property | Value |
|---|---|
| Total pairs | 10,000 HR/LR |
| HR shape | `(1, 150, 150)`, float64, range [0, 1] |
| LR shape | `(1, 75, 75)`, float64 |
| Scale factor | **2× (uniform, isotropic)** |
| Channels | 1 (grayscale) |
| Class | No substructure (single population) |
| Pairing | By filename stem (`sample{N}.npy`) |

**Pixel statistics (500-sample):**

| Region | Pixel fraction | Mean intensity |
|---|---|---|
| Background | 86.4% | 0.024 |
| Ring / arc | 13.6% | 0.301 |
| Ring-to-background contrast | — | 12.5× |

**Einstein radius statistics (200-sample):**

| | Value |
|---|---|
| Mean θ_E | 24.61 px (HR space) |
| Std θ_E | 4.61 px |
| Range | 13.5 – 37.5 px |
| Mean SNR (HR ≈ LR) | 13.4 (Pearson r = 0.9989) |

<br>

<div align="center">
  <img src="assets/2_3_sample_grid.png" width="92%" alt="Sample grid: HR · LR · Bicubic upsampled · Difference"/>
  <br>
  <sub><b>Figure 1.</b> Four randomly sampled HR/LR pairs with bicubic ×2 upsampling and pixel-wise difference HR − bicubic. Einstein ring and arc morphology vary across samples; residuals are concentrated on ring edges.</sub>
</div>

<br>

The dataset contains a single no-substructure population with controlled Gaussian PSF. At 2× scale factor, LR images preserve most spatial frequency content — this makes the classical baselines unusually strong (bicubic SSIM 0.964) and tightens the performance ceiling for deep models.

---

## Scientific Motivation

The Einstein radius θ_E encodes the projected lens mass enclosed within the ring:

```
M_E = (c² / 4G) × (D_s / D_l D_ls) × θ_E²
```

A 1 px systematic shift in the reconstructed θ_E corresponds to a ~4% bias in θ_E (at mean θ_E ≈ 24.6 px) and an ~8% bias in inferred lens mass M_E via the quadratic scaling. Super-resolution is therefore not merely a visual enhancement task — ring-structure preservation is the physically critical metric.

<br>

<div align="center">
  <img src="assets/2_6_radial_profiles.png" width="92%" alt="Einstein ring radial profiles for 6 sample images"/>
  <br>
  <sub><b>Figure 2.</b> Azimuthally averaged radial intensity profiles for 6 HR images with bicubic upsampled LR overlaid. Gold vertical lines mark the HR ring peak θ_E. Mean θ_E = 24.61 px, std = 4.61 px across the dataset (range 13.5–37.5 px).</sub>
</div>

<br>

---

## Pipeline Overview

```
Raw LR (75×75, float64)
      │
      ▼
Per-image min-max normalisation → [0, 1]    (LR only; HR already bounded)
      │
      ▼
LensingSRDataset
  ├── Patch mode  : random 32×32 LR / 64×64 HR crops  (training)
  └── Full-image  : full 75×75 LR / 150×150 HR         (val / test)
      │
      ▼
D4 augmentation (8 symmetries of the square, applied jointly to HR/LR)
      │
      ▼
SR Model  →  PixelShuffle ×2  →  SR output (150×150)
      │
      ▼
Loss: L1 + 0.1 × (1 − SSIM)      [combined default]
      │
      ▼
Evaluation: MSE · PSNR · SSIM · Ring peak shift · Radial profile MSE
```

**Train / val / test split (seed=42):** 8,000 / 1,000 / 1,000

**DataLoader configuration:**

| Loader | Batch size | Mode | Augmentation |
|---|---|---|---|
| train | 16 | Patch 64×64 HR | D4 on |
| val | 4 | Full image | Off |
| test | 1 | Full image | Off |

---

## Architectures

Five architectures benchmarked, spanning 8K to 7.9M parameters:

### SRCNN
3-layer CNN operating in HR space (bicubic pre-upsampled input). Patch extraction (9×9) → nonlinear mapping (1×1) → reconstruction (5×5). The simplest deep SR model; serves as the baseline deep learning result.

| Parameters | Upsampling | Input |
|---|---|---|
| **8,129** | Bicubic pre-upsampling (not learned) | Bicubic-upsampled LR |

### EDSR
Enhanced Deep Residual Network. 16 residual blocks + final conv in body, pixel-shuffle ×2 upsampling head. No batch normalisation (by design). Residual scaling = 0.1. EDSR-baseline configuration (n_feats=64).

| Parameters | Upsampling | Batch norm |
|---|---|---|
| **1,367,553** | Learned pixel-shuffle ×2 | None |

### RCAN
Residual Channel Attention Network. 5 residual groups × 10 residual channel attention blocks. Channel attention uses global average pooling + squeeze-excitation (reduction=16). Reduced configuration relative to the original paper (~15M params).

| Parameters | Upsampling | Attention |
|---|---|---|
| **4,092,297** | Learned pixel-shuffle ×2 | Channel attention (per-group) |

### UNet-SR
Encoder-decoder with 4-stage skip connections and a pixel-shuffle SR head. Uses batch normalisation in encoder/decoder blocks. The only architecture with explicit multi-scale feature reuse via skip connections.

| Parameters | Upsampling | Skip connections |
|---|---|---|
| **7,889,217** | Learned pixel-shuffle ×2 | 4 (e1→d1, e2→d2, e3→d3, e4→d4) |

### SwinIR
Swin Transformer-based image restoration. 4 residual Swin Transformer blocks, depth=6, num_heads=6, window_size=8. Full LR images (75×75) require 5 px reflect padding to satisfy window-size divisibility. Slowest to converge; best patch-trained result.

| Parameters | Upsampling | Attention scope |
|---|---|---|
| **1,410,601** | Learned pixel-shuffle ×2 | 8×8 shifted-window |

---

## Results

### Main Benchmark Table

All models evaluated on the **sealed test set (1,000 images)**. EDSR-FullImg uses full-image training (see [Patch vs Full-Image Ablation](#patch-vs-full-image-ablation) below).

| Model | MSE ↓ | PSNR ↑ (dB) | SSIM ↑ | Ring shift ↓ (px) | Radial MSE ↓ | Params |
|:------|------:|------------:|-------:|------------------:|-------------:|-------:|
| Bicubic | 1.01 × 10⁻⁴ | 40.107 | 0.96419 | 0.212 | 3.21 × 10⁻⁵ | — |
| Lanczos | 0.98 × 10⁻⁴ | 40.243 | 0.96359 | 0.200 | 2.98 × 10⁻⁵ | — |
| SRCNN | 0.77 × 10⁻⁴ | 41.222 | 0.97284 | 0.202 | 1.32 × 10⁻⁵ | 8,129 |
| RCAN | 0.76 × 10⁻⁴ | 41.257 | 0.97379 | 0.223 | 1.43 × 10⁻⁵ | 4.09M |
| UNet-SR | 0.77 × 10⁻⁴ | 41.217 | 0.97397 | 0.204 | 1.42 × 10⁻⁵ | 7.89M |
| EDSR | 0.74 × 10⁻⁴ | 41.414 | 0.97393 | 0.199 | 1.33 × 10⁻⁵ | 1.37M |
| SwinIR | 0.73 × 10⁻⁴ | 41.445 | 0.97424 | 0.204 | 1.28 × 10⁻⁵ | 1.41M |
| **EDSR-FullImg** | **0.67 × 10⁻⁴** | **41.840** | **0.97678** | **0.190** | **1.13 × 10⁻⁵** | 1.37M |

**Key findings:**
- **EDSR with full-image training is the best overall model** across all five metrics.
- **SwinIR is the best patch-trained model**, edging EDSR (patch) on both PSNR and SSIM.
- **SRCNN (8,129 params) achieves 58.9% radial MSE improvement over bicubic** — nearly identical to EDSR (patch) at 168× more parameters.
- **RCAN underperforms EDSR on all metrics** despite 3× more parameters (see [Scientific Discussion](#scientific-discussion) for the attention collapse finding).

### Training Dynamics

<br>

<div align="center">
  <img src="assets/12_2_val_ssim_overlay.png" width="92%" alt="Validation SSIM training curves for all models"/>
  <br>
  <sub><b>Figure 3.</b> Validation SSIM training curves overlaid for all five architectures. CNN models (SRCNN, EDSR, RCAN) reach near-final SSIM within the first 5 epochs. UNet-SR and SwinIR start significantly lower and improve gradually — the cost of training skip-connected and attention-based architectures from scratch on this dataset.</sub>
</div>

<br>

<div align="center">
  <img src="assets/12_3_psnr_ssim_scatter.png" width="68%" alt="PSNR vs SSIM scatter across all models"/>
  <br>
  <sub><b>Figure 4.</b> PSNR vs SSIM scatter for all models on the test set. Rankings are broadly consistent. UNet-SR is the notable outlier — it ranks 2nd on SSIM but 5th on PSNR, indicating structurally faithful but slightly over-smoothed reconstructions.</sub>
</div>

<br>

| Model | Best epoch | Epochs run | Starting val SSIM (ep. 1) |
|---|---|---|---|
| SRCNN | 50 | 50 (no ES) | 0.9699 |
| EDSR | 56 | 76 | 0.9725 |
| RCAN | 4 | 24 | 0.9729 |
| UNet-SR | 96 | 100 | 0.8930 |
| SwinIR | 83 | 103 | 0.9336 |
| EDSR-FullImg | 95 | 100 | — |

### Loss Function Ablation

Conducted on EDSR-baseline. All runs: seed=42, Adam lr=1e-4, CosineAnnealingLR.

<br>

<div align="center">
  <img src="assets/13_2_loss_ablation.png" width="85%" alt="Loss function ablation: val SSIM curves for L2, L1, L1+SSIM, L1+SSIM+Perceptual"/>
  <br>
  <sub><b>Figure 5.</b> Validation SSIM training curves for four loss configurations on EDSR-baseline. Differences are small but consistent — L1+SSIM achieves the highest final SSIM and most stable convergence.</sub>
</div>

<br>

| Loss | MSE ↓ | PSNR ↑ (dB) | SSIM ↑ | Ring shift ↓ (px) |
|---|---|---|---|---|
| L2 only | 7.01 × 10⁻⁵ | **41.60** | 0.97268 | 0.201 |
| L1 only | 7.33 × 10⁻⁵ | 41.40 | 0.97327 | 0.203 |
| **L1 + 0.1·SSIM** | 7.36 × 10⁻⁵ | 41.41 | **0.97393** | **0.199** |
| L1 + 0.1·SSIM + Percept. | 7.40 × 10⁻⁵ | 41.37 | 0.97371 | 0.201 |

L1+SSIM achieves the highest SSIM and smallest ring shift, directly optimising the primary evaluation metric. Selected as the default loss for all subsequent experiments.

### Patch vs Full-Image Ablation

<br>

<div align="center">
  <img src="assets/14_1_fullimage_curves.png" width="80%" alt="Patch vs full-image training: val SSIM and PSNR curves"/>
  <br>
  <sub><b>Figure 6.</b> Patch (blue) vs full-image (orange) training for EDSR. Full-image training converges more slowly (best epoch 95 vs 56) but reaches higher final SSIM and substantially better radial MSE. The complete Einstein ring in every forward pass preserves ring-scale spatial context that 64×64 patches cannot capture.</sub>
</div>

<br>

| | Patch training | Full-image training | Δ |
|---|---|---|---|
| Batch size | 16 | 4 | — |
| Best epoch | 56 | 95 | +39 |
| Test PSNR | 41.414 dB | **41.840 dB** | **+0.426 dB** |
| Test SSIM | 0.97393 | **0.97678** | +0.00285 |
| Ring shift | 0.199 px | **0.190 px** | −4.5% |
| Radial MSE | 1.33 × 10⁻⁵ | **1.13 × 10⁻⁵** | **−15.0%** |

Full-image training outperforms patch training across every metric. The largest gain is in radial MSE (−15%), confirming that **training procedure matters more than architecture choice** on this dataset.

---

## Physical Evaluation — Einstein Ring Metrics

Standard image metrics (MSE, PSNR, SSIM) do not capture physically relevant structure. Two dedicated metrics evaluate ring preservation:

- **Ring peak shift:** distance (px) between the peak of the azimuthally averaged radial profile of the SR output vs HR ground truth. 1 px shift at mean θ_E ≈ 24.6 px implies ~4% bias in θ_E and ~8% bias in inferred M_E.
- **Radial profile MSE:** mean squared error between the full azimuthally averaged profiles of SR and HR — captures both ring position and amplitude fidelity.

### Radial Profile Comparison

<br>

<div align="center">
  <img src="assets/15_1_radial_profiles.png" width="92%" alt="Radial profile comparison across models for 6 test images"/>
  <br>
  <sub><b>Figure 7.</b> Azimuthally averaged radial intensity profiles for 6 test images. HR in black; all SR models overlaid. Gold vertical lines mark the HR ring peak. All SR models preserve ring position accurately. The dominant discrepancy is <b>ring amplitude underestimation</b> — all models slightly smooth the peak intensity, most visibly at high-contrast arcs.</sub>
</div>

<br>

### Ring Peak Shift Distribution

<br>

<div align="center">
  <img src="assets/15_2_ring_shift_distributions.png" width="92%" alt="Ring peak shift distributions for all models across 1,000 test images"/>
  <br>
  <sub><b>Figure 8.</b> Ring peak shift distributions across 1,000 test images for all models. Distributions are structurally identical — the non-zero mean is driven by a small fraction of images where the ring peak falls between 1 px bins. EDSR (full) achieves the lowest mean (0.190 px) and highest zero-shift fraction (87.0%).</sub>
</div>

<br>

| Model | Mean (px) | Std (px) | % zero-shift |
|---|---|---|---|
| Bicubic | 0.212 | 1.058 | 85.1% |
| Lanczos | 0.200 | 1.055 | 86.3% |
| SRCNN | 0.202 | 1.044 | 85.9% |
| EDSR (patch) | 0.199 | 1.042 | 86.1% |
| RCAN | 0.223 | 1.128 | 85.7% |
| UNet-SR | 0.204 | 1.045 | 85.8% |
| SwinIR | 0.204 | 1.115 | 86.7% |
| **EDSR (full)** | **0.190** | **1.039** | **87.0%** |

### Radial Profile MSE

<br>

<div align="center">
  <img src="assets/15_3_radial_mse_distributions.png" width="92%" alt="Radial profile MSE distributions for all models"/>
  <br>
  <sub><b>Figure 9.</b> Radial profile MSE distributions across 1,000 test images. All deep models achieve ~55–65% improvement over bicubic. EDSR (full) achieves the best median and mean. SRCNN at 8,129 parameters performs nearly identically to patch-trained EDSR at 1.37M — diminishing returns from scale are pronounced.</sub>
</div>

<br>

| Model | Mean | Improvement vs Bicubic |
|---|---|---|
| Bicubic | 3.21 × 10⁻⁵ | — |
| Lanczos | 2.98 × 10⁻⁵ | +7.1% |
| SRCNN | 1.32 × 10⁻⁵ | +58.9% |
| RCAN | 1.43 × 10⁻⁵ | +55.5% |
| UNet-SR | 1.42 × 10⁻⁵ | +55.8% |
| EDSR (patch) | 1.33 × 10⁻⁵ | +58.5% |
| SwinIR | 1.28 × 10⁻⁵ | +60.1% |
| **EDSR (full)** | **1.13 × 10⁻⁵** | **+64.9%** |

---

## Qualitative Analysis

### Full Comparison Grid

<br>

<div align="center">
  <img src="assets/16_1_full_comparison_grid.png" width="100%" alt="Full qualitative comparison grid: 4 images × 10 models"/>
  <br>
  <sub><b>Figure 10.</b> Four test images × 10 columns: HR | LR | Bicubic | Lanczos | SRCNN | EDSR (patch) | RCAN | UNet-SR | SwinIR | EDSR (full). Morphologies span two-node arcs, high-contrast elliptical rings, near-circular uniform rings, and strongly asymmetric arcs. At 2× scale, all SR outputs are visually close to HR; differences are concentrated on arc peak amplitude and edge sharpness.</sub>
</div>

<br>

### Difference Maps

<br>

<div align="center">
  <img src="assets/16_2_difference_maps.png" width="92%" alt="Absolute difference maps |HR - SR| for top-3 models"/>
  <br>
  <sub><b>Figure 11.</b> Absolute difference |HR − SR| (hot colormap, bright = large error) for EDSR (patch), SwinIR, and EDSR (full) across 4 test images. Errors are spatially concentrated on the ring — not uniform noise. The background is reconstructed near-perfectly by all models. EDSR (full) achieves consistently lower mean absolute difference across all samples.</sub>
</div>

<br>

| Sample | EDSR (patch) | SwinIR | EDSR (full) |
|---|---|---|---|
| sample9312 | 0.0055 | 0.0055 | **0.0051** |
| sample6211 | 0.0056 | 0.0056 | **0.0052** |
| sample7903 | 0.0057 | 0.0057 | **0.0053** |
| sample61 | 0.0055 | 0.0054 | **0.0051** |

### Failure Cases

<br>

<div align="center">
  <img src="assets/16_3_failure_cases.png" width="92%" alt="Failure cases where EDSR (full) underperforms Lanczos on SSIM"/>
  <br>
  <sub><b>Figure 12.</b> Images where EDSR (full) underperforms Lanczos on SSIM (7/1,000 test images, 0.7%). All failure cases share the same pattern: ring amplitude underestimation at the brightest arc nodes. No systematic morphological failure mode by ellipticity or θ_E was identified.</sub>
</div>

<br>

EDSR (full) underperforms Lanczos on SSIM in **7/1,000 test images (0.7%)**. The failure rate is low and margins are small (max ΔSSIM = −0.0053), but the pattern is consistent — ring amplitude underestimation at high-contrast arc peaks. This is the primary target for future improvement via physics-informed loss terms penalising radial profile MSE directly.

---

## Scientific Discussion

**The architecture plateau.** Within patch-trained models, all architectures converge to a tight performance band (SSIM 0.9728–0.9742, PSNR 41.22–41.45 dB). The gap from bicubic to SRCNN (8,129 params) is +1.1 dB PSNR; the gap from SRCNN to SwinIR (1.41M params) is only +0.22 dB. Diminishing returns from scale are pronounced — the 2× scale factor, high SNR, and controlled PSF are not the bottleneck.

**Training procedure dominates architecture.** The largest single performance gain is from switching EDSR from patch to full-image training: +0.43 dB PSNR, +0.003 SSIM, −15% radial MSE — exceeding any architecture choice within patch-trained models. Full-image training is the recommended default for lensing SR.

**RCAN attention collapse.** Channel attention weights are input-independent (std across 12 test images = 0.0000), suggesting the attention module is not learning useful selectivity on this single-class single-channel dataset. This explains RCAN's underperformance despite 3× more parameters than EDSR.

<br>

<div align="center">
  <img src="assets/9_4_attention_weights.png" width="68%" alt="RCAN channel attention weight heatmap across 12 test images"/>
  <br>
  <sub><b>Figure 13.</b> RCAN channel attention weights across 64 channels for 12 test images. Std across images = 0.0000 for every channel — weights are completely input-independent. The channel attention module provides no selectivity benefit on this single-class single-channel dataset.</sub>
</div>

<br>

**Connection to prior DeepLense SR work.** Pranath Reddy (GSoC 2023) benchmarked EDSR and RCAN on Model-I simulated data (~2,300 pairs), achieving SSIM ~0.574 and PSNR ~30.5 dB. Atal Gupta (GSoC 2024) achieved RCAN SSIM 0.890 on 2,834 simulated pairs (noise+blur degradation). The present work achieves SSIM 0.9768 with 8,000 pairs and full-image training, establishing the simulated-data supervised ceiling for the DeepLense SR task.

---

## Limitations

1. **Single class (no substructure only).** SR models are trained exclusively on no-substructure simulations. Generalisation to CDM subhalo or axion vortex morphologies is not evaluated.

2. **Simulated Gaussian PSF only.** Real telescope images have wavelength-dependent PSFs, detector readout noise, cosmic rays, and atmospheric seeing. The LR generation model (bicubic downsampling of a simulated image) is a controlled simplification.

3. **2× scale factor only.** Performance at 4× or higher — where high-frequency recovery is substantially harder — is not characterised.

4. **RCAN attention collapse.** Channel attention provides no benefit and adds 3× parameter overhead on this single-class single-channel dataset.

5. **Ring amplitude underestimation.** All SR models systematically underestimate peak ring amplitude. This is the dominant remaining error mode and likely requires a physics-informed loss term penalising radial profile MSE directly.

---

## Bridge to Test VI.B

Test VI.A establishes the **supervised SR ceiling under idealised simulation conditions:**

| Key result | Value |
|---|---|
| Best model (full-image EDSR) | SSIM 0.97678 · PSNR 41.840 dB · Radial MSE 1.13 × 10⁻⁵ |
| Best model (patch, SwinIR) | SSIM 0.97424 · PSNR 41.445 dB · Radial MSE 1.28 × 10⁻⁵ |
| Improvement over bicubic | +64.9% radial MSE · +1.73 dB PSNR |
| Key finding | Training procedure > architecture choice |
| Dominant error mode | Ring amplitude underestimation |

Test VI.B asks: how much of this performance survives when the domain shifts from simulated Gaussian-PSF images to real HSC (LR) / HST (HR) telescope pairs, with only 300 paired examples? The `edsr_fullimage.pth` checkpoint from this task is the starting point for VI.B transfer learning — SSIM collapses from 0.977 to 0.471 zero-shot, then recovers to 0.899 via progressive fine-tuning on 240 real pairs.

→ [Test VI.B — Sim-to-Real SR Transfer](../Test_VIB/)

---

## Reproducibility

| | |
|---|---|
| Hardware | NVIDIA A100-SXM4-80GB (Google Colab) |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| Seed | 42 (fixed via `torch`, `numpy`, `cudnn.deterministic=True`) |
| Determinism check | Two independent runs produce identical outputs ✓ |

All checkpoints available on Google Drive:

**Best model checkpoints:**

| Checkpoint | Val SSIM | Size | Description | Download |
|---|---|---|---|---|
| `edsr_fullimage.pth` | **0.97680** | 16.5 MB | EDSR full-image training — best overall model | [↓ Drive](https://drive.google.com/file/d/1ls2Mxi3DjTR9zMfiRvn_nA-idpkW0fNr/view?usp=sharing) |
| `swinir_best.pth` | 0.97431 | 17.9 MB | SwinIR — best patch-trained model | [↓ Drive](https://drive.google.com/file/d/1xkL12yxieCIPESeGrATUamQeu9ZN4gmS/view?usp=sharing) |
| `edsr_best.pth` | 0.97398 | 16.5 MB | EDSR patch training | [↓ Drive](https://drive.google.com/file/d/1nJKTpZ4_cs6KgTkSyfdWaq3N4hWrPR4t/view?usp=sharing) |
| `unetsr_best.pth` | 0.97397 | 94.8 MB | UNet-SR | [↓ Drive](https://drive.google.com/file/d/1gkfzuAHEf6kXkHuoh54txW1LqHPtxdhy/view?usp=sharing) |
| `rcan_best.pth` | 0.97384 | 49.6 MB | RCAN | [↓ Drive](https://drive.google.com/file/d/1Lcx4k38FzNoMXfqAykvnce-8hCojU1Cu/view?usp=sharing) |
| `srcnn_best.pth` | 0.97289 | 0.1 MB | SRCNN | [↓ Drive](https://drive.google.com/file/d/1pExADHiBg87HFSdaPWZtERG6jT1OBe8n/view?usp=sharing) |

**Ablation checkpoints (EDSR-baseline, § 13 loss ablation):**

| Checkpoint | Val SSIM | Loss config | Download |
|---|---|---|---|
| `edsr_ablation_L1_SSIM.pth` | 0.97393 | L1 + 0.1·SSIM (default) | [↓ Drive](https://drive.google.com/file/d/1sAVHIuUArlN9bjs7KVE6aXy7UY1--qz_/view?usp=sharing) |
| `edsr_ablation_L1_SSIM_Percept.pth` | 0.97371 | L1 + 0.1·SSIM + Perceptual | [↓ Drive](https://drive.google.com/file/d/1mGI3sAV1jcV5WDT4oZFreYnOM1eiev1e/view?usp=sharing) |
| `edsr_ablation_L1.pth` | 0.97327 | L1 only | [↓ Drive](https://drive.google.com/file/d/10y20ko0heHQcGRtPDx2YhOPuJyfwNEWt/view?usp=sharing) |
| `edsr_ablation_L2.pth` | 0.97268 | L2 only | [↓ Drive](https://drive.google.com/file/d/13XEsyC5HLVZgnOpSXUc-c7uRvL1gdt1L/view?usp=sharing) |
| `swinir_ablation_L2.pth` | — | SwinIR L2 ablation | [↓ Drive](https://drive.google.com/file/d/1E_N5UxpaUYphRFGheXLUrnFhEj6-xoYb/view?usp=sharing) |

---

## References

- Lim, B., Son, S., Kim, H., Nah, S., & Lee, K. M. (2017). "Enhanced Deep Residual Networks for Single Image Super-Resolution." *CVPRW*.
- Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., & Fu, Y. (2018). "Image Super-Resolution Using Very Deep Residual Channel Attention Networks." *ECCV*.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*.
- Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., & Timofte, R. (2021). "SwinIR: Image Restoration Using Swin Transformer." *ICCVW*.
- Dong, C., Loy, C. C., He, K., & Tang, X. (2015). "Image Super-Resolution Using Deep Convolutional Networks." *TPAMI*.
- Alexander, S., Gleyzer, S., McDonough, E., Toomey, M. W., & Usai, E. (2020). "Deep Learning the Morphology of Dark Matter Substructure." *The Astrophysical Journal*.

---

<div align="center">
<sub>ML4SCI DeepLense · GSoC 2026 · Rafiqus Salehin</sub>
</div>
