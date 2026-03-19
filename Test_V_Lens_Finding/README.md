Section 1  — EDA
  1.1  Dataset statistics: class counts, imbalance ratio
  1.2  Sample visualisation: lenses vs non-lenses per band (g/r/i)
  1.3  Pixel intensity distributions per class per channel
  1.4  Einstein ring detectability analysis

Section 2  — Data pipeline
  2.1  Custom Dataset class for 3-channel .npy loading
  2.2  90:10 stratified train-val split (stratified = critical for imbalance)
  2.3  Weighted random sampler
  2.4  D₄ augmentation pipeline

Section 3  — Baseline: EfficientNet-B2
  3.1  Architecture: 3-channel input, binary head
  3.2  Focal loss implementation
  3.3  Training with early stopping
  3.4  ROC curve + AUC on val set

Section 4  — Extended benchmark
  4.1  ResNet-34
  4.2  DenseNet-121
  4.3  Comparison table

Section 5 — Novel equivariant architectures
  5.1  E-ResNet D₄ (3-channel adaptation)
  5.2  EqDenseNet-C8 (3-channel adaptation — our Test I architecture)
  5.3  Spatial dimension analysis for 64×64 input
  5.4  Equivariance verification on lens images
  5.5  Parameter efficiency comparison vs pretrained models

Section 6  — Ensemble
  6.1  Soft vote: top-3 models
  6.2  ROC curve comparison: individual vs ensemble

Section 7  — Analysis
  7.1  Threshold calibration: Youden's J
  7.2  Precision-Recall curve + PR-AUC (alongside required ROC-AUC)
  7.3  Failure mode: what do misclassified lenses look like?
  7.4  Per-channel importance (g vs r vs i)
