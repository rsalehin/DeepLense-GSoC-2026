## 1. Exploratory Data Analysis (EDA)

### 1.1 Dataset Statistics
- Class counts  
- Imbalance ratio  

### 1.2 Sample Visualization
- Lenses vs non-lenses  
- Visualized per band: **g / r / i**

### 1.3 Pixel Intensity Distributions
- Per class  
- Per channel (g, r, i)

### 1.4 Einstein Ring Detectability Analysis
- Visual and statistical assessment of ring visibility  
- Impact of noise and resolution  

---

## 2. Data Pipeline

### 2.1 Custom Dataset
- Loading **3-channel `.npy` images**  
- Channel structure: *(g, r, i)*  

### 2.2 Train–Validation Split
- **90:10 stratified split**  
- Stratification ensures class balance  

### 2.3 Sampling Strategy
- **Weighted Random Sampler**  
- Addresses class imbalance  

### 2.4 Data Augmentation
- **D₄ group transformations**:
  - Rotations: 0°, 90°, 180°, 270°  
  - Reflections  

---

## 3. Baseline Model: EfficientNet-B2

### 3.1 Architecture
- 3-channel input  
- Binary classification head  

### 3.2 Loss Function
- **Focal Loss** (for imbalance handling)

### 3.3 Training Strategy
- Early stopping  
- Learning rate scheduling  

### 3.4 Evaluation
- ROC curve  
- AUC score on validation set  

---

## 4. Extended Benchmark

### 4.1 ResNet-34
- Adapted for 3-channel input  

### 4.2 DenseNet-121
- Pretrained backbone with modified classifier  

### 4.3 Comparative Results
- Performance comparison table:
  - Accuracy  
  - ROC-AUC  
  - PR-AUC  

---

## 5. Novel Equivariant Architectures

### 5.1 E-ResNet (D₄ Equivariance)
- Rotation and reflection equivariant model  
- Adapted for 3-channel input  

### 5.2 EqDenseNet-C8
- Cyclic group (C₈) equivariance  
- Custom architecture developed for Test I  

### 5.3 Spatial Dimension Analysis
- Input resolution: **64 × 64**  
- Feature map scaling and resolution constraints  

### 5.4 Equivariance Verification
- Empirical validation on lens images  
- Consistency under transformations  

### 5.5 Parameter Efficiency
- Comparison with pretrained CNNs  
- Trade-off: performance vs parameter count  

---

## 6. Ensemble Model

### 6.1 Soft Voting Ensemble
- Top-3 performing models combined  

### 6.2 Performance Comparison
- ROC curves:  
  - Individual models vs ensemble  

---

## 7. Analysis

### 7.1 Threshold Calibration
- **Youden’s J statistic** for optimal threshold selection  

### 7.2 Precision–Recall Analysis
- PR curve  
- PR-AUC  
- Complement to ROC-AUC  

### 7.3 Failure Mode Analysis
- Visual inspection of misclassified lenses  
- Identification of challenging cases  

### 7.4 Channel Importance
- Comparative contribution of:
  - g-band  
  - r-band  
  - i-band  
