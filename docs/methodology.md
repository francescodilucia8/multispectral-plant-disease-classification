# Methodology

This document describes the methodology behind the multispectral plant disease classification pipeline developed for the MSc thesis within the DRONUTS project.

The pipeline is designed to process raw drone acquisitions and produce a structured dataset for supervised learning, followed by model training and evaluation.

---

## 1. Problem Setting

The task is binary classification of plant health (healthy vs diseased) using multispectral imagery captured by drone-mounted sensors.

### Challenges:
- Misalignment between spectral bands
- Overlapping tree canopies
- Background noise (soil, grass)
- Limited and patch-level labels
- Variability across field acquisitions

The pipeline addresses these through a modular preprocessing and training workflow.

---

## 2. Data Representation

Each plant sample consists of:
- 1 RGB image
- 5 multispectral bands:
  - Blue
  - Green
  - Red
  - Red-Edge
  - NIR

Images are stored per plant and grouped by acquisition date. Labels are provided at patch level via external annotation files.

---

## 3. Pipeline Overview

The pipeline is composed of the following stages:

1. Band alignment  
2. Vegetation segmentation  
3. Feature engineering (vegetation indices)  
4. Patch extraction  
5. Dataset construction  
6. Model training and evaluation  

Each stage is designed to reduce noise and improve signal quality for classification.

---

## 4. Band Alignment

### Objective:
Correct spatial misalignment between multispectral bands before feature computation.

### Methods:
- **Phase Correlation** (OpenCV)
  - Applied on central ROI to estimate translation
  - Hanning window used to reduce edge artifacts
  - Alignment accepted based on correlation response threshold

- **Metadata-based alignment**
  - Uses Relative Optical Center offsets from image metadata
  - Aligns all bands to NIR reference

- **ECC Refinement**
  - OpenCV `findTransformECC` for translation-only refinement
  - Improves alignment robustness after initial shift

### Output:
Aligned multispectral stack with consistent pixel correspondence across bands.

---

## 5. Vegetation Segmentation

### Objective:
Isolate the central plant canopy from background.

### Approach:
- Compute **NDVI**:
  
  NDVI = (NIR - RED) / (NIR + RED)

- Apply thresholding to obtain vegetation mask
- Morphological operations:
  - `remove_small_objects`
  - `binary_opening`
  - `binary_closing`
  - `binary_fill_holes`

### Blob Selection:
- Connected components extracted using `regionprops`
- Target canopy selected based on:
  - Distance from image center
  - Area constraints
  - Shape metrics (e.g., solidity)

### Challenges:
- Overlapping canopies (merged blobs)
- NDVI leakage into soil/grass
- Variable plant sizes

---

## 6. Feature Engineering

Vegetation indices are computed to enhance spectral separability.

### Selected indices:
- **GNDVI** = (NIR - GREEN) / (NIR + GREEN)
- **GCI** = (NIR / GREEN) - 1
- **NDREI** = (NIR - RED_EDGE) / (NIR + RED_EDGE)
- **NRI** = (GREEN - RED) / (GREEN + RED)
- **GI** = GREEN / RED

### Notes:
- Percentile-based normalization applied before index computation
- Clipping used to stabilize extreme values

---

## 7. Patch Extraction

### Objective:
Convert each plant into a set of fixed-size samples for training.

### Approach:
- Extract bounding box from segmentation mask
- Divide region into grid patches (e.g., 224×224)
- Resize to fixed resolution

### Variants:
- Masked patches (background removed)
- Unmasked patches (baseline comparison)
- Central crop baseline (segmentation-free)

---

## 8. Dataset Construction

Each patch is stored as a `.npz` file containing:

- RGB channels (3)
- Multispectral bands (5)
- Vegetation indices (5)
- Optional mask channel (1)

### Total channels:
Up to 14 channels depending on configuration.

### Dataset index:
A CSV file tracks:
- plant_id
- patch_id
- label
- file path
- metadata

---

## 9. Model Architecture

### Baseline:
Custom CNN with:
- Conv → BatchNorm → ReLU → MaxPool blocks
- Adaptive average pooling
- Fully connected classification head

### Transfer Learning:
- ResNet18 backbone
- First layer adapted for multi-channel input
- Partial fine-tuning strategies:
  - head-only training
  - partial unfreezing

---

## 10. Training Strategy

### Loss:
- Cross-entropy with class weighting

### Hyperparameters:
- Batch size: 16–128
- Learning rate: ~1e-3
- Weight decay: 1e-4

### Data splitting:
- Group-aware split (by plant or field)
- Train / validation / test separation

---

## 11. Evaluation

Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Additional techniques:
- Threshold tuning (ROC / PR analysis)
- Ablation studies:
  - channel combinations
  - preprocessing variants
  - model architectures

---

## 12. Key Insights

- Vegetation indices significantly improve performance compared to raw multispectral bands alone
- Proper band alignment is critical for stable NDVI computation
- Segmentation quality strongly impacts downstream classification
- Transfer learning improves performance but requires careful channel adaptation

---

## 13. Limitations

- Dataset size is relatively small
- Label granularity limited to patch-level
- Segmentation errors propagate to classification
- Variability across fields affects generalization

---

## 14. Future Work

- End-to-end models integrating segmentation and classification
- Larger and more diverse dataset

---