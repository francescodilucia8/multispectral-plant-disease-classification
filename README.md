# Multispectral Plant Disease Classification

End-to-end machine learning pipeline for plant health classification from multispectral drone imagery.

This project was developed as part of my MSc thesis in Computer Engineering at Politecnico di Torino, within the DRONUTS project. It focuses on automatic classification of healthy vs diseased hazelnut plants using multispectral imagery, image preprocessing, vegetation indices, and CNN-based models.

## Overview

The pipeline processes raw drone acquisitions containing RGB and multispectral bands, extracts meaningful plant-centered patches, and trains deep learning models for disease classification.

Main components:
- multispectral band alignment
- NDVI-based canopy segmentation
- vegetation index computation
- patch dataset generation
- CNN and transfer learning models
- evaluation through ablation studies

## Problem

Detecting plant disease from drone imagery is challenging because of:
- multi-band misalignment
- overlapping tree canopies
- noisy background vegetation
- limited labeled data
- variability across field acquisitions

This project addresses these issues with a structured preprocessing and training pipeline.

## Data

Each plant acquisition contains:
- 1 RGB image
- 5 multispectral bands: Blue, Green, Red, Red-Edge, NIR

The dataset is organized per plant and field acquisition date. Labels are assigned at patch level.

> Note: the original dataset is private and is not included in this repository.

## Pipeline

### 1. Band alignment
Multispectral bands are aligned to improve spatial consistency before feature extraction.

Methods explored:
- phase correlation
- metadata-based alignment
- ECC refinement

### 2. Plant segmentation
The central tree canopy is segmented from the background using:
- NDVI thresholding
- morphological cleanup
- connected component analysis
- central-object scoring heuristics

### 3. Feature engineering
Vegetation indices are computed from aligned spectral bands.

Selected indices:
- GNDVI
- GCI
- NDREI
- NRI
- GI

### 4. Patch extraction
The segmented canopy region is divided into fixed-size patches for classification.

### 5. Modeling
Trained models include:
- custom CNN architectures
- transfer learning approaches based on ResNet18

### 6. Evaluation
Experiments compare:
- different preprocessing strategies
- spectral channel combinations
- vegetation index configurations
- custom CNN vs transfer learning
- threshold tuning and ablation settings

## Repository Structure

```text
src/preprocessing/     alignment, segmentation, vegetation indices
src/dataset/           patch generation and dataset utilities
src/models/            CNN and transfer learning models
src/training/          training and evaluation scripts
src/visualization/     inspection and debugging tools
assets/                figures used in README
configs/               experiment configurations
