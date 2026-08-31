# Self-Supervised Anomaly Detection: PaDiM vs KNN

Academic implementation of **PaDiM** (Patch Distribution Modeling) anomaly detection on **MVTec AD (bottle)** using a **ResNet-50 + MoCo v2** backbone, compared against a **KNN Euclidean** baseline.

## Overview

This project implements and compares two anomaly detection approaches for industrial bottle inspection:

- **SSL + PaDiM**: Models per-patch Gaussian distributions and uses Mahalanobis distance for anomaly scoring
- **SSL + KNN**: Uses Euclidean distance to K nearest neighbors in a learned feature bank

Both methods share the same self-supervised feature extractor (MoCo v2 with frozen ResNet-50 backbone) and operate in a one-class learning paradigm (normal data only).

## Pipeline

```
Preprocessing -> Feature Extraction (3 layers) -> Dimensionality Reduction (1792 -> 550)
-> PaDiM Statistics / KNN Feature Bank -> Anomaly Scoring -> Evaluation
```

## Project Structure

```
anomaly detection/     PaDiM, KNN, inference, Mahalanobis distance
dashboard/             Streamlit dashboard for live inference
evaluation/            Metrics (AUROC, F1, Pixel AUROC, PRO-score)
experimental/          Pipeline variants (v1: no augmentation, v2: with augmentation)
experiments/           End-to-end experiment runners and map generation
feature extractor/     ResNet-50 + MoCo v2 backbone
notebooks/             9-section comparative visualization
prepocessing/          Dataset loading, preprocessing, augmentation
scripts/               Benchmark timing, figure generation
src/                   Configuration
docs/                  Architecture diagrams (draw.io)
output/                Experiment results (timestamped runs)
```

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| IMAGE_SIZE | 224 | ResNet-50 input resolution |
| BATCH_SIZE | 16 | Batch size for dataloader |
| SELECTED_LAYERS | layer1, layer2, layer3 | Multi-layer feature concatenation |
| PADIM_N_DIMS | 550 | Reduced dimensionality (PaDiM optimal) |
| KNN_K | 5 | Number of nearest neighbors |
| GAUSS_SIGMA | 4 | Gaussian smoothing sigma |
| SEED | 1024 | Random seed for reproducibility |

## Quick Start

```bash
conda activate torch-venv

# Run full experiment
python experiments/run_padim.py

# Run with subset of training data
python experiments/run_padim_subset.py --n-train 120
python experiments/run_padim_subset.py --n-train 120 --no-aug

# Launch dashboard
streamlit run dashboard/app.py
```

## Results

All results are from experiment `run_20260612_152608` (K=5, with augmentation).

### Image-Level Metrics

| Metric | PaDiM | KNN | Gap |
|--------|:-----:|:---:|:---:|
| AUROC | 0.9976 | 0.9762 | 0.0214 |
| F1-Score | 0.9841 | 0.9508 | 0.0333 |
| Recall | 0.9841 | 0.9206 | 0.0635 |

### Pixel-Level Metrics

| Metric | PaDiM | KNN | Gap |
|--------|:-----:|:---:|:---:|
| Pixel AUROC | 0.9838 | 0.9755 | 0.0083 |
| PRO-score | 0.9525 | 0.9386 | 0.0139 |

### Inference Time

| Device | PaDiM | KNN | Speedup |
|--------|:-----:|:---:|:-------:|
| GPU (RTX 3060) | 0.016 s | 4.837 s | 302x |
| CPU (Ryzen 7 5800H) | 0.501 s | 20.056 s | 40x |

### Data Efficiency

Training with reduced data (augmentation enabled):

| N_TRAIN | PaDiM AUROC | KNN AUROC | Gap |
|:-------:|:-----------:|:---------:|:---:|
| 209 | 0.9976 | 0.9762 | 0.0214 |
| 180 | 0.9984 | 0.9722 | 0.0262 |
| 150 | 0.9976 | 0.9690 | 0.0286 |
| 120 | 0.9968 | 0.9595 | 0.0373 |

## Dashboard

Streamlit-based live inference dashboard supporting:

- Multi-image upload with sequential processing
- PaDiM anomaly maps with Gaussian smoothing and global normalization
- KNN anomaly maps with chunked Euclidean distance computation
- OOD detection via ComboOOD (Mahalanobis + KNN, threshold = 0.0)
- Contour plot visualization (PCA 550 -> 2D)
- Prediction labels (ANOMALY / NORMAL) based on P95 threshold

## Architecture Diagrams

- `docs/arsitektur_pelatihan.drawio` -- Training phase architecture
- `docs/arsitektur_inferensi.drawio` -- Inference phase architecture with OOD detection

## Dataset

MVTec AD bottle category: 209 training images (normal only), 83 test images across 4 classes (good, broken large, broken small, contamination).
