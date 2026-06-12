# Self-Supervised Anomaly Detection — PaDiM vs KNN

Academic implementation of **PaDiM** (Patch Distribution Modeling Framework) anomaly detection on **MVTec AD (bottle)** using **ResNet-50 + MoCo v2** backbone, compared against **KNN Euclidean** baseline.

## Pipeline

Preprocessing → Feature Extraction (3 layers) → Dim Reduction 1792→550 → PaDiM Statistics → Mahalanobis Distance → Evaluation

## Pipelines

| Variant | Augmentasi | KNN K | Runner |
|---------|:----------:|:-----:|--------|
| **Original** | ✅ Ya | 5 | `experiments/run_padim.py` |
| **experimental/v1** | ❌ Tidak | 5 | `experimental/v1/experiments/run_padim.py` |
| **experimental/v2** | ✅ Ya | 5 | `experimental/v2/experiments/run_padim.py` |

## Results

### Comparison Matrix

| Metric | **Original** (aug, K=5 fix) | **v1** (no aug, K=1) | **v2** (aug, K=5 broken) |
|--------|:---------------------------:|:--------------------:|:-------------------------:|
| **PaDiM AUROC** | **0.9976** | 0.9960 | **0.9976** |
| **KNN AUROC** | 0.9762 | **0.9889** | 0.9817 |
| **Gap PaDiM-KNN** | **0.0214** 🏆 | 0.0071 | 0.0159 |
| **PaDiM Pixel AUROC** | 0.9838 | **0.9854** | 0.9838 |
| **KNN Pixel AUROC** | 0.9755 | **0.9814** | 0.9772 |
| **PaDiM PRO-score** | 0.9525 | **0.9553** | 0.9525 |
| **KNN PRO-score** | 0.9386 | **0.9509** | 0.9421 |
| **PaDiM F1** | 0.9841 | 0.9760 | 0.9841 |
| **KNN F1** | 0.9508 | **0.9760** | 0.9421 |

### Key Takeaways

| Insight | Detail |
|---------|--------|
| Augmentasi bantu KNN | Tanpa augmentasi, KNN naik 0.9817→0.9889 karena feature bank lebih konsisten |
| PaDiM stabil dengan/tanpa augmentasi | PaDiM hanya turun 0.0016 (0.9976→0.9960) tanpa augmentasi |
| K=5 fix melebarkan gap | AUROC gap 0.0159→0.0214 setelah K bug fix |
| Gap recall paling besar | Recall PaDiM 0.984 vs KNN 0.921 (+0.063) — PaDiM lebih sensitif |

## Quick Start

```powershell
conda activate torch-venv
python experiments/run_padim.py                     # original (aug, K=5)
python experimental/v1/experiments/run_padim.py     # tanpa augmentasi (K=5)
python experimental/v2/experiments/run_padim.py     # augmentasi (K=5)
```

## Project Structure

```
├── anomaly detection/      # PaDiM, KNN, inference, Mahalanobis
├── evaluation/             # Metrics (AUROC, F1, Pixel AUROC, PRO-score)
├── experimental/           # Pipeline variants (v1: no aug, v2: K=5)
│   ├── v1/                 # Independent pipeline without augmentations
│   └── v2/                 # Independent pipeline with augmentations, K=5
├── experiments/            # End-to-end experiment runner
├── feature extractor/      # ResNet-50 + MoCo v2 backbone
├── notebooks/              # 9-section comparative visualization
├── prepocessing/           # Dataset loading, preprocessing, augmentation
├── src/                    # Config
└── output/experiments/     # Experiment results (timestamped)
```

## Known Issues

- Preprocessing uses `Resize(224)` instead of paper's `Resize(256)→CenterCrop(224)`
- Data augmentation active during training (not in original PaDiM paper)
- Gaussian smoothing (σ=4) only applied to PaDiM, not KNN
- Only `bottle` category configured
- PRO-score metric implemented (run_20260609_142730)
- Notebook still refers to old experiment run
- Dashboard (Fase 3) and Analysis (Fase 4) not started
- K=5 does not significantly change KNN results for bottle (655K patch bank is too dense)
