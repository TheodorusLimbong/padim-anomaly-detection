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
| **Subset** | ✅/❌ | 5 | `experiments/run_padim_subset.py --n-train N [--no-aug]` |

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

### Data Efficiency Results (13 Jun 2026)

**Dengan Augmentasi:**

| N_TRAIN | PaDiM AUROC | KNN AUROC | Gap |
|:-------:|:-----------:|:---------:|:---:|
| 209 | 0.9976 | 0.9762 | 0.0214 |
| 180 | 0.9984 | 0.9722 | 0.0262 |
| 150 | 0.9976 | 0.9690 | 0.0286 |
| 120 | 0.9968 | 0.9595 | **0.0373** |

**Tanpa Augmentasi:**

| N_TRAIN | PaDiM AUROC | KNN AUROC | Gap |
|:-------:|:-----------:|:---------:|:---:|
| 209 | 0.9960 | 0.9857 | 0.0103 |
| 180 | 0.9960 | 0.9857 | 0.0103 |
| 150 | 0.9952 | 0.9857 | 0.0095 |
| 120 | 0.9952 | 0.9865 | 0.0087 |

**Key insights:** Data efficiency gap hanya terlihat dengan augmentasi (gap melebar 0.021→0.037). Tanpa augmentasi KNN sangat robust — feature bank yang bersih membuat Euclidean distance tetap diskriminatif meski reference berkurang.

### Dashboard (Live Inference — Updated 23 Jun 2026)

Streamlit dashboard untuk live inference — upload gambar, langsung inferensi PaDiM + KNN, anomaly maps, score, prediksi. PaDiM map dinormalisasi menggunakan global pixel min-max dari seluruh test set (bukan per-image) untuk menghindari false positive di pojok gambar normal.

```powershell
streamlit run dashboard/app.py
```

### Key Takeaways

| Insight | Detail |
|---------|--------|
| Augmentasi bantu KNN | Tanpa augmentasi, KNN naik 0.9817→0.9889 karena feature bank lebih konsisten |
| PaDiM stabil dengan/tanpa augmentasi | PaDiM hanya turun 0.0016 (0.9976→0.9960) tanpa augmentasi |
| K=5 fix melebarkan gap | AUROC gap 0.0159→0.0214 setelah K bug fix |
| Gap recall paling besar | Recall PaDiM 0.984 vs KNN 0.921 (+0.063) — PaDiM lebih sensitif |
| Data efficiency hanya valid dgn aug | Tanpa augmentasi, KNN sama robust-nya dengan PaDiM terhadap pengurangan data |
| Global normalization fix (23 Jun) | Dashboard PaDiM map ganti dari per-image ke global min-max — corner artifact hilang |

## Quick Start

```powershell
conda activate torch-venv
python experiments/run_padim.py                     # original (aug, K=5)
python experimental/v1/experiments/run_padim.py     # tanpa augmentasi (K=5)
python experimental/v2/experiments/run_padim.py     # augmentasi (K=5)
python experiments/run_padim_subset.py --n-train N   # subset (N=209,180,150,120)
python experiments/run_padim_subset.py --n-train N --no-aug  # subset tanpa aug
python experiments/generate_maps.py                  # generate anomaly maps
python experiments/compile_subset.py                 # compile results table
python experiments/generate_subset_report.py          # Word report
streamlit run dashboard/app.py                       # dashboard
```

## Project Structure

```
├── anomaly detection/      # PaDiM, KNN, inference, Mahalanobis
├── dashboard/              # Streamlit dashboard (app.py + utils.py)
├── evaluation/             # Metrics (AUROC, F1, Pixel AUROC, PRO-score)
├── experimental/           # Pipeline variants (v1: no aug, v2: K=5)
│   ├── v1/                 # Independent pipeline without augmentations
│   └── v2/                 # Independent pipeline with augmentations, K=5
├── experiments/            # End-to-end experiment runner + map generation
├── feature extractor/      # ResNet-50 + MoCo v2 backbone
├── notebooks/              # 9-section comparative visualization
├── prepocessing/           # Dataset loading, preprocessing, augmentation
├── src/                    # Config
└── output/experiments/     # Experiment results (timestamped runs + subset/)
```

## Known Issues

- Preprocessing uses `Resize(224)` instead of paper's `Resize(256)→CenterCrop(224)`
- Data augmentation active during training (not in original PaDiM paper)
- Gaussian smoothing (σ=4) only applied to PaDiM, not KNN
- Gaussian blur applied AFTER bilinear upsample (224px), not before (56px)
- Only `bottle` category configured
- PRO-score metric implemented (run_20260609_142730)
- Notebook still refers to old experiment run
- 209_aug folder has no artifacts (only metrics.json copied from original)
- K=5 does not significantly change KNN results for bottle (655K patch bank is too dense)
