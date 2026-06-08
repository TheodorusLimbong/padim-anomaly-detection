# Self-Supervised Anomaly Detection — PaDiM vs KNN

Academic implementation of **PaDiM** (Patch Distribution Modeling Framework) anomaly detection on **MVTec AD (bottle)** using **ResNet-50 + MoCo v2** backbone, compared against **KNN (1-NN Euclidean)** baseline.

## Pipeline

Preprocessing → Feature Extraction (3 layers) → Dim Reduction 1792→550 → PaDiM Statistics → Mahalanobis Distance → Evaluation

## Results (run_20260608_145608 — Fair Comparison)

Both methods use **550 dimensions** (random channel selection, same indices).

| Metric | PaDiM | KNN |
|--------|:-----:|:---:|
| AUROC | **0.9976** | 0.9817 |
| F1 | **0.9841** | 0.9421 |
| Pixel AUROC | **0.9838** | 0.9772 |
| Recall | **0.9841** | 0.9048 |
| Inference (83 img) | **19.3s** | 155s |

## Quick Start

```powershell
conda activate torch-venv
python experiments/run_padim.py
```

## Project Structure

```
├── anomaly detection/      # PaDiM, KNN, inference, Mahalanobis
├── evaluation/             # Metrics (AUROC, F1, Pixel AUROC)
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
- Only `bottle` category configured
- PRO-score metric not yet implemented
- Dashboard (Fase 3) and Analysis (Fase 4) not started
