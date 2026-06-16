# Dashboard: PaDiM vs KNN Anomaly Detection — Demo Sidang

## 1. Overview

Dashboard berbasis **Streamlit** untuk memvisualisasikan perbandingan metode PaDiM dan KNN pada dataset MVTec AD bottle. Dashboard ini **tidak melakukan inferensi real-time** — melainkan memuat hasil eksperimen yang sudah dijalankan dari folder `output/experiments/subset/`.

### Tujuan

- Menampilkan perbedaan visual antara PaDiM dan KNN secara jelas saat sidang
- Memperkuat narasi bahwa PaDiM unggul di berbagai aspek (akurasi, visual map, kecepatan, data efficiency)
- Menyediakan navigasi interaktif untuk membandingkan hasil per gambar maupun agregat 83 test images

---

## 2. Tech Stack

| Komponen | Library |
|----------|---------|
| Framework dashboard | Streamlit |
| Charts & plots | Matplotlib, Seaborn |
| Image processing | PIL, Torchvision |
| Data loading | Torch (load .pt files) |
| Layout | Streamlit native (columns, expander, tabs) |

---

## 3. Data Loading

### Sumber Data

Semua data di-load dari `output/experiments/subset/{n_train}_{aug|noaug}/`:

| File | Isi | Shape |
|------|-----|-------|
| `metrics.json` | Semua metrik evaluasi | dict |
| `padim_scores.pt` | PaDiM image scores [83] | [83] |
| `knn_scores.pt` | KNN image scores [83] | [83] |
| `test_features_padim.pt` | Tidak perlu di-load untuk dashboard | — |

Untuk anomaly maps dan ground truth: **tidak disimpan di folder experiment** saat ini. Perlu disimpan saat experiment runner berjalan, ATAU dibuat ulang dari scores + test features.

**Solusi rekomendasi:** Simpan anomaly maps (PaDiM + KNN) sebagai `.pt` file di folder experiment. Tambahkan 1 baris di `compute_padim_scores` / `compute_knn_scores`.

### Flow Loading

1. Scan `output/experiments/subset/` → list folder (209_aug, 209_noaug, ..., 120_noaug)
2. Untuk setiap folder: load `metrics.json`, `padim_scores.pt`, `knn_scores.pt`
3. Load test images dari `dataset/mvtec_anomaly_detection/bottle/test/`
4. Cache semua di `st.session_state` agar tidak reload setiap interaksi

---

## 4. Layout

### Struktur Halaman Tunggal

```
┌─────────────────────────────────────────────────────────────────────┐
│  HEADER: Bar judul + logo                                           │
├──────────┬──────────────────────────────────────────────────────────┤
│          │                                                           │
│  CONTROL │  TAB 1: Anomaly Map Comparison (per image)               │
│  PANEL   │  ┌──────┬──────┬──────┬──────┐                          │
│  (25%)   │  │Orig  │PaDiM │ KNN  │ GT   │                          │
│          │  │Image │ Map  │ Map  │ Mask │                          │
│          │  ├──────┴──────┴──────┴──────┤                          │
│          │  │ Score + Time + Status bar  │                          │
│          │  └──────────────────────────┘                           │
│          │                                                           │
│          │  TAB 2: All Test Evaluation                              │
│          │  ┌──────┬──────┬──────┬──────┐                          │
│          │  │ ROC  │Dist  │Conf  │Metric│                          │
│          │  │Curve │Plot  │Matrix│Table │                          │
│          │  └──────┴──────┴──────┴──────┘                          │
│          │                                                           │
│          │  TAB 3: Data Efficiency                                  │
│          │  ┌───────────────────────────────────────┐               │
│          │  │ N_TRAIN vs AUROC curve                │               │
│          │  └───────────────────────────────────────┘               │
│          │                                                           │
└──────────┴──────────────────────────────────────────────────────────┘
```

---

## 5. Control Panel (Sidebar)

### Komponen

1. **Dropdown "Experiment Run"**
   - Opsi: 209_aug, 180_aug, 150_aug, 140_aug, 120_aug, 209_noaug, ..., 120_noaug
   - Default: 209_aug (hasil terbaik)
   - Switch → update semua chart + maps

2. **Filter Image**
   - Radio button: "All (83)", "Normal Only (20)", "Anomaly Only (63)"
   - Checkbox: "Show only where KNN ❌ but PaDiM ✅"
   - Efek: menyaring gambar yang ditampilkan di Tab 1

3. **Image Navigation**
   - `[◀ Prev]` `[3 / 83]` `[Next ▶]`
   - Slider untuk lompat ke index tertentu
   - Menampilkan nama file gambar saat ini

4. **Image Info Card**
   ```
   Image: broken_large_005.png
   Label: Anomaly (broken_large)
   ──────────────────────────
   PaDiM Score: 0.9214  |  Threshold: 0.4799
   KNN Score:   1.5237  |  Threshold: 1.4786
   ──────────────────────────
   PaDiM Time: 0.23s    |  8× faster
   KNN Time:   1.87s
   ──────────────────────────
   PaDiM: ✅ True Positive
   KNN:   ❌ False Negative
   ```

---

## 6. Tab 1: Anomaly Map Comparison

### Layout 4 Panel

```
┌──────────────┬──────────────┐
│  ORIGINAL    │  PaDiM MAP   │
│  IMAGE       │  (jet cmap)  │
│  (RGB, 224)  │  smooth σ=4  │
├──────────────┼──────────────┤
│  KNN MAP     │  GROUND      │
│  (jet cmap)  │  TRUTH       │
│  tanpa smooth│  (binary)    │
└──────────────┴──────────────┘
```

### Color Scheme

- PaDiM map: `jet` colormap — biru (normal) → merah (anomali)
- KNN map: `jet` colormap — biru (normal) → merah (anomali) — terlihat lebih kasar
- GT mask: hitam (normal) — putih (anomali)

### Overlay (Opsional)

Checkbox: "Show heatmap overlay on original"
- Overlay anomaly map transparan (alpha=0.4) di atas original image
- Membantu dosen melihat lokasi defect yang terdeteksi

### Key Visual

PaDiM map tampak **mulus dan fokus** karena Gaussian smoothing σ=4. KNN map tampak **noisy dan granular** karena tanpa smoothing. Gap ini langsung terlihat tanpa perlu lihat angka.

---

## 7. Tab 2: All Test Evaluation

### Sub-layout 2×2 grid

```
┌──────────────────────┬──────────────────────┐
│  ROC CURVE           │  SCORE DISTRIBUTION  │
│                      │                      │
│  - 2 garis: PaDiM    │  - Boxplot split by  │
│    (biru), KNN (merah)│    method & label    │
│  - AUC value di       │  - 4 box:            │
│    legend             │    PaDiM Normal      │
│  - Diagonal putus-2   │    PaDiM Anomaly     │
│    sebagai baseline   │    KNN Normal        │
│                      │    KNN Anomaly        │
├──────────────────────┼──────────────────────┤
│  CONFUSION MATRIX    │  METRICS TABLE       │
│                      │                      │
│  - 2×2 grid          │  Metric    PaDiM  KNN│
│  - Value + color     │  AUROC    0.998 0.976│
│  - Dibandingkan      │  F1       0.984 0.951│
│    PaDiM vs KNN      │  Recall   0.984 0.921│
│    side-by-side      │  Pixel    0.984 0.976│
│                      │  PRO      0.953 0.939│
│                      │  Time/s   0.23  1.87 │
└──────────────────────┴──────────────────────┘
```

### Detail Chart

**ROC Curve:**
- Sumbu X: False Positive Rate (0–1)
- Sumbu Y: True Positive Rate (0–1)
- PaDiM: garis biru solid, tebal
- KNN: garis merah dashed, tebal
- Legend: "PaDiM (AUROC = 0.9976)", "KNN (AUROC = 0.9762)"
- Baseline diagonal: garis abu-abu putus-putus

**Score Distribution:**
- Boxplot dengan 4 grup
- Warna: biru (PaDiM), merah (KNN)
- Split: Normal vs Anomaly
- Terlihat jelas: PaDiM memiliki **gap lebih lebar** antara normal dan anomaly

---

## 8. Tab 3: Data Efficiency

### Line Chart

```
AUROC
 1.00 ┼     ●────●────●────●────●  PaDiM
      │    ╱
 0.98 ┼   ╱
      │  ╱                          ▲ KNN
 0.96 ┼ ╱
      │╱
 0.94 ┼
      │
 0.92 ┼
      │
 0.90 ┼─────────────────────────────────
      120    150    180    209
                    N_TRAIN
```

- 2 garis: PaDiM (biru, stabil), KNN (merah, menurun)
- 2 panel: Augmentasi (kiri), Tanpa Augmentasi (kanan)
- Anotasi nilai gap pada setiap titik N_TRAIN

### Tabel Pendukung

| N_TRAIN | Aug | PaDiM | KNN | Gap | Δ Gap |
|:-------:|:---:|:-----:|:---:|:---:|:-----:|
| 209 | Ya | 0.9976 | 0.9762 | 0.0214 | — |
| 180 | Ya | 0.9984 | 0.9722 | 0.0262 | +22% |
| 150 | Ya | 0.9976 | 0.9690 | 0.0286 | +34% |
| 120 | Ya | 0.9968 | 0.9595 | **0.0373** | **+74%** |

---

## 9. Error Analysis Feature

### Filter "KNN ❌ but PaDiM ✅"

Button di sidebar yang menyaring gambar-gambar di mana:
- KNN salah mengklasifikasikan (threshold-based)
- PaDiM benar mengklasifikasikan

**Tujuannya:** Menunjukkan secara langsung kasus-kasus di mana PaDiM unggul. Ini adalah fitur paling kuat untuk presentasi sidang.

### Implementasi

```python
def find_padim_wins(padim_scores, knn_scores, labels, padim_thresh, knn_thresh):
    padim_correct = ((padim_scores >= padim_thresh) == (labels == 1))
    knn_correct = ((knn_scores >= knn_thresh) == (labels == 1))
    return padim_correct & ~knn_correct  # PaDiM benar, KNN salah
```

Filter serupa: "Both Correct", "Both Wrong", "PaDiM ❌ KNN ✅"

---

## 10. Inference Time Comparison

Ditampilkan di beberapa tempat:
- **Tab 1** (per image): waktu inference di info card
- **Tab 2** (metrics table): rata-rata waktu per image
- **Sidebar**: total waktu untuk 83 images

**Visual:** Horizontal bar chart:
```
PaDiM:  ████████░░ 19.3s (0.23s/img)
KNN:    ██████████████████████░░ 155s (1.87s/img)
Speedup: 8×
```

---

## 11. File yang Perlu Dimodifikasi

| File | Perubahan |
|------|-----------|
| `experiments/run_padim_subset.py` | Tambah save anomaly maps (`padim_maps.pt`, `knn_maps.pt`) |
| `experiments/run_padim.py` | Sama — tambah save anomaly maps |
| `inference.py` | Return maps juga sudah dilakukan — tapi perlu disimpan ke file |
| **Baru:** `dashboard/app.py` | Main Streamlit app |
| **Baru:** `dashboard/utils.py` | Helper functions (load experiments, plots) |

---

## 12. Urutan Build

1. **Update experiment runner** — simpan anomaly maps ke file `.pt`
2. **Re-run subset experiments** — generate maps untuk semua 9 folder
3. **Build `dashboard/app.py`** — layout, sidebar, tabs
4. **Build `dashboard/utils.py`** — load data, plot functions
5. **Integrasi & test** — navigasi antar gambar, tab switching
6. **Polish** — styling, error handling, responsive layout
7. **Test dengan data real** — pastikan semua 83 images tampil benar

---

## 13. Visual Style Guide

### Warna

| Elemen | Warna | Hex |
|--------|-------|-----|
| PaDiM | Biru | #1f77b4 |
| KNN | Merah | #d62728 |
| Background sidebar | Abu muda | #f0f2f6 |
| Correct | Hijau | #2ca02c |
| Wrong | Merah | #d62728 |
| Colormap anomaly | Jet | — |

### Font

- Default Streamlit (sans-serif)
- Judul tab: bold 18pt
- Info card: monospace 12pt
- Axis labels: 12pt

### Layout Constraint

- Minimum lebar: 1200px (desktop), fallback scroll untuk layar kecil
- Aspect ratio anomaly maps: 1:1 (224×224)
- Grid: 2×2 untuk tab evaluation, 2×2 untuk anomaly maps
