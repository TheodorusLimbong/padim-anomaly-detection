# Self-Supervised Anomaly Detection — PaDiM vs KNN

Academic implementation of **PaDiM** (Patch Distribution Modeling Framework) anomaly detection on **MVTec AD (bottle)** using **ResNet-50 + MoCo v2** backbone, compared against **KNN Euclidean** baseline.

## Inline Code Documentation (16 Jul 2026)

Semua file penting sudah ditambahkan komentar penjelasan (cara jalankan + penjelasan fungsi). Lihat `AGENTS.md` untuk detail mapping diagram arsitektur → file kode.

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

### Dashboard (Live Inference — Updated 24 Jun 2026)

Streamlit dashboard untuk live inference — upload gambar, langsung inferensi PaDiM + KNN, anomaly maps, score, prediksi. PaDiM map dinormalisasi menggunakan global pixel min-max dari seluruh test set (bukan per-image) untuk menghindari false positive di pojok gambar normal.

**Optimasi CPU (einsum→bmm):** PaDiM ~0.4-0.8s on CPU (sebelumnya ~3-8s). KNN ~45-50s dengan pre-computed bank norms + chunk 30K. Sequential display: PaDiM tampil duluan, KNN menyusul.

**GAUSS_SIGMA config:** Disimpan di `src/config.py`, experiment, dan dashboard — konsisten antar run. Feature bank tetap di CPU (peak GPU ~3.8 GB).

**Multi-image upload:** Support upload banyak gambar sekaligus, proses satu persatu (sequential), hasil stack vertikal.

**EVT/POT Threshold (branch terpisah):** Extreme Value Theory untuk threshold estimation — lebih robust dari percentile 95. Fit GPD ke tail distribusi, set threshold untuk desired FPR 1%.

```powershell
streamlit run dashboard/app.py
```

### Proposal Revision Status (7 Jul 2026)

**7 perubahan BAB III — TERVERIFIKASI:** Dimensionality Reduction (baru), renumber 3.4.7→3.4.12, hapus cross-reference, Feature Bank [655.424,550], Mean+Cov_inv, Global min-max, K=5.

**5 poin koreksi dosen:**
| Poin | Status |
|------|:------:|
| 1. Dataset rinci (sumber, jumlah, variabel) | ✅ 70% — perlu sub-bab Variabel Penelitian |
| 2. Metodologi terstruktur | ✅ 90% — minor duplicate numbering 3.1 |
| 3. Latar belakang diperkuat | ✅ 95% — terkuat (data 70-90% human error) |
| 4. Arsitektur detail + diagram | ⚠️ 60% — perlu diagram input→proses→output |
| 5. Rumusan masalah spesifik & selaras | ✅ 85% — 1:1 aligned, minor verb RM1 |

**Sisa BAB IV:** 6 placeholder (Tabel 4.X–4.V, Gambar 4.Y–4.10) belum diisi.

### BAB 4 & 5 Progress (7 Jul 2026)

| Sub-bab | Status | Konten |
|---------|:------:|--------|
| 4.1 Karakteristik Dataset | ✅ | Dataset distribution barchart |
| 4.2 Hasil Preprocessing | ✅ | Resize & augmentation panels |
| 4.3 Hasil Ekstraksi Fitur | ✅ | Feature maps + dim reduction |
| **4.4 Hasil Deteksi Anomali** | **✅** | **∼1.400 kata, 5 tabel, 3 gambar** |
| **4.5 Evaluasi Kinerja Deteksi** | **✅** | **∼1.500 kata, 5 tabel, 4 gambar** |
| **4.6 Analisis Komparatif** | **✅** | **4.6.1 Gaussian smoothing ✅, 4.6.2 Timing benchmark ✅** |
| 4.7 Visualisasi Dashboard | ⏳ | Belum |
| **5.1 Kesimpulan** | **✅** | **3 paragraf, jawab 5 rumusan masalah** |
| **5.2 Saran** | **✅** | **4 poin: ekspansi, SOTA, dataset riil, dashboard** |

Semua gambar di `output/figures/` (13 file total).

### Changes (22 Jul 2026)

| Change | Detail |
|--------|--------|
| OOD threshold | P99.9 66.18 → fixed 0.0 (gap non-bottle -49 vs bottle +0.2) |
| OOD UI | `st.warning()` full width + image col1, tanpa score/caption/emoji |
| Hapus antrian | Tombol reset widget uploader via dynamic key |
| Diagram blueprint | OOD block ditambahkan antara Embedding Processing dan SPLIT |
| Narasi BAB 3 | 3.6.5 Deteksi Out-of-Distribution (ComboOOD) ✅ |
| Narasi BAB 4 | 4.7.3 Deteksi Out-of-Distribution (ComboOOD) ✅ |
| Batasan masalah | Poin 4: "tidak mencakup bottle di luar MVTec AD" |

### Key Takeaways

| Insight | Detail |
|---------|--------|
| Augmentasi bantu KNN | Tanpa augmentasi, KNN naik 0.9817→0.9889 karena feature bank lebih konsisten |
| PaDiM stabil dengan/tanpa augmentasi | PaDiM hanya turun 0.0016 (0.9976→0.9960) tanpa augmentasi |
| K=5 fix melebarkan gap | AUROC gap 0.0159→0.0214 setelah K bug fix |
| Gap recall paling besar | Recall PaDiM 0.984 vs KNN 0.921 (+0.063) — PaDiM lebih sensitif |
| Data efficiency hanya valid dgn aug | Tanpa augmentasi, KNN sama robust-nya dengan PaDiM terhadap pengurangan data |
| Global normalization fix (23 Jun) | Dashboard PaDiM map ganti dari per-image ke global min-max — corner artifact hilang |
| Score normalization fix (24 Jun) | Dashboard PaDiM score normalize pakai pixel-level global min-max — threshold cocok dengan experiment |
| GAUSS_SIGMA config (24 Jun) | Sigma disimpan di config.json, dibaca dashboard — konsisten antara experiment dan dashboard |
| CPU optimization (24 Jun) | einsum→bmm (PaDiM ~0.4-0.8s CPU), precompute bank norms, KNN chunk 30K, sequential display |
| **Timing benchmark (1 Jul)** | **PaDiM GPU 0,016s, KNN GPU 4,837s (302×) — script dashboard-accurate `scripts/benchmark_timing.py`** |

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

## Architecture Blueprint (25 Jun 2026)

`diagram_layout_blueprint.txt` berisi layout diagram arsitektur untuk revisi proposal BAB III. Diagram mencakup 7 fase: Mulai → Data Collection → Pre-processing → Feature Extraction → Split (KNN vs PaDiM) → Perbandingan Kinerja → Selesai.

### Koreksi dari diagram asli (draw.io XML analysis):

| Koreksi | Detail | Ref Kode |
|---------|--------|:--------:|
| Dim Reduction → Reshape | Dim reduction di 4D `[N,1792,56,56]` dulu, baru reshape ke patch `[N,3136,550]` | `run_padim.py:115-128` |
| Upsample → Gaussian | Upsample 56→224 dulu, baru blur σ=4 (known bug, blm difix) | `inference.py:62-64` |
| KNN tanpa smoothing | Tidak ada Gaussian blur di KNN sama sekali | `inference.py:99-111` |
| Augmentasi default ON | Hanya v1 yang non-aug; diagram asli tulis [OPSIONAL] | `build_transform.py:9-13` |

### Rekomendasi revisi teks proposal BAB III:
1. Layer1 [256,56×56] + Layer2 [512,28×28→56×56] + Layer3 [1024,14×14→56×56] → [1792,56×56]
2. Dim reduction 1792→550 channel random select
3. Perbaiki dimensi patch dari `[N,49,2048]` → `[N,3136,550]`
4. Gaussian smoothing σ=4 setelah upsample (catat sebagai known issue)
5. Global min-max normalization pixel-level

### Figures (`output/figures/`)

| File | Untuk | Keterangan | Status |
|------|:-----:|:-----------|:------:|
| `distribusi_dataset_barchart.png` | 4.1 | Bar chart train vs test | ⏳ Belum dibuat |
| `hasil_preprocessing.png` | 4.2.1 | Original → Resize → Normalized | ⏳ Belum dibuat |
| `hasil_augmentasi.png` | 4.2.2 | 5 panel augmentasi (3+2 layout) | ✅ 13 Jul 2026 |
| `feature_maps_table.png` | 4.3.1-4.3.2 | Multi-layer feature maps | ⏳ Belum dibuat |
| `feature_maps_multi_layer.png` | 4.3.1-4.3.2 | Detail per-channel | ⏳ Belum dibuat |
| `dim_reduction_illustration.png` | 4.3.3 | 1792→550 selection | ⏳ Belum dibuat |
| `distribusi_skor_padim.png` | 4.4.1 | Barchart distribusi PaDiM | ⏳ Belum dibuat |
| `distribusi_skor_knn.png` | 4.4.2 | Barchart distribusi KNN | ⏳ Belum dibuat |
| `boxplot_perbandingan.png` | 4.4.3 | Boxplot 4 kelas × 2 metode | ⏳ Belum dibuat |
| `roc_curve_imagelevel.png` | 4.5.1 | ROC Curve PaDiM vs KNN | ⏳ Belum dibuat |
| `anomaly_maps_comparison.png` | 4.5.2 | 4×4 grid anomaly maps | ⏳ Belum dibuat |
| `threshold_tradeoff.png` | 4.5.3 | TPR vs FPR trade-off P90–P99 | ⏳ Belum dibuat |
| `timing_barchart.png` | 4.6.2 | 2-panel barchart PaDiM vs KNN GPU/CPU | ⏳ Belum dibuat |
| `lampiran_dataset.png` | Lampiran | 2×5 grid sampel dataset MVTec AD bottle | ⏳ Belum dibuat |

### Augmentation Figure (13 Jul 2026)

`scripts/gen_augmentation_figure.py` — generate `hasil_augmentasi.png` dengan layout 3 baris atas + 2 baris bawah:
- (a) Original, (b) RandomHorizontalFlip, (c) RandomRotation, (d) ColorJitter, (e) AddGaussianNoise

### Citation Verification (18 Jul 2026)

Referensi dapus BAB 1 diverifikasi dari PDF asli:

| Referensi | Temuan |
|-----------|--------|
| **Prasetyowati (2025)** | Paper tentang **gelas plastik** (bukan botol kaca), DPMO Grade A=62.633,91 (sigma 3,03), Grade B=130.961,47 (sigma 2,63). **Angka 7.813/16.343 tidak ada di paper.** Tahun 2025, bukan 2026. |
| **Imaroh & Mustofa (2022)** | 550.962 defect/3 bulan ✅. Rp 711.816.014 = **Saving Cost** (penghematan dari reduksi 10%), **bukan kerugian biaya**. Defect mencakup banyak jenis, bukan hanya "cacat visual dan kegagalan forming". |

### Proposal Document Analysis (13 Jul 2026)

5 penyimpangan dari teori/code ditemukan di proposal:
| # | Penyimpangan | Status |
|:-:|-------------|:------:|
| 1 | 3.4.5: "layer konvolusi terakhir" (1 layer) → kode pakai 3 layers | ✅ Diperbaiki |
| 2 | 3.7.1: duplikasi judul "Modul Proses Inferensi" 2x | ❌ Belum |
| 3 | Gambar 4.1 dipakai 2x (barchart + sample images) | ❌ Belum |
| 4 | Section 3.5 hilang (skip dari 3.4.12 langsung ke 3.6) | ❌ Belum |
| 5 | K=5 vs paper K=1 — perlu catatan | ❌ Belum |

### Code Restructuring Plan (Proposed 13 Jul 2026)

| Fase | Tindakan | Risiko |
|:----:|----------|:------:|
| **1** | Hapus dead code + folder kosong + file draft | ✅ Aman |
| **2** | Rename `prepocessing/` → `preprocessing/`, cleanup config | 🟡 Sedang |
| **3** | Merge `feature extractor/` + `anomaly detection/` + `evaluation/` ke `src/` | 🔴 Berat |
| **4** | Tambah `__init__.py`, hapus whitespace | ✅ Kosmetik |

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

## OOD Detection — ComboOOD (Fixed 22 Jul 2026)

Dashboard menggunakan **ComboOOD** (Rajasekaran et al., SIAM SDM 2024) untuk menolak gambar non-bottle — semi-parametric framework yang menggabungkan Mahalanobis + KNN distance.

**Formula:** `score = -0.5 × mean(MD²) + (-√550 × log(KNN 1-NN dist))` — unweighted sum.

**Threshold tetap = 0.0:** Gap antara non-bottle (max -49) dan bottle (min +0.2) sangat bersih — threshold 0 memisahkan sempurna tanpa perlu training leave-one-out. Validasi 14 kategori non-bottle MVTec AD: **100% PASS**, seluruh 83 test bottle: **100% LOLOS**.

**UI Dashboard:** OOD reject menampilkan `st.warning()` full width + gambar di kolom 1 (posisi & ukuran sama dengan bottle normal). Kolom 2 & 3 kosong (tidak diproses PaDiM/KNN). Dilengkapi tombol "Hapus antrian" untuk reset widget uploader.

## 🔴 Bug: build_transform.py Syntax Error

`prepocessing/build_transform.py:33` — dua blok `else` akibat Fase 6.1 inline comments. Pipeline original tidak bisa dijalankan. Fix: hapus `else` pertama, pertahankan `else` dengan `ToTensor`.

Eksperimen terakhir sebelum error: `run_20260624_135445`.

## Augmented Dataset

`dataset_augmented/` — 836 gambar (209 original + 209 flip + 209 rotation + 209 color). Ini dokumentasi visual, bukan data training. Training tetap on-the-fly augmentation.

## Advisor Revision (19 Jul 2026)

**"Arsitektur sistem perlu dibuat lebih detail - di luar metodologi penelitian"** — diagram arsitektur teknis terpisah dari BAB III. Blueprint: `docs/diagram_layout_blueprint.txt`. ⏳

**Batasan masalah poin 4 (22 Jul 2026):** Ditambahkan "Penelitian ini tidak mencakup bottle di luar dataset MVTec AD." — menekankan bahwa model hanya berlaku untuk bottle dalam distribusi dataset pelatihan.
