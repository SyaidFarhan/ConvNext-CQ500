# Deteksi dan Klasifikasi Perdarahan Intrakranial dari CT Scan Menggunakan ConvNeXt

Proyek penelitian ini membangun sistem deteksi otomatis untuk **Intracranial Hemorrhage (ICH)** — perdarahan otak — dari gambar CT scan menggunakan deep learning berbasis **ConvNeXt-Tiny**. Sistem dievaluasi pada dataset CQ500 dan divalidasi secara eksternal pada dataset RSNA.

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Arsitektur Model](#arsitektur-model)
- [Pipeline Preprocessing](#pipeline-preprocessing)
- [Konfigurasi Training](#konfigurasi-training)
- [Evaluasi](#evaluasi)
- [Struktur Direktori](#struktur-direktori)
- [Cara Menjalankan](#cara-menjalankan)
- [Dependensi](#dependensi)
- [Hasil](#hasil)

---

## Latar Belakang

Perdarahan intrakranial adalah kondisi darurat medis yang memerlukan diagnosis cepat. Terdapat beberapa subtipe ICH yang masing-masing membutuhkan penanganan berbeda:

| Subtipe | Keterangan |
|---------|-----------|
| **Any ICH** | Deteksi umum — ada/tidaknya perdarahan |
| **IPH** | Intraparenchymal Hemorrhage (dalam jaringan otak) |
| **IVH** | Intraventricular Hemorrhage (dalam ventrikel) |
| **SDH** | Subdural Hematoma (di bawah lapisan dura) |
| **EDH** | Epidural Hematoma (di atas lapisan dura) |
| **SAH** | Subarachnoid Hemorrhage (di ruang subarachnoid) |

Proyek ini membandingkan pendekatan deep learning (ConvNeXt, ResNet50, EfficientNet-B0) dengan machine learning klasik (LightGBM + radiomics) untuk masing-masing subtipe.

---

## Dataset

### CQ500 (Dataset Utama)

- **Sumber**: [qure.ai CQ500 Dataset](http://headctstudy.qure.ai/)
- **Jumlah**: 500 studi CT otak
- **Anotasi**: Multi-radiolog dengan majority voting
- **Label**: Binary per subtipe (ada/tidak ada perdarahan)
- **Pembagian** (patient-level, stratified):
  - 70% training
  - 15% validasi
  - 15% test

### RSNA (Dataset Validasi Eksternal)

- **Sumber**: RSNA Intracranial Hemorrhage Detection Challenge
- **Tujuan**: Uji generalisasi domain dari model yang dilatih di CQ500

### Preprocessing DICOM

1. Parsing file DICOM dan konversi ke Hounsfield Unit (HU)
2. Brain windowing standar (WL=40, WW=80)
3. Resize ke 224×224 piksel
4. Representasi 3-channel: subdural / brain / bone windows
5. Normalisasi ImageNet (mean=`[0.485, 0.456, 0.406]`, std=`[0.229, 0.224, 0.225]`)
6. Cache ke disk (`.npy`) untuk loading cepat

---

## Arsitektur Model

### ConvNeXt-Tiny (Model Utama)

```
Input (224×224×3)
    ↓
ConvNeXt-Tiny Backbone (pretrained ImageNet-1K, ~28M params)
    ↓
GeM Pooling (Generalized Mean Pooling)
    ↓
BatchNorm → Dropout
    ↓
Dense (768 → 256) + GELU
    ↓
Dense (256 → 1) + Sigmoid
    ↓
Output (probabilitas 0–1)
```

**GeM Pooling** dipilih karena lebih efektif dalam mengagregasi fitur spasial dibandingkan Global Average Pooling biasa, terutama untuk lesi fokal berukuran kecil.

### Model Pembanding

| Model | Parameter | Keterangan |
|-------|-----------|-----------|
| ResNet50 | ~25.5M | Baseline klasik |
| EfficientNet-B0 | ~5.3M | Pilihan ringan |
| LightGBM + Radiomics | — | Baseline ML klasik |

---

## Pipeline Preprocessing

File: [ConvNext-CQ500/peprocessing-data.ipynb](ConvNext-CQ500/peprocessing-data.ipynb)

```
File DICOM (CQ500)
    ↓
Parsing metadata & pixel array
    ↓
Konversi HU (slope & intercept dari header DICOM)
    ↓
Windowing (subdural: WL=75/WW=215, brain: WL=40/WW=80, bone: WL=600/WW=2800)
    ↓
Normalisasi per-channel → range [0, 255]
    ↓
Resize 224×224
    ↓
Stack 3-channel → simpan .npy
    ↓
Buat split JSON (train/val/test, patient-level)
```

---

## Konfigurasi Training

### Loss Function

**Focal Loss** dengan parameter:
- `alpha = 0.75` (bobot kelas positif)
- `gamma = 2.0` (focusing parameter)
- `label_smoothing = 0.05`

Dipilih untuk mengatasi ketidakseimbangan kelas yang signifikan pada beberapa subtipe ICH.

### Optimizer & Learning Rate

**AdamW** dengan layer-wise learning rate decay:

| Layer | Learning Rate |
|-------|--------------|
| Early layers (backbone) | ~3e-5 |
| Head (classifier) | ~1.5e-4 |
| Weight decay | 0.05 |

### Augmentasi Data

| Transformasi | Parameter |
|-------------|-----------|
| Horizontal Flip | p=0.5 |
| Rotasi | ±15° |
| Shift/Scale | ±10% |
| Gaussian Noise | σ=0.01 |
| Brightness/Contrast | ±10% |
| Elastic Deformation | — |
| Cutout | 1–3 patches |

### Hyperparameter Lainnya

| Parameter | Nilai |
|-----------|-------|
| Epochs | 40 |
| Batch size | 32 (gradient accumulation ×2 → efektif 64) |
| Mixed precision | AMP (Automatic Mixed Precision) |
| Early stopping | patience=10, monitor=val_AUC |
| Scheduler | CosineAnnealingLR |

---

## Evaluasi

### Metrik Utama

- **ROC-AUC** — metrik primer (threshold-independent)
- Sensitivity (Recall), Specificity
- Precision, F1-Score
- Balanced Accuracy
- MCC (Matthews Correlation Coefficient)
- AUPRC (Area Under Precision-Recall Curve)

### Analisis Statistik

- Confidence interval 95% (bootstrap)
- **DeLong's test** — perbandingan AUC antar model
- **McNemar's test** — perbandingan prediksi biner
- Calibration curve (reliability diagram)

### Agregasi Patient-Level

Prediksi tiap slice diagregasi ke level pasien menggunakan **nilai maksimum probabilitas** di seluruh slice pasien tersebut. Ini lebih relevan secara klinis karena dokter mendiagnosis per pasien, bukan per slice.

---

## Struktur Direktori

```
Project-Ta/
├── README.md
├── ConvNext-CQ500/
│   ├── peprocessing-data.ipynb        # Preprocessing DICOM → .npy + split data
│   ├── Any_ich.ipynb                  # Training & evaluasi Any ICH
│   ├── iph.ipynb                      # Training & evaluasi IPH
│   ├── sdh.ipynb                      # Training & evaluasi SDH
│   ├── sah.ipynb                      # Training & evaluasi SAH
│   ├── lightgbm-vs-convnext.ipynb     # Perbandingan DL vs ML + learning curve
│   ├── testing-and-visualization(6).ipynb  # Validasi eksternal RSNA
│   └── .gitignore
└── .venv/                             # Virtual environment Python 3.12
```

---

## Cara Menjalankan

### 1. Setup Environment

```bash
cd /home/tehes/Documents/Project-Ta
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Preprocessing Data

Jalankan notebook secara berurutan:

```
ConvNext-CQ500/peprocessing-data.ipynb
```

Notebook ini akan:
- Membaca file DICOM dari direktori CQ500
- Melakukan windowing dan normalisasi
- Menyimpan cache `.npy` ke disk
- Membuat file JSON berisi split train/val/test

### 3. Training Model

Jalankan salah satu notebook sesuai subtipe yang ingin dilatih:

```
ConvNext-CQ500/Any_ich.ipynb   → Deteksi umum ICH
ConvNext-CQ500/iph.ipynb       → Intraparenchymal Hemorrhage
ConvNext-CQ500/sdh.ipynb       → Subdural Hematoma
ConvNext-CQ500/sah.ipynb       → Subarachnoid Hemorrhage
```

Setiap notebook akan melatih 3 arsitektur (ConvNeXt, ResNet50, EfficientNet), mengevaluasi, dan menghasilkan visualisasi.

**Estimasi waktu training:** ~30–60 menit per backbone (GPU diperlukan)

### 4. Perbandingan DL vs ML

```
ConvNext-CQ500/lightgbm-vs-convnext.ipynb
```

Notebook ini membandingkan konvneXt dengan LightGBM menggunakan learning curves dan uji statistik untuk menunjukkan titik crossover efisiensi data.

### 5. Validasi Eksternal

```
ConvNext-CQ500/testing-and-visualization(6).ipynb
```

Menguji model yang sudah dilatih di CQ500 pada dataset RSNA untuk mengukur generalisasi domain.

---

## Dependensi

Utama:
- Python 3.12
- PyTorch (dengan CUDA)
- torchvision
- timm (untuk ConvNeXt pretrained)
- pydicom
- scikit-learn
- lightgbm
- pyradiomics
- albumentations
- matplotlib, seaborn
- numpy, pandas

---

## Hasil

Model ConvNeXt-Tiny secara konsisten menghasilkan ROC-AUC tertinggi di antara semua arsitektur yang diuji pada dataset CQ500. Keunggulan deep learning dibandingkan LightGBM semakin signifikan seiring bertambahnya jumlah data training (learning curve analysis).

Validasi eksternal pada dataset RSNA menunjukkan kemampuan generalisasi domain yang baik, mengindikasikan bahwa fitur yang dipelajari bersifat klinis-relevan dan tidak overfitting ke distribusi CQ500.

---

## Reproduksibilitas

Semua random seed telah di-set secara eksplisit (Python, NumPy, PyTorch, CUDA) dan operasi GPU deterministik diaktifkan. Pembagian data dilakukan pada level pasien untuk mencegah data leakage antar split.
