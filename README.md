# Hybrid Wavelet-SVM-LSTM Model for Rainfall Forecasting

## Project Overview

Proyek ini mengembangkan model machine learning hybrid dengan mengombinasikan Wavelet Transform, Support Vector Machine (SVM), dan Long Short-Term Memory (LSTM) untuk meningkatkan performa prediksi pada data time series, khususnya untuk prediksi curah hujan harian. Wavelet Transform digunakan untuk mendekomposisi sinyal cuaca asli menjadi berbagai komponen frekuensi (approximation dan detail). Selanjutnya, SVM dan LSTM digunakan untuk menangkap pola yang berbeda dari data yang telah didekomposisi tersebut beserta fitur cuaca tambahan.

Ablation study juga dilakukan untuk mengevaluasi kontribusi masing-masing komponen model, termasuk model mandiri (SVM, LSTM), model hybrid parsial (Wavelet-SVM, Wavelet-LSTM, SVM-LSTM), dan model hybrid penuh (Wavelet-SVM-LSTM).

## Background

Data time series, terutama data meteorologi seperti curah hujan, seringkali memiliki pola yang kompleks seperti fluktuasi non-linear, noise yang tinggi, dan perubahan ekstrem yang tiba-tiba. Model tradisional seringkali kesulitan untuk menangkap ketergantungan jangka pendek maupun tren jangka panjang secara bersamaan, terutama ketika sinyal sangat tidak stabil.

Wavelet Transform digunakan untuk mendekomposisi sinyal asli ke dalam beberapa komponen, memisahkan tren (approximation) dari noise dan fluktuasi frekuensi tinggi (detail). SVM cocok untuk menangkap pola terstruktur, sedangkan LSTM sangat baik dalam mempelajari ketergantungan temporal sekuensial. Dengan menggabungkan pendekatan ini, model hybrid diharapkan dapat menangkap pola dari sinyal yang didekomposisi dan dependensi temporal secara lebih efektif.

## Objectives

Tujuan dari proyek ini adalah:

1. Mengembangkan model prediksi hybrid menggunakan Wavelet Transform, SVM, dan LSTM untuk prediksi curah hujan.
2. Membandingkan performa model mandiri (baseline) dengan model hybrid.
3. Mengevaluasi kontribusi setiap komponen melalui ablation study.
4. Menganalisis apakah dekomposisi berbasis wavelet dapat meningkatkan performa model.
5. Mengidentifikasi konfigurasi model terbaik berdasarkan metrik evaluasi.

## Dataset

Dataset yang digunakan dalam proyek ini berisi observasi cuaca harian dengan karakteristik sebagai berikut:

| Komponen | Keterangan |
|---|---|
| Nama Dataset | Data Cuaca Harian (`Data Capstone Fix BGt.xlsx`) |
| Sumber | Stasiun Cuaca / BMKG |
| Jumlah Baris | ~2000+ observasi |
| Jumlah Fitur | 4 fitur utama |
| Rentang Waktu | 2018 dan seterusnya |
| Frekuensi | Harian |
| Variabel Target | `Curah Hujan` |
| Jenis Task | Time Series Forecasting (Regresi) |

### Fitur yang Digunakan
- `Curah Hujan` (**Target**)
- `Kelembapan Rata-Rata Udara Harian (%)` 
- `Penyinaran Harian (%)`
- `Penguapan Harian (mm)`

### Contoh Data

| Tanggal | Kelembapan Rata-Rata (%) | Penyinaran Harian (%) | Penguapan Harian (mm) | Curah Hujan (Target) |
|---|---:|---:|---:|---:|
| 2018-01-01 | 87.0 | 0.0 | 5.4 | 0 |
| 2018-01-02 | 85.0 | 100.0 | 3.5 | 0 |
| 2018-01-03 | 77.0 | 65.0 | 4.0 | 0 |
| 2018-01-04 | 80.0 | 43.8 | 5.2 | 0 |
| 2018-01-05 | 81.0 | 56.3 | 2.8 | 6 |

## Project Workflow

Alur kerja proyek ini terdiri dari tahapan berikut:

1. Pengumpulan dan pemuatan data
2. Pembersihan data dan Exploratory Data Analysis (EDA)
3. Pembagian data latih (train) dan uji (test) serta Feature Scaling (MinMaxScaler)
4. Dekomposisi Wavelet pada variabel target
5. Pembuatan sliding window untuk data time series
6. Pengembangan model baseline (SVM, LSTM)
7. Konstruksi model hybrid (Wavelet-SVM, Wavelet-LSTM, SVM-LSTM, Wavelet-SVM-LSTM)
8. Evaluasi model pada data uji
9. Ablation study
10. Analisis hasil eksperimen

```text
Data Cuaca Asli
       ↓
 Preprocessing
       ↓
Wavelet Decomposition
       ↓
Komponen Dekomposisi (Approximation & Detail)
       ↓
 Cabang SVM + Cabang LSTM
       ↓
 Prediksi Hybrid
       ↓
Evaluasi dan Ablation Study
```

## Methodology

### 1. Data Preprocessing
Tahap preprocessing meliputi penanganan missing value, parsing tanggal, scaling fitur numerik menjadi rentang [0,1] menggunakan MinMaxScaler, serta pembentukan urutan windowed data.

### 2. Wavelet Decomposition
Discrete Wavelet Transform (DWT) diterapkan untuk mendekomposisi sinyal target (Curah Hujan) ke dalam beberapa frekuensi. Komponen Approximation mewakili tren utama, sementara Detail mewakili perubahan frekuensi tinggi.

### 3. SVM Model
Support Vector Regression (SVR) digunakan sebagai model baseline dan komponen hybrid. SVM dilatih pada fitur-fitur cuaca dan komponen dekomposisi untuk mempelajari pemetaan yang kompleks.

### 4. LSTM Model
Long Short-Term Memory (LSTM) digunakan untuk menangkap ketergantungan temporal pada data berurutan. Data input diubah bentuknya menjadi urutan 3D sebelum dimasukkan ke dalam jaringan.

### 5. Hybrid Wavelet-SVM-LSTM Model
Model hybrid menggabungkan keunggulan ketiganya. Wavelet mengekstrak pola sinyal yang lebih bersih, sementara SVM dan LSTM memproses informasi tersebut melalui cabangnya masing-masing. Prediksi akhir digabungkan dari output kedua model.

## Model Architecture

Arsitektur hybrid yang diusulkan terdiri dari tiga komponen utama:

| Komponen | Peran |
|---|---|
| Wavelet Transform | Mendekomposisi sinyal asli ke beberapa komponen frekuensi |
| SVM | Mempelajari pola terstruktur dan margin regresi |
| LSTM | Menangkap dependensi temporal sekuensial |
| Hybrid Layer | Menggabungkan output prediksi dari SVM dan LSTM |

### Architecture Flow

```text
       Input Time Series (Data Cuaca)
                     ↓
             Wavelet Transform
                     ↓
      Approximation & Detail Components
                     ↓
             Feature Preparation
                     ↓
       SVM Model           LSTM Model
           ↓                   ↓
       SVM Output          LSTM Output
           ↓                   ↓
           Hybrid Combination
                     ↓
              Final Prediction
```

## Ablation Study

Ablation study dilakukan untuk mengevaluasi dampak setiap komponen dalam arsitektur hybrid yang diusulkan. Berbagai konfigurasi dibandingkan untuk memastikan kontribusi positif dari Wavelet, SVM, dan LSTM.

### Ablation Scenarios

| Skenario | Konfigurasi Model | Tujuan |
|---|---|---|
| Model 1 | SVM only | Model baseline menggunakan SVM tanpa Wavelet dan LSTM |
| Model 2 | LSTM only | Model baseline menggunakan LSTM tanpa Wavelet dan SVM |
| Model 3 | Wavelet + SVM | Mengevaluasi dampak Wavelet Transform pada SVM |
| Model 4 | Wavelet + LSTM | Mengevaluasi dampak Wavelet Transform pada LSTM |
| Model 5 | SVM + LSTM | Mengevaluasi pembelajaran hybrid tanpa Wavelet decomposition |
| Model 6 | Wavelet + SVM + LSTM | Model hybrid penuh yang diusulkan |

### Purpose of Ablation Study
Ablation study ini bertujuan untuk menjawab pertanyaan:
1. Apakah Wavelet Transform mampu meningkatkan performa dengan mereduksi noise?
2. Apakah kombinasi SVM dan LSTM lebih baik daripada model mandiri?
3. Komponen mana yang memberikan kontribusi perbaikan paling besar?
4. Apakah model hybrid penuh merupakan konfigurasi yang terbaik?

## Evaluation Metrics

Model dievaluasi menggunakan metrik berikut:

| Metric | Deskripsi |
|---|---|
| MAE | Mean Absolute Error: Mengukur rata-rata error absolut prediksi |
| RMSE | Root Mean Squared Error: Memberi penalti yang lebih besar untuk error besar |
| MAPE | Mean Absolute Percentage Error: Mengukur error dalam bentuk persentase |
| R² Score | Coefficient of Determination: Mengukur seberapa baik model menjelaskan variasi target |

## Experiment Results

Hasil eksperimen berdasarkan nilai RMSE terbaik (setelah Hyperparameter Tuning):

| Model | RMSE | Keterangan Parameter Terbaik |
|---|---:|---|
| SVM | 0.0934 | time_steps: 4, C: 11.97, gamma: scale |
| LSTM | 0.0659 | time_steps: 3, units: 96, lr: 0.001 |
| Wavelet + SVM | 0.0496 | time_steps: 3, C: 2.49, gamma: scale |
| **Wavelet + LSTM** | **0.0235** | time_steps: 6, units: 64, lr: 0.01 |
| SVM + LSTM | 0.0916 | time_steps: 5, units: 64, svm_C: 98.46 |
| Wavelet + SVM + LSTM | 0.0399 | time_steps: 6, units: 32, svm_C: 7.52 |


### Best Model
Model dengan performa terbaik pada pengujian awal ini adalah:
```text
Wavelet + LSTM (W-LSTM)
```
dengan nilai RMSE:
```text
RMSE : 0.0235
```

## Result Analysis

Berdasarkan hasil eksperimen, integrasi Wavelet Transform sangat signifikan menurunkan nilai RMSE, terlihat dari perbedaan antara Base LSTM (0.0659) dan Wavelet+LSTM (0.0235), serta Base SVM (0.0934) dan Wavelet+SVM (0.0496). Ablation study menunjukkan bahwa Wavelet membantu mereduksi noise sinyal sehingga lebih mudah dipelajari oleh model LSTM maupun SVM. Meskipun arsitektur penuh (Wavelet-SVM-LSTM) memberikan RMSE 0.0399 yang sangat baik, dalam eksperimen saat ini Wavelet+LSTM memberikan error yang sedikit lebih rendah (0.0235).

## Visualization

Proyek ini mencakup visualisasi berikut di dalam notebook:
1. Plot time series asli untuk Curah Hujan
2. Komponen dekomposisi Wavelet
3. Grafik perbandingan nilai Aktual vs Prediksi
4. Performa perbandingan model Ablation Study

## Installation

Clone repository ini:

```bash
git clone https://github.com/username/project-name.git
cd project-name
```

Buat virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```
*(Untuk Windows)*:
```bash
venv\Scripts\activate
```

Instal dependensi:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras-tuner pywt optuna statsmodels
```

## Usage

Anda dapat menjalankan preprocessing, training, dan evaluasi langsung melalui Jupyter Notebook:

```bash
jupyter notebook "comparison_model.ipynb"
```

## Project Structure

```text
hybrid-lstm-svm-weather-forecasting/
│
├── Data Capstone Fix BGt.csv
├── Data Capstone Fix BGt.xlsx
├── comparison_model.ipynb          # Notebook utama berisikan eksperimen model
├── comparison-model no feature column.ipynb
├── README.md                       # Dokumentasi proyek
└── ...
```

## Key Findings

Temuan utama dari proyek ini adalah:
1. Dekomposisi Wavelet secara konsisten meningkatkan kemampuan model (baik SVM maupun LSTM) dalam menangkap pola curah hujan yang bising/volatil.
2. LSTM secara intrinsik berkinerja lebih baik dari SVM pada data time series ini.
3. Model Wavelet-LSTM dan Wavelet-SVM-LSTM adalah dua model dengan performa evaluasi terbaik.
4. Ablation study menegaskan bahwa membuang komponen dekomposisi Wavelet akan menurunkan akurasi model secara drastis.

## Limitations

Keterbatasan dalam proyek ini:
1. Performa model bergantung kuat pada kualitas dan kelengkapan data cuaca.
2. Model hybrid menuntut resource komputasi dan waktu iterasi yang lebih tinggi, terutama saat Hyperparameter tuning.
3. Tuning hyperparameter memberikan dampak yang sangat sensitif pada hasil eksperimen akhir.

## Future Work

Pengembangan selanjutnya dapat mencakup:
1. Menguji arsitektur model ini pada dataset curah hujan wilayah lain.
2. Menggunakan famili wavelet lain (selain pseudo-wavelet default) untuk dekomposisi.
3. Menerapkan tuning parameter yang lebih komprehensif atau algoritma optimasi metaheuristik.
4. Deployment model sebagai dashboard peringatan cuaca.

## References

1. Mallat, S. G. (1989). A theory for multiresolution signal decomposition.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
3. Vapnik, V. (1995). The Nature of Statistical Learning Theory.
