#  LSTM Prediksi Kebutuhan Energi Listrik — Streamlit Dashboard

Dashboard interaktif untuk melatih dan memvisualisasikan model LSTM prediksi energi listrik harian (dataset AEMO).

---

## 📋 Prasyarat

- Python 3.10 atau 3.11 (disarankan)
- Visual Studio Code
- Git (opsional)

---

## 🚀 Cara Menjalankan

### 1. Buat Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

> ⏳ Proses ini memerlukan beberapa menit karena mengunduh TensorFlow.

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`

---

## 📂 Format Dataset CSV

File CSV harus memiliki **dua kolom**:

| Kolom              | Deskripsi                                 |
|--------------------|-------------------------------------------|
| `SETTLEMENTDATE`   | Tanggal (atau nama apapun yang mengandung "date" / "tanggal") |
| `TOTALDEMAND`      | Nilai demand MW (atau nama yang mengandung "demand" / "mw")   |

Contoh isi CSV:
```
SETTLEMENTDATE,TOTALDEMAND
2023-01-01,7823456.5
2023-01-02,7654321.0
...
```

---

## 🖥️ Fitur Dashboard

| Tab                   | Konten                                                         |
|-----------------------|----------------------------------------------------------------|
| 📊 Data Historis      | Time series, rata-rata per bulan, distribusi, preview tabel    |
| 🔧 Pelatihan Model    | Kurva loss training & validation, ringkasan arsitektur         |
| 📈 Evaluasi           | MAE, RMSE, MAPE, scatter aktual vs prediksi, distribusi error  |
| 🔮 Prediksi Masa Depan| Master chart, zoom prediksi, tabel harian                     |
| 💾 Ekspor             | Unduh CSV evaluasi, prediksi, dan ringkasan metrik             |

---

## ⚙️ Parameter yang Dapat Diatur (Sidebar)

| Parameter          | Default  | Keterangan                                     |
|--------------------|----------|------------------------------------------------|
| Lookback Window    | 60 hari  | Jumlah hari historis sebagai input sekuens     |
| Horizon Prediksi   | 30 hari  | Jumlah hari ke depan yang diprediksi           |
| Rasio Data Uji     | 30%      | Persentase data untuk pengujian                |
| LSTM Layer 1       | 128 unit | Ukuran layer LSTM pertama                      |
| LSTM Layer 2       | 64 unit  | Ukuran layer LSTM kedua                        |
| Dropout            | 0.2      | Regularisasi dropout                           |
| Learning Rate      | 0.001    | Laju pembelajaran Adam optimizer              |
| Max Epochs         | 100      | Batas maksimum epoch pelatihan                 |
| Batch Size         | 32       | Ukuran mini-batch                              |
| Early Stop Patience| 15       | Berhenti jika val_loss tidak membaik N epoch   |

---

## 🗂️ Struktur Project

```
lstm_dashboard/
├── app.py              ← Aplikasi Streamlit utama
├── requirements.txt    ← Daftar dependensi Python
└── README.md           ← Dokumentasi ini
```

---

## 🛠️ Tips di VS Code

1. Install ekstensi **Python** dan **Pylance** di VS Code.
2. Pilih interpreter dari virtual environment: `Ctrl+Shift+P` → *Python: Select Interpreter* → pilih `venv`.
3. Untuk menjalankan langsung dari terminal VS Code: `Ctrl+`` ` lalu ketik `streamlit run app.py`.
4. Gunakan **Split Terminal** untuk melihat log sambil tetap bisa mengedit kode.

---

## ❗ Troubleshooting

| Masalah                          | Solusi                                                          |
|----------------------------------|-----------------------------------------------------------------|
| `ModuleNotFoundError`            | Pastikan virtual environment aktif dan sudah `pip install -r requirements.txt` |
| TensorFlow tidak terdeteksi GPU  | Aplikasi tetap berjalan di CPU — tidak ada masalah             |
| Port 8501 sudah digunakan        | Jalankan: `streamlit run app.py --server.port 8502`            |
| Browser tidak terbuka otomatis   | Buka manual: `http://localhost:8501`                           |

---

*Dashboard ini menggunakan model LSTM dua lapis dengan arsitektur:*  
`LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(1)`
