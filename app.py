import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

# Import modul yang sudah kita buat
from utils.config import setup_page_and_styling
from utils.data_handler import load_csv, build_sequences
from utils.model_engine import build_model, predict_future

warnings.filterwarnings("ignore")

# ─── Page Config & CSS (Dipanggil dari modul) ──────────────────────────────
setup_page_and_styling()

# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ LSTM Energi Listrik")
    st.markdown("---")
    
    st.markdown('<div class="section-header">📂 Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (kolom: tanggal, demand)", type=["csv"])
    
    st.markdown('<div class="section-header">⚙️ Konfigurasi Model</div>', unsafe_allow_html=True)
    lookback = st.selectbox("Lookback Window (hari)", [30, 60, 90], index=1)
    horizon  = st.slider("Horizon Prediksi (hari)", 7, 90, 30, 7)
    rasio_test = st.slider("Rasio Data Uji (%)", 10, 40, 30, 5)

    st.markdown("**Arsitektur LSTM**")
    unit1   = st.selectbox("LSTM Layer 1 (unit)", [64, 128, 256], index=1)
    unit2   = st.selectbox("LSTM Layer 2 (unit)", [32, 64, 128], index=1)
    dropout = st.slider("Dropout", 0.1, 0.5, 0.2, 0.05)

    st.markdown("**Pelatihan**")
    lr          = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], value=0.001)
    max_epochs  = st.slider("Max Epochs", 20, 200, 100, 10)
    batch_size  = st.selectbox("Batch Size", [16, 32, 64], index=1)
    patience    = st.slider("Early Stop Patience", 5, 30, 15, 5)
    
    st.markdown("---")
    run_btn = st.button("🚀 Latih & Prediksi", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  HEADER & TABS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("# ⚡ Dashboard Prediksi Kebutuhan Energi Listrik")
st.markdown('<div class="info-banner">Model LSTM dua lapis untuk prediksi kebutuhan energi listrik harian. Upload dataset CSV, atur parameter di sidebar, lalu klik <b>Latih & Prediksi</b>.</div>', unsafe_allow_html=True)

tab_data, tab_train, tab_eval, tab_pred, tab_export = st.tabs([
    "📊 Data Historis", "🔧 Pelatihan Model", "📈 Evaluasi", "🔮 Prediksi Masa Depan", "💾 Ekspor"
])

# =======================================================================
# TRAINING LOGIC (Dipicu Tombol)
# =======================================================================
if run_btn:
    if uploaded is None:
        st.error("⚠️ Silakan upload file CSV terlebih dahulu!")
    else:
        if "df" not in st.session_state:
            st.session_state["df"] = load_csv(uploaded)
        df = st.session_state["df"]
        
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import tensorflow as tf
        
        tf.random.set_seed(42)
        np.random.seed(42)

        with st.spinner("Memproses data dan melatih model LSTM…"):
            scaler = MinMaxScaler()
            nilai_norm = scaler.fit_transform(df[["demand"]].values).flatten()

            X, y = build_sequences(nilai_norm, lookback)
            rasio = rasio_test / 100
            split = int(len(X) * (1 - rasio))
            
            X_train, X_test = X[:split].reshape(-1, lookback, 1), X[split:].reshape(-1, lookback, 1)
            y_train, y_test = y[:split], y[split:]

            model = build_model(lookback, unit1, unit2, dropout, lr)
            es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)

            history = model.fit(
                X_train, y_train, epochs=max_epochs, batch_size=batch_size, 
                validation_split=0.1, callbacks=[es], verbose=1
            )

            # --- Evaluasi & Prediksi (Sisa logika diletakkan di sini sama persis seperti kode lamamu) ---
            # ... [Sisipkan perhitungan MAE, RMSE, MAPE, dan pred_aktual dari kode asli kamu] ...
            
            # Jangan lupa simpan hasil pelatihan ke session_state agar tab lain bisa mengaksesnya
            # st.session_state.update({ ... })
            
            st.success("✅ Pelatihan selesai! Lihat tab Evaluasi & Prediksi.")

# =======================================================================
# KONTEN TABS (Salin blok logika tampilan tab dari kode lama ke sini)
# =======================================================================
with tab_data:
    # [Masukkan kode tampilan tab_data dari script lama]
    pass

# Lanjutkan untuk with tab_train, tab_eval, dst.