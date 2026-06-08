import streamlit as st
import numpy as np
import pandas as pd
import warnings

# Mengimpor modul dari folder utils/
from utils.config import setup_page_and_styling
from utils.data_handler import load_csv, build_sequences
from utils.model_engine import build_model, predict_future
from utils.ui_tabs import render_tab_data, render_tab_train, render_tab_eval, render_tab_pred, render_tab_export

warnings.filterwarnings("ignore")

# ─── Page Config & CSS ──────────────────────────────────────────────────────
setup_page_and_styling()

# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ LSTM Energi Listrik")
    st.markdown("---")

    st.markdown('<div class="section-header"> Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload CSV (kolom: tanggal, demand)",
        type=["csv"],
        # Teks panduan diperbarui untuk mengingatkan pengguna
        help="Pastikan dataset berinterval HARIAN dan mencakup riwayat minimal 6 BULAN. File CSV harus memiliki kolom SETTLEMENTDATE dan TOTALDEMAND (atau tanggal & demand)",
    )

    st.markdown('<div class="section-header">⚙️ Konfigurasi Model</div>', unsafe_allow_html=True)
    lookback = st.selectbox("Lookback Window (hari)", [30, 60, 90], index=1)
    horizon  = st.slider("Horizon Prediksi (hari)", 7, 90, 30, 1)
    rasio_test = st.slider("Rasio Data Uji (%)", 10, 50, 30, 1)

    st.markdown("**Arsitektur LSTM**")
    unit1   = st.selectbox("LSTM Layer 1 (unit)", [64, 128, 256], index=1)
    unit2   = st.selectbox("LSTM Layer 2 (unit)", [32, 64, 128], index=1)
    dropout = st.slider("Dropout", 0.1, 0.5, 0.2, 0.05)

    st.markdown("**Pelatihan**")
    lr          = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], value=0.001)
    max_epochs  = st.slider("Max Epochs", 20, 200, 100, 1)
    batch_size  = st.selectbox("Batch Size", [16, 32, 64], index=1)
    patience    = st.slider("Early Stop Patience", 5, 30, 15, 1)

    st.markdown("---")
    run_btn = st.button("🚀 Latih & Prediksi", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div style="color:#4b5563;font-size:11px;text-align:center;">LSTM Dashboard v1.0<br>AEMO Energy Prediction</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  HEADER & INISIALISASI TABS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("# ⚡ Dashboard Prediksi Kebutuhan Energi Listrik")
st.markdown('<div class="info-banner">Model LSTM dua lapis untuk prediksi kebutuhan energi listrik harian. Upload dataset CSV, atur parameter di sidebar, lalu klik <b>Latih & Prediksi</b>.</div>', unsafe_allow_html=True)

tab_data, tab_train, tab_eval, tab_pred, tab_export = st.tabs([
    "📊 Data Historis", "🔧 Pelatihan Model", "📈 Evaluasi", "🔮 Prediksi Masa Depan", "💾 Ekspor"
])

# ═══════════════════════════════════════════════════════════════════════
#  TRAINING LOGIC (Dipicu Tombol Latih)
# ═══════════════════════════════════════════════════════════════════════
if run_btn:
    if uploaded is None:
        st.error("⚠️ Silakan upload file CSV terlebih dahulu!")
    else:
        if "df" not in st.session_state:
            st.session_state["df"] = load_csv(uploaded)
        df = st.session_state["df"]
        
        # --- BLOK VALIDASI INPUT PENGGUNA ---
        # Memastikan dataset memiliki minimal 180 hari (sekitar 6 bulan)
        if len(df) < 180:
            st.error("⚠️ **Dataset terlalu kecil!** Harap masukkan data dengan rentang waktu minimal 6 bulan (minimal 180 baris data harian) agar model LSTM dapat membentuk sekuens dan dievaluasi dengan baik.")
            st.stop() # Menghentikan proses secara elegan tanpa memicu crash TensorFlow
        # ------------------------------------
        
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        tf.random.set_seed(42)
        np.random.seed(42)

        with st.spinner("Memproses data dan melatih model LSTM…"):
            scaler = MinMaxScaler()
            nilai_norm = scaler.fit_transform(df[["demand"]].values).flatten()

            X, y = build_sequences(nilai_norm, lookback)
            rasio = rasio_test / 100
            split = int(len(X) * (1 - rasio))
            
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            X_train = X_train.reshape(*X_train.shape, 1)
            X_test  = X_test.reshape(*X_test.shape, 1)

            model = build_model(lookback, unit1, unit2, dropout, lr)
            es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)

            history = model.fit(
                X_train, y_train, epochs=max_epochs, batch_size=batch_size,
                validation_split=0.1, callbacks=[es], verbose=1
            )

            pred_norm    = model.predict(X_test, verbose=0).flatten()
            pred_aktual  = scaler.inverse_transform(pred_norm.reshape(-1,1)).flatten()
            y_test_aktual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
            tanggal_test  = df["tanggal"].values[-len(y_test):]

            batas = y_test_aktual.mean() * 0.3
            valid = y_test_aktual > batas
            y_v, p_v = y_test_aktual[valid], pred_aktual[valid]

            mae  = mean_absolute_error(y_v, p_v)
            rmse = np.sqrt(mean_squared_error(y_v, p_v))
            mape = np.mean(np.abs((y_v - p_v)/y_v))*100
            kategori = "Baik" if mape < 10 else ("Cukup" if mape < 20 else "Kurang")
            
            seed = nilai_norm[-lookback:]
            prediksi_depan = predict_future(model, seed, horizon, scaler, lookback)
            tanggal_depan  = pd.date_range(start=df["tanggal"].max() + pd.Timedelta(days=1), periods=horizon)

            # Simpan seluruh variable yang dibutuhkan tab ke session_state
            st.session_state.update({
                "df": df, "model": model, "scaler": scaler, "history": history,
                "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
                "y_test_aktual": y_test_aktual, "pred_aktual": pred_aktual, "tanggal_test": tanggal_test,
                "valid": valid, "y_v": y_v, "p_v": p_v, "mae": mae, "rmse": rmse, "mape": mape,
                "kategori": kategori, "epoch_stop": len(history.history["loss"]),
                "prediksi_depan": prediksi_depan, "tanggal_depan": tanggal_depan,
                "nilai_norm": nilai_norm, "lookback": lookback, "trained": True,
            })

        st.success(f"✅ Pelatihan selesai pada epoch ke-{st.session_state['epoch_stop']}! Lihat tab Evaluasi & Prediksi.")

# ═══════════════════════════════════════════════════════════════════════
#  RENDERING TABS DARI MODUL ui_tabs.py
# ═══════════════════════════════════════════════════════════════════════
with tab_data:
    render_tab_data(uploaded)

with tab_train:
    render_tab_train()

with tab_eval:
    render_tab_eval()

with tab_pred:
    render_tab_pred(horizon)

with tab_export:
    render_tab_export()