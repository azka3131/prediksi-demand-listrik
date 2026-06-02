import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.data_handler import load_csv

def render_tab_data(uploaded):
    if uploaded is None:
        st.markdown('<div class="info-banner">⬆️ Silakan upload file CSV terlebih dahulu melalui sidebar untuk melihat data historis.</div>', unsafe_allow_html=True)
        return

    if "df" not in st.session_state:
        df = load_csv(uploaded)
        st.session_state["df"] = df
    else:
        df = st.session_state["df"]

    # Mengubah jumlah kolom menjadi 6 untuk mengakomodasi Min dan Max
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    stats = [
        ("Total Data",   f"{len(df):,} hari",        ""),
        ("Periode Mulai", df["tanggal"].min().strftime("%d %b %Y"), ""),
        ("Periode Akhir", df["tanggal"].max().strftime("%d %b %Y"), ""),
        ("Rata-rata",    f"{df['demand'].mean()/1e6:.2f}M MW", ""),
        ("Minimum",      f"{df['demand'].min()/1e6:.2f}M MW", ""),
        ("Maksimum",     f"{df['demand'].max()/1e6:.2f}M MW", ""),
    ]
    for col, (lbl, val, sub) in zip([c1,c2,c3,c4,c5,c6], stats):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["tanggal"], df["demand"]/1e6, color="#60a5fa", linewidth=0.9, label="Demand Harian")
    ax.fill_between(df["tanggal"], df["demand"]/1e6, alpha=0.12, color="#60a5fa")
    rata = df["demand"].mean()/1e6
    ax.axhline(rata, color="#f87171", linestyle="--", linewidth=1.2, label=f"Rata-rata: {rata:.2f}M MW")
    ax.set_title("Data Historis Kebutuhan Energi Listrik Harian", fontsize=13, pad=12)
    ax.set_ylabel("Demand (juta MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("#### Rata-rata Demand per Bulan")
    
    df["bulan"] = df["tanggal"].dt.month
    rb = df.groupby("bulan")["demand"].mean()/1e6
    nama = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
    warna = ["#ef4444" if b in [6,7,8] else "#3b82f6" for b in range(1,13)]
    
    # Ukuran figure disamakan dengan chart utama (14, 4) agar full width
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    bars = ax2.bar(nama, rb.values, color=warna, alpha=0.85, edgecolor="#0f1117", linewidth=0.5)
    for b, v in zip(bars, rb.values):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#94a3b8")
    ax2.set_ylabel("Demand (juta MW)")
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    st.markdown("#### Preview Data (10 Terakhir)")
    st.dataframe(
        df.tail(10).assign(demand_MW=df["demand"].tail(10).map("{:,.0f}".format))[["tanggal","demand_MW"]].rename(columns={"tanggal":"Tanggal","demand_MW":"Demand (MW)"}),
        use_container_width=True, hide_index=True
    )


def render_tab_train():
    if not st.session_state.get("trained"):
        st.markdown('<div class="info-banner">⚡ Klik tombol <b>Latih & Prediksi</b> di sidebar untuk memulai pelatihan model.</div>', unsafe_allow_html=True)
        return

    h        = st.session_state["history"]
    ep_stop  = st.session_state["epoch_stop"]
    X_train  = st.session_state["X_train"]
    X_test   = st.session_state["X_test"]

    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val) in zip([c1,c2,c3,c4], [
        ("Total Epoch",   f"{ep_stop}"),
        ("Data Latih",    f"{len(X_train):,} seq"),
        ("Data Uji",      f"{len(X_test):,} seq"),
        ("Best Val Loss", f"{min(h.history['val_loss']):.5f}"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ep = range(1, ep_stop+1)
    ax.plot(ep, h.history["loss"],     color="#60a5fa", linewidth=2, label="Training Loss")
    ax.plot(ep, h.history["val_loss"], color="#f87171", linewidth=2, linestyle="--", label="Validation Loss")
    best = np.argmin(h.history["val_loss"])
    ax.axvline(best+1, color="#fbbf24", linestyle=":", linewidth=1.5, label=f"Best Epoch: {best+1}")
    ax.set_title("Kurva Loss Pelatihan Model LSTM", fontsize=12, pad=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_tab_eval():
    if not st.session_state.get("trained"):
        st.markdown('<div class="info-banner">⚡ Klik tombol <b>Latih & Prediksi</b> di sidebar untuk melihat hasil evaluasi.</div>', unsafe_allow_html=True)
        return

    mae      = st.session_state["mae"]
    rmse     = st.session_state["rmse"]
    mape     = st.session_state["mape"]
    kategori = st.session_state["kategori"]
    y_v      = st.session_state["y_v"]
    p_v      = st.session_state["p_v"]
    tanggal_test = st.session_state["tanggal_test"]
    valid    = st.session_state["valid"]

    badge_cls = {"Baik":"badge-good","Cukup":"badge-mid","Kurang":"badge-bad"}[kategori]
    kelas_mape = "metric-good" if mape<10 else ("metric-mid" if mape<20 else "metric-bad")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">MAE</div><div class="metric-value">{mae:,.0f}</div><div class="metric-sub">MW</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">RMSE</div><div class="metric-value">{rmse:,.0f}</div><div class="metric-sub">MW</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">MAPE</div><div class="metric-value {kelas_mape}">{mape:.2f}%</div><div class="metric-sub">Mean Abs % Error</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">Kategori</div><div class="metric-value"><span class="{badge_cls}">{kategori}</span></div><div class="metric-sub">MAPE < 10% = Baik</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    tanggal_valid = pd.to_datetime(tanggal_test)[valid]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(tanggal_valid, y_v/1e6,   color="#60a5fa", linewidth=2, marker="o", markersize=3, label="Aktual")
    ax.plot(tanggal_valid, p_v/1e6,   color="#f87171", linewidth=2, marker="s", markersize=3, linestyle="--", label="Prediksi")
    ax.fill_between(tanggal_valid, y_v/1e6, p_v/1e6, alpha=0.18, color="#fbbf24", label="Selisih")
    ax.set_title(f"Aktual vs Prediksi  |  MAE={mae:,.0f} MW  |  RMSE={rmse:,.0f} MW  |  MAPE={mape:.2f}%", fontsize=11, pad=10)
    ax.set_ylabel("Demand (juta MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_tab_pred(horizon):
    if not st.session_state.get("trained"):
        st.markdown('<div class="info-banner">⚡ Klik tombol <b>Latih & Prediksi</b> di sidebar untuk melihat prediksi masa depan.</div>', unsafe_allow_html=True)
        return

    prediksi_depan = st.session_state["prediksi_depan"]
    tanggal_depan  = st.session_state["tanggal_depan"]
    df             = st.session_state["df"]
    y_test_aktual  = st.session_state["y_test_aktual"]
    pred_aktual    = st.session_state["pred_aktual"]
    tanggal_test   = st.session_state["tanggal_test"]

    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val) in zip([c1,c2,c3,c4], [
        ("Periode",   f"{len(prediksi_depan)} hari"),
        ("Rata-rata", f"{prediksi_depan.mean()/1e6:.3f}M MW"),
        ("Minimum",   f"{prediksi_depan.min()/1e6:.3f}M MW"),
        ("Maksimum",  f"{prediksi_depan.max()/1e6:.3f}M MW"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df["tanggal"], df["demand"]/1e6, color="#60a5fa", linewidth=0.7, alpha=0.6, label="Data Historis")
    ax.plot(pd.to_datetime(tanggal_test), y_test_aktual/1e6, color="#34d399", linewidth=1.5, marker="o", markersize=2, label="Aktual (data uji)")
    ax.plot(pd.to_datetime(tanggal_test), pred_aktual/1e6,   color="#f87171", linewidth=1.5, marker="s", markersize=2, linestyle="--", label="Prediksi (data uji)")
    ax.plot(tanggal_depan, prediksi_depan/1e6, color="#fbbf24", linewidth=2.5, marker="^", markersize=5, label=f"Prediksi {horizon} Hari ke Depan")
    ax.fill_between(tanggal_depan, prediksi_depan/1e6*0.92, prediksi_depan/1e6*1.08, alpha=0.18, color="#fbbf24", label="Interval ±8%")
    ax.axvline(df["tanggal"].max(), color="#6b7280", linestyle=":", linewidth=1.5)
    ax.set_title("Historis, Evaluasi, dan Prediksi Masa Depan", fontsize=13, pad=12)
    ax.set_ylabel("Demand (juta MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("####  Prediksi Masa Depan")
    
    # Lebar disesuaikan agar membentang secara penuh seperti grafik utama
    fig2, ax2 = plt.subplots(figsize=(15, 4))
    ax2.plot(tanggal_depan, prediksi_depan/1e6, color="#fbbf24", linewidth=2.5, marker="o", markersize=5)
    ax2.fill_between(tanggal_depan, prediksi_depan/1e6*0.92, prediksi_depan/1e6*1.08, alpha=0.18, color="#fbbf24", label="Interval ±8%")
    ax2.set_ylabel("Demand (juta MW)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.tick_params(axis="x", rotation=30); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


def render_tab_export():
    if not st.session_state.get("trained"):
        st.markdown('<div class="info-banner">⚡ Latih model terlebih dahulu untuk mengunduh hasil.</div>', unsafe_allow_html=True)
        return

    st.markdown("### 💾 Unduh Hasil")
    st.markdown("Semua file hasil analisis tersedia untuk diunduh di bawah ini.")

    tanggal_test   = st.session_state["tanggal_test"]
    y_test_aktual  = st.session_state["y_test_aktual"]
    pred_aktual    = st.session_state["pred_aktual"]
    prediksi_depan = st.session_state["prediksi_depan"]
    tanggal_depan  = st.session_state["tanggal_depan"]
    mae  = st.session_state["mae"]
    rmse = st.session_state["rmse"]
    mape = st.session_state["mape"]

    df_eval = pd.DataFrame({
        "tanggal":      pd.to_datetime(tanggal_test),
        "aktual_mw":    y_test_aktual,
        "prediksi_mw":  pred_aktual,
        "error_mw":     y_test_aktual - pred_aktual,
        "error_abs":    np.abs(y_test_aktual - pred_aktual),
        "ape_persen":   np.abs((y_test_aktual - pred_aktual)/y_test_aktual)*100,
    })
    csv_eval = df_eval.to_csv(index=False).encode("utf-8")

    df_future = pd.DataFrame({"tanggal": tanggal_depan, "prediksi_mw": prediksi_depan})
    csv_future = df_future.to_csv(index=False).encode("utf-8")

    df_metrik = pd.DataFrame({
        "metrik": ["MAE","RMSE","MAPE","Kategori"],
        "nilai":  [f"{mae:,.2f} MW", f"{rmse:,.2f} MW", f"{mape:.2f}%", st.session_state["kategori"]],
    })
    csv_metrik = df_metrik.to_csv(index=False).encode("utf-8")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📄 Hasil Evaluasi**")
        st.download_button("⬇️ Unduh hasil_evaluasi.csv", csv_eval, "hasil_evaluasi.csv", "text/csv", use_container_width=True)
    with col2:
        st.markdown("**📄 Prediksi Masa Depan**")
        st.download_button("⬇️ Unduh prediksi_masa_depan.csv", csv_future, "prediksi_masa_depan.csv", "text/csv", use_container_width=True)
    with col3:
        st.markdown("**📄 Ringkasan Metrik**")
        st.download_button("⬇️ Unduh ringkasan_metrik.csv", csv_metrik, "ringkasan_metrik.csv", "text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Preview: Hasil Evaluasi")
    st.dataframe(df_eval.head(20).assign(
        aktual_mw=df_eval["aktual_mw"].head(20).map("{:,.0f}".format),
        prediksi_mw=df_eval["prediksi_mw"].head(20).map("{:,.0f}".format),
        error_mw=df_eval["error_mw"].head(20).map("{:+,.0f}".format),
        ape_persen=df_eval["ape_persen"].head(20).map("{:.2f}%".format),
    ), use_container_width=True, hide_index=True)