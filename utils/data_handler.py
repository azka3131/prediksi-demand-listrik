import pandas as pd
import numpy as np

def load_csv(f):
    f.seek(0)
    df = pd.read_csv(f)
    
    # Mencari kolom tanggal dan demand secara otomatis
    tanggal_col = next((c for c in df.columns if "date" in c.lower() or "tanggal" in c.lower()), df.columns[0])
    demand_col  = next((c for c in df.columns if "demand" in c.lower() or "mw" in c.lower()), df.columns[1])
    
    # Hanya fokus mengambil kolom tanggal dan demand
    df = df[[tanggal_col, demand_col]].copy()
    df.columns = ["tanggal", "demand"]
    
    # -------------------------------------------------------------------------
    # PERBAIKAN WAKTU: Ubah format teks ke datetime, lalu normalisasi.
    # .dt.normalize() mereset semua jam/menit (misal 21:30) menjadi 00:00:00.
    # -------------------------------------------------------------------------
    df["tanggal"] = pd.to_datetime(df["tanggal"]).dt.normalize()
    
    # Hapus baris yang kosong (jika ada)
    df = df.dropna(subset=["tanggal", "demand"])
    
    # Mengelompokkan data berdasarkan hari yang sama dan menghitung rata-ratanya
    # (Mengubah 48 baris interval 30 menit menjadi 1 baris harian)
    df = df.groupby("tanggal")["demand"].mean().reset_index()
    
    # Urutkan secara kronologis
    df = df.sort_values("tanggal").reset_index(drop=True)
    return df

def build_sequences(series_norm, lb):
    X, y = [], []
    for i in range(lb, len(series_norm)):
        X.append(series_norm[i - lb:i])
        y.append(series_norm[i])
    return np.array(X), np.array(y)