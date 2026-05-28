import pandas as pd
import numpy as np

def load_csv(f):
    f.seek(0)
    df = pd.read_csv(f)
    tanggal_col = next((c for c in df.columns if "date" in c.lower() or "tanggal" in c.lower()), df.columns[0])
    demand_col  = next((c for c in df.columns if "demand" in c.lower() or "mw" in c.lower()), df.columns[1])
    
    df = df[[tanggal_col, demand_col]].copy()
    df.columns = ["tanggal", "demand"]
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.dropna(subset=["tanggal", "demand"])
    df = df.groupby("tanggal")["demand"].mean().reset_index()
    df = df.sort_values("tanggal").reset_index(drop=True)
    return df

def build_sequences(series_norm, lb):
    X, y = [], []
    for i in range(lb, len(series_norm)):
        X.append(series_norm[i - lb:i])
        y.append(series_norm[i])
    return np.array(X), np.array(y)