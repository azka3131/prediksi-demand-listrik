import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(lb, u1, u2, do, lr_val):
    model = Sequential([
        LSTM(u1, return_sequences=True, input_shape=(lb, 1)),
        Dropout(do),
        LSTM(u2, return_sequences=False),
        Dropout(do),
        Dense(u2 // 2, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_val), loss="mse")
    return model

def predict_future(model, seed, n, scaler, lb):
    seq = seed.copy()
    out = []
    for _ in range(n):
        inp = seq.reshape(1, lb, 1)
        p   = model.predict(inp, verbose=0)[0, 0]
        out.append(p)
        seq = np.append(seq[1:], p)
    return scaler.inverse_transform(np.array(out).reshape(-1, 1)).flatten()