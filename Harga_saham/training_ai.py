import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# =========================
# 1. LOAD DATA HASIL SPLIT
# =========================
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

print(f"Data Loaded: Train {X_train.shape}, Input Dim {X_train.shape[2]} Features")

# =========================
# 2. FUNCTION MODEL
# =========================
def buat_model(tipe="LSTM"):
    model = Sequential()
    # input_shape[2] otomatis bernilai 6 karena multivariate
    if tipe == "LSTM":
        model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    else:
        model.add(GRU(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))

    model.add(Dropout(0.2)) # Mencegah Overfitting
    model.add(Dense(1))     # Output harga Close

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# =========================
# 3. TRAIN MODEL (Optimasi: Epochs 50)
# =========================
print("\nTraining LSTM (Multivariate)...")
model_lstm = buat_model("LSTM")
history_lstm = model_lstm.fit(X_train, y_train, epochs=50, batch_size=32,
                               validation_data=(X_val, y_val), verbose=2)

print("\nTraining GRU (Multivariate)...")
model_gru = buat_model("GRU")
history_gru = model_gru.fit(X_train, y_train, epochs=50, batch_size=32,
                              validation_data=(X_val, y_val), verbose=2)

# =========================
# 4. EVALUASI
# =========================
def evaluasi(model, nama):
    pred = model.predict(X_test)
    
    # Metrik Evaluasi
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred)

    print(f"\n--- HASIL EVALUASI {nama} ---")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"MAPE : {mape:.2%}")

evaluasi(model_lstm, "LSTM")
evaluasi(model_gru, "GRU")

# =========================
# 5. SIMPAN MODEL
# =========================
model_lstm.save("model_lstm.h5")
model_gru.save("model_gru.h5")

print("\nModel Multivariate (LSTM & GRU) Berhasil Disimpan!")