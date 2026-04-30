import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# 1. LOAD MODEL & DATA
model_lstm = load_model('model_lstm.h5')
model_gru = load_model('model_gru.h5')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# 2. LOAD SCALER (Penting untuk balik ke Rupiah)
scaler = joblib.load('data/scaler.pkl')

# 3. PREDIKSI
pred_lstm_scaled = model_lstm.predict(X_test)
pred_gru_scaled = model_gru.predict(X_test)

# 4. INVERSE TRANSFORM (Kembali ke skala asli)
# Karena kita pakai 6 fitur, kita harus buat dummy array untuk inverse
def ke_harga_asli(pred_scaled):
    # Buat array kosong dengan 6 kolom (sesuai jumlah fitur saat training)
    dummy = np.zeros((len(pred_scaled), 6))
    # Masukkan hasil prediksi ke kolom index 3 (kolom 'Close')
    dummy[:, 3] = pred_scaled.flatten()
    # Inverse transform lalu ambil kolom index 3 saja
    return scaler.inverse_transform(dummy)[:, 3]

y_test_real = ke_harga_asli(y_test.reshape(-1, 1))
pred_lstm_real = ke_harga_asli(pred_lstm_scaled)
pred_gru_real = ke_harga_asli(pred_gru_scaled)

# =========================
# VISUALISASI MULTI-PLOT
# =========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- PLOT 1: PERBANDINGAN HARGA ASLI (RUPIAH) ---
ax1.plot(y_test_real, label='Harga Asli BBCA', color='black', linewidth=1.5)
ax1.plot(pred_lstm_real, label='Prediksi LSTM', color='blue', linestyle='--', alpha=0.8)
ax1.plot(pred_gru_real, label='Prediksi GRU', color='red', linestyle=':', alpha=0.8)
ax1.set_title('Perbandingan Harga Asli vs Prediksi (Skala Rupiah)', fontsize=14)
ax1.set_ylabel('Harga (Rp)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- PLOT 2: ERROR (SELISIH HARGA) ---
# Melihat seberapa jauh melesetnya dalam Rupiah
error_lstm = pred_lstm_real - y_test_real
error_gru = pred_gru_real - y_test_real

ax2.fill_between(range(len(error_lstm)), error_lstm, color='blue', alpha=0.2, label='Error LSTM')
ax2.fill_between(range(len(error_gru)), error_gru, color='red', alpha=0.2, label='Error GRU')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_title('Analisis Residual (Selisih Prediksi - Asli)', fontsize=14)
ax2.set_xlabel('Data Test (Hari)')
ax2.set_ylabel('Selisih (Rp)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# CETAK CONTOH HARGA
print(f"Harga Asli Terakhir: Rp {y_test_real[-1]:.2f}")
print(f"Prediksi GRU Terakhir: Rp {pred_gru_real[-1]:.2f}")