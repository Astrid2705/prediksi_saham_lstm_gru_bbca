import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL & DATA
# =========================
model = load_model('../model_gru.h5')
X_test = np.load('../Data/X_test.npy')
y_test = np.load('../Data/y_test.npy')
scaler = joblib.load('../Data/scaler.pkl')

# =========================
# PREDIKSI
# =========================
pred_scaled = model.predict(X_test)

# Fungsi inverse transform ke harga asli
def ke_harga_asli(pred_scaled):
    dummy = np.zeros((len(pred_scaled), 6))
    dummy[:, 3] = pred_scaled.flatten()
    return scaler.inverse_transform(dummy)[:, 3]

# Ubah ke harga asli
y_test_real = ke_harga_asli(y_test.reshape(-1, 1))
pred_real = ke_harga_asli(pred_scaled)

# =========================
# VISUALISASI
# =========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Grafik prediksi
ax1.plot(y_test_real, label='Harga Asli BBCA', color='black')
ax1.plot(pred_real, label='Prediksi GRU', color='red', linestyle='--')
ax1.set_title('Prediksi Harga Saham BBCA - GRU')
ax1.set_ylabel('Harga (Rp)')
ax1.legend()
ax1.grid(True)

# Grafik error
error = pred_real - y_test_real
ax2.fill_between(range(len(error)), error, color='red', alpha=0.3)
ax2.axhline(y=0, color='black')
ax2.set_title('Residual Error GRU')
ax2.set_xlabel('Hari')
ax2.set_ylabel('Selisih (Rp)')
ax2.grid(True)

plt.tight_layout()

# Simpan gambar ke folder Grafik
plt.savefig('grafik_gru.png', dpi=300, bbox_inches='tight')

plt.show()

print(f"Harga Asli Terakhir : Rp {y_test_real[-1]:.2f}")
print(f"Prediksi GRU       : Rp {pred_real[-1]:.2f}")