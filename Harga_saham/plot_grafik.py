import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model_lstm = load_model('model_lstm.h5')
model_gru = load_model('model_gru.h5')

# =========================
# LOAD DATA TEST
# =========================
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# =========================
# PREDIKSI
# =========================
pred_lstm = model_lstm.predict(X_test)
pred_gru = model_gru.predict(X_test)

# =========================
# PLOT
# =========================
plt.figure(figsize=(12,6))

plt.plot(y_test, label='Harga Asli')
plt.plot(pred_lstm, label='Prediksi LSTM')
plt.plot(pred_gru, label='Prediksi GRU')

plt.title('Perbandingan Harga Asli vs Prediksi')
plt.xlabel('Waktu')
plt.ylabel('Harga (Normalized)')
plt.legend()

plt.show()