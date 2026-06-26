import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib # Disarankan untuk simpan scaler

# =========================
# 1. LOAD DATA CLEAN
# =========================
df = pd.read_csv('Data/dataset_clean.csv')

# =========================
# 2. PILIH FITUR (Update: 6 Variabel)
# =========================
# Sesuaikan nama kolom dengan hasil cleaning tadi
fitur = ['Open', 'High', 'Low', 'Close', 'Volume', 'Laba_Bersih']
data = df[fitur].values

# =========================
# 3. NORMALISASI
# =========================
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Simpan scaler agar bisa dipakai di Web nanti (sangat penting!)
joblib.dump(scaler, 'Data/scaler.pkl')

# =========================
# TAMPILKAN HASIL NORMALISASI
# =========================
df_scaled = pd.DataFrame(data_scaled, columns=fitur)

print("\n===== 5 Data Pertama Setelah Normalisasi =====")
print(df_scaled.head())

print("\n===== Informasi Kolom =====")
print(df_scaled.columns.tolist())

# Simpan hasil normalisasi
df_scaled.to_csv("Data/data_normalisasi.csv", index=False)

print("\nData normalisasi berhasil disimpan di:")
print("Data/data_normalisasi.csv")

# =========================
# 4. TIME SERIES (WINDOW 60)
# =========================
X, y = [], []

for i in range(60, len(data_scaled)):
    # X mengambil ke-6 fitur sebagai input
    X.append(data_scaled[i-60:i])   
    
    # y tetap mengambil kolom 'Close' sebagai target 
    # Index 3 adalah 'Close' dalam daftar fitur kita (Open=0, High=1, Low=2, Close=3...)
    y.append(data_scaled[i, 3])     

X, y = np.array(X), np.array(y)

print("\nShape setelah Update Multivariate:")
print("X shape:", X.shape) # Harus (N, 60, 6)
print("y shape:", y.shape) # Harus (N,)

# =========================
# 5. SPLIT DATA (80/10/10) - Urut Waktu
# =========================
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

# =========================
# 6. SIMPAN DATA SPLIT
# =========================
np.save('Data/X_train.npy', X_train)
np.save('Data/y_train.npy', y_train)
np.save('Data/X_val.npy', X_val)
np.save('Data/y_val.npy', y_val)
np.save('Data/X_test.npy', X_test)
np.save('Data/y_test.npy', y_test)

print("\nPreprocessing 6 Variabel Selesai & Data Tersimpan!")