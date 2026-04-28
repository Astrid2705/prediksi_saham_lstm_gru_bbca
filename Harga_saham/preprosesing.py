import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# =========================
# 1. LOAD DATA CLEAN
# =========================
df = pd.read_csv('data/dataset_clean.csv')

# =========================
# 2. PILIH FITUR
# =========================
fitur = ['Close', 'Laba Bersih (Juataan Rp)']
data = df[fitur].values

# =========================
# 3. NORMALISASI
# =========================
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

print("Contoh data setelah normalisasi:")
print(data_scaled[:5])

# =========================
# 4. TIME SERIES (WINDOW 60)
# =========================
X, y = [], []

for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i])   # 60 hari sebelumnya
    y.append(data_scaled[i, 0])     # target = Close

X, y = np.array(X), np.array(y)

print("\nSebelum split:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 5. SPLIT DATA (80/10/10)
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
# 6. CEK HASIL SPLIT
# =========================
print("\nDetail Split:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# =========================
# 7. SIMPAN DATA SPLIT
# =========================
import numpy as np

np.save('data/X_train.npy', X_train)
np.save('data/y_train.npy', y_train)

np.save('data/X_val.npy', X_val)
np.save('data/y_val.npy', y_val)

np.save('data/X_test.npy', X_test)
np.save('data/y_test.npy', y_test)

print("\nData split berhasil disimpan!")