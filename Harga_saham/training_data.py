import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import os

def clean_and_train():
    # 1. Load dataset
    file_path = 'dataset_siap_training.csv'
    if not os.path.exists(file_path):
        print("Error: Dataset tidak ditemukan.")
        return

    df = pd.read_csv(file_path)
    
    # Deteksi kolom secara otomatis
    col_laba = [c for c in df.columns if 'Laba' in c][0]
    col_harga = [c for c in df.columns if 'Harga' in c][0]
    
    print(f"Preprocessing kolom: {col_laba} dan {col_harga}")

    # Fungsi untuk membersihkan format angka ribuan (menghapus titik)
    def clean_currency(value):
        if isinstance(value, str):
            return float(value.replace('.', '').replace(',', '.'))
        return value

    # Terapkan pembersihan data
    df[col_laba] = df[col_laba].apply(clean_currency)
    df[col_harga] = df[col_harga].apply(clean_currency)
    
    data = df[[col_laba, col_harga]].values
    
    # 2. Normalisasi Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(1, len(scaled_data)):
        X.append(scaled_data[i-1])
        y.append(scaled_data[i, 1])
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 3. Model LSTM
    print("\n--- Training Model LSTM ---")
    model_lstm = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X, y, epochs=50, batch_size=2, verbose=1)
    model_lstm.save('model_lstm_bca.h5')

    # 4. Model GRU
    print("\n--- Training Model GRU ---")
    model_gru = Sequential([
        GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        GRU(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model_gru.compile(optimizer='adam', loss='mean_squared_error')
    model_gru.fit(X, y, epochs=50, batch_size=2, verbose=1)
    model_gru.save('model_gru_bca.h5')

    print("\nSuccess: Model LSTM & GRU berhasil disimpan!")

if __name__ == "__main__":
    clean_and_train()