"""
model_loader.py
---------------
Bertugas memuat model AI (GRU atau LSTM) beserta scaler-nya,
dan menjalankan fungsi prediksi harga saham ke depan.

Isi file ini:
  - load_assets()        : Memuat model .h5 dan scaler .pkl dari disk
  - predict_recursive()  : Memprediksi harga saham untuk beberapa hari ke depan
"""

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model


@st.cache_resource  # Model hanya dimuat sekali, tidak diulang tiap refresh
def load_assets(algo_name):
    """
    Memuat model AI dan scaler berdasarkan algoritma yang dipilih pengguna.

    Parameter:
        algo_name (str): "GRU" atau "LSTM"

    Proses:
        1. Memuat scaler (normalisasi data) dari file scaler.pkl
        2. Memuat file model .h5 sesuai pilihan algoritma
        3. Menyiapkan informasi metrik evaluasi model (MAPE, RMSE, MAE)

    Mengembalikan: (model, scaler, info_metrik)

    Catatan: @st.cache_resource memastikan model tidak dimuat ulang
             setiap kali pengguna berinteraksi dengan halaman.
    """
    scaler = joblib.load('../Data/scaler.pkl')

    if algo_name == "GRU":
        model = load_model('../model_gru.h5')
        info = {
            "mape": "1.89%",
            "rmse": "0.017241",
            "mae": "0.012537"
        }
    else:  # LSTM
        model = load_model('../model_lstm.h5')
        info = {
            "mape": "3.31%",
            "rmse": "0.027024",
            "mae": "0.021892"
        }

    return model, scaler, info


def predict_recursive(model, scaler, data_input, days=5):
    """
    Melakukan prediksi harga saham secara rekursif (berantai) untuk
    sejumlah hari ke depan.

    Parameter:
        model       : Model AI yang sudah dimuat (GRU atau LSTM)
        scaler      : Scaler untuk normalisasi dan denormalisasi data
        data_input  : Array numpy berisi data historis (minimal 60 baris, 6 kolom)
        days (int)  : Jumlah hari ke depan yang ingin diprediksi (default: 5)

    Cara kerja:
        - Ambil 60 data terakhir sebagai input awal
        - Normalisasi data → masukkan ke model → dapatkan prediksi
        - Hasil prediksi ditambahkan ke batch → dipakai untuk prediksi hari berikutnya
        - Ulangi sebanyak 'days' kali (rekursif/berantai)

    Mengembalikan: list berisi harga prediksi (dalam rupiah, sudah de-normalisasi)
    """
    current_batch = data_input[-60:].copy()  # Ambil 60 data terakhir
    preds = []

    for _ in range(days):
        # Normalisasi data input
        scaled = scaler.transform(current_batch)

        # Bentuk ulang menjadi (1, 60, 6) sesuai input model
        inp = scaled.reshape((1, 60, 6))

        # Prediksi menghasilkan nilai ternormalisasi
        p_scaled = model.predict(inp, verbose=0)

        # Buat array dummy untuk inverse transform (hanya kolom Close / indeks 3)
        dummy = np.zeros((1, 6))
        dummy[0, 3] = p_scaled[0, 0]

        # Kembalikan ke skala asli (rupiah)
        p = scaler.inverse_transform(dummy)[0, 3]
        preds.append(p)

        # Geser batch: hapus baris pertama, tambah prediksi sebagai baris baru
        new_row = current_batch[-1].copy()
        new_row[3] = p  # Update nilai Close dengan hasil prediksi
        current_batch = np.append(current_batch[1:], [new_row], axis=0)

    return preds
