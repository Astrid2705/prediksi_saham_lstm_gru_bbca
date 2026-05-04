import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Skripsi Saham - Komparasi LSTM vs GRU", layout="wide")

# ==========================================
# 2. LOAD ASSETS (Model & Scaler)
# ==========================================
@st.cache_resource
def load_assets(algo_name):
    # Sesuaikan path sesuai struktur folder Anda
    scaler = joblib.load('../Data/scaler.pkl') 
    
    if algo_name == "GRU":
        model = load_model('../model_gru.h5')
        info = {
            "mape": "1.89%",
            "rmse": "0.0172",
            "desc": "GRU menggunakan Reset & Update Gate. Lebih efisien dalam komputasi untuk data ini."
        }
    else:
        model = load_model('../model_lstm.h5')
        info = {
            "mape": "2.45%", 
            "rmse": "0.0210",
            "desc": "LSTM menggunakan 3 Gate (Input, Forget, Output) untuk memori jangka panjang."
        }
    return model, scaler, info

# ==========================================
# 3. LOGIKA DATA & PREDIKSI
# ==========================================
def get_live_data():
    ticker = "BBCA.JK"
    # Menarik data 100 hari terakhir
    data = yf.download(ticker, period="100d", interval="1d", auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60).copy()
    df['Laba_Bersih'] = 0.0237864 # Data fundamental statis (dummy skripsi)
    return df

def predict_recursive(model, scaler, data_input, days=5):
    current_batch = data_input[-60:].copy()
    preds = []
    
    for _ in range(days):
        scaled_input = scaler.transform(current_batch)
        inp = scaled_input.reshape((1, 60, 6))
        p_scaled = model.predict(inp, verbose=0)
        
        # Denormalisasi
        dummy = np.zeros((1, 6))
        dummy[0, 3] = p_scaled[0, 0]
        p_final = scaler.inverse_transform(dummy)[0, 3]
        preds.append(p_final)
        
        # Update window
        new_row = current_batch[-1].copy()
        new_row[3] = p_final 
        current_batch = np.append(current_batch[1:], [new_row], axis=0)
        
    return preds

# ==========================================
# 4. SIDEBAR (Kontrol & Historical)
# ==========================================
st.sidebar.header("Kontrol Sistem")
pilihan_algo = st.sidebar.selectbox("Pilih Algoritma Komparasi", ["GRU", "LSTM"])

# Load model berdasarkan pilihan sidebar
model, scaler, model_info = load_assets(pilihan_algo)

st.sidebar.divider()

# Informasi Model
st.sidebar.header("Informasi Model Terpilih")
st.sidebar.success(f"Algoritma: **{pilihan_algo}**")
st.sidebar.write(f"**Akurasi (MAPE):** {model_info['mape']}")
st.sidebar.write(f"**RMSE:** {model_info['rmse']}")
st.sidebar.info(model_info['desc'])

st.sidebar.divider()

# Historical Data Table di Sidebar
st.sidebar.header("Historical Data (BBCA.JK)")
try:
    df_side = yf.download("BBCA.JK", period="5d", interval="1d", auto_adjust=True)
    if isinstance(df_side.columns, pd.MultiIndex):
        df_side.columns = df_side.columns.get_level_values(0)
    
    df_display = df_side[['Close', 'Volume']].copy()
    df_display.index = df_display.index.strftime('%d/%m/%Y')
    
    st.sidebar.dataframe(
        df_display.sort_index(ascending=False).style.format({
            "Close": "{:,.0f}",
            "Volume": "{:,.0f}"
        }), 
        use_container_width=True
    )
    st.sidebar.caption("Menampilkan 5 hari bursa terakhir.")
except:
    st.sidebar.error("Gagal memuat data historis.")

# ==========================================
# 5. MAIN DASHBOARD
# ==========================================
st.title("Dashboard Prediksi Harga Saham BBCA")
st.caption(f"Aplikasi Real-Time Menggunakan Algoritma {pilihan_algo}")

try:
    df_live = get_live_data()
    latest_date = df_live.index[-1]
    latest_close = float(df_live['Close'].iloc[-1])

    # Row 1: Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Harga Terakhir (Live)", f"Rp {latest_close:,.0f}")
    with m2:
        st.metric("Model Aktif", pilihan_algo)
    with m3:
        st.metric("Target Prediksi", "Close Price")

    # Jalankan Prediksi
    future_preds = predict_recursive(model, scaler, df_live.values, days=5)
    
    # Hitung Tanggal Mendatang (Menghindari Sabtu-Minggu)
    dates_future = []
    curr = latest_date
    while len(dates_future) < 5:
        curr += timedelta(days=1)
        if curr.weekday() < 5:
            dates_future.append(curr)

    # Row 2: Interactive Line Chart
    st.subheader(f"Tren Harga Historis vs Estimasi {pilihan_algo}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_live.index, 
        y=df_live['Close'],
        mode='lines',
        name='Harga Historis',
        line=dict(color='#1f77b4', width=2),
        customdata=df_live[['Open', 'High', 'Low', 'Volume']],
        hovertemplate="<b>Tanggal: %{x}</b><br>Close: Rp %{y:,.0f}<extra></extra>"
    ))

    # Garis Prediksi
    connect_date = [latest_date] + dates_future
    connect_price = [latest_close] + future_preds

    fig.add_trace(go.Scatter(
        x=connect_date,
        y=connect_price,
        mode='lines+markers',
        name=f'Estimasi {pilihan_algo}',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=6),
        hovertemplate="<b>Estimasi: %{x}</b><br>Harga: Rp %{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(template="plotly_dark", height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Row 3: Prediction Table
    st.divider()
    st.subheader("Rincian Estimasi 5 Hari Kerja Mendatang")

    res_df = pd.DataFrame({
        "Tanggal": [d.strftime("%A, %d %B %Y") for d in dates_future],
        "Estimasi Harga": future_preds,
        "Perubahan": [p - ([latest_close] + future_preds)[i] for i, p in enumerate(future_preds)]
    })

    def color_delta(val):
        color = '#28a745' if val > 0 else '#dc3545'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        res_df.style.format({"Estimasi Harga": "Rp {:,.2f}", "Perubahan": "Rp {:+,.2f}"})
        .applymap(color_delta, subset=['Perubahan']),
        use_container_width=True
    )

except Exception as e:
    st.error(f"Terjadi kesalahan sistem: {e}")