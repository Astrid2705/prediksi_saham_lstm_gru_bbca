import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import base64
from pathlib import Path
from datetime import datetime
import pytz

# ==========================================
# KONFIGURASI & LOAD CSS
# ==========================================
st.set_page_config(page_title="Skripsi Saham - BBCA", layout="wide")

# Load CSS dari file eksternal
def load_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        # Fallback CSS jika file tidak ditemukan
        st.markdown("""
        <style>
            .main { background-color: #f8f7f4; }
            .card { background-color: white; border-radius: 16px; padding: 20px; margin-bottom: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.08); }
        </style>
        """, unsafe_allow_html=True)

load_css()

# ==========================================
# FUNGSI UTILITY
# ==========================================
def get_image_base64(image_path):
    """Mengkonversi gambar ke base64 untuk ditampilkan di HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return None

@st.cache_resource
def load_assets(algo_name):
    scaler = joblib.load('../Data/scaler.pkl')
    if algo_name == "GRU":
        model = load_model('../model_gru.h5')
        info = {
            "mape": "1.89%",
            "rmse": "0.017241",
            "mae": "0.012537"
        }
    else:
        model = load_model('../model_lstm.h5')
        info = {
            "mape": "3.31%",
            "rmse": "0.027024",
            "mae": "0.021892"
        }
    return model, scaler, info

def get_live_data():
    try:
        with st.spinner("Mengambil data saham BBCA..."):
            data = yf.download("BBCA.JK", period="100d", interval="1d",
                             auto_adjust=True, progress=False, timeout=10)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        df = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60).copy()
        df['Laba_Bersih'] = 0.0237864
        return df
    except Exception as e:
        st.error(f"❌ Gagal mengambil data saham: {e}")
        return None

def get_stock_change_info(df):
    if df is None or len(df) < 2:
        return None, None, None, None
    
    latest = df['Close'].iloc[-1]
    previous = df['Close'].iloc[-2]
    change = latest - previous
    change_pct = (change / previous) * 100
    return latest, change, change_pct, previous

@st.cache_data(ttl=3600)
def get_usd_idr():
    try:
        usd = yf.download("USDIDR=X", period="5d", progress=False)
        return float(usd['Close'].iloc[-1]) if not usd.empty else None
    except:
        return None

@st.cache_data(ttl=86400)
def get_inflasi_terbaru():
    return {
        "nilai": "2.42%",
        "periode": "April 2026",
        "jenis": "YoY (Tahunan)",
        "update": "05 Mei 2026"
    }

def predict_recursive(model, scaler, data_input, days=5):
    current_batch = data_input[-60:].copy()
    preds = []
    for _ in range(days):
        scaled = scaler.transform(current_batch)
        inp = scaled.reshape((1, 60, 6))
        p_scaled = model.predict(inp, verbose=0)
        dummy = np.zeros((1, 6))
        dummy[0, 3] = p_scaled[0, 0]
        p = scaler.inverse_transform(dummy)[0, 3]
        preds.append(p)
        new_row = current_batch[-1].copy()
        new_row[3] = p
        current_batch = np.append(current_batch[1:], [new_row], axis=0)
    return preds

def create_sidebar_header():
    """Membuat header sidebar dengan logo BBCA"""
    logo_path = Path(__file__).parent / "image" / "BBCA.png"
    
    if logo_path.exists():
        img_base64 = get_image_base64(logo_path)
        if img_base64:
            st.markdown(f"""
            <div class="sidebar-header">
                <div class="logo-container">
                    <img src="data:image/png;base64,{img_base64}" 
                         style="width: 70px; height: 70px; object-fit: contain;">
                </div>
                <h2>BBCA</h2>
                <p>PREDIKSI SAHAM</p>
            </div>
            """, unsafe_allow_html=True)
            return True
    # Fallback jika logo tidak ditemukan
    st.markdown("""
    <div class="sidebar-header">
        <div class="logo-container">
            <div style="width: 60px; height: 60px; background: #0f2b3d; border-radius: 30px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 20px;">BCA</div>
        </div>
        <h2>BBCA</h2>
        <p>PREDIKSI SAHAM</p>
    </div>
    """, unsafe_allow_html=True)
    return False

def create_kontrol_sistem_card():
    """Membuat card Kontrol Sistem"""
    st.markdown("""
    <div class="kontrol-card">
        <div class="kontrol-title">
            <span>⚙️</span> KONTROL SISTEM
        </div>
        <span class="algo-label">Pilih Algoritma Komparasi</span>
    </div>
    """, unsafe_allow_html=True)

def create_model_info_card(algo, info):
    """Membuat card Informasi Model"""
    st.markdown(f"""
    <div class="model-card">
        <div class="model-label">INFORMASI MODEL</div>
        <div class="model-name">Model Aktif<br>{algo}</div>
        <div class="model-metrics">
            <span>MAPE<br><strong>{info['mape']}</strong></span>
            <span>RMSE<br><strong>{info['rmse']}</strong></span>
            <span>MAE<br><strong>{info['mae']}</strong></span>
        </div>
        <div class="model-desc">
            {algo} menggunakan Reset & Update Gate. <br>
            Lebih efisien dalam komputasi untuk data pada penelitian ini.
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_live_price_card(latest_close, price_change, price_change_pct, change_class, change_sign, change_symbol, current_time, is_market_open):

    # === WIT (Waktu Indonesia Timur) ===
    wit = pytz.timezone('Asia/Jayapura')
    current_time = datetime.now(wit).strftime("%d %B %Y %H:%M:%S WIT")
    
    change_symbol = "↓" if price_change < 0 else "↑"
    change_color = "#ef4444" if price_change < 0 else "#22c55e"
    
    """Membuat card Harga Terakhir Live"""
    st.markdown(f"""
    <div class="live-price-box">
        <div class="live-label">
            <span>💰 HARGA TERAKHIR (LIVE)</span>
            <span class="ticker">BBCA.JK</span>
        </div>
        <div class="price-number">Rp {latest_close:,.0f}</div>
        <div class="price-change {change_class}">
            <span>{change_sign}{price_change:,.0f} ({price_change_pct:+.2f}%)</span> 
            <span style="font-size:16px;">{change_symbol}</span>
        </div>
        <div class="update-time">
            Update: <strong>{current_time}</strong>
            <span class="market-status">● {is_market_open}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_status_sistem_card():
    """Membuat card Status Sistem"""
    st.markdown("""
    <div class="system-status">
        <div class="status-title">
            <span>🖥️</span> STATUS SISTEM
        </div>
        <div class="status-text">
            ✅ Sistem Aktif<br>
            Realtime data from Yahoo Finance
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar_footer():
    """Membuat footer sidebar"""
    st.markdown("""
    <div class="sidebar-footer">
        © 2025 Skripsi Teknik Informatika<br>
        Prediksi Harga Saham BBCA (LSTM vs GRU)
    </div>
    """, unsafe_allow_html=True)

def create_chart(df, latest_date, future_dates, latest_close, preds, algo, chart_type):
    """Membuat grafik interaktif"""
    fig = go.Figure()
    
    if chart_type == "Candlestick (OHLC)":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='OHLC Historis',
            increasing_line_color='#16a34a',
            decreasing_line_color='#dc2626',
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Open: Rp %{customdata[0]:,.2f}<br>High: Rp %{customdata[1]:,.2f}<br>Low: Rp %{customdata[2]:,.2f}<br>Close: Rp %{customdata[3]:,.2f}<extra></extra>",
            customdata=np.stack((df['Open'], df['High'], df['Low'], df['Close']), axis=1)
        ))
    elif chart_type == "Line Chart":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Harga Close', line=dict(width=4, color='#0f172a')))
    elif chart_type == "Area Chart":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', fill='tozeroy', name='Harga Close', line=dict(width=3, color='#3b82f6')))
    elif chart_type == "Bar Chart (Close)":
        fig.add_trace(go.Bar(x=df.index, y=df['Close'], name='Close Price', marker_color='#0f172a'))
    elif chart_type == "Line + Markers":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines+markers', name='Harga Close', line=dict(width=3, color='#0f172a'), marker=dict(size=6)))
    
    # Prediksi
    fig.add_trace(go.Scatter(
        x=[latest_date] + future_dates,
        y=[latest_close] + preds,
        mode='lines+markers',
        name=f'Prediksi {algo}',
        line=dict(width=4, dash='dash', color='#d4af37'),
        marker=dict(size=9, symbol='diamond'),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Prediksi: Rp %{y:,.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        template="plotly_white",
        height=650,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30)
    )
    
    return fig

# ==========================================
# LOAD DATA
# ==========================================
df = get_live_data()
if df is None or df.empty:
    st.error("Gagal memuat data saham. Silakan refresh lagi.")
    st.stop()

latest_date = df.index[-1]
latest_close, price_change, price_change_pct, previous_close = get_stock_change_info(df)

if price_change_pct is None:
    price_change_pct = 0
    price_change = 0

current_time = datetime.now().strftime("%d %b %Y %H:%M:%S WIB")
is_market_open = "Market Open" if datetime.now().hour < 15 else "Market Closed"

change_class = "price-change-positive" if price_change >= 0 else "price-change-negative"
change_symbol = "↑" if price_change >= 0 else "↓"
change_sign = "+" if price_change >= 0 else ""

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    create_sidebar_header()
    create_kontrol_sistem_card()
    
    # Selectbox untuk pilih algoritma
    algo = st.selectbox("", ["GRU", "LSTM"], label_visibility="collapsed")
    model, scaler, info = load_assets(algo)
    
    create_model_info_card(algo, info)
    create_live_price_card(latest_close, price_change, price_change_pct, change_class, change_sign, change_symbol, current_time, is_market_open)
    create_status_sistem_card()
    create_sidebar_footer()

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title("📊 Dashboard Prediksi Saham BBCA")

if st.button("🔄 Refresh Data Saham", type="primary"):
    st.cache_data.clear()
    st.rerun()

usd_idr = get_usd_idr()
inflasi_data = get_inflasi_terbaru()

# PREDIKSI
preds = predict_recursive(model, scaler, df.values, 5)

future_dates = []
d = latest_date + timedelta(days=1)
while len(future_dates) < 5:
    if d.weekday() < 5:
        future_dates.append(d)
    d += timedelta(days=1)

# METRICS
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Harga Terakhir", f"Rp {latest_close:,.0f}")
c2.metric("Model", algo)

if preds and future_dates:
    next_pred = preds[0]
    next_date = future_dates[0]
    delta = next_pred - latest_close
    c3.metric(f"Prediksi {next_date.strftime('%a, %d %b')}",
              f"Rp {next_pred:,.0f}", f"{delta:,.0f}")

c4.metric("Kurs USD/IDR", f"{usd_idr:,.0f}" if usd_idr else "N/A")
c5.metric(
    label="Inflasi Indonesia",
    value=inflasi_data["nilai"],
    delta=inflasi_data["periode"],
    help=f"{inflasi_data['jenis']} • Update: {inflasi_data['update']}"
)

# GRAFIK
st.markdown('<div class="card">', unsafe_allow_html=True)
col_title, col_chart_type = st.columns([6, 4])
with col_title:
    st.subheader("📈 Harga Saham BBCA & Prediksi")
with col_chart_type:
    chart_type = st.selectbox(
        "Tipe Grafik",
        ["Candlestick (OHLC)", "Line Chart", "Area Chart",
         "Bar Chart (Close)", "Line + Markers"],
        index=0,
        label_visibility="collapsed"
    )

fig = create_chart(df, latest_date, future_dates, latest_close, preds, algo, chart_type)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ESTIMASI 5 HARI & TABEL
left, right = st.columns([7, 3])

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔮 Estimasi 5 Hari Trading")
    for i in range(5):
        harga = preds[i]
        delta = harga - latest_close
        warna = "#16a34a" if delta > 0 else "#dc2626"
        simbol = "▲" if delta > 0 else "▼"
        tanda = "+" if delta > 0 else ""
        st.markdown(f"""
        <div class="mini-card" style="border-left-color: {warna};">
            <b>{future_dates[i].strftime("%a, %d %b")}</b><br>
            <span style="font-size: 19px; font-weight: 600;">Rp {harga:,.2f}</span><br>
            <span style="color:{warna}; font-weight:bold; font-size:15px;">
                {simbol} {tanda}{delta:,.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Data Historical (60 Hari Terakhir)")

    hist = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # urutkan dulu saat masih datetime
    hist = hist.sort_index(ascending=False)

    # baru ubah format tanggal
    hist.index = hist.index.strftime('%d/%m/%Y')
    hist.index.name = "Tanggal"

    st.dataframe(
        hist.style.format({
            "Open": "Rp {:,.2f}",
            "High": "Rp {:,.2f}",
            "Low": "Rp {:,.2f}",
            "Close": "Rp {:,.2f}",
            "Volume": "{:,.0f}"
        }),
        use_container_width=True,
        height=400
    )

    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Dashboard Prediksi Saham BBCA • Hover di Candlestick untuk melihat OHLCV")