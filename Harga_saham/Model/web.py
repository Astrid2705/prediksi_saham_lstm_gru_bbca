"""
web.py
------
FILE UTAMA / DASHBOARD UTAMA

Layout halaman:
  1. Konfigurasi awal + auto-refresh 60 detik
  2. Sidebar kiri
  3. Info bar pojok kanan atas (fixed): market status, jam, refresh count
  4. Judul dashboard + tombol refresh data live
  5. 5 kartu metrik (Prediksi, Inflasi, Kurs, Harga Terakhir, Model Aktif)
  6. Header grafik (judul + legend + radio periode) | tipe grafik
  7. Grafik historis | Kartu estimasi 5 hari
  8. Tabel historical 60 hari
  9. Badge fitur unggulan

Cara menjalankan:
    streamlit run web.py
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import streamlit.components.v1 as components
from config import setup_page
from utils import (
    get_live_data,
    get_live_price,
    get_market_status,
    get_stock_change_info,
    get_usd_idr,
    get_inflasi_terbaru,
    get_laporan_keuangan,
)
from model_loader import (
    load_assets,
    predict_recursive,
    predict_historical
)
from sidebar import (
    create_sidebar_header,
    create_sidebar_mini_icons,
    create_kontrol_sistem_card,
    create_model_info_card,
    create_live_price_card,
    create_status_sistem_card,
    create_sidebar_footer,
)
from charts import create_chart

# =============================================================================
# 1. KONFIGURASI AWAL
# =============================================================================
setup_page()

if "refresh_count" not in st.session_state:
    st.session_state["refresh_count"] = 0

# =============================================================================
# 2. LOAD DATA SAHAM
# =============================================================================
df_full = get_live_data()
if df_full is None or df_full.empty:
    st.error("Gagal memuat data saham. Silakan refresh lagi.")
    st.stop()

df_model = df_full.tail(60)
latest_date = df_full.index[-1]


# =============================================================================
# EPS YAHOO FINANCE
# =============================================================================
try:
    ticker = yf.Ticker("BBCA.JK")

    eps_value = ticker.info.get("trailingEps")

except Exception:
    eps_value = None

# =============================================================================
# AUTO UPDATE KHUSUS DATA LIVE (tanpa refresh seluruh halaman)
# =============================================================================
if "refresh_count" not in st.session_state:
    st.session_state["refresh_count"] = 0

@st.fragment(run_every="30s")
def auto_update_live_data():
    st.session_state["refresh_count"] += 1

    live_price, live_change, live_change_pct = get_live_price()
    market = get_market_status()
    usd_idr = get_usd_idr()

    return live_price, live_change, live_change_pct, market, usd_idr


live_placeholder = st.empty()

with live_placeholder:
    live_price, live_change, live_change_pct, market, usd_idr = auto_update_live_data()

if live_price is not None:
    latest_close = live_price
    price_change = live_change
    price_change_pct = live_change_pct
else:
    latest_close, price_change, price_change_pct, _ = get_stock_change_info(df_full)

if price_change_pct is None:
    price_change_pct = 0.0
    price_change = 0.0

wit = pytz.timezone("Asia/Jayapura")
now_wit = datetime.now(wit)
time_str = now_wit.strftime("%H:%M:%S WIT")
date_str = now_wit.strftime("%d %B %Y %H:%M:%S WIT")

is_open = market["is_open"]
market_label = market["label"]

change_symbol = "↑" if price_change >= 0 else "↓"
change_sign = "+" if price_change >= 0 else ""
_refresh_count = st.session_state["refresh_count"]

# =============================================================================
# 3. SIDEBAR
# =============================================================================
with st.sidebar:
    create_sidebar_header()
    create_sidebar_mini_icons()
    create_kontrol_sistem_card()

    algo = st.selectbox("", ["GRU", "LSTM"], label_visibility="collapsed")
    model, scaler, info = load_assets(algo)

    create_model_info_card(algo, info)
    create_live_price_card(
        latest_close,
        price_change,
        price_change_pct,
        "",
        change_sign,
        change_symbol,
        date_str,
        market,
    )
    create_status_sistem_card()
    create_sidebar_footer()

# =============================================================================
# 4. PREDIKSI
# =============================================================================
preds = predict_recursive(model, scaler, df_model.values, days=5)

# =============================================================================
# HISTORICAL PREDICTION (BACKTEST)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_hist_prediction(data):
    _, pred = predict_historical(
        model,
        scaler,
        data
    )
    return pred

# cukup ambil data maksimal 1 tahun
data_hist = df_full.tail(425)

pred_hist = get_hist_prediction(
    data_hist.values
)

hist_dates = data_hist.index[60:]
future_dates = []
d = latest_date + timedelta(days=1)
while len(future_dates) < 5:
    if d.weekday() < 5:
        future_dates.append(d)
    d += timedelta(days=1)

inflasi_data = get_inflasi_terbaru()
next_pred = preds[0] if preds else latest_close
next_date = future_dates[0]
pred_delta = next_pred - latest_close
pred_delta_sign = "+" if pred_delta >= 0 else ""
pred_delta_color = "#16a34a" if pred_delta >= 0 else "#dc2626"
pred_arrow = "↑" if pred_delta >= 0 else "↓"
kurs_val = f"Rp {usd_idr:,.0f}" if usd_idr else "N/A"

# =============================================================================
# 5. INFO BAR
# =============================================================================
refresh_info = f"Auto-update #{_refresh_count} · 30 dtk"
st.markdown(
    f"""
<div class="info-bar-fixed">
<span class="{market['badge_class']}"><span style="color:{market['dot_color']};">●</span>&nbsp;{market_label}</span>
<span class="top-time">🕐 {time_str}</span>
<span class="top-autorefresh">⟳ {refresh_info}</span>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# 6. HEADER
# =============================================================================
st.markdown(
    f"""
<div class="dashboard-header-area">
<div class="dashboard-title">Dashboard Prediksi Harga Saham BBCA</div>
<div class="dashboard-subtitle">Aplikasi Real-Time Menggunakan Algoritma <span class="algo-highlight">{algo}</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# 7. CARD METRIK (URUTAN BARU)
# Harga Live | Model Aktif | Prediksi | Kurs | Inflasi
# =============================================================================
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="card-icon icon-blue">📈</div>
        <span class="card-label">Harga Live</span>
        <span class="card-value">Rp {latest_close:,.0f}</span>
        <span class="{'card-change-pos' if price_change >= 0 else 'card-change-neg'}">{change_sign}{price_change:,.0f} ({price_change_pct:+.2f}%) {change_symbol}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="card-icon icon-purple">🤖</div>
        <span class="card-label">Model Aktif</span>
        <span class="card-value">{algo}</span>
        <span class="card-sub">Algoritma Terpilih</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="card-icon icon-blue">🔮</div>
        <span class="card-label">Prediksi {next_date.strftime('%d %b')}</span>
        <span class="card-value-sm">Rp {next_pred:,.0f}</span>
        <span style="color:{pred_delta_color};">{pred_delta_sign}{pred_delta:,.0f} {pred_arrow}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="card-icon icon-teal">💱</div>
        <span class="card-label">Kurs USD/IDR</span>
        <span class="card-value-sm">{kurs_val}</span>
        <span class="card-sub">Live</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="card-icon icon-pink">📊</div>
        <span class="card-label">Inflasi</span>
        <span class="card-value-sm">{inflasi_data['nilai']}</span>
        <span class="card-sub">{inflasi_data['periode']}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )




# =============================================================================
# 8. GRAFIK + KARTU ESTIMASI
# =============================================================================
if "periode_grafik" not in st.session_state:
    st.session_state["periode_grafik"] = "3 Bulan"

period_map = {
    "1 Minggu": 7,
    "3 Minggu": 21,
    "1 Bulan": 30,
    "3 Bulan": 90,
    "6 Bulan": 180,
    "1 Tahun": 365
}
show_eps = st.session_state["periode_grafik"] in [
    "3 Bulan",
    "6 Bulan",
    "1 Tahun"
]
# ── Header baris: judul+legend+radio (kiri) | tipe chart (kanan) ─────────────

st.markdown("""
    <h2 style="
        text-align:center;
        font-size:30px;
        font-weight:700;
        color:#1e293b;
        margin-top:10px;
        margin-bottom:20px;
    ">
        Grafik Saham (BBCA.JK)
    </h2>
    """, unsafe_allow_html=True)

col_chart_header, col_chart_type = st.columns([7, 3])

with col_chart_header:

    # Radio periode — tepat di bawah judul grafik
    selected_period = st.radio(
        "Periode",
        options=list(period_map.keys()),
        index=list(period_map.keys()).index(st.session_state["periode_grafik"]),
        horizontal=True,
        label_visibility="collapsed",
        key="periode_grafik"
    )
    

with col_chart_type:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    chart_type = st.selectbox(
        "Tipe Grafik",
        [
            "Line Chart",
            "Candlestick (OHLC)",
            "Area Chart",
            "Bar Chart (Close)",
            "Line + Markers",
        ],
    )

n_days = period_map[selected_period]
df_chart = df_full.tail(n_days)

start_date = df_chart.index[0]

filtered_dates = []
filtered_preds = []

for d, p in zip(hist_dates, pred_hist):
    if d >= start_date:
        filtered_dates.append(d)
        filtered_preds.append(p)


# ── Grafik (kiri) | Kartu Estimasi (kanan) ───────────────────────────────────
col_grafik, col_estimasi = st.columns([7, 3])

with col_grafik:

    eps_df = pd.DataFrame()

    if show_eps:
        try:
            ticker = yf.Ticker("BBCA.JK")

            q = ticker.quarterly_financials

            eps = q.loc["Basic EPS"]
            net_income = q.loc["Net Income"]

            SHARES = 123_275_000_000

            eps_data = []

            for dt in net_income.index:

                eps_yahoo = eps.get(dt, np.nan)

                if pd.notna(eps_yahoo):
                    eps_final = eps_yahoo
                else:
                    eps_final = net_income[dt] / SHARES

                eps_data.append({
                    "Tanggal": dt,
                    "EPS": round(eps_final, 2)
                })

            eps_df = pd.DataFrame(eps_data)

            eps_df = eps_df.sort_values("Tanggal")

            eps_df = eps_df[
                eps_df["Tanggal"] >= df_chart.index.min()
            ]

        except Exception as e:
            print(e)
            eps_df = pd.DataFrame()
    
    fig = create_chart(
        df_chart,
        latest_date,
        future_dates,
        latest_close,
        preds,
        filtered_preds,
        filtered_dates,
        algo,
        chart_type,
        show_eps,
        eps_df
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

with col_estimasi:
    # Bangun baris tabel dalam satu string agar tidak ada tag HTML terputus
    rows_html = ""
    for i in range(5):
        h = preds[i]
        delta = h - latest_close
        tanda = "+" if delta >= 0 else ""
        kelas = "change-pos" if delta >= 0 else "change-neg"
        simbol = "▲" if delta >= 0 else "▼"
        bg_row = "#f0fdf4" if delta >= 0 else "#fff5f5"
        rows_html += (
            f"<tr style='background:{bg_row};'>"
            f"<td style='color:#94a3b8;font-weight:600;'>{i + 1}</td>"
            f"<td style='font-weight:600;'>{future_dates[i].strftime('%a, %d %b')}</td>"
            f"<td style='font-weight:800;'>Rp {h:,.0f}</td>"
            f"<td class='{kelas}'>{tanda}{delta:,.0f} {simbol}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <div class="estimation-container">
            <div class="est-header">
                <span class="section-title" style="margin-bottom:0;">
                    📅 Estimasi 5 Hari Kerja
                </span>
                <span class="hist-badge">{algo}</span>
            </div>
            <div class="est-ref-row">
                Referensi harga terakhir:&nbsp;
                <strong>Rp {latest_close:,.0f}</strong>
            </div>
            <table class="est-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Tanggal</th>
                        <th>Estimasi</th>
                        <th>Selisih</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>    
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)
# =============================================================================
# HISTORICAL DATA
# =============================================================================

st.markdown("---")

with st.container():

    st.markdown("""
    <h2 style="
        text-align:center;
        font-size:24px;
        font-weight:700;
        color:#1e293b;
        margin-top:10px;
        margin-bottom:5px;
    ">
        Historical Data (BBCA.JK)
    </h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="
        text-align:center;
        color:#64748b;
        font-size:14px;
        margin-bottom:15px;
    ">
        60 Hari Terakhir
    </p>
    """, unsafe_allow_html=True)

    hist = df_full[
        ["Open", "High", "Low", "Close", "Volume"]
    ].tail(60).copy()

    try:
        ticker = yf.Ticker("BBCA.JK")

        net_income = ticker.quarterly_financials.loc["Net Income"]

        latest_net_income = net_income.iloc[0]

        hist["Net Income"] = latest_net_income

    except:
        hist["Net Income"] = np.nan

    hist = hist.sort_index(ascending=False)

    hist.index = hist.index.strftime("%d/%m/%Y")
    hist.index.name = "Tanggal"

    st.dataframe(
        hist.style.format(
            {
            "Open": "Rp {:,.2f}",
            "High": "Rp {:,.2f}",
            "Low": "Rp {:,.2f}",
            "Close": "Rp {:,.2f}",
            "Volume": "{:,.0f}",
            "Net Income": "{:,.0f}"
            }
        ),
        use_container_width=True,
        height=353,
    )
