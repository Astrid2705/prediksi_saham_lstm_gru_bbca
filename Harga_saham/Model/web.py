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
)
from model_loader import load_assets, predict_recursive
from sidebar import (
    create_sidebar_header,
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

# Auto-refresh setiap 60 detik via JavaScript — tidak perlu package tambahan.
components.html(
    "<script>setTimeout(function(){window.parent.location.reload();}, 60000);</script>",
    height=0,
)

if "refresh_count" not in st.session_state:
    st.session_state["refresh_count"] = 0
else:
    st.session_state["refresh_count"] += 1
_refresh_count = st.session_state["refresh_count"]


# =============================================================================
# 2. LOAD DATA SAHAM
# =============================================================================
df_full = get_live_data()
if df_full is None or df_full.empty:
    st.error("Gagal memuat data saham. Silakan refresh lagi.")
    st.stop()

df_model    = df_full.tail(60)
latest_date = df_full.index[-1]

live_price, live_change, live_change_pct = get_live_price()
if live_price is not None:
    latest_close     = live_price
    price_change     = live_change
    price_change_pct = live_change_pct
else:
    latest_close, price_change, price_change_pct, _ = get_stock_change_info(df_full)

if price_change_pct is None:
    price_change_pct = 0.0
    price_change     = 0.0

wit      = pytz.timezone("Asia/Jayapura")
now_wit  = datetime.now(wit)
time_str = now_wit.strftime("%H:%M:%S WIT")
date_str = now_wit.strftime("%d %B %Y %H:%M:%S WIT")

market       = get_market_status()
is_open      = market["is_open"]
market_label = market["label"]

change_class  = "price-change-positive" if price_change >= 0 else "price-change-negative"
change_symbol = "↑" if price_change >= 0 else "↓"
change_sign   = "+" if price_change >= 0 else ""


# =============================================================================
# 3. SIDEBAR
# =============================================================================
with st.sidebar:
    create_sidebar_header()
    create_kontrol_sistem_card()

    algo = st.selectbox("", ["GRU", "LSTM"], label_visibility="collapsed")
    model, scaler, info = load_assets(algo)

    create_model_info_card(algo, info)
    create_live_price_card(
        latest_close, price_change, price_change_pct,
        change_class, change_sign, change_symbol,
        date_str, market,
    )
    create_status_sistem_card()
    create_sidebar_footer()


# =============================================================================
# 4. PREDIKSI & DATA TAMBAHAN
# =============================================================================
preds = predict_recursive(model, scaler, df_model.values, days=5)

future_dates = []
d = latest_date + timedelta(days=1)
while len(future_dates) < 5:
    if d.weekday() < 5:
        future_dates.append(d)
    d += timedelta(days=1)

usd_idr      = get_usd_idr()
inflasi_data = get_inflasi_terbaru()

next_pred        = preds[0] if preds else latest_close
next_date        = future_dates[0] if future_dates else latest_date + timedelta(days=1)
pred_delta       = next_pred - latest_close
pred_delta_sign  = "+" if pred_delta >= 0 else ""
pred_delta_color = "#16a34a" if pred_delta >= 0 else "#dc2626"
pred_arrow       = "↑" if pred_delta >= 0 else "↓"
kurs_val         = f"Rp {usd_idr:,.0f}" if usd_idr else "N/A"


# =============================================================================
# 5. INFO BAR — POJOK KANAN ATAS (fixed, tidak mengambil ruang layout)
# =============================================================================
refresh_info = (
    f"Auto-refresh #{_refresh_count} · 60 dtk"
    if _refresh_count > 0
    else "Auto-refresh aktif · 60 dtk"
)
st.markdown(
    f"""
    <div class="info-bar-fixed">
        <span class="{market['badge_class']}">
            <span style="color:{market['dot_color']};">●</span>&nbsp;{market_label}
        </span>
        <span class="top-time">🕐 {time_str}</span>
        <span class="top-autorefresh">⟳ {refresh_info}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 6. JUDUL DASHBOARD + TOMBOL REFRESH DATA LIVE
# =============================================================================
col_title, col_btn = st.columns([11, 1])

with col_title:
    st.markdown(
        f"""
        <div class="dashboard-header-area">
            <div class="dashboard-title">Dashboard Prediksi Harga Saham BBCA</div>
            <div class="dashboard-subtitle">
                Aplikasi Real-Time Menggunakan Algoritma
                <span class="algo-highlight">{algo}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_btn:
    # Spacer kecil agar tombol sejajar vertikal dengan judul
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    if st.button(
        "🔄",
        help="Refresh data live saja (harga, status pasar, kurs) — tampilan tidak berubah",
        use_container_width=True,
    ):
        # Hanya hapus cache data live — bukan seluruh cache
        get_live_price.clear()
        get_market_status.clear()
        get_usd_idr.clear()
        st.rerun()


# =============================================================================
# 7. KARTU METRIK — 5 kartu, 1 baris
# Urutan: Prediksi | Inflasi | Kurs | Harga Terakhir | Model Aktif
# =============================================================================
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="card-icon icon-blue">🔮</div>
            <span class="card-label">Prediksi {next_date.strftime('%a, %d %b')}</span>
            <span class="card-value-sm">Rp {next_pred:,.0f}</span>
            <span style="font-size:12px;font-weight:600;color:{pred_delta_color};">
                {pred_delta_sign}{pred_delta:,.0f} {pred_arrow}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="card-icon icon-pink">📊</div>
            <span class="card-label">Inflasi Indonesia</span>
            <span class="card-value-sm">{inflasi_data['nilai']}</span>
            <span class="card-sub">{inflasi_data['periode']} · {inflasi_data['jenis']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="card-icon icon-teal">💱</div>
            <span class="card-label">Kurs USD / IDR</span>
            <span class="card-value-sm">{kurs_val}</span>
            <span class="card-sub">Live · Yahoo Finance</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="card-icon icon-blue">📈</div>
            <span class="card-label">Harga Terakhir (Live)</span>
            <span class="card-value">Rp {latest_close:,.0f}</span>
            <span class="{'card-change-pos' if price_change >= 0 else 'card-change-neg'}">
                {change_sign}{price_change:,.0f} ({price_change_pct:+.2f}%) {change_symbol}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="card-icon icon-purple">🤖</div>
            <span class="card-label">Model Aktif</span>
            <span class="card-value" style="color:#7c3aed;">{algo}</span>
            <span class="card-sub">Algoritma Terpilih</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)


# =============================================================================
# 8. GRAFIK + KARTU ESTIMASI
# =============================================================================
if "periode_grafik" not in st.session_state:
    st.session_state["periode_grafik"] = "3M"

period_map = {"1M": 30, "3M": 90, "6M": 180, "1T": 365, "Semua": len(df_full)}

# ── Header baris: judul+legend+radio (kiri) | tipe chart (kanan) ─────────────
col_chart_header, col_chart_type = st.columns([7, 3])

with col_chart_header:
    st.markdown(
        f"""
        <div class="chart-header-block">
            <div class="chart-title-text">Tren Harga Historis vs Estimasi {algo}</div>
            <div class="chart-legend-row">
                <span class="legend-item">
                    <span class="legend-line-blue"></span> Harga Historis
                </span>
                <span class="legend-item">
                    <span class="legend-line-orange"></span> Estimasi {algo}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Radio periode — tepat di bawah judul grafik
    selected_period = st.radio(
        "Periode",
        options=list(period_map.keys()),
        index=list(period_map.keys()).index(st.session_state["periode_grafik"]),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["periode_grafik"] = selected_period

with col_chart_type:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    chart_type = st.selectbox(
        "Tipe Grafik",
        ["Line Chart", "Candlestick (OHLC)", "Area Chart",
         "Bar Chart (Close)", "Line + Markers"],
    )

n_days   = period_map[selected_period]
df_chart = df_full.tail(n_days)

# ── Grafik (kiri) | Kartu Estimasi (kanan) ───────────────────────────────────
col_grafik, col_estimasi = st.columns([6, 4])

with col_grafik:
    fig = create_chart(
        df_chart, latest_date, future_dates,
        latest_close, preds, algo, chart_type,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col_estimasi:
    # Bangun baris tabel dalam satu string agar tidak ada tag HTML terputus
    rows_html = ""
    for i in range(5):
        h      = preds[i]
        delta  = h - latest_close
        tanda  = "+" if delta >= 0 else ""
        kelas  = "change-pos" if delta >= 0 else "change-neg"
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
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)


# =============================================================================
# 9. TABEL HISTORICAL DATA — 60 hari terakhir
# =============================================================================
st.markdown(
    """
    <div class="historical-container">
        <div class="hist-header-row">
            <span class="section-title" style="margin-bottom:0;">
                Historical Data (BBCA.JK)
            </span>
            <span class="hist-badge">60 Hari Terakhir</span>
        </div>
    """,
    unsafe_allow_html=True,
)

hist = df_full[["Open", "High", "Low", "Close", "Volume"]].tail(60).copy()
hist = hist.sort_index(ascending=False)
hist.index = hist.index.strftime("%d/%m/%Y")
hist.index.name = "Tanggal"

st.dataframe(
    hist.style.format({
        "Open":   "Rp {:,.2f}",
        "High":   "Rp {:,.2f}",
        "Low":    "Rp {:,.2f}",
        "Close":  "Rp {:,.2f}",
        "Volume": "{:,.0f}",
    }),
    use_container_width=True,
    height=400,
)

st.markdown(
    """
        <span class="hist-source">📡 Data diambil dari Yahoo Finance (BBCA.JK)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)


# =============================================================================
# 10. BADGE FITUR UNGGULAN
# =============================================================================
b1, b2, b3, b4 = st.columns(4)

features = [
    ("⚡", "icon-orange", "Real-Time Data",         "Data diperbarui langsung dari pasar"),
    ("🧠", "icon-purple", "Deep Learning",           "Arsitektur GRU/LSTM"),
    ("🎯", "icon-green",  "Prediksi Akurat",         "Model dioptimasi untuk hasil terbaik"),
    ("📊", "icon-blue",   "Visualisasi Interaktif",  "Grafik modern & responsif"),
]

for col, (icon, ic, title, desc) in zip([b1, b2, b3, b4], features):
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon {ic}">{icon}</div>
                <div class="feature-text">
                    <span class="feature-title">{title}</span>
                    <span class="feature-desc">{desc}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)