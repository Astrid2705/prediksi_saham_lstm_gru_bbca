"""
sidebar.py
----------
Berisi semua fungsi yang menampilkan komponen UI di SIDEBAR kiri.

Isi file ini:
  - create_sidebar_header()       : Logo dan nama BBCA di atas sidebar
  - create_kontrol_sistem_card()  : Judul bagian "Kontrol Sistem"
  - create_model_info_card()      : Informasi model aktif beserta metrik evaluasinya
  - create_live_price_card()      : Harga saham terakhir secara live + status pasar
  - create_status_sistem_card()   : Status koneksi sistem
  - create_sidebar_footer()       : Teks hak cipta di bagian bawah
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import pytz

from utils import get_image_base64


def create_sidebar_header():
    logo_path = Path(__file__).parent / "image" / "BBCA.png"
    if logo_path.exists():
        img_base64 = get_image_base64(logo_path)
        if img_base64:
            st.markdown(f"""
            <div class="sidebar-header">
                <div class="logo-container">
                    <img src="data:image/png;base64,{img_base64}"
                         style="width:70px;height:70px;object-fit:contain;border-radius:14px;">
                </div>
                <h2>BBCA</h2>
                <p>PREDIKSI SAHAM</p>
            </div>
            """, unsafe_allow_html=True)
            return True

    st.markdown("""
    <div class="sidebar-header">
        <div class="logo-container">
            <div class="logo-box">BCA</div>
        </div>
        <h2>BBCA</h2>
        <p>PREDIKSI SAHAM</p>
    </div>
    """, unsafe_allow_html=True)
    return False

def create_sidebar_mini_icons():
    st.markdown("""
    <div class="sidebar-mini-icons">
        <div title="Dashboard">⌂</div>
        <div title="Analytics">◈</div>
        <div title="Model">◎</div>
        <div title="Settings">⚙</div>
    </div>
    """, unsafe_allow_html=True)

def create_kontrol_sistem_card():
    st.markdown("""
    <div class="kontrol-card">
        <div class="kontrol-title">⚙️ KONTROL SISTEM</div>
        <span class="algo-label">Pilih Algoritma Komparasi</span>
    </div>
    """, unsafe_allow_html=True)


def create_model_info_card(algo, info):
    st.markdown(f"""
    <div class="model-card">
        <div class="model-label">INFORMASI MODEL</div>
        <div style="margin-bottom:10px;">
            <span style="font-size:12px;color:rgba(255,255,255,0.6);">Model Aktif</span><br>
            <span class="model-badge">{algo}</span>
        </div>
        <div class="model-metrics-row">
            <div class="metric-box">
                <span class="metric-label">MAPE</span>
                <span class="metric-value">{info['mape']}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">RMSE</span>
                <span class="metric-value">{info['rmse']}</span>
            </div>
        </div>
        <div class="metric-box" style="margin-bottom:0;">
            <span class="metric-label">MAE</span>
            <span class="metric-value">{info['mae']}</span>
        </div>
        <div class="model-desc" style="margin-top:10px;">
            {algo} menggunakan Reset &amp; Update Gate.<br>
            Lebih efisien dalam komputasi untuk data pada penelitian ini.
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_live_price_card(latest_close, price_change, price_change_pct,
                           change_class, change_sign, change_symbol,
                           current_time, market):
    """
    Parameter:
        market : Dict dari get_market_status() — berisi state, label,
                 dot_class, badge_class, is_open, dot_color.

    Status pasar tampil otomatis:
        🟢 Market Buka  (hijau)  — jam reguler IDX
        🟡 Pra-Pasar    (kuning) — sebelum sesi buka
        🔵 Pasca-Pasar  (biru)   — setelah sesi tutup
        🔴 Market Tutup (merah)  — malam / weekend / libur
    """
    wit           = pytz.timezone('Asia/Jayapura')
    current_time  = datetime.now(wit).strftime("%d %b %Y %H:%M:%S WIT")
    change_symbol = "↓" if price_change < 0 else "↑"

    state_icon = {
        "REGULAR": "🟢",
        "PRE":     "🟡",
        "POST":    "🔵",
        "CLOSED":  "🔴",
    }.get(market.get("state", "CLOSED"), "🔴")

    dot_class    = market.get("dot_class",  "market-dot-close")
    market_label = market.get("label",      "Market Tutup")

    st.markdown(f"""
    <div class="live-price-box">
        <span class="live-section-label">{state_icon} HARGA TERAKHIR (LIVE)</span>
        <span class="live-ticker">BBCA.JK</span>
        <div class="live-price-number">Rp {latest_close:,.0f}</div>
        <span class="{change_class}">
            {change_sign}{price_change:,.0f} ({price_change_pct:+.2f}%) {change_symbol}
        </span>
        <span class="live-update-time">
            Update: {current_time}<br>
            <span class="{dot_class}">● {market_label}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


def create_status_sistem_card():
    st.markdown("""
    <div class="system-status">
        <span class="status-section-label">🖥️ STATUS SISTEM</span>
        <div class="status-row">
            <div class="status-dot"></div>
            <span class="status-text">Sistem Aktif</span>
        </div>
        <div class="status-sub">
            📡 Realtime data from Yahoo Finance
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_sidebar_footer():
    st.markdown("""
    <div class="sidebar-footer">
        © 2025 Skripsi Teknik Informatika<br>
        Prediksi Harga Saham BBCA (LSTM vs GRU)
    </div>
    """, unsafe_allow_html=True)