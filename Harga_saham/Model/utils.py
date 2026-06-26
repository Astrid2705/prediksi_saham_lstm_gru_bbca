"""
utils.py
--------
Kumpulan fungsi utilitas / pembantu untuk mengambil
dan mengolah data eksternal maupun lokal.
"""

import base64
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR.parent / "Data" / "laporan keuangan bca.csv"

print("CSV_PATH =", CSV_PATH)
print("FILE ADA =", CSV_PATH.exists())
# =========================================================
# IMAGE
# =========================================================
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return encoded
    except Exception:
        return None


# =========================================================
# MARKET STATUS
# =========================================================
def _parse_market_state(state: str) -> dict:
    state = str(state).upper()

    mapping = {
        "REGULAR": {
            "state": "REGULAR",
            "is_open": True,
            "label": "Market Buka",
            "dot_class": "market-dot-open",
            "badge_class": "market-badge-open",
            "dot_color": "#16a34a",
        },
        "PRE": {
            "state": "PRE",
            "is_open": False,
            "label": "Pra-Pasar",
            "dot_class": "market-dot-pre",
            "badge_class": "market-badge-pre",
            "dot_color": "#d97706",
        },
        "POST": {
            "state": "POST",
            "is_open": False,
            "label": "Market Tutup",
            "dot_class": "market-dot-post",
            "badge_class": "market-badge-post",
            "dot_color": "#2563eb",
        },
    }

    if state in ("PREPRE",):
        state = "PRE"

    if state in ("POSTPOST",):
        state = "POST"

    return mapping.get(
        state,
        {
            "state": "CLOSED",
            "is_open": False,
            "label": "Market Tutup",
            "dot_class": "market-dot-close",
            "badge_class": "market-badge-closed",
            "dot_color": "#dc2626",
        },
    )


@st.cache_data(ttl=60, show_spinner=False)
def get_market_status() -> dict:

    # =============================
    # Method 1 - fast_info
    # =============================
    try:
        ticker = yf.Ticker("BBCA.JK")
        state = ticker.fast_info.get("market_state")

        if state:
            return _parse_market_state(state)

    except Exception:
        pass

    # =============================
    # Method 2 - info
    # =============================
    try:
        ticker = yf.Ticker("BBCA.JK")
        state = ticker.info.get("marketState")

        if state:
            return _parse_market_state(state)

    except Exception:
        pass

    # =============================
    # Fallback Manual WIB
    # =============================
    wib = pytz.timezone("Asia/Jakarta")
    now = datetime.now(wib)

    hari = now.weekday()
    menit = now.hour * 60 + now.minute

    # Sabtu / Minggu
    if hari >= 5:
        return _parse_market_state("CLOSED") 

    # Jam Bursa IDX
    if (9 * 60 <= menit < 12 * 60) or (13 * 60 + 30 <= menit < 15 * 60):
        return _parse_market_state("REGULAR")

    if 8 * 60 + 45 <= menit < 9 * 60:
        return _parse_market_state("PRE")

    if 15 * 60 <= menit < 15 * 60 + 30:
        return _parse_market_state("POST")

    return _parse_market_state("CLOSED")


# =========================================================
# LIVE PRICE
# =========================================================
@st.cache_data(ttl=60, show_spinner=False)
def get_live_price():

    # =============================
    # Method 1 - info
    # =============================
    try:
        ticker = yf.Ticker("BBCA.JK")
        info = ticker.info

        harga = info.get("regularMarketPrice") or info.get("currentPrice")
        prev = info.get("regularMarketPreviousClose") or info.get("previousClose")

        if harga and prev and prev > 0:
            selisih = harga - prev
            persen = (selisih / prev) * 100

            return float(harga), float(selisih), float(persen)

    except Exception:
        pass

    # =============================
    # Method 2 - fast_info
    # =============================
    try:
        fi = yf.Ticker("BBCA.JK").fast_info

        harga = fi.get("last_price")
        prev = fi.get("previous_close")

        if harga and prev and prev > 0:
            selisih = harga - prev
            persen = (selisih / prev) * 100

            return float(harga), float(selisih), float(persen)

    except Exception:
        pass

    # =============================
    # Method 3 - Intraday
    # =============================
    try:
        df = yf.download(
            "BBCA.JK",
            period="2d",
            interval="1m",
            auto_adjust=True,
            progress=False,
            timeout=10,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if not df.empty and "Close" in df.columns:

            harga = float(df["Close"].iloc[-1])

            hari_ini = df.index[-1].date()

            df_kemarin = df[
                pd.to_datetime(df.index).date < hari_ini
            ]

            if not df_kemarin.empty:

                prev = float(df_kemarin["Close"].iloc[-1])

                selisih = harga - prev
                persen = (selisih / prev) * 100

                return harga, selisih, persen

    except Exception:
        pass

    # =============================
    # Method 4 - Daily
    # =============================
    try:
        data = yf.download(
            "BBCA.JK",
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=10,
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if not data.empty and len(data) >= 2:

            harga = float(data["Close"].iloc[-1])
            prev = float(data["Close"].iloc[-2])

            selisih = harga - prev
            persen = (selisih / prev) * 100

            return harga, selisih, persen

    except Exception:
        pass

    return None, None, None


# =========================================================
# HISTORICAL DATA
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_live_data():

    try:
        with st.spinner("Mengambil data saham BBCA..."):

            data = yf.download(
                "BBCA.JK",
                period="2y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=15,
            )

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_cols = ["Open", "High", "Low", "Close", "Volume"]

        df = data[required_cols].copy()

        df["Laba_Bersih"] = 0.0237864

        return df

    except Exception as e:
        st.error(f"Gagal mengambil data saham: {e}")
        return None


# =========================================================
# STOCK CHANGE
# =========================================================
def get_stock_change_info(df):

    if df is None or df.empty or len(df) < 2:
        return None, None, None, None

    latest = float(df["Close"].iloc[-1])
    previous = float(df["Close"].iloc[-2])

    change = latest - previous
    change_pct = (change / previous) * 100

    return latest, change, change_pct, previous


# =========================================================
# USD IDR
# =========================================================
@st.cache_data(ttl=60, show_spinner=False)
def get_usd_idr():

    try:
        usd = yf.download(
            "USDIDR=X",
            period="5d",
            progress=False,
        )

        if isinstance(usd.columns, pd.MultiIndex):
            usd.columns = usd.columns.get_level_values(0)

        if usd.empty:
            return None

        return float(usd["Close"].iloc[-1])

    except Exception:
        return None


# =========================================================
# INFLASI
# =========================================================
@st.cache_data(ttl=10, show_spinner=False)
def get_inflasi_terbaru():

    return {
        "nilai": "2.51%",
        "periode": "juni 2026",
        "jenis": "YoY (Tahunan)",
        "update": "Data statis (manual input untuk kebutuhan sistem)",
    }
# =========================================================
# LAPORAN KEUANGAN
# =========================================================
@st.cache_data(ttl=10, show_spinner=False)
def get_laporan_keuangan():

    try:
        df = pd.read_csv(
            CSV_PATH,
            sep=";"
        )

        df.columns = [
            "Periode",
            "Total_Aset",
            "Kredit",
            "Laba_Bersih",
            "DPK"
        ]

        # Hilangkan titik pemisah ribuan
        for col in ["Total_Aset", "Kredit", "Laba_Bersih", "DPK"]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        return df

    except Exception as e:
        st.error(f"Gagal membaca laporan keuangan: {e}")
        return None