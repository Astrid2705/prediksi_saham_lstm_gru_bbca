"""
config.py
---------
Bertugas mengatur konfigurasi awal halaman Streamlit
dan memuat file CSS eksternal (style.css).

Fungsi di sini dipanggil PERTAMA KALI saat web.py dijalankan,
sebelum komponen lain ditampilkan.
"""

import streamlit as st
from pathlib import Path


# =========================================================
# SETUP HALAMAN
# =========================================================
def setup_page():
    """
    Mengatur konfigurasi dasar halaman Streamlit.
    Harus dipanggil paling awal di web.py
    """

    st.set_page_config(
        page_title="Skripsi Saham - BBCA",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css()


# =========================================================
# LOAD CSS
# =========================================================
def load_css():
    """
    Membaca file style.css lalu memasukkan CSS ke Streamlit.

    Jika file CSS tidak ditemukan atau error encoding,
    maka otomatis memakai fallback CSS.
    """

    css_path = Path(__file__).parent / "style.css"

    # =====================================================
    # JIKA FILE CSS ADA
    # =====================================================
    if css_path.exists():

        try:
            # PENTING:
            # encoding='utf-8' untuk menghindari UnicodeDecodeError
            with open(css_path, "r", encoding="utf-8") as f:
                css = f.read()

            st.markdown(
                f"<style>{css}</style>",
                unsafe_allow_html=True
            )

        except UnicodeDecodeError:
            st.error(
                "Encoding style.css bermasalah. "
                "Simpan file sebagai UTF-8 di VS Code."
            )

        except Exception as e:
            st.error(f"Gagal membaca CSS: {e}")

    # =====================================================
    # FALLBACK CSS
    # =====================================================
    else:

        st.markdown("""
        <style>

        /* =================================================
           GLOBAL
        ================================================= */

        .main {
            background-color: #f8f7f4;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        h1, h2, h3, h4 {
            color: #111827;
            font-weight: 700;
        }

        p, span, label {
            color: #374151;
        }

        /* =================================================
           CARD
        ================================================= */

        .card {
            background: white;
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 20px;

            box-shadow:
                0 10px 25px rgba(0,0,0,0.08);

            transition: all 0.3s ease;
        }

        .card:hover {
            transform: translateY(-2px);

            box-shadow:
                0 14px 30px rgba(0,0,0,0.12);
        }

        /* =================================================
           METRIC CARD
        ================================================= */

        .metric-card {
            background: white;
            border-radius: 18px;
            padding: 22px;

            text-align: center;

            box-shadow:
                0 10px 25px rgba(0,0,0,0.08);
        }

        .metric-title {
            font-size: 14px;
            color: #6b7280;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #111827;
        }

        /* =================================================
           BUTTON
        ================================================= */

        .stButton button {
            border-radius: 12px;
            border: none;

            background: #111827;
            color: white;

            padding: 10px 18px;
            font-weight: 600;

            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background: #1f2937;
            transform: scale(1.02);
        }

        /* =================================================
           SIDEBAR
        ================================================= */

        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        /* =================================================
           INPUT
        ================================================= */

        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div {
            border-radius: 10px;
        }

        /* =================================================
           TABLE
        ================================================= */

        .dataframe {
            border-radius: 12px;
            overflow: hidden;
        }

        /* =================================================
           ALERT
        ================================================= */

        .stAlert {
            border-radius: 14px;
        }

        </style>
        """, unsafe_allow_html=True)