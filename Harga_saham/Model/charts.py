"""
charts.py
---------
Bertugas membuat grafik/visualisasi interaktif menggunakan Plotly.

Isi file ini:
  - create_chart() : Membuat grafik harga historis + garis prediksi

Grafik mendukung 5 jenis tampilan yang bisa dipilih pengguna:
  1. Candlestick (OHLC)  — Batang lilin dengan Open, High, Low, Close
  2. Line Chart          — Garis sederhana (tampilan default)
  3. Area Chart          — Garis dengan area terisi di bawahnya
  4. Bar Chart (Close)   — Batang berdasarkan harga Close saja
  5. Line + Markers      — Garis dengan titik-titik penanda

Semua jenis grafik dilengkapi tooltip OHLC (Open, High, Low, Close)
saat kursor diarahkan ke titik data manapun.
"""

import plotly.graph_objects as go
import numpy as np


def create_chart(
    df,
    latest_date,
    future_dates,
    latest_close,
    preds,
    pred_hist,
    hist_dates,
    algo,
    chart_type,
    show_eps,
    eps_df
):
    """
    Membuat grafik interaktif yang menggabungkan data historis
    dan garis prediksi 5 hari ke depan.

    Parameter:
        df           : DataFrame berisi data historis (sudah difilter sesuai periode)
        latest_date  : Tanggal data terakhir (titik awal garis prediksi)
        future_dates : List tanggal 5 hari ke depan (hari kerja saja)
        latest_close : Harga penutupan terakhir (rupiah)
        preds        : List hasil prediksi harga untuk 5 hari ke depan
        algo         : Nama algoritma untuk label di legenda ("GRU" / "LSTM")
        chart_type   : Jenis grafik yang dipilih pengguna (string)

    Mengembalikan: objek Figure Plotly yang siap ditampilkan dengan st.plotly_chart()
    """
    fig = go.Figure()

    # Siapkan customdata OHLC untuk semua jenis grafik agar tooltip seragam
    ohlc_data = np.stack((df['Open'], df['High'], df['Low'], df['Close']), axis=1)

    ohlc_hover = (
        "<b>%{x|%d %b %Y}</b><br>"
        "Open:  Rp %{customdata[0]:,.2f}<br>"
        "High:  Rp %{customdata[1]:,.2f}<br>"
        "Low:   Rp %{customdata[2]:,.2f}<br>"
        "Close: Rp %{customdata[3]:,.2f}<extra></extra>"
    )

    # ── Tampilan grafik historis sesuai pilihan pengguna ──────────────────────

    if chart_type == "Candlestick (OHLC)":
        # Grafik lilin: menampilkan 4 harga sekaligus (Open, High, Low, Close)
        # Hijau = harga naik, Merah = harga turun
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='OHLC Historis',
            increasing_line_color='#16a34a',
            decreasing_line_color='#dc2626',
            customdata=ohlc_data,
            hovertemplate=ohlc_hover
        ))

    elif chart_type == "Line Chart":
        # Grafik garis bersih berdasarkan harga Close
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines',
            name='Harga Historis',
            line=dict(width=2.5, color='#2563eb'),
            customdata=ohlc_data,
            hovertemplate=ohlc_hover
        ))

    elif chart_type == "Area Chart":
        # Grafik garis dengan area berwarna biru di bawahnya
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(37,99,235,0.08)',
            name='Harga Historis',
            line=dict(width=2.5, color='#2563eb'),
            customdata=ohlc_data,
            hovertemplate=ohlc_hover
        ))

    elif chart_type == "Bar Chart (Close)":
        # Grafik batang: hijau jika harga naik dari hari sebelumnya, merah jika turun
        bar_colors = [
            '#16a34a' if df['Close'].iloc[i] >= df['Close'].iloc[i - 1] else '#dc2626'
            for i in range(len(df))
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df['Close'],
            name='Close Price',
            marker_color=bar_colors,
            customdata=ohlc_data,
            hovertemplate=ohlc_hover
        ))

    elif chart_type == "Line + Markers":
        # Grafik garis dengan titik bulat di setiap data point
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines+markers',
            name='Harga Historis',
            line=dict(width=2.5, color='#2563eb'),
            marker=dict(size=5, color='#2563eb'),
            customdata=ohlc_data,
            hovertemplate=ohlc_hover
        ))

     # ── Prediksi historis model ───────────────────────────────────────────────
    if len(pred_hist) > 0:
        fig.add_trace(
            go.Scatter(
                x=hist_dates,
                y=pred_hist,
                mode='lines',
                name=f'Prediksi Historis {algo}',
                line=dict(
                    color='#ef4444',
                    width=2,
                    dash='dash'
                ),
                hovertemplate=
                "<b>%{x|%d %b %Y}</b><br>"
                "Prediksi: Rp %{y:,.0f}<extra></extra>"
            )
        )    

    # ── Garis vertikal pemisah historis vs prediksi ───────────────────────────
    # Garis putus-putus di titik tanggal terakhir sebagai batas visual
    fig.add_vline(
        x=latest_date,
        line_width=1.5,
        line_dash="dash",
        line_color="rgba(100,116,139,0.5)"
    )

    # Anotasi label "Terakhir" di titik transisi
    fig.add_annotation(
        x=latest_date,
        y=latest_close,
        text=f"Terakhir<br>{latest_date.strftime('%d %b %Y')}<br><b>Rp {latest_close:,.0f}</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor="#64748b",
        ax=40, ay=-50,
        font=dict(size=11, color="#1e293b"),
        bgcolor="white",
        bordercolor="#e2e8f0",
        borderwidth=1,
        borderpad=6
    )

    # ── Garis prediksi (warna emas, selalu di atas grafik historis) ───────────
    fig.add_trace(go.Scatter(
        x=[latest_date] + future_dates,
        y=[latest_close] + preds,
        mode='lines+markers',
        name=f'Estimasi {algo}',
        line=dict(width=2.5, dash='dash', color='#d97706'),
        marker=dict(size=8, symbol='circle', color='#d97706',
                    line=dict(width=2, color='white')),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Estimasi: Rp %{y:,.2f}<extra></extra>"
    ))

    # ── Label bawah: "Data Historis ← | → Prediksi 5 Hari Kerja" ─────────────
    # Posisi tengah area historis
    mid_hist_idx = len(df) // 2
    if mid_hist_idx < len(df):
        mid_hist_date = df.index[mid_hist_idx]
        fig.add_annotation(
            x=mid_hist_date, y=df['Close'].min(),
            yref="y", xref="x",
            text="◄ Data Historis",
            showarrow=False,
            font=dict(size=11, color="#2563eb"),
            yanchor="top",
            yshift=-30
        )

    # Posisi tengah area prediksi
    if future_dates:
        mid_pred_date = future_dates[len(future_dates) // 2]
        fig.add_annotation(
            x=mid_pred_date, y=df['Close'].min(),
            yref="y", xref="x",
            text=f"Prediksi 5 Hari Kerja ►",
            showarrow=False,
            font=dict(size=11, color="#d97706"),
            yanchor="top",
            yshift=-30
        )

    # =====================================================
    # EPS QUARTERLY
    # =====================================================
    if show_eps and not eps_df.empty:                    # ← WAJIB ada guard ini
        eps_min = eps_df["EPS"].min()
        eps_max = eps_df["EPS"].max()
        eps_pad = (eps_max - eps_min) * 2               # padding 200%

        fig.add_trace(
            go.Scatter(
                x=eps_df["Tanggal"],
                y=eps_df["EPS"],
                mode="lines+markers",
                name="EPS (Quarterly)",
                line=dict(
                    shape="hv",
                    color="#16a34a",
                    width=1.5,
                    dash="dot"
                ),
                marker=dict(
                    size=10,
                    color="#16a34a",
                    line=dict(width=2, color="white")
                ),
                yaxis="y2"
            )
        )
    else:
        eps_min, eps_max, eps_pad = 0, 1, 2             # ← nilai default kalau EPS tidak ditampilkan

    # ── Pengaturan tampilan grafik ─────────────────────────────────────────────
    fig.update_layout(
        template="plotly_white",
        height=420,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=12)
        ),
        margin=dict(t=20, b=50, l=10, r=10),
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            tickformat="%d %b\n%Y",
            tickfont=dict(size=11, color="#64748b"),
            showline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            tickprefix="Rp ",
            tickformat=",.0f",
            tickfont=dict(size=11, color="#64748b"),
            showline=False
        ),
        yaxis2=dict(
            title="EPS",
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(size=11, color="#16a34a"),
            range=[eps_min - eps_pad, eps_max + eps_pad]
        )
    )

    return fig