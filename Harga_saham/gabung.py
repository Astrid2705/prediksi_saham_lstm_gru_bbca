import pandas as pd
import yfinance as yf
import os

def prepare_dataset():
    input_file = 'Data/laporan keuangan bca.csv'
    output_file = 'dataset_siap_training.csv'
    ticker_symbol = "BBCA.JK"
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} tidak ditemukan.")
        return

    try:
        # Load data
        df = pd.read_csv(input_file, sep=';')

        # Rapikan nama kolom
        df.columns = df.columns.str.strip()

        # Rename kolom (perbaiki typo!)
        df = df.rename(columns={
            'Laba Bersih (Jutaan Rp)': 'Laba_Bersih'
        })

        # Konversi tanggal (handle format Mar-18)
        df['Periode'] = pd.to_datetime(df['Periode'], format='%b-%y', errors='coerce')

        # Hapus baris yang gagal parsing tanggal
        df.sort_values('Periode').reset_index(drop=True)

        # Download data saham
        print(f"Downloading historical data for {ticker_symbol}...")
        stock_data = yf.download(ticker_symbol, start="2015-01-01")['Close']

        if stock_data.empty:
            print("Error: Gagal mengambil data saham.")
            return

        # Pastikan index datetime & urut
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data = stock_data.sort_index()

        # Ambil harga saham terdekat sebelum tanggal laporan
        df['Harga_Close'] = df['Periode'].apply(lambda x: stock_data.asof(x))

        # Hapus data kosong
        df = df.dropna()

        # Simpan hasil
        df.to_csv(output_file, index=False)

        print(f"Success: Dataset disimpan di {output_file}")
        print(df.head())

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    prepare_dataset()