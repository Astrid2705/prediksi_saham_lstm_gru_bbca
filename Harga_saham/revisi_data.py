import yfinance as yf
import pandas as pd

# 1. AMBIL DATA DARI INTERNET (YAHOO FINANCE)
print("Sedang mengambil data saham terbaru...")
data_saham = yf.download("BBCA.JK", start="2015-01-01")

data_saham.columns = data_saham.columns.get_level_values(0)

data_saham = data_saham.reset_index()

# Pastikan format tanggalnya bersih
data_saham['Date'] = pd.to_datetime(data_saham['Date']).dt.tz_localize(None)

# 2. BACA DATA LABA DARI FILE 
print("Membaca data laba dari file lama...")
data_lama = pd.read_csv('Data/dataset_siap_training.csv')

# Bersihkan angka laba (hapus titik agar jadi angka asli)
kolom_laba = 'Laba_Bersih'
data_lama[kolom_laba] = data_lama[kolom_laba].astype(str).str.replace('.', '', regex=False)
data_lama[kolom_laba] = pd.to_numeric(data_lama[kolom_laba])

# Pastikan kolom 'Periode' di file kamu jadi format tanggal
data_lama['Periode'] = pd.to_datetime(data_lama['Periode']).dt.tz_localize(None)

# 3. GABUNGKAN DATA SAHAM DAN DATA LABA
print("Menggabungkan data...")
dataset_lengkap = pd.merge(data_saham, data_lama[['Periode', kolom_laba]], 
                           left_on='Date', right_on='Periode', how='left')

# Isi bagian yang kosong (karena laba biasanya nggak berubah tiap hari)
dataset_lengkap[kolom_laba] = dataset_lengkap[kolom_laba].ffill()

# Hapus kolom periode biar nggak double
dataset_lengkap = dataset_lengkap.drop(columns=['Periode'])

# 4. SIMPAN HASILNYA
dataset_lengkap.to_csv("Data/dataset_final_skripsi.csv", index=False)

print("Selesai! File 'dataset_final_skripsi.csv' sudah jadi dan siap dipakai.")