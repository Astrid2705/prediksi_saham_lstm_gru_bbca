import pandas as pd

# 1. Load dataset
df = pd.read_csv('Data/dataset_final_skripsi.csv')

# 2. Konversi Tanggal (Penting untuk Time-Series)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date') # Pastikan data urut waktu

# 3. Identifikasi Kolom Utama (OHLCV + Laba)
# Pastikan nama kolom di bawah ini sesuai dengan file CSV kamu
kolom_teknikal = ['Open', 'High', 'Low', 'Close', 'Volume']
kolom_laba = [c for c in df.columns if 'Laba' in c][0]

# 4. Bersihkan Seluruh Kolom Teknikal (Banyak kolom sekaligus)
for col in kolom_teknikal:
    if df[col].dtype == 'object': # Jika tipe datanya masih teks/string
        df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 5. Bersihkan Kolom Laba
df[kolom_laba] = df[kolom_laba].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df[kolom_laba] = pd.to_numeric(df[kolom_laba], errors='coerce')

# 6. Terapkan Forward Filling (FFILL) untuk Laba
# Mengisi baris kosong laba harian dengan nilai laba kuartal terakhir yang muncul
df[kolom_laba] = df[kolom_laba].ffill()

# 7. Hapus data kosong yang tersisa (biasanya di baris paling awal sebelum laba pertama rilis)
df = df.dropna(subset=kolom_teknikal + [kolom_laba])

# 8. Cek hasil akhir
print("Kolom yang tersedia:", df.columns.tolist())
print("\nCek data kosong (NaN):")
print(df.isnull().sum())

print("\nJumlah data setelah cleaning:", len(df))
print(df[kolom_teknikal + [kolom_laba]].head())

# 9. Simpan hasil cleaning untuk tahap Training
df.to_csv('Data/dataset_clean.csv', index=False)

print("\nCleaning selesai! Data OHLCV + Laba sudah siap di dataset_clean.csv")