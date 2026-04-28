import pandas as pd

# 1. Load dataset (yang SUDAH kamu gabungkan)
df = pd.read_csv('data/dataset_final_skripsi.csv')

# 2. Cek kolom
print("Kolom dataset:")
print(df.columns)

# 3. Bersihkan kolom laba
kolom_laba = [c for c in df.columns if 'Laba' in c][0]

df[kolom_laba] = df[kolom_laba].astype(str)\
    .str.replace('.', '', regex=False)\
    .str.replace(',', '.', regex=False)

df[kolom_laba] = pd.to_numeric(df[kolom_laba], errors='coerce')

# 4. Pastikan kolom Close numerik
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

# 5. Hapus data kosong
df = df.dropna()

# 6. Cek hasil
print("\nCek NaN:")
print(df.isnull().sum())

print("\nJumlah data:", len(df))
print(df.head())

# 7. Simpan hasil cleaning
df.to_csv('data/dataset_clean.csv', index=False)

print("\nCleaning selesai! File: dataset_clean.csv")