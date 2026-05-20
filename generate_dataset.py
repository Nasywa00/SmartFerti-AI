import pandas as pd
import numpy as np
from datetime import datetime

# Set seed untuk reproducibility
np.random.seed(42)

# Parameter dataset
n_samples = 500
crops = ['padi', 'jagung', 'cabai', 'tomat']
soil_conditions = ['berat', 'sedang', 'ringan']
seasons = ['musim hujan', 'musim kemarau', 'musim transisi']

# Generate dataset
data = {
    'crop': np.random.choice(crops, n_samples),
    'soil_condition': np.random.choice(soil_conditions, n_samples),
    'season': np.random.choice(seasons, n_samples),
    'age_days': np.random.randint(30, 120, n_samples),
    'land_area_ha': np.round(np.random.uniform(0.5, 5.0, n_samples), 2)
}

df = pd.DataFrame(data)

# Generate target variables (N, P, K) berdasarkan crop dan kondisi lainnya
# Nilai N, P, K dalam kg/ha
df['N'] = 0.0
df['P'] = 0.0
df['K'] = 0.0

# Tentukan kebutuhan nutrisi berdasarkan jenis tanaman
nutrient_needs = {
    'padi': {'N': (120, 150), 'P': (60, 90), 'K': (80, 120)},
    'jagung': {'N': (150, 200), 'P': (70, 100), 'K': (100, 150)},
    'cabai': {'N': (100, 130), 'P': (50, 80), 'K': (120, 180)},
    'tomat': {'N': (110, 150), 'P': (60, 90), 'K': (110, 150)}
}

# Adjustment factor berdasarkan soil condition
soil_adjustment = {
    'berat': 1.1,      # Tanah berat membutuhkan lebih banyak nutrisi
    'sedang': 1.0,     # Baseline
    'ringan': 0.9      # Tanah ringan membutuhkan lebih sedikit
}

# Adjustment factor berdasarkan age_days
def age_factor(age):
    if age < 45:
        return 0.6  # Fase vegetatif awal, kebutuhan lebih rendah
    elif age < 75:
        return 1.0  # Fase vegetatif aktif, kebutuhan normal
    else:
        return 1.2  # Fase generatif, kebutuhan lebih tinggi

# Hitung N, P, K untuk setiap baris
for idx, row in df.iterrows():
    crop_type = row['crop']
    soil_type = row['soil_condition']
    age = row['age_days']
    
    # Dapatkan range kebutuhan nutrisi untuk jenis tanaman
    n_range = nutrient_needs[crop_type]['N']
    p_range = nutrient_needs[crop_type]['P']
    k_range = nutrient_needs[crop_type]['K']
    
    # Hitung base value (tengah range)
    n_base = (n_range[0] + n_range[1]) / 2
    p_base = (p_range[0] + p_range[1]) / 2
    k_base = (k_range[0] + k_range[1]) / 2
    
    # Apply adjustment factors
    soil_adj = soil_adjustment[soil_type]
    age_adj = age_factor(age)
    
    # Tambahkan variasi random (±10%)
    n_value = n_base * soil_adj * age_adj * np.random.uniform(0.9, 1.1)
    p_value = p_base * soil_adj * age_adj * np.random.uniform(0.9, 1.1)
    k_value = k_base * soil_adj * age_adj * np.random.uniform(0.9, 1.1)
    
    df.at[idx, 'N'] = round(n_value, 2)
    df.at[idx, 'P'] = round(p_value, 2)
    df.at[idx, 'K'] = round(k_value, 2)

# Simpan ke CSV
csv_path = 'd:/Nasywa/Sem-6/presisi/SmartFerti AI/dataset.csv'
df.to_csv(csv_path, index=False)

print(f"✓ Dataset berhasil dibuat dengan {len(df)} sampel")
print(f"✓ File disimpan di: {csv_path}")
print(f"\nInfo Dataset:")
print(f"  - Jumlah sampel: {len(df)}")
print(f"  - Jenis tanaman: {', '.join(crops)}")
print(f"  - Fitur: crop, soil_condition, season, age_days, land_area_ha")
print(f"  - Target: N, P, K (kg/ha)")
print(f"\n5 baris pertama:")
print(df.head())
print(f"\nStatistik dataset:")
print(df.describe())
