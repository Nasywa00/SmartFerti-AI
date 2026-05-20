import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("PELATIHAN MODEL MACHINE LEARNING - SMART FERTI AI")
print("=" * 70)

# 1. Load dataset
print("\n[1] Memuat dataset...")
df = pd.read_csv('dataset.csv')
print(f"    ✓ Dataset dimuat: {df.shape[0]} sampel, {df.shape[1]} fitur")
print(f"    ✓ Fitur: {list(df.columns)}")

# 2. Exploratory Data Analysis
print("\n[2] Analisis Data:")
print(f"    Unique crops: {df['crop'].unique()}")
print(f"    Unique soil conditions: {df['soil_condition'].unique()}")
print(f"    Unique seasons: {df['season'].unique()}")
print(f"    Age range: {df['age_days'].min()}-{df['age_days'].max()} hari")
print(f"    Land area range: {df['land_area_ha'].min()}-{df['land_area_ha'].max()} ha")

# 3. Data Preprocessing - Encoding fitur kategori
print("\n[3] Preprocessing Data:")
print("    Melakukan encoding pada fitur kategori...")

# Buat copy untuk preprocessing
df_processed = df.copy()

# Inisialisasi label encoders
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_season = LabelEncoder()

# Fit dan transform fitur kategori
df_processed['crop_encoded'] = le_crop.fit_transform(df_processed['crop'])
df_processed['soil_condition_encoded'] = le_soil.fit_transform(df_processed['soil_condition'])
df_processed['season_encoded'] = le_season.fit_transform(df_processed['season'])

print(f"    ✓ Crop encoding: {dict(zip(le_crop.classes_, le_crop.transform(le_crop.classes_)))}")
print(f"    ✓ Soil condition encoding: {dict(zip(le_soil.classes_, le_soil.transform(le_soil.classes_)))}")
print(f"    ✓ Season encoding: {dict(zip(le_season.classes_, le_season.transform(le_season.classes_)))}")

# Pilih fitur dan target
X = df_processed[['crop_encoded', 'soil_condition_encoded', 'season_encoded', 'age_days', 'land_area_ha']]
y = df_processed[['N', 'P', 'K']]

print(f"    ✓ Fitur shape: {X.shape}")
print(f"    ✓ Target shape: {y.shape}")

# 4. Split data menjadi training dan testing
print("\n[4] Split Data:")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    ✓ Training set: {X_train.shape[0]} sampel ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"    ✓ Testing set: {X_test.shape[0]} sampel ({X_test.shape[0]/len(X)*100:.1f}%)")

# 5. Training Model
print("\n[5] Training Model:")
print("    Membuat model RandomForestRegressor dengan MultiOutputRegressor...")

# Buat base model
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Wrapper dengan MultiOutputRegressor
model = MultiOutputRegressor(rf_model)

# Training
print("    Training sedang berjalan...")
model.fit(X_train, y_train)
print("    ✓ Model training selesai!")

# 6. Evaluasi Model
print("\n[6] Evaluasi Model:")

# Prediksi
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics untuk training set
print("\n    Training Set Performance:")
for i, target in enumerate(['N', 'P', 'K']):
    mse = mean_squared_error(y_train.iloc[:, i], y_pred_train[:, i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train.iloc[:, i], y_pred_train[:, i])
    r2 = r2_score(y_train.iloc[:, i], y_pred_train[:, i])
    print(f"      {target}: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

# Metrics untuk testing set
print("\n    Testing Set Performance:")
for i, target in enumerate(['N', 'P', 'K']):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred_test[:, i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred_test[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred_test[:, i])
    print(f"      {target}: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

# 7. Feature Importance
print("\n[7] Feature Importance (Average across all outputs):")
feature_names = ['crop', 'soil_condition', 'season', 'age_days', 'land_area_ha']
feature_importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
for feat, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
    print(f"    {feat}: {importance:.4f}")

# 8. Simpan model
print("\n[8] Menyimpan Model:")
model_path = 'model.pkl'
joblib.dump(model, model_path)
print(f"    ✓ Model disimpan ke: {model_path}")

# Simpan encoders juga
encoders = {
    'crop_encoder': le_crop,
    'soil_encoder': le_soil,
    'season_encoder': le_season,
    'feature_names': feature_names,
    'target_names': ['N', 'P', 'K']
}
encoders_path = 'encoders.pkl'
joblib.dump(encoders, encoders_path)
print(f"    ✓ Encoders disimpan ke: {encoders_path}")

# 9. Sample Prediction
print("\n[9] Sample Prediksi (5 sampel pertama dari test set):")
print(f"    {'Actual N':>10} {'Pred N':>10} {'Actual P':>10} {'Pred P':>10} {'Actual K':>10} {'Pred K':>10}")
print("    " + "-" * 66)
for i in range(min(5, len(y_test))):
    print(f"    {y_test.iloc[i, 0]:>10.2f} {y_pred_test[i, 0]:>10.2f} "
          f"{y_test.iloc[i, 1]:>10.2f} {y_pred_test[i, 1]:>10.2f} "
          f"{y_test.iloc[i, 2]:>10.2f} {y_pred_test[i, 2]:>10.2f}")

print("\n" + "=" * 70)
print("PELATIHAN SELESAI!")
print("=" * 70)
