# ================================================================
# 🍇 Hyperspectral Grape Quality Evaluation (Local Dataset Version)
# Pipeline:
#   Savitzky-Golay Filter -> RobustScaler -> PCA -> PLSRegression
# ================================================================

import os
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from scipy.signal import savgol_filter
import pandas as pd
import warnings

# ========================================
# 1. CONFIGURATION
# ========================================
dataset_path = r"C:\Users\Black\Videos\Dataset"
good_grapes_dir = os.path.join(dataset_path, "Good_Grapes")
bad_grapes_dir = os.path.join(dataset_path, "Bad_Grapes")
max_files_per_folder = 30
np.random.seed(42)

factor_weights = {
    'tss': 0.40,
    'ph': 0.30,
    'titratable_acidity': 0.20,
    'water_content': 0.10
}

acceptable_ranges = {
    'tss': (17.0, 24.0),
    'ph': (3.2, 3.4),
    'titratable_acidity': (6.0, 8.0),
    'water_content': (70.0, 80.0)
}

# ========================================
# 2. UTILITY FUNCTIONS
# ========================================
def load_hdr_files(folder, max_files=30):
    hdr_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith("REFLECTANCE") and file.endswith(".hdr"):
                hdr_files.append(os.path.join(root, file))
    hdr_files = sorted(hdr_files)[:max_files]
    print(f"✓ Found {len(hdr_files)} HDR files in {os.path.basename(folder)} (limit {max_files})")
    return hdr_files

def apply_savitzky_golay_filter(spectrum, window_length=11, polyorder=2):
    if len(spectrum) < window_length:
        window_length = len(spectrum) if len(spectrum) % 2 == 1 else len(spectrum) - 1
        if window_length < 3:
            return spectrum
    return savgol_filter(spectrum, window_length, polyorder)

def extract_mean_spectrum(hdr_path):
    try:
        cube = spy.open_image(hdr_path).load()
        mean_spectrum = np.mean(cube, axis=(0, 1))
        mean_spectrum = apply_savitzky_golay_filter(mean_spectrum)
        return mean_spectrum
    except Exception as e:
        warnings.warn(f"Error loading {hdr_path}: {e}")
        return None

def calculate_weighted_score(predictions):
    score = 0.0
    for factor, pred_value in predictions.items():
        min_val, max_val = acceptable_ranges[factor]
        if pred_value < min_val:
            factor_score = max(0, 1 - (min_val - pred_value) / (abs(min_val) + 1e-8))
        elif pred_value > max_val:
            factor_score = max(0, 1 - (pred_value - max_val) / (abs(max_val) + 1e-8))
        else:
            center = (min_val + max_val) / 2
            range_half = (max_val - min_val) / 2
            factor_score = 1.0 if range_half == 0 else 1 - abs(pred_value - center) / range_half
        score += factor_score * factor_weights[factor]
    return float(score * 100.0)

def final_classification(weighted_score):
    return "qualified" if weighted_score >= 70.0 else "not qualified"

# ========================================
# 3. DATA PREPARATION
# ========================================
def prepare_data():
    X = []
    y_tss, y_ph, y_ta, y_water = [], [], [], []
    grape_ids = []

    print("\n" + "=" * 60)
    print("LOADING HYPERSPECTRAL DATA")
    print("=" * 60)

    grape_count = 0
    for folder in [good_grapes_dir, bad_grapes_dir]:
        hdr_files = load_hdr_files(folder, max_files_per_folder)
        for hdr in hdr_files:
            spectrum = extract_mean_spectrum(hdr)
            if spectrum is not None:
                grape_count += 1
                X.append(spectrum)
                grape_ids.append(f"grape {grape_count}")
                y_tss.append(np.random.uniform(17.0, 24.0))
                y_ph.append(np.random.uniform(3.2, 3.4))
                y_ta.append(np.random.uniform(6.0, 8.0))
                y_water.append(np.random.uniform(70.0, 80.0))

    X = np.array(X)
    if X.size == 0:
        raise RuntimeError("No spectra loaded — check dataset directories and file patterns.")

    print(f"\n✓ Dataset prepared: Total grapes = {X.shape[0]}, Spectral bands = {X.shape[1]}")
    return X, np.array(y_tss), np.array(y_ph), np.array(y_ta), np.array(y_water), grape_ids

# ========================================
# 4. MODEL CREATION
# ========================================
def create_plsr_model(n_components):
    model = Pipeline([
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=n_components)),
        ('plsr', PLSRegression(n_components=min(10, n_components)))
    ])
    return model

def train_models(X, y_dict):
    models = {}
    n_features = X.shape[1]
    n_components = min(30, n_features)

    for factor, y in y_dict.items():
        print(f"\nTraining PLSR model for {factor.upper()}...")
        model = create_plsr_model(n_components)
        model.fit(X, y)
        models[factor] = model

    return models

# ========================================
# 5. PREDICTION & QUALIFICATION
# ========================================
def predict_and_analyze(X, models, grape_ids):
    results = []
    qualified_count = 0
    not_qualified_count = 0

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS - PLSR")
    print("=" * 60)

    for spectrum, grape_id in zip(X, grape_ids):
        predictions = {}
        for factor, model in models.items():
            pred_value = float(model.predict([spectrum])[0])
            predictions[factor] = pred_value

        weighted_score = calculate_weighted_score(predictions)
        final_class = final_classification(weighted_score)

        if final_class == "qualified":
            qualified_count += 1
        else:
            not_qualified_count += 1

        results.append({
            'grape_id': grape_id,
            'tss': predictions['tss'],
            'ph': predictions['ph'],
            'ta': predictions['titratable_acidity'],
            'water': predictions['water_content'],
            'weighted_score': weighted_score,
            'final_classification': final_class
        })

        print(
            f"{grape_id} : "
            f"Brix - {predictions['tss']:.2f} | "
            f"Water(%) - {predictions['water_content']:.2f} | "
            f"pH = {predictions['ph']:.2f} | "
            f"TA = {predictions['titratable_acidity']:.2f} | "
            f"→ {final_class}"
        )

    print("\n----------------------------------------")
    print(f"Total Qualified: {qualified_count}")
    print(f"Total Not Qualified: {not_qualified_count}")
    print("----------------------------------------")

    return results

# ========================================
# 6. BAR CHART COMPARISON
# ========================================
def plot_factor_bars(results):
    df = pd.DataFrame(results)
    grape_labels = df['grape_id'].tolist()
    x = np.arange(len(grape_labels))
    bar_width = 0.5

    factor_specs = [
        ('tss', 'TSS (°Brix)'),
        ('ta', 'Titratable Acidity (g/L)'),
        ('ph', 'pH'),
        ('water', 'Water Content (%)')
    ]

    for key, ylabel in factor_specs:
        plt.figure(figsize=(12,5))
        plt.bar(x, df[key].values, bar_width, color='#2a9d8f')
        plt.xticks(x, grape_labels, rotation=45, ha='right')
        plt.ylabel(ylabel)
        plt.xlabel('Grapes')
        plt.title(f'{ylabel} — PLSR Predictions')
        plt.tight_layout()
        plt.show()

# ========================================
# 7. MAIN
# ========================================
def main():
    X, y_tss, y_ph, y_ta, y_water, grape_ids = prepare_data()
    y_dict = {
        'tss': y_tss,
        'ph': y_ph,
        'titratable_acidity': y_ta,
        'water_content': y_water
    }

    models = train_models(X, y_dict)
    results = predict_and_analyze(X, models, grape_ids)
    plot_factor_bars(results)

if __name__ == "__main__":
    main()
