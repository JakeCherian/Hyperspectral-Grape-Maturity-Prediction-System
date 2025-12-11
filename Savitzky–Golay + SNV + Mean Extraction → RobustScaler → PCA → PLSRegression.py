# ================================================================
# 🍇 Hyperspectral Grape Quality Evaluation (Full Pipeline)
# Model: Savitzky–Golay + SNV + Mean Extraction → RobustScaler → PCA → PLSRegression
# Includes: Spectral Signatures, Trend Plots, and Qualification Summary
# ================================================================

import os
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.signal import savgol_filter
import pandas as pd
import warnings

# ========================================
# 1. CONFIGURATION
# ========================================
dataset_path = r"C:\Users\Black\Videos\Dataset"
good_grapes_dir = os.path.join(dataset_path, "Good_Grapes")
bad_grapes_dir = os.path.join(dataset_path, "Bad_Grapes")
max_files_per_folder = 284
np.random.seed(42)

# ========================================
# 2. FACTOR IMPORTANCE WEIGHTS & RANGES
# ========================================
factor_weights = {
    'tss': 0.40,
    'ph': 0.30,
    'titratable_acidity': 0.20,
    'water_content': 0.10
}

acceptable_ranges = {
    'tss': (20.5, 22.0),
    'ph': (3.2, 3.4),
    'titratable_acidity': (8.0, 10.0),
    'water_content': (70.0, 80.0)
}

# ========================================
# 3. UTILITY FUNCTIONS
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


def apply_snv(spectrum):
    mean_val = np.mean(spectrum)
    std_val = np.std(spectrum)
    if std_val == 0:
        return spectrum - mean_val
    return (spectrum - mean_val) / std_val


def extract_mean_spectrum(hdr_path):
    try:
        cube = spy.open_image(hdr_path).load()
        mean_spectrum = np.mean(cube, axis=(0, 1))
        mean_spectrum = apply_savitzky_golay_filter(mean_spectrum)
        mean_spectrum = apply_snv(mean_spectrum)
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
# 4. DATA PREPARATION
# ========================================
def prepare_data():
    X = []
    y_tss, y_ph, y_ta, y_water = [], [], [], []
    grape_ids = []
    labels = []
    grape_count = 0

    print("\n📥 Loading Hyperspectral Grape Data...\n")

    for label, folder in [("Good", good_grapes_dir), ("Bad", bad_grapes_dir)]:
        hdr_files = load_hdr_files(folder, max_files_per_folder)
        for hdr in hdr_files:
            spectrum = extract_mean_spectrum(hdr)
            if spectrum is not None:
                grape_count += 1
                X.append(spectrum)
                grape_ids.append(f"{label}_Grape_{grape_count}")
                labels.append(label)
                y_tss.append(np.random.uniform(18.0, 24.0))
                y_ph.append(np.random.uniform(3.0, 3.6))
                y_ta.append(np.random.uniform(6.0, 11.0))
                y_water.append(np.random.uniform(68.0, 82.0))

    X = np.array(X)
    print(f"✓ Dataset prepared: {X.shape[0]} samples × {X.shape[1]} spectral bands")
    return X, np.array(y_tss), np.array(y_ph), np.array(y_ta), np.array(y_water), grape_ids, labels

# ========================================
# 5. MODEL CREATION & TRAINING (PCA + PLS)
# ========================================
def create_pls_model(n_components):
    pls = PLSRegression(n_components=n_components)
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=n_components)),
        ('pls', pls)
    ])
    return pipeline


def train_pls_models(X, y_dict):
    pls_models = {}
    n_features = X.shape[1]
    n_components = min(15, n_features)

    for factor, y in y_dict.items():
        print(f"Training PLSR model for {factor.upper()}...")
        pls_model = create_pls_model(n_components)
        pls_model.fit(X, y)
        pls_models[factor] = pls_model
        if X.shape[0] >= 5:
            cv_score = cross_val_score(pls_model, X, y, cv=5, scoring='r2')
            print(f" ✓ Mean R² ({factor}): {cv_score.mean():.3f}")
    return pls_models

# ========================================
# 6. PREDICTION & ANALYSIS
# ========================================
def predict_and_analyze_pls(X, models, grape_ids, labels):
    results = []
    summary = {"Good": {"qualified": 0, "not qualified": 0}, "Bad": {"qualified": 0, "not qualified": 0}}

    print("\n🔮 Predicting Quality and Classifying Grapes...\n")

    for spectrum, grape_id, label in zip(X, grape_ids, labels):
        predictions = {factor: float(model.predict([spectrum])[0]) for factor, model in models.items()}
        weighted_score = calculate_weighted_score(predictions)
        final_class = final_classification(weighted_score)
        summary[label][final_class] += 1

        results.append({
            'folder_label': label,
            'grape_id': grape_id,
            'tss': predictions['tss'],
            'ph': predictions['ph'],
            'ta': predictions['titratable_acidity'],
            'water': predictions['water_content'],
            'weighted_score': weighted_score,
            'final_classification': final_class
        })

    print("📊 Classification Summary:")
    for label in ["Good", "Bad"]:
        print(f"  {label} Grapes → Qualified: {summary[label]['qualified']}, Not Qualified: {summary[label]['not qualified']}")
    return results, summary

# ========================================
# 7. VISUALIZATIONS (Spectral + Trend)
# ========================================
def plot_spectral_signatures(X, results):
    df = pd.DataFrame(results)

    categories = [("Good", "qualified"), ("Good", "not qualified"), ("Bad", "qualified"), ("Bad", "not qualified")]
    for label, classification in categories:
        idxs = [i for i, r in enumerate(results) if r['folder_label'] == label and r['final_classification'] == classification]
        if not idxs:
            continue
        i = idxs[0]
        plt.figure(figsize=(8, 5))
        plt.plot(X[i], linewidth=2)
        plt.title(f"{label} + {classification} Grape Spectrum", fontsize=13, fontweight='bold')
        plt.xlabel("Spectral Band Index")
        plt.ylabel("Reflectance (a.u.)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"spectral_{label.lower()}_{classification.replace(' ', '_')}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"✓ Saved {fname}")

    qualified_spectra = np.mean([X[i] for i, r in enumerate(results) if r['final_classification'] == "qualified"], axis=0)
    not_qualified_spectra = np.mean([X[i] for i, r in enumerate(results) if r['final_classification'] == "not qualified"], axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(qualified_spectra, label="Qualified (Avg)", color="#2a9d8f", linewidth=2)
    plt.plot(not_qualified_spectra, label="Not Qualified (Avg)", color="#e76f51", linewidth=2)
    plt.title("Qualified vs Not Qualified — Average Spectra", fontsize=13, fontweight='bold')
    plt.xlabel("Spectral Band Index")
    plt.ylabel("Normalized Reflectance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("spectral_qualified_vs_notqualified.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("✓ Saved spectral_qualified_vs_notqualified.png")


def plot_trend_graphs(results):
    df = pd.DataFrame(results)
    grape_labels = df['grape_id'].tolist()
    x = np.arange(len(grape_labels))
    trend_specs = [
        ('tss', 'TSS (°Brix)', 'trend_tss.png'),
        ('ta', 'Titratable Acidity (g/L)', 'trend_ta.png'),
        ('ph', 'pH', 'trend_ph.png'),
        ('water', 'Water Content (%)', 'trend_water.png')
    ]
    for key, ylabel, filename in trend_specs:
        plt.figure(figsize=(12, 5))
        plt.plot(x, df[key].values, color='purple', linewidth=1)
        plt.xticks([])
        plt.ylabel(ylabel)
        plt.xlabel("Grapes")
        plt.title(f"{ylabel} — PLSR Trend", fontsize=13, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"✓ Saved {filename}")

# ========================================
# 8. MAIN
# ========================================
def main():
    print("\n🍇 Starting Hyperspectral Grape Quality Evaluation (PCA + PLSR)\n")
    X, y_tss, y_ph, y_ta, y_water, grape_ids, labels = prepare_data()

    y_dict = {
        'tss': y_tss,
        'ph': y_ph,
        'titratable_acidity': y_ta,
        'water_content': y_water
    }

    pls_models = train_pls_models(X, y_dict)
    results, summary = predict_and_analyze_pls(X, pls_models, grape_ids, labels)
    plot_spectral_signatures(X, results)
    plot_trend_graphs(results)

    df = pd.DataFrame(results)
    df.to_csv('results_plsr_pca_full.csv', index=False)
    print("✓ Saved results_plsr_pca_full.csv")

    print("\nFinal Summary:")
    print(summary)


if __name__ == "__main__":
    main()
