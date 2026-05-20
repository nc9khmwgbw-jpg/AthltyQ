import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

# ─── Chemins ────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent.parent.parent
BENCHMARK_DIR = ROOT / "LM" / "models" / "benchmark"
DATA_PATH     = ROOT / "data" / "processed" / "features_dataset.csv"
SAVE_DIR      = BENCHMARK_DIR / "Polynomial_Regression"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IDENTITY_COLS = [
    'Nom', 'Player_ID', 'Match_Date', 'Event_ID', 'Home_Team', 'Away_Team',
    'Score_Home', 'Score_Away', 'Tournament', 'Equipe', 'Position', 'Poste_Cat',
    'Risk_Category', 'Target_Injury_Occurred', 'Current_Injury',
    'Fatigue_Realisee', 'Fatigue_Reelle_Match_T'
]

# ─── Hyper-paramètres ─────────────────────────────────────────────────────
# PCA réduit les features à N composantes avant Polynomial
# → évite l'explosion mémoire (N features → N*(N+3)/2 avec degree=2)
PCA_N_COMPONENTS   = 20          # garde 20 composantes principales
POLY_DEGREE        = 2           # degré polynomial
# RidgeCV cherche automatiquement le meilleur alpha dans cette liste
RIDGE_ALPHAS       = np.array([0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
CV_FOLDS           = 5


def _load_data():
    df      = pd.read_csv(DATA_PATH)
    col_nom = 'Nom' if 'Nom' in df.columns else 'Player_Name'
    X = df.drop(columns=IDENTITY_COLS + ['Target_Fatigue'], errors='ignore')
    y = df['Target_Fatigue']
    model_cols = joblib.load(BENCHMARK_DIR / "model_columns.joblib")
    X = X.select_dtypes(include='number').reindex(columns=model_cols, fill_value=0).fillna(0)
    return df, col_nom, X, y


def _split(df, col_nom, X, y):
    train_p = set(joblib.load(BENCHMARK_DIR / "train_players.joblib"))
    test_p  = set(joblib.load(BENCHMARK_DIR / "test_players.joblib"))
    mt = df[col_nom].isin(train_p)
    mv = df[col_nom].isin(test_p)
    return X[mt], y[mt], X[mv], y[mv]


def _print_cv_results(scores, label="CV"):
    print(f"  {label}  →  mean={scores.mean():.4f}  std=±{scores.std():.4f}  "
          f"[{scores.min():.4f} … {scores.max():.4f}]")


def train_polynomial_regression():
    print("\n" + "═"*65)
    print("  📐 ENTRAÎNEMENT : RÉGRESSION POLYNOMIALE  (VERSION PRO)")
    print("═"*65)

    # ── 1. Données ────────────────────────────────────────────────────────
    df, col_nom, X, y = _load_data()
    X_train, y_train, X_test, y_test = _split(df, col_nom, X, y)
    print(f"\n  Train : {X_train.shape[0]:,} lignes  |  Test : {X_test.shape[0]:,} lignes")
    print(f"  Features d'entrée : {X_train.shape[1]}")

    # ── 2. Construction du pipeline complet ──────────────────────────────
    # NOTE : Le scaler est INTÉGRÉ dans le pipeline.
    # On n'utilise plus benchmark_scaler.joblib pour ce modèle.
    # Cela garantit que le modèle est auto-suffisant à la prédiction.
    n_comp = min(PCA_N_COMPONENTS, X_train.shape[1], X_train.shape[0] - 1)
    print("\n  Architecture du pipeline :")
    print(f"    StandardScaler → PCA(n={n_comp}) → PolynomialFeatures(deg={POLY_DEGREE}) → RidgeCV")

    poly_model = Pipeline([
        ('scaler',      StandardScaler()),
        ('pca',         PCA(n_components=n_comp, random_state=42)),
        ('poly',        PolynomialFeatures(degree=POLY_DEGREE,
                                           include_bias=False,
                                           interaction_only=False)),
        ('ridge',       RidgeCV(alphas=RIDGE_ALPHAS, cv=5)),  # type: ignore[arg-type]
    ])

    # ── 3. Entraînement ───────────────────────────────────────────────────
    print("\n  ⏳ Entraînement en cours...")
    t0 = time.time()
    poly_model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  ✅ Terminé en {elapsed:.2f}s")

    best_alpha = poly_model.named_steps['ridge'].alpha_
    print(f"  🎯 Alpha optimal (RidgeCV) : {best_alpha}")

    # ── 4. Scores train vs test (détection overfitting) ───────────────────
    y_pred_train = poly_model.predict(X_train)
    y_pred_test  = poly_model.predict(X_test)

    r2_train   = r2_score(y_train, y_pred_train)
    r2_test    = r2_score(y_test,  y_pred_test)
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))

    overfit_gap = r2_train - r2_test
    overfit_flag = "⚠️  Sur-apprentissage détecté !" if overfit_gap > 0.10 else "✅ Pas de sur-apprentissage"

    print("\n" + "─"*45)
    print("  🎯 RÉSULTATS POLYNOMIAL REGRESSION")
    print("─"*45)
    print(f"  R² Train  : {r2_train:.4f}")
    print(f"  R² Test   : {r2_test:.4f}   (écart : {overfit_gap:+.4f})  {overfit_flag}")
    print(f"  MAE Test  : {mae_test:.4f}")
    print(f"  RMSE Test : {rmse_test:.4f}")

    # ── 5. Cross-Validation 5-fold (sur train set uniquement) ─────────────
    print(f"\n  🔄 Cross-Validation {CV_FOLDS}-fold (sur train set) :")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    cv_r2   = cross_val_score(poly_model, X_train, y_train, cv=kf, scoring='r2',                    n_jobs=-1)
    cv_mae  = cross_val_score(poly_model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_rmse = cross_val_score(poly_model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)

    _print_cv_results(cv_r2,   "R²  ")
    _print_cv_results(-cv_mae,  "MAE ")
    _print_cv_results(-cv_rmse, "RMSE")

    # ── 6. Analyse des résidus ────────────────────────────────────────────
    residuals = y_test.values - y_pred_test
    print("\n  📉 Analyse des résidus (test set) :")
    print(f"     Moyenne  : {residuals.mean():.4f}  (idéal = 0)")
    print(f"     Std      : {residuals.std():.4f}")
    print(f"     Max abs  : {np.abs(residuals).max():.4f}")
    pct_10 = np.mean(np.abs(residuals) <= 10) * 100
    print(f"     Dans ±10% fatigue : {pct_10:.1f}% des prédictions")

    # ── 6. Importance des variables (Back-projection PCA -> Original) ─────
    # Comme on utilise PCA + Ridge, l'importance est : coefficients * loadings
    try:
        pca = poly_model.named_steps['pca']
        ridge = poly_model.named_steps['ridge']
        
        # On prend les coefficients du premier degré (les 20 premières features poly après PCA)
        # poly_features = [f1, f2, ..., f20, f1^2, f1*f2, ...]
        # Les loadings PCA sont (n_components, n_features_origin)
        loadings = pca.components_ 
        ridge_coefs = ridge.coef_[:n_comp] # on prend les coefs linéaires uniquement
        
        # Projection : importance_originale = sum(abs(ridge_coef * loading))
        imp_orig = np.sum(np.abs(ridge_coefs[:, np.newaxis] * loadings), axis=0)
        
        feat_imp = []
        for i, col in enumerate(X_train.columns):
            feat_imp.append({"feature": col, "importance": float(imp_orig[i])})
            
        top_features = sorted(feat_imp, key=lambda x: x['importance'], reverse=True)[:10]
    except Exception as e:
        print(f"  ⚠️ Impossible d'extraire l'importance : {e}")
        top_features = []

    # ── 7. Sauvegarde ────────────────────────────────────────────────────
    joblib.dump(poly_model, SAVE_DIR / "model_poly.joblib")

    metrics = {
        "model"         : "Polynomial Regression",
        "R2_train"      : round(r2_train,  4),
        "R2_test"       : round(r2_test,   4),
        "MAE"           : round(mae_test,  4),
        "RMSE"          : round(rmse_test, 4),
        "overfit_gap"   : round(overfit_gap, 4),
        "cv_r2_mean"    : round(cv_r2.mean(),  4),
        "cv_r2_std"     : round(cv_r2.std(),   4),
        "cv_mae_mean"   : round((-cv_mae).mean(),  4),
        "cv_rmse_mean"  : round((-cv_rmse).mean(), 4),
        "best_alpha"    : float(best_alpha),
        "pca_components": int(n_comp),
        "residuals_mean": round(float(residuals.mean()), 4),
        "residuals_std" : round(float(residuals.std()),  4),
        "pct_within_10" : round(pct_10, 2),
        "train_time_s"  : round(elapsed, 2),
        "top_features"  : top_features
    }

    with open(SAVE_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n  💾 Modèle et métriques sauvegardés dans : {SAVE_DIR}")
    print("═"*65 + "\n")


if __name__ == "__main__":
    train_polynomial_regression()