import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold,
                                      cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Chemins ────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent.parent.parent
BENCHMARK_DIR = ROOT / "LM" / "models" / "benchmark"
DATA_PATH     = ROOT / "data" / "processed" / "features_dataset.csv"
SAVE_DIR      = BENCHMARK_DIR / "Random_Forest"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IDENTITY_COLS = [
    'Nom', 'Player_ID', 'Match_Date', 'Event_ID', 'Home_Team', 'Away_Team',
    'Score_Home', 'Score_Away', 'Tournament', 'Equipe', 'Position', 'Poste_Cat',
    'Risk_Category', 'Target_Injury_Occurred', 'Current_Injury',
    'Fatigue_Realisee', 'Fatigue_Reelle_Match_T'
]

# ─── Espace de recherche des hyperparamètres ─────────────────────────────
# Moins de combinaisons = plus rapide, mais couvre les axes importants
PARAM_GRID = {
    'rf__n_estimators'     : [100, 200],
    'rf__max_depth'        : [10, 20, None],
    'rf__min_samples_leaf' : [2, 5, 10],
    'rf__max_features'     : ['sqrt', 0.5],
}
CV_FOLDS = 5


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


def _confidence_interval_from_trees(rf_model, X_scaled, percentile=90):
    """
    Calcule un intervalle de confiance basé sur les prédictions
    de chaque arbre individuel de la forêt.
    Retourne (lower, upper) pour chaque prédiction.
    """
    tree_preds = np.array([tree.predict(X_scaled) for tree in rf_model.estimators_])
    lo = np.percentile(tree_preds, (100 - percentile) / 2, axis=0)
    hi = np.percentile(tree_preds, 100 - (100 - percentile) / 2, axis=0)
    return lo, hi


def train_random_forest():
    print("\n" + "═"*65)
    print("  🌲 ENTRAÎNEMENT : RANDOM FOREST  (VERSION PRO)")
    print("═"*65)

    # ── 1. Données ────────────────────────────────────────────────────────
    df, col_nom, X, y = _load_data()
    X_train, y_train, X_test, y_test = _split(df, col_nom, X, y)
    print(f"\n  Train : {X_train.shape[0]:,} lignes  |  Test : {X_test.shape[0]:,} lignes")
    print(f"  Features : {X_train.shape[1]}")

    # ── 2. Pipeline de base (scaler intégré) ──────────────────────────────
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf',     RandomForestRegressor(
            oob_score   = True,       # ✅ Score out-of-bag gratuit
            random_state= 42,
            n_jobs      = 1,
        ))
    ])

    # ── 3. GridSearchCV pour les hyperparamètres optimaux ─────────────────
    print(f"\n  🔍 Recherche des hyperparamètres (GridSearchCV {CV_FOLDS}-fold)...")
    print(f"     Combinaisons à tester : "
          f"{np.prod([len(v) for v in PARAM_GRID.values()])}")

    kf   = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    grid = GridSearchCV(
        base_pipeline,
        PARAM_GRID,
        cv         = kf,
        scoring    = 'r2',
        n_jobs     = -1,
        refit      = True,
        verbose    = 0,
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_params = {k.replace('rf__', ''): v
                   for k, v in grid.best_params_.items()}
    print(f"  ✅ Recherche terminée en {elapsed:.1f}s")
    print(f"  🏆 Meilleurs hyperparamètres :")
    for k, v in best_params.items():
        print(f"     • {k} = {v}")
    print(f"  📈 Meilleur CV R² (validation) : {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    # ── 4. OOB Score (Validation interne du modèle final refitté sur 100% du Train)
    rf_fitted = best_model.named_steps['rf']
    oob_score = rf_fitted.oob_score_
    print(f"\n  🌳 OOB Score (Validation interne sur Train Set) : {oob_score:.4f}")

    # ── 5. Scores train vs test (détection overfitting) ───────────────────
    y_pred_train = best_model.predict(X_train)
    y_pred_test  = best_model.predict(X_test)

    r2_train   = r2_score(y_train, y_pred_train)
    r2_test    = r2_score(y_test,  y_pred_test)
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
    overfit_gap = r2_train - r2_test

    overfit_flag = ("⚠️  Sur-apprentissage détecté !"
                    if overfit_gap > 0.10 else "✅ Pas de sur-apprentissage")

    print("\n" + "─"*45)
    print("  🎯 RÉSULTATS RANDOM FOREST")
    print("─"*45)
    print(f"  R² Train  : {r2_train:.4f}")
    print(f"  R² Test   : {r2_test:.4f}   (écart : {overfit_gap:+.4f})  {overfit_flag}")
    print(f"  R² OOB    : {oob_score:.4f}")
    print(f"  MAE Test  : {mae_test:.4f}")
    print(f"  RMSE Test : {rmse_test:.4f}")

    # ── 6. Cross-Validation finale avec le meilleur modèle ────────────────
    print(f"\n  🔄 Cross-Validation {CV_FOLDS}-fold finale (train set) :")
    cv_r2   = cross_val_score(best_model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
    cv_mae  = cross_val_score(best_model, X_train, y_train, cv=kf,
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_rmse = cross_val_score(best_model, X_train, y_train, cv=kf,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    _print_cv_results(cv_r2,   "R²  ")
    _print_cv_results(-cv_mae,  "MAE ")
    _print_cv_results(-cv_rmse, "RMSE")

    # ── 7. Intervalle de confiance par arbres ─────────────────────────────
    scaler_fitted = best_model.named_steps['scaler']
    X_test_scaled = scaler_fitted.transform(X_test)
    lo, hi = _confidence_interval_from_trees(rf_fitted, X_test_scaled, percentile=90)
    avg_interval_width = np.mean(hi - lo)
    print(f"\n  📊 Intervalle de confiance à 90% :")
    print(f"     Largeur moyenne : ±{avg_interval_width/2:.2f}% de fatigue")

    # ── 8. Analyse des résidus ────────────────────────────────────────────
    residuals = y_test.values - y_pred_test
    pct_10    = np.mean(np.abs(residuals) <= 10) * 100
    print(f"\n  📉 Analyse des résidus :")
    print(f"     Moyenne  : {residuals.mean():.4f}  (idéal = 0)")
    print(f"     Std      : {residuals.std():.4f}")
    print(f"     Dans ±10% fatigue : {pct_10:.1f}% des prédictions")

    # ── 9. Feature importance top 10 ─────────────────────────────────────
    model_cols  = joblib.load(BENCHMARK_DIR / "model_columns.joblib")
    importances = rf_fitted.feature_importances_
    indices     = np.argsort(importances)[::-1][:10]
    top_features = [
        {"feature": model_cols[i], "importance": round(float(importances[i]), 6)}
        for i in indices
    ]
    print(f"\n  🌟 Top 5 variables importantes :")
    for i, tf in enumerate(top_features[:5]):
        bar = "█" * int(tf['importance'] * 200)
        print(f"     {i+1}. {tf['feature']:<40} {tf['importance']:.4f}  {bar}")

    # ── 10. Sauvegarde ────────────────────────────────────────────────────
    joblib.dump(best_model, SAVE_DIR / "model_rf.joblib")

    metrics = {
        "model"            : "Random Forest",
        "R2_train"         : round(r2_train,   4),
        "R2_test"          : round(r2_test,    4),
        "R2_oob"           : round(oob_score,  4),
        "MAE"              : round(mae_test,   4),
        "RMSE"             : round(rmse_test,  4),
        "overfit_gap"      : round(overfit_gap, 4),
        "cv_r2_mean"       : round(cv_r2.mean(),  4),
        "cv_r2_std"        : round(cv_r2.std(),   4),
        "cv_mae_mean"      : round((-cv_mae).mean(),  4),
        "cv_rmse_mean"     : round((-cv_rmse).mean(), 4),
        "best_params"      : best_params,
        "ci_90_avg_width"  : round(float(avg_interval_width), 4),
        "residuals_mean"   : round(float(residuals.mean()), 4),
        "residuals_std"    : round(float(residuals.std()),  4),
        "pct_within_10"    : round(pct_10, 2),
        "top_features"     : top_features,
        "train_time_s"     : round(elapsed, 2),
    }

    with open(SAVE_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n  💾 Modèle et métriques sauvegardés dans : {SAVE_DIR}")
    print("═"*65 + "\n")


if __name__ == "__main__":
    train_random_forest()
