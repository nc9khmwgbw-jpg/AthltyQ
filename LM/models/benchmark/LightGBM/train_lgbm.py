import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Optuna (optionnel mais recommandé) ──────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    OPTUNA_AVAILABLE = False
    print("  ℹ️  Optuna non installé. Utilisation des hyperparamètres par défaut.")
    print("     Pour activer : pip install optuna")

# ─── SHAP (optionnel mais recommandé) ────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False
    print("  ℹ️  SHAP non installé. Explicabilité désactivée.")
    print("     Pour activer : pip install shap")

# ─── Chemins ────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent.parent.parent
BENCHMARK_DIR = ROOT / "LM" / "models" / "benchmark"
DATA_PATH     = ROOT / "data" / "processed" / "features_dataset.csv"
SAVE_DIR      = BENCHMARK_DIR / "LightGBM"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IDENTITY_COLS = [
    'Nom', 'Player_ID', 'Match_Date', 'Event_ID', 'Home_Team', 'Away_Team',
    'Score_Home', 'Score_Away', 'Tournament', 'Equipe', 'Position', 'Poste_Cat',
    'Risk_Category', 'Target_Injury_Occurred', 'Current_Injury',
    'Fatigue_Realisee', 'Fatigue_Reelle_Match_T'
]

CV_FOLDS        = 5
OPTUNA_TRIALS   = 50      # Nombre d'essais Optuna (augmenter pour de meilleurs résultats)
OPTUNA_TIMEOUT  = 300     # Limite en secondes (5 min)


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


def _optuna_search(X_train_scaled: np.ndarray, y_train: pd.Series) -> tuple[dict, float]:
    """
    Recherche des meilleurs hyperparamètres avec Optuna.
    Utilise une validation croisée interne sur le train set.
    """
    assert optuna is not None, "Optuna doit être installé pour appeler cette fonction"
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators',    100, 1000),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'num_leaves'       : trial.suggest_int('num_leaves',      15, 300),
            'max_depth'        : -1,   # LightGBM gère seul avec num_leaves
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'subsample'        : trial.suggest_float('subsample',   0.5, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state'     : 42,
            'n_jobs'           : -1,
            'verbose'          : -1,
        }
        scores = []
        for train_idx, val_idx in kf.split(X_train_scaled):  # type: ignore[arg-type]
            Xt, Xv = X_train_scaled[train_idx], X_train_scaled[val_idx]
            yt, yv = y_train.values[train_idx], y_train.values[val_idx]
            m = lgb.LGBMRegressor(**params)
            m.fit(Xt, yt,
                  eval_set=[(Xv, yv)],
                  callbacks=[
                      lgb.early_stopping(stopping_rounds=30, verbose=False),
                      lgb.log_evaluation(period=-1),
                  ])
            scores.append(r2_score(yv, m.predict(Xv)))  # type: ignore[arg-type]
        return np.mean(scores)

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,
                   n_trials   = OPTUNA_TRIALS,
                   timeout    = OPTUNA_TIMEOUT,
                   show_progress_bar=False)

    best = study.best_params
    best['max_depth'] = -1
    best['random_state'] = 42
    best['n_jobs'] = -1
    best['verbose'] = -1
    return best, study.best_value


def train_lightgbm():
    print("\n" + "═"*65)
    print("  🚀 ENTRAÎNEMENT : LIGHTGBM  (VERSION PRO)")
    print("═"*65)

    # ── 1. Données ────────────────────────────────────────────────────────
    df, col_nom, X, y = _load_data()
    X_train, y_train, X_test, y_test = _split(df, col_nom, X, y)
    print(f"\n  Train : {X_train.shape[0]:,} lignes  |  Test : {X_test.shape[0]:,} lignes")
    print(f"  Features : {X_train.shape[1]}")

    # ── 2. Scaler (pour Optuna et pipeline final) ─────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── 3. Recherche des hyperparamètres ──────────────────────────────────
    if OPTUNA_AVAILABLE:
        print(f"\n  🔍 Optimisation Optuna ({OPTUNA_TRIALS} essais, timeout {OPTUNA_TIMEOUT}s)...")
        t0 = time.time()
        best_params, best_cv_score = _optuna_search(X_train_scaled, y_train)  # type: ignore[arg-type]
        elapsed_optuna = time.time() - t0
        print(f"  ✅ Optuna terminé en {elapsed_optuna:.1f}s")
        print(f"  📈 Meilleur CV R² Optuna : {best_cv_score:.4f}")
        print("  🏆 Meilleurs hyperparamètres :")
        for k, v in best_params.items():
            if k not in ('random_state', 'n_jobs', 'verbose', 'max_depth'):
                print(f"     • {k} = {v}")
    else:
        # Hyperparamètres par défaut optimisés manuellement (correction bug max_depth)
        best_params = {
            'n_estimators'     : 500,
            'learning_rate'    : 0.05,
            'num_leaves'       : 63,       # ✅ max_depth=-1, num_leaves contrôle la complexité
            'max_depth'        : -1,        # ✅ Correction : -1 = illimité, cohérent avec num_leaves
            'min_child_samples': 20,
            'reg_alpha'        : 0.1,
            'reg_lambda'       : 0.1,
            'subsample'        : 0.8,
            'colsample_bytree' : 0.8,
            'random_state'     : 42,
            'n_jobs'           : -1,
            'verbose'          : -1,
        }
        elapsed_optuna = 0
        best_cv_score  = None

    # ── 4. Entraînement final avec early stopping ─────────────────────────
    print("\n  ⏳ Entraînement final avec early stopping...")

    # Séparation d'un validation set depuis le train pour early stopping
    val_size = int(0.15 * len(X_train_scaled))
    X_es_train = X_train_scaled[val_size:]
    X_es_val   = X_train_scaled[:val_size]
    y_es_train = y_train.values[val_size:]
    y_es_val   = y_train.values[:val_size]

    lgbm_final = lgb.LGBMRegressor(**best_params)  # type: ignore[call-overload]
    t0 = time.time()
    lgbm_final.fit(
        X_es_train, y_es_train,
        eval_set=[(X_es_val, y_es_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
    )
    elapsed_fit = time.time() - t0
    n_iter_used = lgbm_final.best_iteration_ or best_params['n_estimators']
    print(f"  ✅ Terminé en {elapsed_fit:.2f}s")
    print(f"  🌱 Arbres utilisés (early stopping) : {n_iter_used}")

    # ── 5. Construction du pipeline final déployable ──────────────────────
    # On re-entraîne sur TOUT le train set avec les meilleurs params + n_estimators fixé
    best_params_final = dict(best_params)
    best_params_final['n_estimators'] = n_iter_used
    best_params_final.pop('verbose', None)
    best_params_final['verbose'] = -1

    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lgbm',   lgb.LGBMRegressor(**best_params_final))  # type: ignore[call-overload]
    ])
    final_pipeline.fit(X_train, y_train)
    print("  ✅ Pipeline final (scaler + LGBM) entraîné sur 100% du train set.")

    # ── 6. Scores train vs test (détection overfitting) ───────────────────
    y_pred_train = final_pipeline.predict(X_train)
    y_pred_test  = final_pipeline.predict(X_test)

    r2_train  = r2_score(y_train, y_pred_train)
    r2_test   = r2_score(y_test,  y_pred_test)
    mae_test  = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    overfit_gap = r2_train - r2_test

    overfit_flag = ("⚠️  Sur-apprentissage détecté !"
                    if overfit_gap > 0.10 else "✅ Pas de sur-apprentissage")

    print("\n" + "─"*45)
    print("  🎯 RÉSULTATS LIGHTGBM")
    print("─"*45)
    print(f"  R² Train  : {r2_train:.4f}")
    print(f"  R² Test   : {r2_test:.4f}   (écart : {overfit_gap:+.4f})  {overfit_flag}")
    print(f"  MAE Test  : {mae_test:.4f}")
    print(f"  RMSE Test : {rmse_test:.4f}")

    # ── 7. Cross-Validation 5-fold ────────────────────────────────────────
    print(f"\n  🔄 Cross-Validation {CV_FOLDS}-fold (train set) :")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_r2   = cross_val_score(final_pipeline, X_train, y_train, cv=kf,
                               scoring='r2', n_jobs=-1)
    cv_mae  = cross_val_score(final_pipeline, X_train, y_train, cv=kf,
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_rmse = cross_val_score(final_pipeline, X_train, y_train, cv=kf,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    _print_cv_results(cv_r2,   "R²  ")
    _print_cv_results(-cv_mae,  "MAE ")
    _print_cv_results(-cv_rmse, "RMSE")

    # ── 8. SHAP values (explicabilité) ────────────────────────────────────
    shap_summary = []
    if SHAP_AVAILABLE:
        print("\n  🔬 Calcul des SHAP values (sur 200 échantillons test)...")
        model_cols = joblib.load(BENCHMARK_DIR / "model_columns.joblib")
        scaler_fitted = final_pipeline.named_steps['scaler']
        lgbm_fitted   = final_pipeline.named_steps['lgbm']

        # Limiter à 200 pour la vitesse
        assert shap is not None, "SHAP doit être installé pour appeler cette section"
        n_shap = min(200, len(X_test))
        X_shap = scaler_fitted.transform(X_test.iloc[:n_shap])
        explainer  = shap.TreeExplainer(lgbm_fitted)
        shap_vals  = explainer.shap_values(X_shap)
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_indices  = np.argsort(mean_abs_shap)[::-1][:10]
        shap_summary  = [
            {"feature": model_cols[i], "shap_importance": round(float(mean_abs_shap[i]), 6)}
            for i in shap_indices
        ]
        print("  🌟 Top 5 SHAP (impact moyen absolu sur la prédiction) :")
        for i, s in enumerate(shap_summary[:5]):
            bar = "█" * int(s['shap_importance'] * 10)
            print(f"     {i+1}. {s['feature']:<40} {s['shap_importance']:.4f}  {bar}")

        # Sauvegarde de l'explainer pour utilisation future
        joblib.dump(explainer, SAVE_DIR / "shap_explainer.joblib")
    else:
        # Feature importance native LightGBM
        model_cols  = joblib.load(BENCHMARK_DIR / "model_columns.joblib")
        lgbm_fitted = final_pipeline.named_steps['lgbm']
        importances = lgbm_fitted.feature_importances_
        indices     = np.argsort(importances)[::-1][:10]
        shap_summary = [
            {"feature": model_cols[i], "shap_importance": round(float(importances[i]), 6)}
            for i in indices
        ]
        print("\n  🌟 Top 5 variables (feature importance LGBM) :")
        for i, s in enumerate(shap_summary[:5]):
            print(f"     {i+1}. {s['feature']:<40} {s['shap_importance']:.4f}")

    # ── 9. Analyse des résidus ────────────────────────────────────────────
    residuals = y_test.values - y_pred_test
    pct_10    = np.mean(np.abs(residuals) <= 10) * 100
    print("\n  📉 Analyse des résidus :")
    print(f"     Moyenne  : {residuals.mean():.4f}  (idéal = 0)")
    print(f"     Std      : {residuals.std():.4f}")
    print(f"     Dans ±10% fatigue : {pct_10:.1f}% des prédictions")

    # ── 10. Sauvegarde ────────────────────────────────────────────────────
    joblib.dump(final_pipeline, SAVE_DIR / "model_lgbm.joblib")

    metrics = {
        "model"              : "LightGBM",
        "R2_train"           : round(r2_train,  4),
        "R2_test"            : round(r2_test,   4),
        "MAE"                : round(mae_test,  4),
        "RMSE"               : round(rmse_test, 4),
        "overfit_gap"        : round(overfit_gap, 4),
        "cv_r2_mean"         : round(cv_r2.mean(),  4),
        "cv_r2_std"          : round(cv_r2.std(),   4),
        "cv_mae_mean"        : round((-cv_mae).mean(),  4),
        "cv_rmse_mean"       : round((-cv_rmse).mean(), 4),
        "best_params"        : {k: v for k, v in best_params_final.items()
                                if k not in ('random_state', 'n_jobs', 'verbose')},
        "optuna_best_cv_r2"  : round(best_cv_score, 4) if best_cv_score else None,
        "n_estimators_used"  : int(n_iter_used),
        "residuals_mean"     : round(float(residuals.mean()), 4),
        "residuals_std"      : round(float(residuals.std()),  4),
        "pct_within_10"      : round(pct_10, 2),
        "top_features"       : shap_summary,
        "shap_available"     : SHAP_AVAILABLE,
        "train_time_s"       : round(elapsed_fit, 2),
    }

    with open(SAVE_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n  💾 Modèle et métriques sauvegardés dans : {SAVE_DIR}")
    print("═"*65 + "\n")


if __name__ == "__main__":
    train_lightgbm()