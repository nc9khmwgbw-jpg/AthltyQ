"""
AthlytIQ — Apprentissage Automatique des Poids de Similarité (VERSION PRO)
============================================================================
Remplace les POSITION_WEIGHTS codés en dur par des poids appris par ML.

Méthode :
  Pour chaque poste (ATT, AG, AD, MOF, MC, MDF, CB, LB, RB) :
    1. Sélectionne les joueurs de ce poste (TRAIN SET uniquement — 80%)
    2. Entraîne un Ridge Regression pour prédire Rating_MA10
    3. Les coefficients normalisés deviennent les poids de similarité
    4. Optuna optimise le hyperparamètre alpha de Ridge
    5. Cross-Validation 5-fold pour la fiabilité
    6. Évaluation sur le TEST SET (20% joueurs jamais vus)

  FIX : AD et RB sont fusionnés avec AG et LB respectivement
        (même profil tactique, trop peu de joueurs pour un modèle séparé)

Usage :
    .venv/bin/python LM2/benchmark/train_weights.py
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ─── Optuna (optionnel) ─────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False
    print("  ℹ️  Optuna non installé. Utilisation des hyperparamètres par défaut.")

# ─── Chemins ────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = ROOT / "LM2" / "benchmark"

# Postes de sortie (9 postes granulaires)
POSTES_OUTPUT = ["ATT", "AG", "AD", "MOF", "MC", "MDF", "CB", "LB", "RB"]

# Postes d'entraînement : AD fusionne avec AG, RB fusionne avec LB
# (même profil tactique — gauche/droite non distinguable statistiquement)
MERGE_MAP = {"AD": "AG", "RB": "LB"}
POSTES_TRAIN = ["ATT", "AG", "MOF", "MC", "MDF", "CB", "LB"]

TARGET_FEATURE = "Rating_MA10"

OPTUNA_TRIALS  = 30
OPTUNA_TIMEOUT = 120
CV_FOLDS       = 5
MIN_PLAYERS    = 15  # Minimum pour entraîner un modèle fiable


def _optuna_search_alpha(X: np.ndarray, y: np.ndarray) -> float:
    """Trouve le meilleur alpha pour Ridge via Optuna."""
    assert optuna is not None
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT,
                   show_progress_bar=False)
    return study.best_params["alpha"]


def train_position_weights() -> None:
    print("\n" + "═" * 70)
    print("🧠  APPRENTISSAGE DES POIDS DE SIMILARITÉ PAR POSTE  (VERSION PRO)")
    print("═" * 70)

    # ── 1. Chargement des profils et du split ────────────────────────────
    profiles = joblib.load(BENCHMARK_DIR / "scouting_profiles.joblib")
    features = joblib.load(BENCHMARK_DIR / "scouting_features.joblib")
    train_idx = joblib.load(BENCHMARK_DIR / "scouting_train_idx.joblib")
    test_idx = joblib.load(BENCHMARK_DIR / "scouting_test_idx.joblib")

    print(f"\n  📂 {len(profiles)} profils chargés | {len(features)} features")
    print(f"  👥 Split : {len(train_idx)} train / {len(test_idx)} test")

    if "Poste_Cat" not in profiles.columns:
        print("  ❌ Colonne Poste_Cat absente.")
        return

    # Créer la colonne de poste d'entraînement (fusionner AD→AG, RB→LB)
    profiles["Poste_Train"] = profiles["Poste_Cat"].map(
        lambda p: MERGE_MAP.get(p, p)
    )

    # Features d'entrée = toutes sauf la target
    input_features = [f for f in features if f != TARGET_FEATURE]
    print(f"  🎯 Target : {TARGET_FEATURE}")
    print(f"  📊 Input features : {len(input_features)}")
    print(f"  🔀 Fusion : AD → AG, RB → LB (profils latéraux identiques)")

    # ── 2. Apprentissage par poste (TRAIN SET uniquement) ────────────────
    learned_weights = {}
    all_metrics = {}
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    t0_global = time.time()

    for poste in POSTES_TRAIN:
        # Sélectionner les joueurs de ce poste dans le TRAIN SET uniquement
        mask_poste = profiles["Poste_Train"] == poste
        mask_train = profiles.index.isin([profiles.index[i] for i in train_idx])
        mask_test = profiles.index.isin([profiles.index[i] for i in test_idx])

        train_mask = mask_poste & mask_train
        test_mask = mask_poste & mask_test

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train < MIN_PLAYERS:
            print(f"\n  ⚠️  {poste} : {n_train} joueurs train — trop peu, poids par défaut")
            learned_weights[poste] = {f: 1.0 for f in features}
            all_metrics[poste] = {
                "n_train": int(n_train), "n_test": int(n_test),
                "r2_cv": None, "r2_test": None, "method": "default"
            }
            continue

        X_train = profiles.loc[train_mask, input_features].fillna(0).values
        y_train = profiles.loc[train_mask, TARGET_FEATURE].fillna(0).values
        X_test_p = profiles.loc[test_mask, input_features].fillna(0).values
        y_test_p = profiles.loc[test_mask, TARGET_FEATURE].fillna(0).values

        # Normaliser (fitté sur train uniquement)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test_p) if n_test > 0 else None

        # Optuna pour alpha
        if OPTUNA_AVAILABLE and n_train >= 20:
            alpha = _optuna_search_alpha(X_train_s, y_train)
        else:
            alpha = 1.0

        # Entraîner Ridge (TRAIN SET uniquement)
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train)

        # Cross-Validation (TRAIN SET)
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=kf, scoring="r2")
        r2_cv_mean = cv_scores.mean()
        r2_cv_std = cv_scores.std()

        # Score Train
        r2_train = r2_score(y_train, model.predict(X_train_s))

        # Score Test (20% joueurs JAMAIS VUS)
        r2_test = None
        mae_test = None
        if X_test_s is not None and n_test >= 3:
            y_pred_test = model.predict(X_test_s)
            r2_test = r2_score(y_test_p, y_pred_test)
            mae_test = mean_absolute_error(y_test_p, y_pred_test)

        # Détecter overfitting
        overfit_gap = r2_train - (r2_test if r2_test is not None else r2_cv_mean)
        overfit_flag = "⚠️ Overfitting" if overfit_gap > 0.15 else "✅ OK"

        # Extraire les coefficients comme poids
        coeffs = np.abs(model.coef_)
        if coeffs.max() > 0:
            coeffs_norm = coeffs / coeffs.max()
        else:
            coeffs_norm = np.ones_like(coeffs)

        weights = {TARGET_FEATURE: 1.0}
        for i, f in enumerate(input_features):
            weights[f] = round(float(coeffs_norm[i]) * 2.5, 3)

        learned_weights[poste] = weights
        all_metrics[poste] = {
            "n_train": int(n_train),
            "n_test": int(n_test),
            "alpha": round(alpha, 6),
            "r2_train": round(r2_train, 4),
            "r2_cv_mean": round(r2_cv_mean, 4),
            "r2_cv_std": round(r2_cv_std, 4),
            "r2_test": round(r2_test, 4) if r2_test is not None else None,
            "mae_test": round(mae_test, 4) if mae_test is not None else None,
            "overfit_gap": round(overfit_gap, 4),
            "method": "optuna" if OPTUNA_AVAILABLE and n_train >= 20 else "default",
        }

        print(f"\n  {'─' * 55}")
        print(f"  📌 {poste} ({n_train} train / {n_test} test)")
        print(f"     Alpha    : {alpha:.4f}")
        print(f"     R² Train : {r2_train:.4f}")
        print(f"     R² CV    : {r2_cv_mean:.4f} ± {r2_cv_std:.4f}")
        if r2_test is not None:
            print(f"     R² Test  : {r2_test:.4f}  (écart: {overfit_gap:+.4f})  {overfit_flag}")
            print(f"     MAE Test : {mae_test:.4f}")
        else:
            print(f"     R² Test  : N/A (pas assez de joueurs test)")

        # Top 3 features
        top3_idx = np.argsort(coeffs_norm)[::-1][:3]
        for rank, idx in enumerate(top3_idx):
            print(f"     Top {rank+1}: {input_features[idx]:30s} → poids {weights[input_features[idx]]:.3f}")

    # ── 3. Copier les poids fusionnés (AD=AG, RB=LB) ────────────────────
    for merged, source in MERGE_MAP.items():
        if source in learned_weights:
            learned_weights[merged] = dict(learned_weights[source])
            all_metrics[merged] = dict(all_metrics.get(source, {}))
            all_metrics[merged]["note"] = f"Copié depuis {source} (fusion latérale)"
            print(f"\n  🔗 {merged} ← Poids copiés depuis {source}")

    elapsed = time.time() - t0_global

    # ── 4. Sauvegarde ────────────────────────────────────────────────────
    joblib.dump(learned_weights, BENCHMARK_DIR / "learned_weights.joblib")

    report = {
        "postes": all_metrics,
        "total_time_s": round(elapsed, 2),
        "optuna_available": OPTUNA_AVAILABLE,
        "n_trials_per_poste": OPTUNA_TRIALS,
        "cv_folds": CV_FOLDS,
        "merge_map": MERGE_MAP,
        "min_players_threshold": MIN_PLAYERS,
    }
    with open(BENCHMARK_DIR / "weights_metrics.json", "w") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # ── 5. Résumé final ──────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"✅ POIDS APPRIS POUR {len(POSTES_TRAIN)} POSTES (+ {len(MERGE_MAP)} copiés)")
    print(f"   Temps total : {elapsed:.1f}s")
    print(f"\n  📊 RÉSUMÉ TRAIN vs TEST :")
    print(f"  {'Poste':6s} {'Train':>6s} {'Test':>6s} {'R²_Train':>10s} {'R²_CV':>10s} {'R²_Test':>10s} {'Écart':>8s}")
    print(f"  {'─'*58}")
    for p in POSTES_TRAIN:
        m = all_metrics.get(p, {})
        r2t = f"{m.get('r2_train', 0):.4f}" if m.get('r2_train') is not None else "N/A"
        r2cv = f"{m.get('r2_cv_mean', 0):.4f}" if m.get('r2_cv_mean') is not None else "N/A"
        r2te = f"{m.get('r2_test', 0):.4f}" if m.get('r2_test') is not None else "N/A"
        gap = f"{m.get('overfit_gap', 0):+.4f}" if m.get('overfit_gap') is not None else "N/A"
        print(f"  {p:6s} {m.get('n_train', 0):6d} {m.get('n_test', 0):6d} {r2t:>10s} {r2cv:>10s} {r2te:>10s} {gap:>8s}")

    print(f"\n  💾 Artefacts sauvegardés :")
    print(f"     • learned_weights.joblib (9 postes)")
    print(f"     • weights_metrics.json")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    train_position_weights()
