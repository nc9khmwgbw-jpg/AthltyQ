"""
AthlytIQ — Génération des 14 graphiques post-entraînement
==========================================================
Couvre les 3 modèles de régression (RF, LGBM, Poly) + le modèle de blessure.

Catégories :
  1. Performance & métriques comparatives      → graphes 01 à 04
  2. Prédictions vs valeurs réelles             → graphes 05 à 07
  3. Distribution des erreurs                   → graphes 08 à 09
  4. Importance des variables                   → graphes 10 à 12
  5. Modèle de blessure (classification)        → graphes 13 à 14

Usage :
  python generate_model_graphs.py
  python generate_model_graphs.py --output_dir ./mes_graphes
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    auc, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error,
    precision_recall_curve, r2_score, roc_curve,
)
from sklearn.model_selection import (
    KFold, cross_val_score, learning_curve, train_test_split,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Palette & style ────────────────────────────────────────────────────────
plt.style.use("dark_background")

BG       = "#1A1A2E"   # fond principal
BG2      = "#16213E"   # fond secondaire (axes)
GRID_C   = "#2A2A4A"   # couleur grille

C_RF     = "#FF9F1C"   # orange   → Random Forest
C_LGBM   = "#2EC4B6"   # teal     → LightGBM
C_POLY   = "#A78BFA"   # violet   → Polynomial Regression
C_INJ    = "#F87171"   # rouge    → Injury model
C_GOOD   = "#4ADE80"   # vert     → bon résultat
C_NEU    = "#94A3B8"   # gris     → neutre / baseline

MODEL_COLORS = {"Random Forest": C_RF, "LightGBM": C_LGBM, "Polynomial": C_POLY}
MODEL_SHORT  = {"Random Forest": "RF", "LightGBM": "LGBM", "Polynomial": "Poly"}

SAVE_DPI = 180
FIG_BG   = BG


def _style_ax(ax, grid=True):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=C_NEU, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    if grid:
        ax.grid(color=GRID_C, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.title.set_color("white")
    ax.xaxis.label.set_color(C_NEU)
    ax.yaxis.label.set_color(C_NEU)


def _save(fig, path):
    fig.patch.set_facecolor(FIG_BG)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    print(f"  ✅  {Path(path).name}")


# ════════════════════════════════════════════════════════════════════════════
# DONNÉES SYNTHÉTIQUES (remplace les données réelles si absentes)
# ════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n=3000, seed=42):
    """
    Génère un jeu de données synthétique imitant features_dataset.csv.
    Retourne X_train, X_test, y_train, y_test + noms de features.
    """
    rng = np.random.default_rng(seed)
    feature_names = [
        "Minutes_MA7", "Distance_P90", "Sprints_MA5",
        "Rating_MA15", "ACWR", "Cumulative_Min_21d",
        "Days_Rest", "Age", "Usage_Factor", "Form_Score",
    ]
    n_feat = len(feature_names)
    X_raw = rng.standard_normal((n, n_feat))

    # Target fatigue (0-100), corrélée aux features
    coefs = rng.uniform(-1, 1, n_feat)
    noise = rng.standard_normal(n) * 8
    y_raw = 50 + X_raw @ coefs * 5 + noise
    y_reg = np.clip(y_raw, 0, 100)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y_reg, test_size=0.2, random_state=seed
    )
    return X_tr, X_te, y_tr, y_te, feature_names


def train_regression_models(X_train, X_test, y_train, y_test):
    """Entraîne RF, LGBM simulé (Ridge ici) et Poly sur les données."""
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train)
    Xte_sc = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_rf_train = rf.predict(X_train)
    y_rf_test  = rf.predict(X_test)

    # LGBM simulé par un GradientBoosting léger (évite la dépendance si absent)
    try:
        from lightgbm import LGBMRegressor
        lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=63,
                             random_state=42, verbose=-1, n_jobs=-1)
        lgbm.fit(Xtr_sc, y_train)
        y_lgbm_train = lgbm.predict(Xtr_sc)
        y_lgbm_test  = lgbm.predict(Xte_sc)
        lgbm_fi      = lgbm.feature_importances_
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        lgbm = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        lgbm.fit(Xtr_sc, y_train)
        y_lgbm_train = lgbm.predict(Xtr_sc)
        y_lgbm_test  = lgbm.predict(Xte_sc)
        lgbm_fi      = lgbm.feature_importances_

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    poly_pipe = Pipeline([
        ("sc",   StandardScaler()),
        ("pca",  PCA(n_components=8, random_state=42)),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", RidgeCV(alphas=[0.1, 1, 10, 100])),
    ])
    poly_pipe.fit(X_train, y_train)
    y_poly_train = poly_pipe.predict(X_train)
    y_poly_test  = poly_pipe.predict(X_test)

    models_data = {
        "Random Forest": {
            "model": rf, "scaler": None,
            "y_train_pred": y_rf_train, "y_test_pred": y_rf_test,
            "feature_importances": rf.feature_importances_,
            "X_train": X_train, "X_test": X_test,
        },
        "LightGBM": {
            "model": lgbm, "scaler": scaler,
            "y_train_pred": y_lgbm_train, "y_test_pred": y_lgbm_test,
            "feature_importances": lgbm_fi,
            "X_train": Xtr_sc, "X_test": Xte_sc,
        },
        "Polynomial": {
            "model": poly_pipe, "scaler": None,
            "y_train_pred": y_poly_train, "y_test_pred": y_poly_test,
            "feature_importances": None,
            "X_train": X_train, "X_test": X_test,
        },
    }
    return models_data


def compute_metrics(models_data, y_train, y_test):
    """Calcule R², MAE, RMSE, overfitting gap pour chaque modèle."""
    metrics = {}
    for name, d in models_data.items():
        r2_tr  = r2_score(y_train, d["y_train_pred"])
        r2_te  = r2_score(y_test,  d["y_test_pred"])
        mae    = mean_absolute_error(y_test, d["y_test_pred"])
        rmse   = np.sqrt(mean_squared_error(y_test, d["y_test_pred"]))
        resid  = y_test - d["y_test_pred"]
        pct10  = np.mean(np.abs(resid) <= 10) * 100
        metrics[name] = {
            "R2_train": r2_tr, "R2_test": r2_te,
            "MAE": mae, "RMSE": rmse,
            "overfit_gap": r2_tr - r2_te,
            "residuals": resid, "pct_within_10": pct10,
        }
    return metrics


def compute_cv_scores(models_data, X_train, y_train, cv=5):
    """Cross-validation R² pour chaque modèle."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = {}
    for name, d in models_data.items():
        scores = cross_val_score(
            d["model"], d["X_train"], y_train, cv=kf, scoring="r2", n_jobs=-1
        )
        cv_scores[name] = scores
    return cv_scores


# ════════════════════════════════════════════════════════════════════════════
#  CATÉGORIE 1 — Performance & métriques (graphes 01–04)
# ════════════════════════════════════════════════════════════════════════════

def graph_01_metrics_comparison(metrics, out_dir):
    """Grouped bar : R², MAE, RMSE pour les 3 modèles."""
    names  = list(metrics.keys())
    colors = [MODEL_COLORS[n] for n in names]
    r2s    = [metrics[n]["R2_test"]  for n in names]
    maes   = [metrics[n]["MAE"]      for n in names]
    rmses  = [metrics[n]["RMSE"]     for n in names]

    x   = np.arange(len(names))
    w   = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparaison des métriques — Test Set", fontsize=16, color="white", y=1.02)

    for ax, vals, title, fmt, best_fn in zip(
        axes,
        [r2s, maes, rmses],
        ["R² (↑ meilleur)", "MAE (↓ meilleur)", "RMSE (↓ meilleur)"],
        [".3f", ".2f", ".2f"],
        [max, min, min],
    ):
        bars = ax.bar(x, vals, color=colors, width=0.55, zorder=3, edgecolor=BG, linewidth=0.8)
        best_idx = vals.index(best_fn(vals))
        bars[best_idx].set_edgecolor(C_GOOD)
        bars[best_idx].set_linewidth(2.5)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[n] for n in names], fontsize=12)
        ax.set_title(title, fontsize=13, pad=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=11,
                    color="white", fontweight="bold")
        _style_ax(ax)

    plt.tight_layout()
    _save(fig, out_dir / "01_metrics_comparison.png")


def graph_02_radar_metrics(metrics, cv_scores, out_dir):
    """Radar spider : R², MAE normalisé, RMSE normalisé, CV R², stabilité CV."""
    names = list(metrics.keys())
    labels = ["R² test", "MAE\n(inversé)", "RMSE\n(inversé)", "CV R²", "Stabilité CV"]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Normalisation 0-1 (1 = meilleur)
    def norm_inv(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn: return [0.5] * len(vals)
        return [1 - (v - mn) / (mx - mn) for v in vals]

    r2_vals   = [metrics[n]["R2_test"]  for n in names]
    mae_vals  = [metrics[n]["MAE"]      for n in names]
    rmse_vals = [metrics[n]["RMSE"]     for n in names]
    cv_means  = [cv_scores[n].mean()    for n in names]
    cv_stds   = [cv_scores[n].std()     for n in names]

    all_data = {
        n: [
            r2_vals[i],
            norm_inv(mae_vals)[i],
            norm_inv(rmse_vals)[i],
            cv_means[i],
            1 - norm_inv(cv_stds)[i],
        ]
        for i, n in enumerate(names)
    }

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(BG2)
    fig.patch.set_facecolor(BG)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11, color=C_NEU)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8, color=C_NEU)
    ax.grid(color=GRID_C, linewidth=0.8)
    ax.spines["polar"].set_color(GRID_C)

    for name, vals in all_data.items():
        vals_plot = vals + vals[:1]
        c = MODEL_COLORS[name]
        ax.plot(angles, vals_plot, color=c, linewidth=2.5, label=name)
        ax.fill(angles, vals_plot, color=c, alpha=0.12)

    ax.set_title("Radar — Performance multi-métriques", fontsize=14, color="white", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              facecolor=BG2, edgecolor=GRID_C, labelcolor="white", fontsize=11)
    _save(fig, out_dir / "02_radar_metrics.png")


def graph_03_train_vs_test_r2(metrics, out_dir):
    """Barres doublées Train / Test R² — visualise l'overfitting gap."""
    names = list(metrics.keys())
    x     = np.arange(len(names))
    w     = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_tr = ax.bar(x - w/2, [metrics[n]["R2_train"] for n in names],
                     width=w, label="Train", color=[MODEL_COLORS[n] for n in names],
                     alpha=0.5, zorder=3, edgecolor=BG)
    bars_te = ax.bar(x + w/2, [metrics[n]["R2_test"]  for n in names],
                     width=w, label="Test",  color=[MODEL_COLORS[n] for n in names],
                     alpha=1.0, zorder=3, edgecolor=BG)

    # Flèches indiquant le gap
    for i, n in enumerate(names):
        gap = metrics[n]["overfit_gap"]
        y_tr = metrics[n]["R2_train"]
        y_te = metrics[n]["R2_test"]
        color_gap = "#F87171" if gap > 0.10 else C_GOOD
        ax.annotate("", xy=(x[i] + w/2, y_te), xytext=(x[i] - w/2, y_tr),
                    arrowprops=dict(arrowstyle="-|>", color=color_gap, lw=1.5))
        ax.text(x[i], max(y_tr, y_te) + 0.015, f"Δ={gap:+.3f}",
                ha="center", fontsize=9.5, color=color_gap, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("R² Train vs Test — Détection du sur-apprentissage", fontsize=14)
    legend = ax.legend(facecolor=BG2, edgecolor=GRID_C, labelcolor="white", fontsize=11)

    # Ligne seuil overfitting
    ax.axhline(y=1.0, color=GRID_C, linewidth=0.8, linestyle="--")
    _style_ax(ax)
    plt.tight_layout()
    _save(fig, out_dir / "03_train_vs_test_r2.png")


def graph_04_cv_boxplot(cv_scores, out_dir):
    """Boxplot des scores R² sur les 5 folds CV pour chaque modèle."""
    names  = list(cv_scores.keys())
    data   = [cv_scores[n] for n in names]
    colors = [MODEL_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        data, patch_artist=True, notch=True,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(color=C_NEU, linewidth=1.5),
        capprops=dict(color=C_NEU, linewidth=1.5),
        flierprops=dict(markerfacecolor=C_NEU, marker="o", markersize=5),
        boxprops=dict(linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor(color)

    # Points individuels (jitter)
    for i, (d, c) in enumerate(zip(data, colors), 1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, color=c, s=50, zorder=5, alpha=0.9)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("R² (fold)", fontsize=12)
    ax.set_title("Cross-Validation 5-fold — Stabilité par modèle", fontsize=14)
    _style_ax(ax)
    plt.tight_layout()
    _save(fig, out_dir / "04_cv_boxplot.png")


# ════════════════════════════════════════════════════════════════════════════
#  CATÉGORIE 2 — Prédictions vs valeurs réelles (graphes 05–07)
# ════════════════════════════════════════════════════════════════════════════

def graph_05_scatter_pred_vs_real(models_data, y_test, out_dir):
    """Scatter Prédit vs Réel avec diagonale y=x pour chaque modèle."""
    names = list(models_data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Prédictions vs Valeurs réelles — Test Set", fontsize=16, color="white", y=1.02)

    for ax, name in zip(axes, names):
        y_pred = models_data[name]["y_test_pred"]
        c = MODEL_COLORS[name]
        r2 = r2_score(y_test, y_pred)

        # Hexbin pour densité
        hb = ax.hexbin(y_test, y_pred, gridsize=35, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Densité")

        # Diagonale parfaite
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], color=C_GOOD, linestyle="--", linewidth=2, label="y = ŷ")

        # Ligne de régression
        m, b = np.polyfit(y_test, y_pred, 1)
        ax.plot([lo, hi], [m*lo+b, m*hi+b], color=c, linewidth=2, label="Régression")

        ax.set_xlabel("Valeur réelle (fatigue %)")
        ax.set_ylabel("Valeur prédite (fatigue %)")
        ax.set_title(f"{name}\nR² = {r2:.4f}", fontsize=13)
        ax.legend(facecolor=BG2, edgecolor=GRID_C, labelcolor="white", fontsize=9)
        _style_ax(ax)

    plt.tight_layout()
    _save(fig, out_dir / "05_scatter_pred_vs_real.png")


def graph_06_residuals_plot(models_data, y_test, out_dir):
    """Résidus (y − ŷ) vs valeurs prédites — détecte les biais systématiques."""
    names = list(models_data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Analyse des résidus — Test Set", fontsize=16, color="white", y=1.02)

    for ax, name in zip(axes, names):
        y_pred = models_data[name]["y_test_pred"]
        resid  = y_test - y_pred
        c = MODEL_COLORS[name]

        ax.scatter(y_pred, resid, color=c, alpha=0.35, s=15, zorder=3)

        # Ligne zéro
        ax.axhline(0, color=C_GOOD, linestyle="--", linewidth=2)

        # Bande ±1 std
        std = resid.std()
        ax.axhline(+std, color=C_NEU, linestyle=":", linewidth=1, alpha=0.7)
        ax.axhline(-std, color=C_NEU, linestyle=":", linewidth=1, alpha=0.7)
        ax.fill_between(ax.get_xlim(), -std, std, color=C_NEU, alpha=0.05)

        # Loess smooth (approximé par LOWESS scipy)
        from scipy.ndimage import uniform_filter1d
        order = np.argsort(y_pred)
        smooth = uniform_filter1d(resid[order], size=max(10, len(resid)//20))
        ax.plot(y_pred[order], smooth, color="#FBBF24", linewidth=2.5, label="Tendance")

        mae  = np.abs(resid).mean()
        bias = resid.mean()
        ax.set_xlabel("Valeur prédite (fatigue %)")
        ax.set_ylabel("Résidu (réel − prédit)")
        ax.set_title(f"{name}\nBiais={bias:.2f}  MAE={mae:.2f}", fontsize=13)
        ax.legend(facecolor=BG2, edgecolor=GRID_C, labelcolor="white", fontsize=9)
        _style_ax(ax)

    plt.tight_layout()
    _save(fig, out_dir / "06_residuals_plot.png")


def graph_07_confidence_interval_rf(models_data, X_test, y_test, out_dir):
    """
    Intervalle de confiance 90% basé sur les arbres individuels (RF uniquement).
    Montre les prédictions triées avec leur bande d'incertitude.
    """
    rf_model = models_data["Random Forest"]["model"]
    y_pred   = models_data["Random Forest"]["y_test_pred"]

    # Prédictions individuelles de chaque arbre
    tree_preds = np.array([tree.predict(X_test) for tree in rf_model.estimators_])
    lo = np.percentile(tree_preds, 5,  axis=0)
    hi = np.percentile(tree_preds, 95, axis=0)

    # Trier par valeur réelle
    order   = np.argsort(y_test)
    n_show  = min(150, len(y_test))
    idx     = order[:n_show]

    x_plot = np.arange(n_show)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(x_plot, lo[idx], hi[idx], color=C_RF, alpha=0.25, label="IC 90% (arbres)")
    ax.plot(x_plot, y_pred[idx], color=C_RF,   linewidth=2,   label="Prédiction RF", zorder=4)
    ax.plot(x_plot, y_test[idx], color=C_GOOD, linewidth=1.5, linestyle="--", label="Valeur réelle", zorder=5)

    avg_width = np.mean(hi - lo)
    ax.set_xlabel("Joueurs triés par fatigue réelle")
    ax.set_ylabel("Fatigue (%)")
    ax.set_title(f"Random Forest — Intervalle de confiance 90% par arbre\n"
                 f"Largeur moyenne : ±{avg_width/2:.1f}%  (n={len(y_test)} points)", fontsize=14)
    ax.legend(facecolor=BG2, edgecolor=GRID_C, labelcolor="white", fontsize=11)
    _style_ax(ax)
    plt.tight_layout()
    _save(fig, out_dir / "07_confidence_interval_rf.png")


# ════════════════════════════════════════════════════════════════════════════
#  CATÉGORIE 3 — Distribution des erreurs (graphes 08–09)
# ════════════════════════════════════════════════════════════════════════════

def graph_08_residuals_histogram(models_data, y_test, out_dir):
    """Histogramme + KDE des résidus pour chaque modèle."""
    names = list(models_data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Distribution des résidus — Idéal : cloche centrée sur 0", fontsize=15, color="white", y=1.02)

    for ax, name in zip(axes, names):
        resid = y_test - models_data[name]["y_test_pred"]
        c = MODEL_COLORS[name]

        ax.hist(resid, bins=35, color=c, alpha=0.55, density=True, zorder=3, edgecolor=BG)

        # KDE
        kde_x = np.linspace(resid.min(), resid.max(), 300)
        kde   = stats.gaussian_kde(resid)
        ax.plot(kde_x, kde(kde_x), color=c, linewidth=2.5, label="KDE")

        # Courbe normale théorique
        mu, std = resid.mean(), resid.std()
        norm_y  = stats.norm.pdf(kde_x, mu, std)
        ax.plot(kde_x, norm_y, color="white", linewidth=1.5, linestyle="--", label="Normale théorique", alpha=0.7)

        # Ligne mu
        ax.axvline(mu, color="#FBBF24", linewidth=2, linestyle="-.", label=f"μ={mu:.2f}")
        ax.axvline(0,  color=C_GOOD,   linewidth=1.5, linestyle="--", alpha=0.6, label="0")

        ax.set_xlabel("Résidu (réel − prédit)")
        ax.set_ylabel("Densité")
        ax.set_title(f"{name}\nμ={mu:.2f}  σ={std:.2f}", fontsize=13)
        ax.legend(facecolor=BG2, edgecolor=GRID_C, labelcolor="white", fontsize=9)
        _style_ax(ax)

    plt.tight_layout()
    _save(fig, out_dir / "08_residuals_histogram.png")


def graph_09_pct_within_thresholds(metrics, out_dir):
    """
    Barres horizontales : % de prédictions dans ±5%, ±10%, ±15%, ±20%.
    Métrique métier clé pour l'usage terrain (entraîneur / staff médical).
    """
    names      = list(metrics.keys())
    thresholds = [5, 10, 15, 20]

    # Recalcul des % par seuil à partir des résidus
    pct_data = {n: [] for n in names}
    for n in names:
        resid = metrics[n]["residuals"]
        for th in thresholds:
            pct_data[n].append(np.mean(np.abs(resid) <= th) * 100)

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos  = np.arange(len(thresholds))
    height = 0.22

    for i, name in enumerate(names):
        offset = (i - 1) * height
        bars   = ax.barh(
            y_pos + offset, pct_data[name],
            height=height, color=MODEL_COLORS[name],
            label=name, alpha=0.85, edgecolor=BG, zorder=3,
        )
        for bar, val in zip(bars, pct_data[name]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9.5, color="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"±{th}% fatigue" for th in thresholds], fontsize=12)
    ax.set_xlabel("% de prédictions dans le seuil", fontsize=12)
    ax.set_xlim(0, 108)
    ax.set_title("% de prédictions dans ±N% de fatigue — Métrique terrain", fontsize=14)
    ax.legend(facecolor=BG2, edgecolor=GRID_C, labelcolor="white", fontsize=11)
    ax.axvline(80, color=C_GOOD, linewidth=1.5, linestyle="--", alpha=0.6, label="Objectif 80%")
    _style_ax(ax, grid=False)
    ax.grid(axis="x", color=GRID_C, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    _save(fig, out_dir / "09_pct_within_thresholds.png")


# ════════════════════════════════════════════════════════════════════════════
#  CATÉGORIE 4 — Importance des variables (graphes 10–12)
# ════════════════════════════════════════════════════════════════════════════

def graph_10_feature_importance_rf(models_data, feature_names, out_dir):
    """Barres horizontales — Feature importance RF (top 10)."""
    fi      = models_data["Random Forest"]["feature_importances_"] if "feature_importances_" in models_data["Random Forest"] else models_data["Random Forest"]["feature_importances"]
    indices = np.argsort(fi)
    top_n   = min(10, len(indices))
    idx     = indices[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        [feature_names[i] for i in idx],
        fi[idx] * 100,
        color=C_RF, edgecolor=BG, alpha=0.85, zorder=3,
    )
    for bar, val in zip(bars, fi[idx] * 100):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, color="white")

    ax.set_xlabel("Importance relative (%)", fontsize=12)
    ax.set_title("Random Forest — Top 10 variables importantes\n(MDI : Mean Decrease in Impurity)", fontsize=13)
    _style_ax(ax)
    plt.tight_layout()
    _save(fig, out_dir / "10_feature_importance_rf.png")


def graph_11_shap_lgbm(models_data, feature_names, out_dir):
    """
    SHAP summary plot (beeswarm) pour LightGBM.
    Fallback vers feature importance native si SHAP absent.
    """
    lgbm_data = models_data["LightGBM"]
    lgbm_model = lgbm_data["model"]
    X_test_sc  = lgbm_data["X_test"]

    try:
        import shap
        explainer = shap.TreeExplainer(lgbm_model)
        n_shap    = min(300, X_test_sc.shape[0])
        shap_vals = explainer.shap_values(X_test_sc[:n_shap])
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        order     = np.argsort(mean_abs)

        fig, ax = plt.subplots(figsize=(12, 7))

        # Beeswarm manuel
        rng = np.random.default_rng(42)
        for rank, feat_idx in enumerate(order[-10:]):
            shap_col  = shap_vals[:n_shap, feat_idx]
            feat_vals = X_test_sc[:n_shap, feat_idx]
            # Normaliser couleur sur valeur feature
            norm_feat = (feat_vals - feat_vals.min()) / ((feat_vals.max() - feat_vals.min()) + 1e-9)
            colors_pt = plt.cm.coolwarm(norm_feat)
            jitter    = rng.uniform(-0.25, 0.25, len(shap_col))
            ax.scatter(shap_col, np.full(len(shap_col), rank) + jitter,
                       c=colors_pt, s=12, alpha=0.6, zorder=3)

        ax.set_yticks(range(10))
        ax.set_yticklabels([feature_names[i] for i in order[-10:]], fontsize=11)
        ax.axvline(0, color="white", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Valeur SHAP (impact sur la prédiction de fatigue)", fontsize=12)
        ax.set_title("LightGBM — SHAP Beeswarm (rouge = valeur feature élevée)", fontsize=14)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Valeur de la feature (normalisée)")
        cbar.ax.yaxis.label.set_color(C_NEU)
        cbar.ax.tick_params(colors=C_NEU)
        _style_ax(ax)

    except ImportError:
        # Fallback : feature importance native
        fi      = lgbm_data["feature_importances"]
        indices = np.argsort(fi)[-10:]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(
            [feature_names[i] for i in indices],
            fi[indices],
            color=C_LGBM, edgecolor=BG, alpha=0.85, zorder=3,
        )
        for bar, val in zip(bars, fi[indices]):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}", va="center", fontsize=10, color="white")
        ax.set_xlabel("Importance (split count)", fontsize=12)
        ax.set_title("LightGBM — Feature importance native\n(SHAP non disponible — pip install shap)", fontsize=13)
        _style_ax(ax)

    plt.tight_layout()
    _save(fig, out_dir / "11_shap_lgbm.png")


def graph_12_backprojection_poly(models_data, feature_names, out_dir):
    """
    Back-projection PCA → features originales pour le modèle Polynomial.
    Importance = Σ|coef_Ridge * loading_PCA|.
    """
    pipe = models_data["Polynomial"]["model"]

    try:
        pca   = pipe.named_steps["pca"]
        ridge = pipe.named_steps["ridge"]

        loadings   = pca.components_                       # (n_comp, n_feat)
        n_comp     = loadings.shape[0]
        # Les features poly : [linéaires, croisées, carrées]
        # On prend les n_comp premiers coefficients (termes linéaires)
        ridge_coefs = ridge.coef_
        linear_coefs = ridge_coefs[:n_comp]

        imp_orig = np.sum(np.abs(linear_coefs[:, np.newaxis] * loadings), axis=0)
        order    = np.argsort(imp_orig)

        fig, ax = plt.subplots(figsize=(10, 6))
        vals    = imp_orig[order[-10:]]
        labels  = [feature_names[i] for i in order[-10:]]

        # Normaliser pour pourcentage
        vals_pct = vals / vals.sum() * 100

        cmap   = plt.cm.get_cmap("Purples")
        colors = [cmap(0.4 + 0.6 * v / vals_pct.max()) for v in vals_pct]
        bars   = ax.barh(labels, vals_pct, color=colors, edgecolor=BG, alpha=0.9, zorder=3)
        for bar, v in zip(bars, vals_pct):
            ax.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=10, color="white")

        ax.set_xlabel("Importance relative reconstruite (%)", fontsize=12)
        ax.set_title("Régression Polynomiale — Back-projection PCA\n"
                     r"Importance $\approx$ $\Sigma$|coef$_{Ridge}$ × loading$_{PCA}$|", fontsize=13)
        _style_ax(ax)

    except Exception as e:
        # Fallback si le pipeline n'a pas la structure attendue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Back-projection indisponible\n{e}", ha="center", va="center",
                fontsize=12, color=C_NEU, transform=ax.transAxes)
        ax.set_title("Régression Polynomiale — Feature importance", fontsize=13)
        _style_ax(ax, grid=False)

    plt.tight_layout()
    _save(fig, out_dir / "12_backprojection_poly.png")


# ════════════════════════════════════════════════════════════════════════════
#  CATÉGORIE 5 — Modèle de blessure / classification (graphes 13–14)
# ════════════════════════════════════════════════════════════════════════════

def _build_injury_data(seed=42):
    """Génère des données de classification déséquilibrées (blessures = 10%)."""
    feat_names = ["ACWR", "Fatigue_Index", "Congestion_Risk",
                  "Cumulative_Min_21d", "Days_Rest",
                  "Form_Score", "Age_Risk_Factor", "Usage_Factor"]
    X, y = make_classification(
        n_samples=4000, n_features=len(feat_names),
        n_informative=6, n_redundant=2,
        weights=[0.90, 0.10], random_state=seed, flip_y=0.03,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                               random_state=seed, stratify=y)
    # RF classifier avec class_weight='balanced' (comme injury_predictor.py)
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=8,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    y_proba = clf.predict_proba(X_te)[:, 1]
    y_pred  = clf.predict(X_te)
    return clf, X_te, y_te, y_pred, y_proba, feat_names


def graph_13_roc_auc(out_dir):
    """Courbe ROC-AUC avec zone ombrée pour le modèle de blessure."""
    clf, X_te, y_te, y_pred, y_proba, feat_names = _build_injury_data()

    fpr, tpr, thresholds = roc_curve(y_te, y_proba)
    roc_auc = auc(fpr, tpr)
    ap_score = average_precision_score(y_te, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Modèle de blessure — Évaluation (classification)", fontsize=15, color="white")

    # — Courbe ROC
    ax = axes[0]
    ax.fill_between(fpr, tpr, color=C_INJ, alpha=0.2)
    ax.plot(fpr, tpr, color=C_INJ, linewidth=3, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=C_NEU, linestyle="--", linewidth=1.5, label="Hasard (AUC=0.5)")

    # Point opérationnel (seuil 0.5)
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    ax.scatter(fpr[idx_05], tpr[idx_05], color="white", s=100, zorder=5,
               label=f"Seuil 0.5\n(FPR={fpr[idx_05]:.2f}, TPR={tpr[idx_05]:.2f})")

    ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs (TPR / Recall)", fontsize=12)
    ax.set_title("Courbe ROC", fontsize=13)
    ax.legend(facecolor=BG2, edgecolor=GRID_C, labelcolor="white", fontsize=10)
    _style_ax(ax)

    # — Courbe Precision-Recall
    ax2 = axes[1]
    prec, rec, _ = precision_recall_curve(y_te, y_proba)
    baseline      = y_te.mean()
    ax2.fill_between(rec, prec, color=C_LGBM, alpha=0.2)
    ax2.plot(rec, prec, color=C_LGBM, linewidth=3, label=f"AP = {ap_score:.3f}")
    ax2.axhline(baseline, color=C_NEU, linestyle="--", linewidth=1.5,
                label=f"Baseline (prévalence = {baseline:.0%})")
    ax2.set_xlabel("Recall (Vrais Positifs / Blessés réels)", fontsize=12)
    ax2.set_ylabel("Précision", fontsize=12)
    ax2.set_title("Courbe Precision-Recall\n(critique pour classes déséquilibrées)", fontsize=13)
    ax2.legend(facecolor=BG2, edgecolor=GRID_C, labelcolor="white", fontsize=10)
    _style_ax(ax2)

    plt.tight_layout()
    _save(fig, out_dir / "13_roc_auc_injury.png")


def graph_14_confusion_matrix(out_dir):
    """Matrice de confusion annotée + métriques détaillées pour le modèle de blessure."""
    clf, X_te, y_te, y_pred, y_proba, feat_names = _build_injury_data()
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    brier       = brier_score_loss(y_te, y_proba)

    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

    # — Heatmap matrice de confusion
    ax_cm = fig.add_subplot(gs[0])
    labels = np.array([
        [f"VN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"VP\n{tp}"],
    ])
    background = np.array([[tn, fp], [fn, tp]], dtype=float)
    # Normaliser par ligne pour voir taux
    background_norm = background / background.sum(axis=1, keepdims=True)

    sns.heatmap(
        background_norm, annot=labels, fmt="", cmap="RdYlGn",
        ax=ax_cm, vmin=0, vmax=1, linewidths=2, linecolor=BG,
        annot_kws={"size": 16, "color": "black", "weight": "bold"},
        cbar_kws={"label": "Taux par ligne"},
    )
    ax_cm.set_xticklabels(["Prédit : Sain", "Prédit : Blessé"], fontsize=12)
    ax_cm.set_yticklabels(["Réel : Sain", "Réel : Blessé"],   fontsize=12, rotation=0)
    ax_cm.set_title("Matrice de confusion\n(normalisée par ligne)", fontsize=13, color="white")
    ax_cm.title.set_color("white")
    ax_cm.tick_params(colors="white")

    # — Panel métriques
    ax_m = fig.add_subplot(gs[1])
    ax_m.set_facecolor(BG2)
    ax_m.axis("off")

    metrics_list = [
        ("Recall (Sensibilité)",  recall,     "Blessés correctement détectés"),
        ("Précision",             precision,  "Alarmes qui sont vraies"),
        ("F1-Score",              f1,         "Équilibre Precision-Recall"),
        ("Spécificité",           specificity,"Sains correctement classés"),
        ("Brier Score",           brier,      "Calibration de probabilité (↓ mieux)"),
    ]
    y_start = 0.88
    ax_m.text(0.5, 0.97, "Métriques détaillées", ha="center", va="top",
              fontsize=14, color="white", fontweight="bold",
              transform=ax_m.transAxes)

    colors_val = [C_GOOD, C_GOOD, C_GOOD, C_GOOD, C_INJ]
    for i, (label, val, desc) in enumerate(metrics_list):
        y = y_start - i * 0.16
        cv = colors_val[i] if (val > 0.5 if i < 4 else val < 0.15) else C_INJ
        ax_m.text(0.05, y, label, transform=ax_m.transAxes, fontsize=12, color=C_NEU)
        ax_m.text(0.55, y, f"{val:.3f}", transform=ax_m.transAxes, fontsize=14,
                  color=cv, fontweight="bold")
        ax_m.text(0.05, y - 0.065, desc, transform=ax_m.transAxes, fontsize=9,
                  color="#64748B", style="italic")

    # Annotation coût des erreurs
    ax_m.text(0.05, 0.08, "⚠  FN = blessure non détectée (coût élevé)",
              transform=ax_m.transAxes, fontsize=9, color=C_INJ)
    ax_m.text(0.05, 0.03, "✓  FP = repos inutile (coût faible)",
              transform=ax_m.transAxes, fontsize=9, color=C_GOOD)

    fig.suptitle("Modèle de blessure — Analyse complète", fontsize=16, color="white", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "14_confusion_matrix_injury.png")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(output_dir: str = "./graphs_athlytiq"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'═'*60}")
    print("  AthlytIQ — Génération des 14 graphiques post-entraînement")
    print(f"{'═'*60}")
    print(f"  Dossier de sortie : {out.resolve()}\n")

    # ── Données & modèles ──────────────────────────────────────────
    print("► Génération des données synthétiques...")
    X_tr, X_te, y_tr, y_te = generate_synthetic_data()[:4]
    feat_names = generate_synthetic_data()[4]

    print("► Entraînement des 3 modèles de régression...")
    models_data = train_regression_models(X_tr, X_te, y_tr, y_te)

    print("► Calcul des métriques & cross-validation...")
    metrics   = compute_metrics(models_data, y_tr, y_te)
    cv_scores = compute_cv_scores(models_data, X_tr, y_tr)

    print("\n► Génération des graphiques :\n")

    # ── Catégorie 1 : Performance & métriques ─────────────────────
    print("  [1/5] Performance & métriques comparatives")
    graph_01_metrics_comparison(metrics, out)
    graph_02_radar_metrics(metrics, cv_scores, out)
    graph_03_train_vs_test_r2(metrics, out)
    graph_04_cv_boxplot(cv_scores, out)

    # ── Catégorie 2 : Prédictions vs réel ─────────────────────────
    print("  [2/5] Prédictions vs valeurs réelles")
    graph_05_scatter_pred_vs_real(models_data, y_te, out)
    graph_06_residuals_plot(models_data, y_te, out)
    graph_07_confidence_interval_rf(models_data, X_te, y_te, out)

    # ── Catégorie 3 : Distribution des erreurs ────────────────────
    print("  [3/5] Distribution des erreurs")
    graph_08_residuals_histogram(models_data, y_te, out)
    graph_09_pct_within_thresholds(metrics, out)

    # ── Catégorie 4 : Feature importance ─────────────────────────
    print("  [4/5] Importance des variables")
    graph_10_feature_importance_rf(models_data, feat_names, out)
    graph_11_shap_lgbm(models_data, feat_names, out)
    graph_12_backprojection_poly(models_data, feat_names, out)

    # ── Catégorie 5 : Modèle de blessure ─────────────────────────
    print("  [5/5] Modèle de blessure (classification)")
    graph_13_roc_auc(out)
    graph_14_confusion_matrix(out)

    print(f"\n{'═'*60}")
    print(f"  ✅  14 graphiques sauvegardés dans : {out.resolve()}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AthlytIQ — génération des 14 graphiques ML")
    parser.add_argument("--output_dir", default="./graphs_athlytiq",
                        help="Dossier de sortie (défaut : ./graphs_athlytiq)")
    args = parser.parse_args()
    main(args.output_dir)