import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ─── Chemins ────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARK_DIR = ROOT / "LM" / "models" / "benchmark"
PLOTS_DIR     = BENCHMARK_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

IDENTITY_COLS = [
    'Nom', 'Player_ID', 'Match_Date', 'Event_ID', 'Home_Team', 'Away_Team',
    'Score_Home', 'Score_Away', 'Tournament', 'Equipe', 'Position', 'Poste_Cat',
    'Risk_Category', 'Target_Injury_Occurred', 'Current_Injury',
    'Fatigue_Realisee', 'Fatigue_Reelle_Match_T'
]

MODEL_DIRS = {
    "Polynomial Regression": "Polynomial_Regression",
    "Random Forest"        : "Random_Forest",
    "LightGBM"             : "LightGBM",
}

COLORS = {
    "Polynomial Regression": "#E07B54",
    "Random Forest"        : "#5B8DB8",
    "LightGBM"             : "#6AAB6A",
}


# ─── Helpers ────────────────────────────────────────────────────────────────
def load_all_metrics():
    metrics = []
    for name, folder in MODEL_DIRS.items():
        path = BENCHMARK_DIR / folder / "metrics.json"
        if path.exists():
            with open(path) as f:
                m = json.load(f)
            # Compatibilité : ancien format sans R2_test
            if 'R2_test' not in m and 'R2' in m:
                m['R2_test'] = m['R2']
            if 'R2_train' not in m:
                m['R2_train'] = None
            metrics.append(m)
    return metrics


def load_test_predictions(metrics):
    """Charge les données de test et calcule les prédictions des 3 modèles."""
    data_path = ROOT / "data" / "processed" / "features_dataset.csv"
    if not data_path.exists():
        return None, None, {}

    df = pd.read_csv(data_path)
    col_nom = 'Nom' if 'Nom' in df.columns else 'Player_Name'
    X = df.drop(columns=IDENTITY_COLS + ['Target_Fatigue'], errors='ignore')
    y = df['Target_Fatigue']
    model_cols = joblib.load(BENCHMARK_DIR / "model_columns.joblib")
    X = X.select_dtypes(include='number').reindex(columns=model_cols, fill_value=0).fillna(0)

    test_players = set(joblib.load(BENCHMARK_DIR / "test_players.joblib"))
    mask_test    = df[col_nom].isin(test_players)
    X_test, y_test = X[mask_test], y[mask_test]

    preds = {}
    for name, folder in MODEL_DIRS.items():
        # Essayer les deux noms possibles
        for candidate in [
            BENCHMARK_DIR / folder / "model_poly.joblib",
            BENCHMARK_DIR / folder / "model_rf.joblib",
            BENCHMARK_DIR / folder / "model_lgbm.joblib",
        ]:
            if candidate.exists():
                model = joblib.load(candidate)
                # Si le pipeline contient un scaler, on n'en a pas besoin d'externe
                try:
                    preds[name] = model.predict(X_test)
                except Exception:
                    # Fallback : ancien scaler externe
                    scaler = joblib.load(BENCHMARK_DIR / "benchmark_scaler.joblib")
                    X_scaled = scaler.transform(X_test)
                    preds[name] = model.predict(X_scaled)
                break

    return X_test, y_test, preds


# ─── Graphique 1 : Métriques d'erreur + R² ──────────────────────────────────
def plot_metrics_overview(metrics):
    df = pd.DataFrame(metrics)
    models = df['model'].tolist()
    colors = [COLORS.get(m, '#888888') for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparaison des Modèles IA — Métriques Globales",
                 fontsize=15, fontweight='bold', y=1.02)

    def bar_chart(ax, col, title, ylabel, higher_better=False):
        vals = df[col].fillna(0).values
        bars = ax.bar(models, vals, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
        for bar, val in zip(bars, vals):
            y_pos = bar.get_height() + abs(bar.get_height()) * 0.02
            color = '#27ae60' if (higher_better and val == max(vals)) else \
                    '#e74c3c' if (not higher_better and val == min(vals)) else 'black'
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=color)

    bar_chart(axes[0], 'MAE',    "Marge d'Erreur Moyenne (MAE)\n[Plus bas = Plus précis]",  "% fatigue", higher_better=False)
    bar_chart(axes[1], 'RMSE',   "Pénalité de Grosses Erreurs (RMSE)\n[Évalue la dangerosité des écarts]", "% fatigue", higher_better=False)
    bar_chart(axes[2], 'R2_test',"Fiabilité Globale (R²)\n[1.0 = Perfection | 0.0 = Hasard]", "Score R²",  higher_better=True)

    # AJOUT : Note pédagogique ultra-détaillée
    plt.figtext(0.5, -0.08, 
                "📘 GLOSSAIRE DES MÉTRIQUES :\n"
                "• R² (R-Squared) : La 'note' de compréhension de l'IA. 0.30 signifie que l'IA explique 30% des variations de fatigue.\n"
                "• MAE : L'erreur moyenne. Si MAE=12, l'IA se trompe en moyenne de 12 points sur une jauge de 100.\n"
                "• RMSE : Identique à la MAE mais 'punit' plus fort les grosses erreurs. Un RMSE proche de la MAE = Modèle stable.\n"
                "• LAMBDA/ALPHA : Les 'freins' mathématiques qui empêchent l'IA de sur-apprendre des détails inutiles.",
                ha="center", fontsize=9, bbox={"facecolor":"orange", "alpha":0.1, "pad":10})

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "1_metriques_globales.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ 1_metriques_globales.png")


# ─── Graphique 2 : Train vs Test R² (overfitting) ────────────────────────────
def plot_overfitting_analysis(metrics):
    df = pd.DataFrame(metrics)
    models = df['model'].tolist()

    # Ne tracer que si R2_train est disponible
    if df['R2_train'].isnull().all():
        print("  ⚠️  R2_train non disponible (ancien format). Skipping graphique 2.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.35

    r2_train = df['R2_train'].fillna(0).values
    r2_test  = df['R2_test'].fillna(0).values

    bars1 = ax.bar(x - w/2, r2_train, w, label='R² Train',  # type: ignore[arg-type]
                   color=[COLORS.get(m, '#888') for m in models], alpha=0.9)
    bars2 = ax.bar(x + w/2, r2_test,  w, label='R² Test',   # type: ignore[arg-type]
                   color=[COLORS.get(m, '#888') for m in models], alpha=0.45,
                   hatch='//', edgecolor='white')

    for bar, val in zip(list(bars1) + list(bars2), list(r2_train) + list(r2_test)):
        yp = max(0, bar.get_height()) + 0.01
        ax.text(bar.get_x() + bar.get_width()/2., yp,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Flèches montrant l'écart (overfitting)
    for i, (rt, rv) in enumerate(zip(r2_train, r2_test)):
        if rt is not None and rv is not None and rt > rv + 0.05:
            ax.annotate('', xy=(x[i] + w/2, rv), xytext=(x[i] - w/2, rt),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
            gap = rt - rv
            ax.text(x[i] + 0.05, (rt + rv)/2, f'Δ={gap:.2f}',
                    fontsize=8, color='red', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_title("Détection d'Overfitting — R² Train vs R² Test",
                 fontsize=13, fontweight='bold')
    ax.set_ylabel("Score R²")
    ax.legend(fontsize=10)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    
    # AJOUT : Note pédagogique
    plt.figtext(0.5, -0.05, 
                "💡 DIAGNOSTIC D'OVERFITTING :\n"
                "L'overfitting (sur-apprentissage) survient quand un modèle apprend par cœur le passé (Train) mais échoue sur le futur (Test).\n"
                "• Un écart (Δ) inférieur à 0.15 est considéré comme sain pour des données de performance physique.",
                ha="center", fontsize=10, bbox={"facecolor":"blue", "alpha":0.1, "pad":8})

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "2_overfitting_train_vs_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ 2_overfitting_train_vs_test.png")


# ─── Graphique 3 : Cross-Validation ─────────────────────────────────────────
def plot_cv_results(metrics):
    df = pd.DataFrame(metrics)

    has_cv = 'cv_r2_mean' in df.columns and not df['cv_r2_mean'].isnull().all()
    if not has_cv:
        print("  ⚠️  CV scores non disponibles. Skipping graphique 3.")
        return

    models = df['model'].tolist()
    means  = df['cv_r2_mean'].fillna(0).values
    stds   = df['cv_r2_std'].fillna(0).values
    colors = [COLORS.get(m, '#888888') for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, means, color=colors, edgecolor='white',  # type: ignore[arg-type]
                  linewidth=0.8, alpha=0.9, zorder=2)
    ax.errorbar(models, means, yerr=stds * 2,  # type: ignore[arg-type]  # Intervalle à 2 std (~95%)
                fmt='none', color='black', capsize=8, capthick=2,
                linewidth=2, zorder=3, label='±2 std (≈95%)')

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2.,
                mean + std * 2 + 0.01,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title("Cross-Validation 5-Fold — R² moyen ± 2 std\n(C'est la métrique qui compte vraiment)",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("R² moyen (CV)")
    ax.axhline(0, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, zorder=1)

    # AJOUT : Note pédagogique ultra-détaillée
    plt.figtext(0.5, -0.08, 
                "📘 COMPRENDRE LA VALIDATION CROISÉE :\n"
                "• CV 5-FOLD : On divise les joueurs en 5 groupes. On entraîne sur 4 et on teste sur le 5ème. On répète 5 fois.\n"
                "• MEAN (Moyenne) : La performance réelle que vous aurez sur de nouveaux joueurs jamais vus.\n"
                "• STD (Écart-type) : La stabilité. Un STD faible (ex: ±0.01) signifie que l'IA est fiable partout.\n"
                "• BARRE NOIRE : Représente la zone de confiance. Plus elle est courte, plus le modèle est solide.",
                ha="center", fontsize=9, bbox={"facecolor":"green", "alpha":0.1, "pad":10})

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "3_cross_validation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ 3_cross_validation.png")


# ─── Graphique 4 : Scatter prédictions vs réalité ────────────────────────────
def plot_scatter_predictions(metrics, preds_dict, y_test):
    if not preds_dict or y_test is None:
        print("  ⚠️  Prédictions non disponibles. Skipping graphique 4.")
        return

    fig, axes = plt.subplots(1, len(preds_dict), figsize=(7 * len(preds_dict), 6), sharey=False)
    if len(preds_dict) == 1:
        axes = [axes]

    for ax, (name, pred) in zip(axes, preds_dict.items()):
        color = COLORS.get(name, '#888888')
        r2    = next((m.get('R2_test', m.get('R2')) for m in metrics if m['model'] == name), None)
        mae   = next((m.get('MAE')  for m in metrics if m['model'] == name), None)
        pct10 = next((m.get('pct_within_10') for m in metrics if m['model'] == name), None)

        # Scatter
        ax.scatter(y_test, pred, alpha=0.3, color=color, edgecolors='none', s=20, zorder=2)

        # Ligne parfaite
        lims = [max(0, min(y_test.min(), pred.min()) - 5),
                min(105, max(y_test.max(), pred.max()) + 5)]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Idéal (Y=X)', zorder=3)

        # Ligne de tendance
        try:
            z = np.polyfit(y_test, pred, 1)
            p = np.poly1d(z)
            xs = np.linspace(lims[0], lims[1], 100)
            ax.plot(xs, p(xs), '--', color='darkorange', lw=1.5,
                    label='Tendance modèle', zorder=3)
        except Exception:
            pass

        title = f"{name}"
        if r2  is not None: title += f"\nR²={r2:.3f}"
        if mae is not None: title += f"  MAE={mae:.2f}"
        if pct10 is not None: title += f"\n{pct10:.0f}% dans ±10%"

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Vraie Fatigue (%)", fontsize=10)
        ax.set_ylabel("Fatigue Prédite (%)", fontsize=9)
        ax.set_xlim(lims)
        ax.set_ylim(-20, 120)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.2, zorder=1)

    # AJOUT : Note pédagogique
    plt.figtext(0.5, -0.02, 
                "💡 ANALYSE DE CORRÉLATION :\n"
                "• Ligne Rouge (Y=X) : La perfection. Plus les points sont proches de cette ligne, plus l'IA est proche de la réalité.\n"
                "• Dispersion : Une dispersion large indique que certains profils de joueurs atypiques sont plus difficiles à prédire.",
                ha="center", fontsize=10, bbox={"facecolor":"purple", "alpha":0.1, "pad":8})

    plt.suptitle("Prédictions vs Réalité — Test Set",
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "4_predictions_vs_realite.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ 4_predictions_vs_realite.png")


# ─── Graphique 5 : Distribution des résidus ──────────────────────────────────
def plot_residuals(metrics, preds_dict, y_test):
    if not preds_dict or y_test is None:
        print("  ⚠️  Prédictions non disponibles. Skipping graphique 5.")
        return

    fig, axes = plt.subplots(1, len(preds_dict), figsize=(6 * len(preds_dict), 5))
    if len(preds_dict) == 1:
        axes = [axes]

    for ax, (name, pred) in zip(axes, preds_dict.items()):
        color     = COLORS.get(name, '#888888')
        residuals = y_test.values - pred

        sns.histplot(residuals, kde=True, ax=ax, color=color, alpha=0.7, bins=30)
        ax.axvline(0,                 color='red',   linestyle='--', lw=1.5, label='Idéal (0)')
        ax.axvline(residuals.mean(),  color='orange', linestyle='-',  lw=1.5,
                   label=f'Moyenne ({residuals.mean():.2f})')

        ax.set_title(f"Résidus — {name}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Erreur (Réel - Prédit)", fontsize=9)
        ax.set_ylabel("Fréquence", fontsize=9)
        ax.legend(fontsize=8)

        # Annotation skewness
        skew = scipy_stats.skew(residuals)
        ax.text(0.97, 0.96, f'Skewness: {skew:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='gray')

    # AJOUT : Note pédagogique
    plt.figtext(0.5, -0.05, 
                "💡 ANALYSE DES ERREURS (RÉSIDUS) :\n"
                "• Centre à 0 : Signifie que le modèle ne surestime pas (biais positif) ni ne sous-estime (biais négatif) la fatigue.\n"
                "• Cloche de Gauss : Plus la cloche est haute et étroite, plus les erreurs sont petites et concentrées.",
                ha="center", fontsize=10, bbox={"facecolor":"red", "alpha":0.1, "pad":8})

    plt.suptitle("Distribution des Résidus — Test Set",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "5_distribution_residus.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ 5_distribution_residus.png")


# ─── Graphique 6 : Feature Importance / SHAP ─────────────────────────────────
def plot_feature_importance(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    palette_map = {
        "Polynomial Regression": "Reds_r",
        "Random Forest"        : "Blues_r",
        "LightGBM"             : "Greens_r",
    }

    for ax, (name, folder) in zip(axes, MODEL_DIRS.items()):
        m = next((x for x in metrics if x['model'] == name), None)
        if not m or 'top_features' not in m:
            ax.set_title(f"{name}\n(données non disponibles)", fontsize=11)
            ax.axis('off')
            continue

        df_feat = pd.DataFrame(m['top_features'])
        feat_col = 'importance' if 'importance' in df_feat.columns else 'shap_importance'
        df_feat = df_feat.sort_values(feat_col, ascending=False).head(10)

        sns.barplot(data=df_feat, x=feat_col, y='feature',
                    hue='feature', legend=False,
                    ax=ax, palette=palette_map.get(name, 'viridis'))

        label_type = "SHAP (impact moyen)" if 'shap_importance' in df_feat.columns \
                     else "Importance relative"
        ax.set_title(f"Top 10 Variables\n{name}", fontsize=12, fontweight='bold')
        ax.set_xlabel(label_type, fontsize=10)
        ax.set_ylabel("")
        ax.grid(axis='x', alpha=0.3)

    # AJOUT : Note pédagogique
    plt.figtext(0.5, -0.05, 
                "💡 COMPRÉHENSION DES VARIABLES (SHAP) :\n"
                "• SHAP Values : Mesurent l'impact réel de chaque donnée sur le score final de fatigue.\n"
                "• Hiérarchie : Les variables en haut de la liste sont les leviers principaux utilisés par l'IA pour son diagnostic.",
                ha="center", fontsize=10, bbox={"facecolor":"cyan", "alpha":0.1, "pad":8})

    plt.suptitle("Variables les Plus Importantes par Modèle",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "6_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ 6_feature_importance.png")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*65)
    print("  🎨 GÉNÉRATION DES VISUELS  (VERSION PRO)")
    print("═"*65 + "\n")

    metrics = load_all_metrics()
    if not metrics:
        print("❌ Aucun metrics.json trouvé. Lancez les entraînements d'abord.")
        raise SystemExit(1)

    print(f"  📂 {len(metrics)} modèles trouvés : {[m['model'] for m in metrics]}\n")

    # Chargement des prédictions (optionnel si le dataset est disponible)
    _, y_test, preds_dict = load_test_predictions(metrics)

    plot_metrics_overview(metrics)
    plot_overfitting_analysis(metrics)
    plot_cv_results(metrics)
    plot_scatter_predictions(metrics, preds_dict, y_test)
    plot_residuals(metrics, preds_dict, y_test)
    plot_feature_importance(metrics)

    print(f"\n✅ 6 graphiques sauvegardés dans : {PLOTS_DIR}")
    print("═"*65 + "\n")