"""
AthlytIQ — Rapports Visuels Scouting (VERSION PRO)
=====================================================
Génère 6 graphiques d'analyse automatiques pour le module de scouting,
calqué sur la philosophie de LM/models/benchmark/generate_visuals.py.

Graphiques :
  1. Distribution des archétypes (PCA 2D colorée par cluster)
  2. Radar comparatif des profils moyens par cluster
  3. Matrice de confusion positionnelle (Poste_Cat vs Cluster)
  4. Heatmap des poids appris par poste
  5. Top 10 paires les plus similaires par poste
  6. Comparaison des métriques de similarité

Usage :
    .venv/bin/python LM2/benchmark/generate_scouting_visuals.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

warnings.filterwarnings("ignore")

# ─── Chemins ────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = ROOT / "LM2" / "benchmark"
PLOTS_DIR     = BENCHMARK_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

POSTES = ["ATT", "AG", "AD", "MOF", "MC", "MDF", "CB", "LB", "RB"]

# Palette de couleurs pour les clusters
CLUSTER_COLORS = [
    "#E07B54", "#5B8DB8", "#6AAB6A", "#D4A843", "#9B59B6",
    "#E74C3C", "#1ABC9C", "#3498DB", "#F39C12", "#8E44AD",
    "#2ECC71", "#E67E22", "#16A085", "#C0392B",
]

POSTE_COLORS = {
    "ATT": "#E74C3C", "AG": "#E07B54", "AD": "#F39C12",
    "MOF": "#3498DB", "MC": "#5B8DB8", "MDF": "#2C3E50",
    "CB": "#27AE60", "LB": "#6AAB6A", "RB": "#1ABC9C",
}


def _safe_load(filename):
    path = BENCHMARK_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None


# ─── Graphique 1 : PCA 2D des archétypes ────────────────────────────────────
def plot_archetypes_pca(profiles, features):
    """Visualise les clusters dans un espace PCA 2D."""
    if "Cluster_ID" not in profiles.columns:
        print("  ⚠️  Clusters non disponibles. Skipping graphique 1.")
        return

    X = profiles[features].fillna(0).values
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))

    clusters = profiles["Cluster_ID"].values
    unique_clusters = sorted(profiles["Cluster_ID"].unique())

    for c in unique_clusters:
        mask = clusters == c
        label = profiles.loc[mask, "Archetype"].iloc[0] if "Archetype" in profiles.columns else f"Cluster {c}"
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label,
                   alpha=0.6, edgecolors="white", s=50, linewidth=0.5)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("Carte des Archétypes de Joueurs (PCA 2D)", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

    plt.figtext(0.5, -0.02,
                "💡 Chaque point est un joueur. Les couleurs représentent les archétypes découverts par l'IA.\n"
                "Des joueurs proches sur le graphique ont des profils statistiques très similaires.",
                ha="center", fontsize=9, bbox={"facecolor": "blue", "alpha": 0.1, "pad": 8})

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "1_archetypes_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 1_archetypes_pca.png")


# ─── Graphique 2 : Radar des clusters ───────────────────────────────────────
def plot_cluster_radar(profiles, features):
    """Radar comparatif des profils moyens par cluster."""
    if "Cluster_ID" not in profiles.columns:
        print("  ⚠️  Clusters non disponibles. Skipping graphique 2.")
        return

    unique_clusters = sorted(profiles["Cluster_ID"].unique())
    n_clusters = len(unique_clusters)

    # Prendre les 8 features les plus discriminantes pour la lisibilité
    display_features = features[:8]
    n_vars = len(display_features)

    # Calculer les moyennes normalisées
    cluster_means = {}
    for c in unique_clusters:
        mask = profiles["Cluster_ID"] == c
        means = profiles.loc[mask, display_features].mean()
        # Normaliser entre 0 et 1
        global_min = profiles[display_features].min()
        global_max = profiles[display_features].max()
        norm = (means - global_min) / (global_max - global_min + 1e-8)
        cluster_means[c] = norm.values

    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for c in unique_clusters:
        vals = cluster_means[c].tolist() + [cluster_means[c][0]]
        label = profiles.loc[profiles["Cluster_ID"] == c, "Archetype"].iloc[0] \
            if "Archetype" in profiles.columns else f"Cluster {c}"
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax.plot(angles, vals, "o-", linewidth=2, label=label, color=color, alpha=0.7)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    short_labels = [f.replace("_P90", "").replace("_MA10", "").replace("_", " ") for f in display_features]
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_title("Profils Moyens par Archétype", fontsize=14, fontweight="bold", pad=20)
    ax.legend(bbox_to_anchor=(1.3, 1.0), fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "2_cluster_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 2_cluster_radar.png")


# ─── Graphique 3 : Matrice Poste vs Cluster ──────────────────────────────────
def plot_position_cluster_matrix(profiles):
    """Matrice de confusion entre les postes réels et les clusters assignés."""
    if "Cluster_ID" not in profiles.columns or "Poste_Cat" not in profiles.columns:
        print("  ⚠️  Données insuffisantes. Skipping graphique 3.")
        return

    ct = pd.crosstab(profiles["Poste_Cat"], profiles["Cluster_ID"], normalize="index")

    # Renommer les colonnes avec les archétypes
    if "Archetype" in profiles.columns:
        col_map = {}
        for c in ct.columns:
            mask = profiles["Cluster_ID"] == c
            if mask.any():
                col_map[c] = profiles.loc[mask, "Archetype"].iloc[0]
        ct.columns = [col_map.get(c, f"C{c}") for c in ct.columns]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(ct, annot=True, fmt=".0%", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "% de joueurs"})
    ax.set_title("Distribution des Postes par Archétype", fontsize=14, fontweight="bold")
    ax.set_xlabel("Archétype (Cluster)", fontsize=11)
    ax.set_ylabel("Poste Réel", fontsize=11)

    plt.figtext(0.5, -0.02,
                "💡 Cette matrice montre comment les 9 postes se répartissent dans les archétypes découverts.\n"
                "Un archétype qui contient majoritairement des CB et MDF sera un profil défensif pur.",
                ha="center", fontsize=9, bbox={"facecolor": "orange", "alpha": 0.1, "pad": 8})

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "3_position_cluster_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 3_position_cluster_matrix.png")


# ─── Graphique 4 : Heatmap des poids appris ─────────────────────────────────
def plot_weights_heatmap(features):
    """Heatmap des poids de similarité appris par poste."""
    weights_path = BENCHMARK_DIR / "learned_weights.joblib"
    if not weights_path.exists():
        print("  ⚠️  Poids appris non disponibles. Skipping graphique 4.")
        return

    learned_weights = joblib.load(weights_path)

    # Construire la matrice
    rows = []
    for poste in POSTES:
        if poste in learned_weights:
            row = [learned_weights[poste].get(f, 0) for f in features]
        else:
            row = [1.0] * len(features)
        rows.append(row)

    matrix = np.array(rows)
    short_features = [f.replace("_P90", "").replace("_MA10", "").replace("_", "\n") for f in features]

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(matrix, xticklabels=short_features, yticklabels=POSTES,
                annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Poids de Similarité"})
    ax.set_title("Poids de Similarité Appris par l'IA (par Poste × Feature)",
                 fontsize=14, fontweight="bold")

    plt.figtext(0.5, -0.03,
                "💡 LECTURE : Un poids élevé (vert) signifie que l'IA considère cette feature comme CRUCIALE pour comparer des joueurs à ce poste.\n"
                "Un poids faible (rouge) signifie que cette feature est peu pertinente pour ce poste.",
                ha="center", fontsize=9, bbox={"facecolor": "green", "alpha": 0.1, "pad": 8})

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "4_weights_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 4_weights_heatmap.png")


# ─── Graphique 5 : Distribution des scores de similarité ────────────────────
def plot_similarity_distribution(profiles, features):
    """Distribution des scores de similarité cosinus intra-cluster vs inter-cluster."""
    if "Cluster_ID" not in profiles.columns:
        print("  ⚠️  Clusters non disponibles. Skipping graphique 5.")
        return

    X = profiles[features].fillna(0).values
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    cos_sim = cosine_similarity(X_scaled)
    clusters = profiles["Cluster_ID"].values

    intra_scores = []
    inter_scores = []

    # Échantillonner pour la performance (max 500 paires)
    n = len(profiles)
    np.random.seed(42)
    sample_pairs = min(500, n * (n - 1) // 2)
    sampled = 0

    for i in range(n):
        for j in range(i + 1, n):
            if sampled >= sample_pairs:
                break
            score = cos_sim[i, j]
            if clusters[i] == clusters[j]:
                intra_scores.append(score)
            else:
                inter_scores.append(score)
            sampled += 1
        if sampled >= sample_pairs:
            break

    fig, ax = plt.subplots(figsize=(10, 6))
    if intra_scores:
        ax.hist(intra_scores, bins=30, alpha=0.7, color="#27AE60", label=f"Même archétype (n={len(intra_scores)})")
    if inter_scores:
        ax.hist(inter_scores, bins=30, alpha=0.7, color="#E74C3C", label=f"Archétypes différents (n={len(inter_scores)})")

    ax.set_xlabel("Score de Similarité Cosinus", fontsize=11)
    ax.set_ylabel("Fréquence", fontsize=11)
    ax.set_title("Validation du Clustering — Similarité Intra vs Inter Archétype",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    # Moyennes
    if intra_scores and inter_scores:
        ax.axvline(np.mean(intra_scores), color="#27AE60", linestyle="--", linewidth=2,
                   label=f"Moyenne intra = {np.mean(intra_scores):.3f}")
        ax.axvline(np.mean(inter_scores), color="#E74C3C", linestyle="--", linewidth=2,
                   label=f"Moyenne inter = {np.mean(inter_scores):.3f}")
        ax.legend(fontsize=10)

    plt.figtext(0.5, -0.02,
                "💡 Un bon clustering montre une séparation claire : les joueurs du même archétype (vert)\n"
                "ont des scores de similarité plus élevés que les joueurs d'archétypes différents (rouge).",
                ha="center", fontsize=9, bbox={"facecolor": "cyan", "alpha": 0.1, "pad": 8})

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "5_similarity_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 5_similarity_distribution.png")


# ─── Graphique 6 : Résumé global du benchmark scouting ──────────────────────
def plot_benchmark_summary():
    """Résumé visuel des métriques du benchmark scouting."""
    # Charger les métriques
    weights_meta = {}
    cluster_meta = {}

    wp = BENCHMARK_DIR / "weights_metrics.json"
    cp = BENCHMARK_DIR / "cluster_labels.json"

    if wp.exists():
        with open(wp) as f:
            weights_meta = json.load(f)
    if cp.exists():
        with open(cp) as f:
            cluster_meta = json.load(f)

    if not weights_meta and not cluster_meta:
        print("  ⚠️  Métriques non disponibles. Skipping graphique 6.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1 : R² CV par poste (poids appris)
    if "postes" in weights_meta:
        postes_data = weights_meta["postes"]
        valid = {k: v for k, v in postes_data.items() if v.get("r2_cv_mean") is not None}
        if valid:
            names = list(valid.keys())
            r2_vals = [valid[n]["r2_cv_mean"] for n in names]
            colors = [POSTE_COLORS.get(n, "#888") for n in names]

            bars = axes[0].bar(names, r2_vals, color=colors, edgecolor="white")
            for bar, val in zip(bars, r2_vals):
                axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                             f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

            axes[0].set_title("R² Cross-Validation par Poste\n(Qualité des poids appris)",
                              fontsize=12, fontweight="bold")
            axes[0].set_ylabel("R² CV (5-fold)")
            axes[0].axhline(0, color="red", linestyle="--", alpha=0.3)
            axes[0].grid(axis="y", alpha=0.2)
        else:
            axes[0].text(0.5, 0.5, "Aucune donnée", ha="center", va="center", fontsize=14)
            axes[0].set_title("R² par Poste", fontsize=12, fontweight="bold")

    # Panel 2 : Distribution des joueurs par cluster
    if "clusters" in cluster_meta:
        clusters = cluster_meta["clusters"]
        labels = [clusters[k]["label"] for k in sorted(clusters.keys(), key=int)]
        sizes = [clusters[k]["n_players"] for k in sorted(clusters.keys(), key=int)]
        colors = [CLUSTER_COLORS[int(k) % len(CLUSTER_COLORS)] for k in sorted(clusters.keys(), key=int)]

        axes[1].barh(labels, sizes, color=colors, edgecolor="white")
        for i, (s, l) in enumerate(zip(sizes, labels)):
            axes[1].text(s + 1, i, str(s), va="center", fontsize=9, fontweight="bold")

        axes[1].set_title(f"Archétypes Découverts ({len(labels)} clusters)\nSilhouette = {cluster_meta.get('best_silhouette', 0):.4f}",
                          fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Nombre de joueurs")
        axes[1].invert_yaxis()
        axes[1].grid(axis="x", alpha=0.2)

    plt.suptitle("📊 Benchmark Scouting — Résumé Global", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "6_benchmark_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 6_benchmark_summary.png")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("  🎨 GÉNÉRATION DES VISUELS SCOUTING  (VERSION PRO)")
    print("═" * 70 + "\n")

    profiles = _safe_load("scouting_profiles.joblib")
    features = _safe_load("scouting_features.joblib")

    if profiles is None or features is None:
        print("❌ Profils non trouvés. Exécutez setup_scouting_data.py d'abord.")
        raise SystemExit(1)

    print(f"  📂 {len(profiles)} profils chargés | {len(features)} features\n")

    plot_archetypes_pca(profiles, features)
    plot_cluster_radar(profiles, features)
    plot_position_cluster_matrix(profiles)
    plot_weights_heatmap(features)
    plot_similarity_distribution(profiles, features)
    plot_benchmark_summary()

    print(f"\n✅ 6 graphiques sauvegardés dans : {PLOTS_DIR}")
    print("═" * 70 + "\n")
