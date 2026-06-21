"""
AthlytIQ — Clustering des Archétypes de Joueurs (VERSION PRO)
===============================================================
Découvre automatiquement les archétypes de joueurs via K-Means.

Pipeline :
  1. Charge les profils normalisés
  2. Détecte le nombre optimal de clusters (Elbow + Silhouette)
  3. Entraîne K-Means
  4. Labellise automatiquement chaque cluster (analyse des centroids)
  5. Sauvegarde le modèle et les labels

Usage :
    .venv/bin/python LM2/benchmark/train_clusters.py
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Chemins ────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = ROOT / "LM2" / "benchmark"

# Plage de clusters à tester
K_MIN = 4
K_MAX = 14


# ─── Labellisation automatique des clusters ─────────────────────────────────
def _auto_label_cluster(centroid: np.ndarray, features: list) -> str:
    """
    Génère un label humain pour un cluster en analysant quelles features
    ont les valeurs les plus élevées dans le centroid (déjà normalisé).
    """
    feat_vals = {f: centroid[i] for i, f in enumerate(features)}

    # Trouver les 3 features les plus fortes
    sorted_feats = sorted(feat_vals.items(), key=lambda x: x[1], reverse=True)
    top3 = [f[0] for f in sorted_feats[:3]]

    # Logique de nommage basée sur les features dominantes
    if "xG_P90" in top3 and "Dribbles_P90" in top3:
        return "Ailier Explosif"
    elif "xG_P90" in top3 and "xG_P90" == top3[0]:
        return "Finisseur de Surface"
    elif "Key_Passes_P90" in top3 and "xA_P90" in top3:
        return "Playmaker Créatif"
    elif "Defensive_Actions_P90" in top3 and "distanceRun" in top3:
        return "Sentinelle Box-to-Box"
    elif "Defensive_Actions_P90" in top3 and "Possession_Security" in top3:
        return "Récupérateur Défensif"
    elif "distanceRun" in top3 and "Dribbles_P90" in top3:
        return "Latéral Offensif"
    elif "Pass_Accuracy" in top3 and "Possession_Security" in top3:
        return "Métronome Technique"
    elif "Rating_MA10" in top3 and "Minutes_Played" in top3:
        return "Titulaire Indiscutable"
    elif "Fatigue_Index" in top3 or "Medical_Risk_Score" in top3:
        return "Profil à Risque"
    elif "Age" in top3 and feat_vals.get("Age", 0) > 0.5:
        return "Vétéran Expérimenté"
    elif "Rating_Trend" in top3 and feat_vals.get("Rating_Trend", 0) > 0:
        return "Talent en Progression"
    elif "distanceRun" in top3:
        return "Coureur Infatigable"
    else:
        return "Profil Polyvalent"


def train_clusters() -> None:
    print("\n" + "═" * 70)
    print("🔬  CLUSTERING DES ARCHÉTYPES DE JOUEURS  (VERSION PRO)")
    print("═" * 70)

    # ── 1. Chargement ────────────────────────────────────────────────────
    profiles = joblib.load(BENCHMARK_DIR / "scouting_profiles.joblib")
    features = joblib.load(BENCHMARK_DIR / "scouting_features.joblib")
    print(f"\n  📂 {len(profiles)} profils chargés | {len(features)} features")

    X = profiles[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 2. Détection du K optimal (Elbow + Silhouette) ───────────────────
    print(f"\n  🔍 Recherche du nombre optimal de clusters (K={K_MIN}..{K_MAX})...")
    t0 = time.time()

    inertias = []
    silhouettes = []
    k_range = range(K_MIN, K_MAX + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"     K={k:2d} → Inertie={km.inertia_:10.1f}  Silhouette={sil:.4f}")

    # Meilleur K = celui qui maximise le silhouette score
    best_k_idx = np.argmax(silhouettes)
    best_k = list(k_range)[best_k_idx]
    best_sil = silhouettes[best_k_idx]

    elapsed_search = time.time() - t0
    print(f"\n  ✅ K optimal détecté : {best_k} (Silhouette = {best_sil:.4f})")
    print(f"     Recherche terminée en {elapsed_search:.1f}s")

    # ── 3. Entraînement final K-Means ────────────────────────────────────
    print(f"\n  ⚙️  Entraînement K-Means final (K={best_k})...")
    t0 = time.time()
    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=20, max_iter=500)
    cluster_labels = final_km.fit_predict(X_scaled)
    elapsed_fit = time.time() - t0

    # ── 4. Labellisation automatique ─────────────────────────────────────
    print(f"\n  🏷️  Labellisation automatique des {best_k} archétypes :")
    cluster_names = {}
    cluster_stats = {}

    for c in range(best_k):
        centroid = final_km.cluster_centers_[c]
        label = _auto_label_cluster(centroid, features)
        n_players = (cluster_labels == c).sum()
        cluster_names[str(c)] = label
        cluster_stats[str(c)] = {
            "label": label,
            "n_players": int(n_players),
            "pct": round(100 * n_players / len(profiles), 1),
        }

        # Postes dominants dans ce cluster
        if "Poste_Cat" in profiles.columns:
            postes_in = profiles.loc[cluster_labels == c, "Poste_Cat"].value_counts()
            top_postes = postes_in.head(3).to_dict()
            cluster_stats[str(c)]["dominant_postes"] = top_postes

        print(f"     Cluster {c} : {label:30s} ({n_players:3d} joueurs, {100*n_players/len(profiles):.1f}%)")

    # ── 5. Assigner les clusters aux profils ─────────────────────────────
    profiles = profiles.copy()
    profiles["Cluster_ID"] = cluster_labels
    profiles["Archetype"] = [cluster_names[str(c)] for c in cluster_labels]

    # ── 6. Sauvegarde ────────────────────────────────────────────────────
    joblib.dump(final_km, BENCHMARK_DIR / "clusters_model.joblib")
    joblib.dump(scaler, BENCHMARK_DIR / "clusters_scaler.joblib")
    joblib.dump(profiles, BENCHMARK_DIR / "scouting_profiles.joblib")  # Mis à jour

    report = {
        "best_k": best_k,
        "best_silhouette": round(best_sil, 4),
        "search_time_s": round(elapsed_search, 2),
        "fit_time_s": round(elapsed_fit, 2),
        "k_search": {
            str(k): {"inertia": round(i, 2), "silhouette": round(s, 4)}
            for k, i, s in zip(k_range, inertias, silhouettes)
        },
        "clusters": cluster_stats,
    }

    with open(BENCHMARK_DIR / "cluster_labels.json", "w") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"\n{'═' * 70}")
    print(f"✅ {best_k} ARCHÉTYPES DÉCOUVERTS (Silhouette = {best_sil:.4f})")
    print(f"   Temps total : {elapsed_search + elapsed_fit:.1f}s")
    print(f"\n  💾 Artefacts sauvegardés :")
    print(f"     • clusters_model.joblib")
    print(f"     • clusters_scaler.joblib")
    print(f"     • cluster_labels.json")
    print(f"     • scouting_profiles.joblib (mis à jour avec Cluster_ID + Archetype)")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    train_clusters()
