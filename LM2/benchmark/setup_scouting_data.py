"""
AthlytIQ — Préparation des Données de Scouting (VERSION PRO)
==============================================================
Équivalent de LM/models/benchmark/setup_data.py pour le module de similarité.

Responsabilités :
  1. Charger features_dataset.csv
  2. Agréger les stats par joueur (moyenne exponentielle sur 15 derniers matchs)
  3. Calculer les percentiles par feature
  4. Split validation : 80% joueurs pour entraînement des poids, 20% pour test
  5. Sauvegarder les artefacts (profils, scaler, méta)

Usage :
    .venv/bin/python LM2/benchmark/setup_scouting_data.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─── Chemins ────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
DATA_PATH     = ROOT / "data" / "processed" / "features_dataset.csv"
BENCHMARK_DIR = ROOT / "LM2" / "benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# ─── 15 Features Orthogonales pour le Scouting ──────────────────────────────
SCOUTING_FEATURES = [
    "Rating_MA10",
    "xG_P90",
    "xA_P90",
    "Pass_Accuracy",
    "Defensive_Actions_P90",
    "distanceRun",
    "Possession_Security",
    "Dribbles_P90",
    "Key_Passes_P90",
    "Fatigue_Index",
    "Medical_Risk_Score",
    "Age",
    "Rating_Trend",
    "Rating_STD5",
    "Minutes_Played",
]


def _build_ewm_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le profil agrégé de chaque joueur via moyenne exponentielle
    sur ses 15 derniers matchs (les matchs récents pèsent plus).
    """
    df = df.sort_values("Match_Date")

    # Sécuriser les features
    for f in SCOUTING_FEATURES:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0)
        else:
            df[f] = 0.0

    profils = []
    for nom, group in df.groupby("Nom"):
        last_n = group.tail(15)
        if len(last_n) >= 2:
            profil = last_n[SCOUTING_FEATURES].ewm(span=5, min_periods=1).mean().iloc[-1]
        else:
            profil = last_n[SCOUTING_FEATURES].iloc[-1]

        # Métadonnées du dernier match
        meta = last_n.iloc[-1]
        profil["Nom"] = nom
        profil["nb_matchs"] = len(group)
        for col in ["Team", "Equipe", "Poste_Cat", "League", "Age"]:
            if col in meta.index:
                profil[col] = meta[col]

        profils.append(profil)

    return pd.DataFrame(profils)


def setup_scouting_data() -> None:
    print("\n" + "═" * 70)
    print("🛠️  PRÉPARATION DES DONNÉES DE SCOUTING  (VERSION PRO)")
    print("═" * 70)

    # ── 1. Chargement ────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["Match_Date"] = pd.to_datetime(df["Match_Date"], errors="coerce")
    print(f"\n  📂 Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    # ── 2. Construction des profils agrégés ──────────────────────────────
    print("  ⚙️  Construction des profils joueurs (EWM 15 matchs)...")
    profiles = _build_ewm_profiles(df)
    n_joueurs = len(profiles)
    print(f"  📊 {n_joueurs} profils joueurs construits")

    # ── 3. Vérification Poste_Cat ────────────────────────────────────────
    if "Poste_Cat" in profiles.columns:
        dist = profiles["Poste_Cat"].value_counts()
        print(f"\n  📌 Distribution des postes :")
        for pos, count in dist.items():
            print(f"     {pos:4s} : {count:3d} joueurs ({100*count/n_joueurs:.1f}%)")
    else:
        print("  ⚠️  Colonne Poste_Cat absente — classification par défaut MC")
        profiles["Poste_Cat"] = "MC"

    # ── 4. Normalisation ─────────────────────────────────────────────────
    X = profiles[SCOUTING_FEATURES].copy()
    missing = X.isnull().sum()
    cols_nan = missing[missing > 0]
    if len(cols_nan) > 0:
        print(f"\n  ⚠️  {len(cols_nan)} features avec NaN (imputées à 0)")
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 5. Split par joueur (80/20) ──────────────────────────────────────
    joueurs = profiles["Nom"].values.astype(str)
    rng = np.random.RandomState(42)
    indices = np.arange(len(joueurs))
    rng.shuffle(indices)
    split_idx = int(len(joueurs) * 0.8)

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    print(f"\n  👥 Split joueurs : {len(train_idx)} train / {len(test_idx)} test")

    # ── 6. Calcul des percentiles (rang par feature) ─────────────────────
    percentiles = pd.DataFrame(index=profiles.index)
    for f in SCOUTING_FEATURES:
        if X[f].std() > 0:
            percentiles[f] = X[f].rank(pct=True)
        else:
            percentiles[f] = 0.5

    # ── 7. Sauvegarde des artefacts ──────────────────────────────────────
    joblib.dump(profiles, BENCHMARK_DIR / "scouting_profiles.joblib")
    joblib.dump(X_scaled, BENCHMARK_DIR / "scouting_X_scaled.joblib")
    joblib.dump(scaler, BENCHMARK_DIR / "scouting_scaler.joblib")
    joblib.dump(percentiles.values, BENCHMARK_DIR / "scouting_percentiles.joblib")
    joblib.dump(SCOUTING_FEATURES, BENCHMARK_DIR / "scouting_features.joblib")
    joblib.dump(list(train_idx), BENCHMARK_DIR / "scouting_train_idx.joblib")
    joblib.dump(list(test_idx), BENCHMARK_DIR / "scouting_test_idx.joblib")

    meta = {
        "n_joueurs": int(n_joueurs),
        "n_features": len(SCOUTING_FEATURES),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "features": SCOUTING_FEATURES,
        "postes": profiles["Poste_Cat"].value_counts().to_dict()
        if "Poste_Cat" in profiles.columns
        else {},
    }
    with open(BENCHMARK_DIR / "scouting_meta.json", "w") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"\n  💾 Artefacts sauvegardés dans : {BENCHMARK_DIR}")
    print(f"     • scouting_profiles.joblib ({n_joueurs} profils)")
    print(f"     • scouting_scaler.joblib")
    print(f"     • scouting_percentiles.joblib")
    print(f"     • scouting_meta.json")
    print("\n✅ Préparation scouting terminée.")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    setup_scouting_data()
