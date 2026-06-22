import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from sklearn.preprocessing import StandardScaler

# ─── Chemins ────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH     = ROOT / "data" / "processed" / "features_dataset.csv"
BENCHMARK_DIR = ROOT / "LM" / "models" / "benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

IDENTITY_COLS = [
    'Nom', 'Player_ID', 'Match_Date', 'Event_ID', 'Home_Team', 'Away_Team',
    'Score_Home', 'Score_Away', 'Tournament', 'Equipe', 'Position', 'Poste_Cat',
    'Risk_Category', 'Target_Injury_Occurred', 'Current_Injury',
    'Fatigue_Realisee', 'Fatigue_Reelle_Match_T'
]

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _col_nom(df: pd.DataFrame) -> str:
    return 'Nom' if 'Nom' in df.columns else 'Player_Name'


def _analyse_target(y: pd.Series) -> None:
    """Affiche les statistiques de la variable cible et détecte les outliers."""
    q1, q3   = y.quantile(0.25), y.quantile(0.75)
    iqr      = q3 - q1
    low_cut  = q1 - 1.5 * iqr
    high_cut = q3 + 1.5 * iqr
    outliers = y[(y < low_cut) | (y > high_cut)]

    print(f"\n  📊 Distribution de Target_Fatigue :")
    print(f"     Min={y.min():.2f}  Max={y.max():.2f}  "
          f"Moyenne={y.mean():.2f}  Médiane={y.median():.2f}  Std={y.std():.2f}")
    print(f"     Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}")
    if len(outliers) > 0:
        print(f"  ⚠️  {len(outliers)} outliers détectés "
              f"(< {low_cut:.1f} ou > {high_cut:.1f}) "
              f"— soit {100*len(outliers)/len(y):.1f}% des données")
        print(f"     → Ces valeurs ne sont PAS supprimées mais signalées.")
    else:
        print(f"  ✅ Aucun outlier extrême détecté.")


def setup_benchmark_data() -> None:
    print("\n" + "═"*70)
    print("🛠️  PRÉPARATION DES DONNÉES DU BENCHMARK IA  (VERSION PRO)")
    print("═"*70)

    # ── 1. Chargement ────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"\n  📂 Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    col_nom = _col_nom(df)

    # ── 2. Préparation des features ──────────────────────────────────────────
    X = df.drop(columns=IDENTITY_COLS + ['Target_Fatigue'], errors='ignore')
    y = df['Target_Fatigue']

    # Conserver uniquement les colonnes numériques
    X = X.select_dtypes(include='number')

    # Rapport sur les valeurs manquantes AVANT imputation
    missing = X.isnull().sum()
    cols_with_nan = missing[missing > 0]
    if len(cols_with_nan) > 0:
        print(f"\n  ⚠️  {len(cols_with_nan)} colonnes avec NaN (imputées à 0) :")
        for col, n in cols_with_nan.items():
            print(f"     • {col}: {n} NaN ({100*n/len(X):.1f}%)")
    X = X.fillna(0)

    # Analyse de la target
    _analyse_target(y)

    # ── 3. Split par joueur (80 / 20 — stratifié sur joueur) ─────────────────
    joueurs = np.array(df[col_nom].dropna().unique(), dtype=str)

    rng = np.random.RandomState(42)
    rng.shuffle(joueurs)
    split_idx      = int(len(joueurs) * 0.8)
    joueurs_train  = joueurs[:split_idx]
    joueurs_test   = joueurs[split_idx:]

    mask_train = df[col_nom].isin(set(joueurs_train))
    mask_test  = df[col_nom].isin(set(joueurs_test))

    print(f"\n  👥 Split joueurs  : {len(joueurs_train)} train / {len(joueurs_test)} test")
    print(f"  📋 Split lignes   : {mask_train.sum():,} train / {mask_test.sum():,} test "
          f"({100*mask_test.sum()/len(df):.1f}%)")

    # ── 4. Vérification de l'absence de fuite de données ────────────────────
    overlap = set(joueurs_train) & set(joueurs_test)
    assert len(overlap) == 0, f"FUITE DE DONNÉES : {len(overlap)} joueurs dans train ET test !"
    print(f"  ✅ Zéro fuite de données confirmée.")

    # ── 5. Scaler global (fitté uniquement sur train) ────────────────────────
    X_train = X[mask_train]
    scaler  = StandardScaler()
    scaler.fit(X_train)

    # ── 6. Sauvegarde ────────────────────────────────────────────────────────
    joblib.dump(list(joueurs_train), BENCHMARK_DIR / "train_players.joblib")
    joblib.dump(list(joueurs_test),  BENCHMARK_DIR / "test_players.joblib")
    joblib.dump(X.columns.tolist(),  BENCHMARK_DIR / "model_columns.joblib")
    joblib.dump(scaler,              BENCHMARK_DIR / "benchmark_scaler.joblib")

    # Sauvegarder des métadonnées utiles
    meta = {
        "n_features"    : int(X.shape[1]),
        "n_train_rows"  : int(mask_train.sum()),
        "n_test_rows"   : int(mask_test.sum()),
        "n_train_players": int(len(joueurs_train)),
        "n_test_players" : int(len(joueurs_test)),
        "target_mean"   : float(y.mean()),
        "target_std"    : float(y.std()),
        "target_min"    : float(y.min()),
        "target_max"    : float(y.max()),
    }
    with open(BENCHMARK_DIR / "data_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\n  💾 Tous les artefacts sauvegardés dans : {BENCHMARK_DIR}")
    print(f"     • train_players.joblib  • test_players.joblib")
    print(f"     • model_columns.joblib  • benchmark_scaler.joblib")
    print(f"     • data_meta.json")
    print("\n✅ Préparation terminée. Les 3 modèles peuvent maintenant s'entraîner.")
    print("═"*70 + "\n")


if __name__ == "__main__":
    setup_benchmark_data()
