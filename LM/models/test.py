"""
AthlytIQ — Script de Test Strict
==================================
Évalue le modèle sur les 20% de joueurs mis de côté lors de l'entraînement.

RÈGLE ABSOLUE :
  - train.py  →  entraîne sur 80% des joueurs  →  sauvegarde train_players.joblib
  - test.py   →  teste  sur 20% des joueurs     →  lit test_players.joblib
  ⚠️  Aucun joueur commun entre les deux sets (0% de fuite garanti)
"""

import warnings
from typing import Set, List, Optional, Any
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ─── Chemins (identiques à train.py) ──────────────────────────────────────────
ROOT            = Path(__file__).resolve().parents[2]
DATA_PATH       = ROOT / "data" / "processed" / "features_dataset.csv"
SAVE_DIR        = ROOT / "LM" / "models" / "saved"
MODEL_PATH      = SAVE_DIR / "fatigue_model.joblib"
SCALER_PATH     = SAVE_DIR / "fatigue_scaler.joblib"
COLS_PATH       = SAVE_DIR / "model_columns.joblib"
TRAIN_PLAYERS_PATH = SAVE_DIR / "train_players.joblib"
TEST_PLAYERS_PATH  = SAVE_DIR / "test_players.joblib"

IDENTITY_COLS = [
    'Nom', 'Player_ID', 'Match_Date', 'Event_ID', 'Home_Team', 'Away_Team',
    'Score_Home', 'Score_Away', 'Tournament', 'Equipe', 'Position', 'Poste_Cat',
    'Risk_Category', 'Target_Injury_Occurred', 'Current_Injury',
    'Fatigue_Realisee', 'Fatigue_Reelle_Match_T'
]

# ──────────────────────────────────────────────────────────────────────────────

def tester_modele(joueur: Optional[str] = None) -> None:
    """
    Paramètre optionnel `joueur` :
      - None  → teste sur TOUS les 20% de joueurs de test (vue globale)
      - "Nom" → teste sur CE joueur uniquement (doit être dans le set de test)
    """
    print("\n" + "═"*70)
    print("🔬 AthlytIQ — TEST STRICT SUR JOUEURS JAMAIS VUS")
    print("═"*70)

    # 1. Vérification des fichiers requis
    for p, label in [(MODEL_PATH, "Modèle"), (SCALER_PATH, "Scaler"),
                     (COLS_PATH, "Colonnes"), (TRAIN_PLAYERS_PATH, "Liste train"),
                     (TEST_PLAYERS_PATH, "Liste test"), (DATA_PATH, "Dataset")]:
        if not p.exists():
            print(f"❌ {label} introuvable : {p}")
            print("   → Lance d'abord : .venv/bin/python LM/models/train.py")
            return

    # 2. Chargement du modèle et des listes
    model: Any = joblib.load(MODEL_PATH)
    scaler: Any = joblib.load(SCALER_PATH)
    model_cols: List[str] = joblib.load(COLS_PATH)  # type: ignore
    train_players: Set[str] = set(joblib.load(TRAIN_PLAYERS_PATH)) # type: ignore
    test_players: Set[str] = set(joblib.load(TEST_PLAYERS_PATH))   # type: ignore

    print(f"\n✅ Modèle chargé.")
    print(f"   Train : {len(train_players)} joueurs  |  Test : {len(test_players)} joueurs")

    # Vérification stricte 0% fuite
    fuite = train_players & test_players
    if fuite:
        print(f"❌ FUITE DÉTECTÉE : {len(fuite)} joueurs communs ! Relance train.py.")
        return
    print(f"   Fuite de données : 0 joueurs communs ✅")

    # 3. Chargement du dataset
    df = pd.read_csv(DATA_PATH)
    col_nom = 'Nom' if 'Nom' in df.columns else 'Player_Name'

    # 4. Sélection du joueur ou de tous les joueurs de test
    df_test: pd.DataFrame
    if joueur:
        # Recherche partielle insensible à la casse
        unique_names = df[col_nom].dropna().unique()
        noms_trouves = [str(n) for n in unique_names if joueur.lower() in str(n).lower()]
        if not noms_trouves:
            print(f"\n❌ Joueur '{joueur}' introuvable dans le dataset.")
            return

        vrai_nom = noms_trouves[0]

        if vrai_nom in train_players:
            print(f"\n🚫 REFUSÉ : '{vrai_nom}' fait partie des joueurs D'ENTRAÎNEMENT.")
            print(f"   → Le modèle a déjà vu ce joueur — test non fiable.")
            print(f"   → Choisis parmi les {len(test_players)} joueurs de test.")
            return

        if vrai_nom not in test_players:
            print(f"\n⚠️  '{vrai_nom}' n'est ni dans le train ni dans le test.")
            print(f"   → Ce joueur n'existait pas lors du dernier entraînement.")
            return

        print(f"\n✅ '{vrai_nom}' est dans le set de TEST ✅")
        df_test = df[df[col_nom] == vrai_nom].copy()

    else:
        # Tous les joueurs du set de test
        df_test = df[df[col_nom].isin(test_players)].copy()

    if df_test.empty:
        print("\n❌ Aucune donnée de test disponible.")
        return

    print(f"\n📊 Données de test : {len(df_test)} matchs / {df_test[col_nom].nunique()} joueurs")

    # 5. Préparation des features (même logique que train.py)
    X_test = df_test.drop(columns=IDENTITY_COLS + ['Target_Fatigue'], errors='ignore')
    y_test = df_test['Target_Fatigue']
    X_test = X_test.select_dtypes(include='number').reindex(columns=model_cols, fill_value=0)
    X_test = X_test.fillna(0)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 100)

    # 6. Métriques globales
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    print("\n" + "─"*70)
    print("📈 RÉSULTATS GLOBAUX (Joueurs Jamais Vus) :")
    print(f"   Marge d'erreur (MAE)      : {mae:.2f}%")
    print(f"   Erreur quadratique (RMSE) : {rmse:.2f}%")
    print(f"   Fiabilité (R²)            : {r2:.2f}")
    print("─"*70)

    # 7. Vue par joueur (si mode individuel ou nombre réduit)
    joueurs_test = df_test[col_nom].unique()
    if joueur or len(joueurs_test) <= 20:
        print("\n📋 DÉTAIL PAR JOUEUR :")
        print(f"{'Joueur':<28} | {'Matchs':<7} | {'MAE':<7} | {'RMSE':<7} | Statut")
        print("─"*70)

        resultats = []
        for nom in sorted(list(joueurs_test)):
            mask = df_test[col_nom] == nom
            y_j = y_test[mask]
            # y_pred est un array numpy, on utilise les indices entiers du masque
            indices = np.where(mask.values)[0]
            y_p = y_pred[indices]
            
            if len(y_j) == 0:
                continue
            mae_j = float(mean_absolute_error(y_j, y_p))
            rmse_j = float(np.sqrt(mean_squared_error(y_j, y_p)))
            status = "🟢" if mae_j < 10 else ("🟠" if mae_j < 20 else "🔴")
            resultats.append((nom, len(y_j), mae_j, rmse_j, status))

        for nom, n, mae_j, rmse_j, status in sorted(resultats, key=lambda x: x[2]):
            print(f"{nom:<28} | {n:<7} | {mae_j:<7.2f} | {rmse_j:<7.2f}% | {status}")

        print("─"*70)

    # 8. Distribution des erreurs
    erreurs = np.abs(y_test.values - y_pred)
    print(f"\n📊 Distribution des erreurs :")
    print(f"   < 10%  (Excellente) : {(erreurs < 10).sum():>4} matchs  ({100*(erreurs<10).mean():.1f}%)")
    print(f"   10-20% (Bonne)      : {((erreurs>=10)&(erreurs<20)).sum():>4} matchs  ({100*((erreurs>=10)&(erreurs<20)).mean():.1f}%)")
    print(f"   > 20%  (À améliorer): {(erreurs >= 20).sum():>4} matchs  ({100*(erreurs>=20).mean():.1f}%)")
    print("═"*70 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Mode joueur unique : python test.py "Harvey Barnes"
        nom_arg = " ".join(sys.argv[1:])
        tester_modele(joueur=nom_arg)
    else:
        # Mode global : tous les 20% de joueurs de test
        tester_modele()
