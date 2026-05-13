import pandas as pd
import joblib
from pathlib import Path
import sys
import os
from typing import Set, List, Optional, Any

# Ajout du chemin racine pour les imports
sys.path.append(os.getcwd())

try:
    from LM.models.fatigue_predictor import FatiguePredictor
except ImportError:
    FatiguePredictor = None  # type: ignore

# Chemins
ROOT        = Path(__file__).resolve().parents[1]
DATA_PATH   = ROOT / "data" / "processed" / "features_dataset.csv"
TRAIN_PATH  = ROOT / "LM" / "models" / "saved" / "train_players.joblib"
TEST_PATH   = ROOT / "LM" / "models" / "saved" / "test_players.joblib"


def test_joueur(nom_joueur: str) -> None:
    print(f"\n🧠 DIAGNOSTIC IA — {nom_joueur.upper()}")
    print("="*60)

    # ─── GARDE STRICTE : Vérification Train/Test ─────────────────────────
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        print("❌ Listes train/test introuvables.")
        print("   → Lance d'abord : .venv/bin/python LM/models/train.py")
        return

    train_players: Set[str] = set(joblib.load(TRAIN_PATH))  # type: ignore
    test_players: Set[str] = set(joblib.load(TEST_PATH))    # type: ignore

    if not DATA_PATH.exists():
        print(f"❌ Dataset introuvable : {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    if 'Nom' not in df.columns:
        print("❌ La colonne 'Nom' est absente du dataset.")
        return

    matches = df['Nom'].dropna().unique()
    noms_trouves: List[str] = [str(n) for n in matches if nom_joueur.lower() in str(n).lower()]

    if not noms_trouves:
        print(f"❌ Joueur '{nom_joueur}' non trouvé dans le dataset.")
        return

    vrai_nom = noms_trouves[0]

    # Vérification : est-il dans le train ou dans le test ?
    if vrai_nom in train_players:
        print(f"🚫 REFUSÉ : '{vrai_nom}' fait partie des joueurs D'ENTRAÎNEMENT.")
        print(f"   → Le modèle a déjà vu ce joueur. Ce test ne serait pas fiable.")
        print(f"   → Choisis un joueur parmi les {len(test_players)} joueurs de test.")
        return
    elif vrai_nom in test_players:
        print(f"✅ '{vrai_nom}' est dans le set de TEST (jamais vu par le modèle) ✅")
    else:
        print(f"⚠️  '{vrai_nom}' n'est ni dans le train ni dans le test connus.")
        print(f"   → Ce joueur a peut-être été ajouté après l'entraînement.")
    # ──────────────────────────────────────────────────────────────────────

    player_data = df[df['Nom'] == vrai_nom].sort_values('Match_Date')
    
    if player_data.empty:
        print(f"❌ Aucune donnée pour {vrai_nom}")
        return

    dernier_match = player_data.iloc[-1:]

    print(f"📊 Nombre de matchs en historique : {len(player_data)}")
    
    # Utilisation de .item() ou .values[0] avec précaution pour l'IDE
    match_date = dernier_match['Match_Date'].values[0]
    acwr = dernier_match['ACWR'].values[0]
    days_rest = dernier_match['Days_Rest'].values[0]

    print(f"📅 Dernier match analysé          : {match_date}")
    print(f"⚡ ACWR Actuel                    : {float(acwr):.2f}")
    print(f"⏳ Repos                          : {days_rest} jours")

    if FatiguePredictor is None:
        print("❌ FatiguePredictor non chargé.")
        return

    predictor = FatiguePredictor()

    print("\n📈 ÉVOLUTION DE LA PRÉDICTION (5 derniers matchs) :")
    print("-" * 65)
    print(f"{'Date':<12} | {'Note':<6} | {'Repos':<6} | {'ACWR':<6} | {'Fatigue IA (n+1)':<12}")
    print("-" * 65)

    for _, row in player_data.tail(5).iterrows():
        # On force la conversion en DataFrame pour predictor.predict
        row_df = pd.DataFrame([row])
        pred_arr = predictor.predict(row_df)
        
        if pred_arr is None or len(pred_arr) == 0:
            print(f"⚠️ Erreur de prédiction pour {row.get('Match_Date')}")
            continue
            
        pred = float(pred_arr[0])
        
        status = "🟢"
        if pred > 75:   status = "🔴"
        elif pred > 45: status = "🟠"
        
        display_date = str(row['Match_Date']).split('T')[0]
        rating = float(row.get('Rating', 0))
        rest = float(row.get('Days_Rest', 0))
        acwr_val = float(row.get('ACWR', 0))
        
        print(f"{display_date:<12} | {rating:<6.1f} | {rest:<6.1f} | {acwr_val:<6.2f} | {status} {pred:>5.1f}%")

    print("-" * 65)
    print("💡 Interprétation : La fatigue prédite est celle attendue pour le match SUIVANT.")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    # Si un nom est passé en argument : test_ia.py "Nom Joueur"
    if len(sys.argv) > 1:
        nom_arg = " ".join(sys.argv[1:])
        test_joueur(nom_arg)
    else:
        # Afficher les joueurs disponibles dans le set de TEST
        if TEST_PATH.exists():
            test_players_list = joblib.load(TEST_PATH)
            if isinstance(test_players_list, (list, set)):
                print(f"\n📋 JOUEURS DISPONIBLES POUR LE TEST ({len(test_players_list)} au total) :")
                print("─" * 40)
                sorted_players = sorted(list(test_players_list))
                for p in sorted_players[:20]:
                    print(f"  • {p}")
                print(f"  ... (et {max(0, len(sorted_players)-20)} autres)")
                print("\n💡 Usage : .venv/bin/python scratch/test_ia.py \"Harvey Barnes\"")
                print("─" * 40)

                # Test sur deux joueurs de test connus
                test_joueur("Harvey Barnes")
                test_joueur("Mason Mount")
        else:
            print("❌ Fichier de test introuvable. Lance d'abord l'entraînement.")
