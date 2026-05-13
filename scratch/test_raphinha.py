
import pandas as pd
import joblib
from pathlib import Path
import sys
import os

# Ajouter le chemin pour importer nos modules
sys.path.append(os.getcwd())

from LM.models.fatigue_predictor import FatiguePredictor

def test_joueur(nom_joueur):
    print(f"\n🧠 DIAGNOSTIC IA — {nom_joueur.upper()}")
    print("="*50)
    
    # 1. Charger le dataset traité
    df = pd.read_csv("data/processed/features_dataset.csv")
    
    # 2. Filtrer pour le joueur
    # On gère les noms qui pourraient varier (ex: Raphinha)
    player_data = df[df['Nom'].str.contains(nom_joueur, case=False, na=False)].sort_values('Match_Date')
    
    if player_data.empty:
        print(f"❌ Joueur '{nom_joueur}' non trouvé dans le dataset.")
        return

    # Prendre le match le plus récent
    dernier_match = player_data.iloc[-1:]
    date_match = dernier_match['Match_Date'].values[0]
    
    print(f"📅 Dernier match analysé : {date_match}")
    print(f"📊 Rating MA15 : {dernier_match['Rating_MA15'].values[0]:.2f}")
    print(f"⚡ ACWR Actuel : {dernier_match['ACWR'].values[0]:.2f}")
    print(f"⏳ Repos : {dernier_match['Days_Rest'].values[0]} jours")
    
    # 3. Utiliser le Predictor
    predictor = FatiguePredictor()
    
    print("\n📈 ÉVOLUTION DE LA PRÉDICTION (5 derniers matchs) :")
    print("-" * 65)
    print(f"{'Date':<12} | {'Note':<6} | {'Repos':<6} | {'ACWR':<6} | {'Fatigue IA (n+1)':<12}")
    print("-" * 65)

    # Analyser les 5 derniers matchs
    derniers_matchs = player_data.tail(5)
    
    for i, row in derniers_matchs.iterrows():
        # Prédiction pour le match n+1 à partir de ce match
        pred = predictor.predict(pd.DataFrame([row]))[0]
        
        # Formatage couleur (visuel console)
        status = "🟢"
        if pred > 75: status = "🔴"
        elif pred > 45: status = "🟠"
        
        print(f"{row['Match_Date'].split('T')[0]:<12} | {row['Rating']:<6.1f} | {row['Days_Rest']:<6.1f} | {row['ACWR']:<6.2f} | {status} {pred:>5.1f}%")

    # Récupérer la toute dernière prédiction pour la chaîne d'accumulation
    last_pred = predictor.predict(pd.DataFrame([derniers_matchs.iloc[-1]]))[0]

    print("-" * 65)

    # --- SECTION : PRÉVISIONS (SIMULATION DU FUTUR) ---
    print("\n🔮 PRÉVISIONS (Prochains Matchs - Simulation) :")
    print("Scénario : 90 min / Note 7.0 / Charge constante")
    print("-" * 65)
    
    # Simuler le match dans 4 jours (J+4)
    future_match_1 = derniers_matchs.iloc[-1].copy()
    future_match_1['Match_Date'] = "2026-03-26" # Simulation
    future_match_1['Days_Rest'] = 4.0
    future_match_1['Rating'] = 7.0
    future_match_1['Fatigue_Lag1'] = last_pred 
    
    pred_future_1 = predictor.predict(pd.DataFrame([future_match_1]))[0]
    
    # Simuler le match suivant dans 3 jours (J+7)
    future_match_2 = future_match_1.copy()
    future_match_2['Match_Date'] = "2026-03-29"
    future_match_2['Days_Rest'] = 3.0
    future_match_2['Rating'] = 7.0
    future_match_2['Fatigue_Lag1'] = pred_future_1
    
    pred_future_2 = predictor.predict(pd.DataFrame([future_match_2]))[0]

    def get_status(p):
        if p > 75: return "🔴"
        if p > 45: return "🟠"
        return "🟢"

    print(f"Match J+4 (Prochain)  : {get_status(pred_future_1)} {pred_future_1:>5.1f}%")
    print(f"Match J+7 (Suivant)   : {get_status(pred_future_2)} {pred_future_2:>5.1f}%")
    print("-" * 65)
    print("💡 Note : La fatigue J+7 augmente car le repos est plus court (3 jours).")
    print("═" * 65 + "\n")

if __name__ == "__main__":
    test_joueur("Raphinha")
