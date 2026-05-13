import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Ajouter la racine du projet au path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from LM.models.fatigue_predictor import FatiguePredictor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def examen_final_20_pourcent():
    print("\n" + "═"*70)
    print("🏁 EXAMEN FINAL DES 20% — AthlytIQ PERFORMANCE")
    print("═"*70)

    # 1. Chargement des données
    data_path = Path("data/processed/features_dataset.csv")
    if not data_path.exists():
        print("❌ Dataset introuvable.")
        return

    df = pd.read_csv(data_path)
    
    # On prend un échantillon aléatoire de 20% (Test Set)
    test_df = df.sample(frac=0.2, random_state=42)
    print(f"📊 Test sur {len(test_df)} matchs inconnus...")

    # 2. Inférence avec le Predictor
    predictor = FatiguePredictor()
    
    # La cible réelle
    y_true = test_df['Target_Fatigue'].values
    
    # La prédiction de l'IA
    print("🧠 L'IA analyse les profils...")
    y_pred = predictor.predict(test_df)

    # 3. Calcul des statistiques de précision
    mae = mean_absolute_error(y_true, y_pred)
    precision = 100 - mae
    
    # Calcul des alertes bien détectées (Zone Rouge > 70%)
    real_alerts = (y_true > 70)
    pred_alerts = (y_pred > 70)
    detection_rate = (np.sum(real_alerts & pred_alerts) / np.sum(real_alerts)) * 100

    print("\n" + "📈 RAPPORT D'EXAMEN :")
    print(f"   ✅ Précision Moyenne : {precision:.2f}%")
    print(f"   ✅ Marge d'erreur : {mae:.2f} points")
    print(f"   🚨 Taux de détection des Alertes Rouges : {detection_rate:.1f}%")
    print(f"   🎯 Fiabilité (R²) : {r2_score(y_true, y_pred):.2f}")

    # 4. Exemple de top détections
    print("\n" + "🔍 EXEMPLES DE PRÉDICTIONS RÉUSSIES :")
    test_df['Prediction'] = y_pred
    success = test_df[['Nom', 'Match_Date', 'Target_Fatigue', 'Prediction']].tail(5)
    print(success.to_string(index=False))

    print("\n" + "═"*70)

if __name__ == "__main__":
    examen_final_20_pourcent()
