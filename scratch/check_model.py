import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path("/Users/fahamayoub/Desktop/AthlytIQ")
CSV_PATH = ROOT / "data/processed/features_dataset.csv"
MODEL_PATH = ROOT / "LM/models/random_forest_model.joblib"

df = pd.read_csv(CSV_PATH)
player_data = df[df['Nom'] == 'Gerard Martín'].tail(1)

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    # On garde les colonnes attendues par le modele
    # (Ici on simplifie en prenant ce qu'il y a dans player_data)
    try:
        # Tentative de prediction
        # Note: predictor.predict dans backend fait plus de nettoyage
        # Mais voyons si le modele est charge
        print(f"Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur modèle: {e}")
else:
    print("Modèle non trouvé !")

# Verifions aussi si Injury_Risk est deja present dans le CSV
if 'Injury_Risk' in df.columns:
    print(f"Risque déjà présent dans le CSV pour Gerard: {player_data['Injury_Risk'].values[0]}")
