import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/Users/fahamayoub/Desktop/AthlytIQ")
CSV_PATH = ROOT / "data/processed/features_dataset.csv"

df = pd.read_csv(CSV_PATH)
player_df = df[df['Nom'] == 'Gerard Martín'].sort_values('Match_Date')

print(f"Analyse des Risques pour Gerard Martín :")
print("-" * 50)
for _, row in player_df.tail(5).iterrows():
    # Simulation du calcul actuel dans backend.py
    days_since = float(row['Jours_Depuis_Blessure']) if not pd.isna(row['Jours_Depuis_Blessure']) else 365
    recency = 100 * np.exp(-days_since / 30.0)
    
    # Trauma index (valeur brute dans le CSV)
    trauma_val = float(row['Trauma_Index']) if 'Trauma_Index' in row else 0
    nb_blessures = float(row['Nb_Blessures_Musculaires_12m']) if 'Nb_Blessures_Musculaires_12m' in row else 0
    
    total_trauma_score = ((trauma_val + nb_blessures) * 20) + recency
    
    # Fatigue (simulée ou réelle si présente)
    fatigue = 50.0 # Moyenne par défaut si non prédit
    
    risk = (fatigue * 0.50 + total_trauma_score * 0.30) / 100.0
    risk = min(1.0, risk)
    
    print(f"Date: {row['Match_Date']}")
    print(f"  - Jours depuis blessure: {days_since} (Score Récence: {recency:.2f})")
    print(f"  - Trauma Index CSV: {trauma_val}")
    print(f"  - Nb Blessures 12m: {nb_blessures}")
    print(f"  - Score Trauma Total: {total_trauma_score:.2f}")
    print(f"  - Risque Médical Final: {risk*100:.1f}%")
    print("-" * 30)
