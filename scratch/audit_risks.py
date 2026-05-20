import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/Users/fahamayoub/Desktop/AthlytIQ")
CSV_PATH = ROOT / "data/processed/features_dataset.csv"

df = pd.read_csv(CSV_PATH)
df['Match_Date'] = pd.to_datetime(df['Match_Date'], errors='coerce')
df = df.sort_values('Match_Date', ascending=False).drop_duplicates('Nom')

# 1. Fatigue IA (On simule avec Fatigue_Lag1 car on n'a pas le modele ici)
df['Fatigue_IA'] = df['Fatigue_Lag1'].fillna(50.0)

# 2. Trauma
days_since = pd.to_numeric(df['Jours_Depuis_Blessure'], errors='coerce').fillna(365)
injury_recency_score = 100 * np.exp(-days_since.values / 30.0)

trauma_cols = ['Nb_Blessures_Musculaires_12m', 'Trauma_Index']
present_trauma_cols = [c for c in trauma_cols if c in df.columns]
trauma_vals = df[present_trauma_cols].sum(axis=1).fillna(0).values
trauma_score = (trauma_vals * 20) + injury_recency_score

# 3. ACWR
acwr_val = pd.to_numeric(df['ACWR'], errors='coerce').fillna(1.0).values
acwr_stress = np.clip((np.abs(acwr_val - 1.0) * 50.0), 0, 30)

# 4. Final Risk
fatigue_part = df['Fatigue_IA'].values * 0.50
trauma_part  = trauma_score * 0.30
acwr_part    = acwr_stress * 0.20

medical_risk = (fatigue_part + trauma_part + acwr_part) / 100.0
df['Calculated_Risk'] = np.clip(medical_risk, 0, 1)

high_risk_players = df[df['Calculated_Risk'] >= 0.99][['Nom', 'Calculated_Risk', 'Fatigue_IA', 'Jours_Depuis_Blessure', 'Nb_Blessures_Musculaires_12m']]
print(f"Joueurs à 100% de risque ({len(high_risk_players)} joueurs) :")
print(high_risk_players.head(20))

gerard = df[df['Nom'].str.contains("Gerard Martín", na=False)][['Nom', 'Calculated_Risk', 'Fatigue_IA', 'Jours_Depuis_Blessure']]
print("\nStats pour Gerard Martín :")
print(gerard)
