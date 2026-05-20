
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ROOT = Path("/Users/fahamayoub/Desktop/AthlytIQ")
sys.path.insert(0, str(ROOT))

from LM.models.fatigue_predictor import FatiguePredictor

predictor = FatiguePredictor()
df = pd.read_csv(ROOT / "data" / "processed" / "features_dataset.csv")
df['Match_Date'] = pd.to_datetime(df['Match_Date'], errors='coerce')

# Filter for Gerard Martín
player_df = df[df['Nom'] == "Gerard Martín"].sort_values('Match_Date', ascending=False).head(1)

if player_df.empty:
    print("Gerard Martín not found")
    sys.exit(1)

pred = predictor.predict(player_df)
print(f"Prediction for {player_df.iloc[0]['Nom']}: {pred}")

# Calculate trauma
trauma_cols = ['Nb_Blessures_Musculaires_12m', 'Trauma_Index']
present_trauma_cols = [c for c in trauma_cols if c in player_df.columns]
days_since = pd.to_numeric(player_df['Jours_Depuis_Blessure'], errors='coerce').fillna(365).values[0]
injury_recency_score = 100 * np.exp(-days_since / 30.0)

if present_trauma_cols:
    trauma_sum = player_df[present_trauma_cols].sum(axis=1).values[0]
    trauma_score = (trauma_sum * 20) + injury_recency_score
else:
    trauma_score = injury_recency_score

# ACWR
acwr = pd.to_numeric(player_df['ACWR'], errors='coerce').fillna(1.0).values[0]
acwr_stress = np.clip((np.abs(acwr - 1.0) * 50.0), 0, 30)

fatigue = pred[0] if pred is not None else 0

fatigue_part = fatigue * 0.50
trauma_part = trauma_score * 0.30
acwr_part = acwr_stress * 0.20

total_score = (fatigue_part + trauma_part + acwr_part)
medical_risk = total_score / 100.0

print(f"Fatigue Part (50%): {fatigue_part}")
print(f"Trauma Part (30%): {trauma_part} (Trauma Score: {trauma_score}, Days Since: {days_since})")
print(f"ACWR Part (20%): {acwr_part} (ACWR: {acwr})")
print(f"Total Score (0-100): {total_score}")
print(f"Medical Risk (0-1): {medical_risk}")
