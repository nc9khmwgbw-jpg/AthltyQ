import sys
from pathlib import Path
ROOT = Path('/Users/fahamayoub/Desktop/AthlytIQ/DASHBOARD').parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from DASHBOARD.backend import get_data

results, features = get_data()
features['Match_Date'] = pd.to_datetime(features['Match_Date'])
all_dates = sorted(features['Match_Date'].dropna().unique())
last_14_dates = all_dates[-14:] if len(all_dates) >= 14 else all_dates
date_to_j = {d: f"J-{len(last_14_dates) - i}" for i, d in enumerate(last_14_dates)}
recent_features = features[features['Match_Date'].isin(last_14_dates)]

balde = recent_features[recent_features['Nom'].str.contains('Balde', na=False)]
print("Balde matches in last 14 dates:")
for _, row in balde.iterrows():
    m_date = row['Match_Date']
    mins = float(row.get('Minutes_Played', 0))
    rating = float(row.get('Rating', 5.0))
    intensity_raw = (mins / 90.0) * (rating / 10.0) * 100 * 1.2
    print(f"Date: {m_date} ({date_to_j[m_date]}), Mins: {mins}, Rating: {rating}, Raw Intensity: {intensity_raw}")

