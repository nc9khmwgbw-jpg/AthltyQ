import sys
from pathlib import Path
import pandas as pd
ROOT = Path('/Users/fahamayoub/Desktop/AthlytIQ/DASHBOARD').parent
sys.path.insert(0, str(ROOT))
from DASHBOARD.backend import get_data

results, features = get_data()
features['Match_Date'] = pd.to_datetime(features['Match_Date'])

yamal = features[features['Nom'] == 'Lamine Yamal']
print("Total rows for Lamine Yamal in features:", len(yamal))
print("Unique match dates for Lamine Yamal:", sorted(yamal['Match_Date'].dropna().unique()))
