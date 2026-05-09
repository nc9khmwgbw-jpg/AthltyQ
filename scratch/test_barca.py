import sys
from pathlib import Path
ROOT = Path('/Users/fahamayoub/Desktop/AthlytIQ/DASHBOARD').parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from DASHBOARD.backend import get_data

results, features = get_data()
features['Match_Date'] = pd.to_datetime(features['Match_Date'])

barca = features[features['Team'].str.contains('Barcelona', na=False, case=False)]
print("Total rows for Barcelona:", len(barca))
dates = barca['Match_Date'].dropna().unique()
print("Unique match dates for Barcelona:", sorted(dates))
