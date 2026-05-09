import sys
from pathlib import Path
import pandas as pd

ROOT = Path('/Users/fahamayoub/Desktop/AthlytIQ/DASHBOARD').parent
PROCESSED_PATH = ROOT / "data" / "processed" / "features_dataset.csv"

if PROCESSED_PATH.exists():
    df = pd.read_csv(PROCESSED_PATH, low_memory=False)
    print("Columns:", df.columns.tolist())
    print("Sample players:", df['Nom'].unique()[:5])
    df['Match_Date'] = pd.to_datetime(df['Match_Date'])
    dates = sorted(df['Match_Date'].unique())
    print("Last 5 dates:", dates[-5:])
