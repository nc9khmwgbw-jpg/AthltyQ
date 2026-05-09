import pandas as pd
import numpy as np

# Simulate some data
df = pd.DataFrame({
    'Nom': ['A', 'B', 'A', 'B'],
    'Match_Date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02']),
    'Minutes_Played': [90, 45, 90, 90],
    'Rating': [7.0, 6.0, 8.0, 7.5]
})

all_dates = sorted(df['Match_Date'].dropna().unique())
last_14_dates = all_dates[-14:]
date_to_j = {d: f"J-{len(last_14_dates) - i}" for i, d in enumerate(last_14_dates)}
print(date_to_j)
