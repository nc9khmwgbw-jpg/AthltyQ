import pandas as pd
import sys
from pathlib import Path

ROOT = Path('/Users/fahamayoub/Desktop/AthlytIQ')
sys.path.insert(0, str(ROOT))
from DASHBOARD.backend import get_data, normalize_team_name

res, feat = get_data()
if feat is not None:
    koln = feat[feat['Team'] == '1 Fc Koln']
    print(f"Number of Koln players: {len(koln['Nom'].unique())}")
    for _, row in koln.head(5).iterrows():
        c = str(row.get('Team', ''))
        h = str(row.get('Home_Team', ''))
        a = str(row.get('Away_Team', ''))
        
        c_norm = normalize_team_name(c)
        h_norm = normalize_team_name(h)
        a_norm = normalize_team_name(a)
        
        is_match = (c_norm in h_norm) or (c_norm in a_norm) or (h_norm in c_norm) or (a_norm in c_norm)
        print(f"Team: '{c}' ({c_norm}), Home: '{h}' ({h_norm}), Away: '{a}' ({a_norm}) -> is_club_match: {is_match}")
