import sys
from pathlib import Path

ROOT = Path('/Users/fahamayoub/Desktop/AthlytIQ')
sys.path.insert(0, str(ROOT))
from DASHBOARD.backend import get_player_data

res = get_player_data()
players = res['players']
koln_players = [p for p in players if 'koln' in p['team'].lower() or 'köln' in p['team'].lower()]
if koln_players:
    p = koln_players[0]
    print(f"Player: {p['name']}, Team: {p['team']}")
    print(f"Historique jours count: {len(p['historique_jours'])}")
    for date, data in p['historique_jours'].items():
        print(f"Date: {date}, Data: {data}")
else:
    print("No Koln players found")
