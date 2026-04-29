import requests
import pandas as pd
import time
import random
from pathlib import Path

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...',
    'Referer': 'https://www.sofascore.com/'
}

def get_player_matches(player_id):
    url = f"https://api.sofascore.com/api/v1/player/{player_id}/events/last/0"
    response = requests.get(url, headers=HEADERS)
    return response.json().get('events', [])

def get_match_stats(event_id, player_id):
    url = f"https://api.sofascore.com/api/v1/event/{event_id}/player/{player_id}/statistics"
    response = requests.get(url, headers=HEADERS)
    return response.json() if response.status_code == 200 else None
