"""
AthlytIQ — Scraper Match-par-Match (Selenium + API Intercept)
===============================================================
Utilise Selenium pour contourner le blocage anti-bot de SofaScore,
puis exploite les appels API internes du navigateur pour extraire
les statistiques match par match (Statistiques Techniques + Physiques).

Stratégie AMÉLIORÉE: Double requête API (Statistics + Performance).
"""

import os
import re
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager


# ══════════════════════════════════════════════════════════════════════
# 0. UTILITAIRE DE RECHERCHE DE STATS (INDISPENSABLE POUR LE PHYSIQUE)
# ══════════════════════════════════════════════════════════════════════

def find_stat(obj, key):
    """Cherche une stat dans l'objet principal ou dans les 'groups' de l'API."""
    if not obj: return None
    if key in obj: return obj[key]
    if 'groups' in obj:
        for group in obj['groups']:
            for item in group.get('statisticsItems', []):
                if item.get('key') == key:
                    return item.get('value')
    return None


# ══════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION DU NAVIGATEUR (STEALTH)
# ══════════════════════════════════════════════════════════════════════

def creer_driver(headless=True):
    options = Options()
    options.page_load_strategy = 'eager'

    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-GB")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
    )

    return driver


# ══════════════════════════════════════════════════════════════════════
# 2. RÉCUPÉRATION DES JOUEURS DEPUIS LE CSV EXISTANT
# ══════════════════════════════════════════════════════════════════════

def charger_joueurs_depuis_csv(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    joueurs = []
    for _, row in df.iterrows():
        nom = row.get('Nom', '')
        player_id = row.get('Stat_id', row.get('SofaScore_ID', row.get('ID_SofaScore', row.get('id', row.get('ID', None)))))
        slug = row.get('Slug', 'player')
        if nom and pd.notna(player_id):
            joueurs.append({'name': nom, 'id': int(player_id), 'url': f"https://www.sofascore.com/player/{slug}/{int(player_id)}"})
    return joueurs


# ══════════════════════════════════════════════════════════════════════
# 3. EXTRACTION DES MATCHS VIA SELENIUM + JS FETCH
# ══════════════════════════════════════════════════════════════════════

def extraire_matchs_joueur(driver, player_id, player_name, nb_pages=10, saison_debut='2024-08-01', total_match_limit=15):
    all_match_data_pool = []
    date_cutoff = datetime.strptime(saison_debut, '%Y-%m-%d')

    try:
        driver.get(f"https://www.sofascore.com/player/player/{player_id}")
        time.sleep(2)
    except: pass

    # Collecte du pool de matchs
    for page in range(nb_pages):
        js_script = f"return await fetch('https://api.sofascore.com/api/v1/player/{player_id}/events/last/{page}').then(r => r.json()).catch(e => ({{}}));"
        try:
            result = driver.execute_script(js_script)
            events = result.get('events', [])
            if not events: break
            
            for event in events:
                if event.get('status', {}).get('type') != 'finished': continue
                ts = event.get('startTimestamp', 0)
                dt = datetime.fromtimestamp(ts)
                if dt < date_cutoff: continue 
                
                all_match_data_pool.append({
                    'id': event.get('id'), 'date': dt.strftime('%Y-%m-%d'), 'dt_obj': dt,
                    'home': event.get('homeTeam', {}).get('name', 'Unknown'),
                    'away': event.get('awayTeam', {}).get('name', 'Unknown'),
                    'tournament': event.get('tournament', {}).get('name', 'Unknown'),
                    'home_score': event.get('homeScore', {}).get('current', 0),
                    'away_score': event.get('awayScore', {}).get('current', 0)
                })
        except: break

    if not all_match_data_pool: return []

    # Tri par date décroissante pour prendre les plus récents (Avril d'abord)
    all_match_data_pool.sort(key=lambda x: x['dt_obj'], reverse=True)
    targets = all_match_data_pool[:total_match_limit]
    
    print(f"      🎯 {len(targets)} derniers matchs identifiés. Récupération Stats + Physique...")

    final_results = []
    for m in targets:
        event_id = m['id']
        print(f"      📊 Stats: {m['date']} — {m['home']} vs {m['away']}")
        
        # 1. Stats Techniques
        s_script = f"return await fetch('https://api.sofascore.com/api/v1/event/{event_id}/player/{player_id}/statistics').then(r => r.json()).catch(e => ({{}}));"
        try:
            stats_res = driver.execute_script(s_script)
            stats = stats_res.get('statistics', {})
            if not stats: continue
            mins = stats.get('minutesPlayed', 0)

            # 2. Stats Physiques (Opta)
            p_script = f"return await fetch('https://api.sofascore.com/api/v1/event/{event_id}/player/{player_id}/performance').then(r => r.json()).catch(e => ({{}}));"
            dist_metres = 0.0
            sprints = 0.0
            try:
                perf_res = driver.execute_script(p_script)
                dist_metres = float(perf_res.get('totalDistance') or find_stat(perf_res, 'distanceRun') or 0)
                sprints = float(find_stat(perf_res, 'sprints') or find_stat(perf_res, 'highIntensityRuns') or 0)
            except: pass

            # Calculs KPIs
            work_rate = round(dist_metres / mins, 2) if (dist_metres > 0 and mins > 0) else 0.0
            
            final_results.append({
                'Nom': player_name, 'Match_Date': m['date'], 'Home_Team': m['home'], 'Away_Team': m['away'],
                'Rating': round(float(stats.get('rating', 0)), 2) if stats.get('rating') else None,
                'Minutes_Played': mins,
                'distanceRun': dist_metres if dist_metres > 0 else None,
                'sprints': sprints if sprints > 0 else None,
                'kpi_work_rate': work_rate,
                'Goals': stats.get('goals', 0), 'Assists': stats.get('assists', 0),
                'Expected_Goals': round(float(stats.get('expectedGoals', 0)), 2),
                'Expected_Assists': round(float(stats.get('expectedAssists', 0)), 2),
                'Accurate_Passes': stats.get('accuratePass', 0), 'Total_Passes': stats.get('totalPass', 0),
                'Key_Passes': stats.get('keyPass', 0), 'Tackles': stats.get('tackle', 0),
                'Interceptions': stats.get('interceptionWon', 0), 'Clearances': stats.get('clearance', 0),
                'Ball_Recovery': stats.get('ballRecovery', 0), 'Touches': stats.get('touches', 0)
            })
            time.sleep(0.5)
        except: continue

    final_results.sort(key=lambda x: x['Match_Date']) # Tri final pour le CSV
    return final_results

# ══════════════════════════════════════════════════════════════════════
# 4. FONCTIONS DE DÉCOUVERTE
# ══════════════════════════════════════════════════════════════════════

def get_teams_in_league(league_id, season_id, driver=None):
    url = f"https://api.sofascore.com/api/v1/unique-tournament/{league_id}/season/{season_id}/teams"
    import requests
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        return [{'id': t['id'], 'name': t['name']} for t in response.json().get('teams', [])]
    except: return []

def get_players_in_team(team_id, driver=None):
    """Récupère les joueurs via l'API en utilisant le contexte du navigateur pour éviter le blocage."""
    if driver:
        script = f"return await fetch('https://api.sofascore.com/api/v1/team/{team_id}/players').then(r => r.json()).catch(() => ({{}}));"
        try:
            data = driver.execute_script(script)
            players = data.get('players', [])
            return [{'id': p['player']['id'], 'name': p['player']['name']} for p in players]
        except: pass
    
    # Fallback requests (si pas de driver)
    url = f"https://api.sofascore.com/api/v1/team/{team_id}/players"
    import requests
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        return [{'id': p['player']['id'], 'name': p['player']['name']} for p in response.json().get('players', [])]
    except: return []

if __name__ == "__main__":
    print("🚀 Moteur AthlytIQ Physique Prêt.")
