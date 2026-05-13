"""
AthlytIQ — Patch Age (via scraper_league IDs)
===============================================
Stratégie : Visite toutes les équipes déjà scrapées (via l'API Sofascore),
récupère les player_id pour chaque joueur, puis injecte l'âge dans les CSV
existants sans rescraper les matchs.

Usage : .venv/bin/python DATA_PIPELINE/SCRAPPING/scripts/patch_age.py
"""

import sys
import time
import pandas as pd
from pathlib import Path
from unidecode import unidecode  # type: ignore

ROOT    = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "raw" / "sofascore"

sys.path.append(str(ROOT / "DATA_PIPELINE" / "SCRAPPING" / "scripts"))
try:
    from sofascore_match_scraper import creer_driver, get_players_in_team, get_player_age  # type: ignore
except ImportError as e:
    print(f"Impossible d'importer le moteur : {e}")
    sys.exit(1)


# ── Même config de ligues que scraper_league.py ──────────────────────────────
LEAGUES = {
    "LaLiga":          {"id": 8,   "url": "https://www.sofascore.com/tournament/football/spain/laliga/8"},
    "Premier":         {"id": 17,  "url": "https://www.sofascore.com/tournament/football/england/premier-league/17"},
    "Ligue 1":         {"id": 34,  "url": "https://www.sofascore.com/tournament/football/france/ligue-1/34"},
    "SerieA":          {"id": 23,  "url": "https://www.sofascore.com/tournament/football/italy/serie-a/23"},
    "Bundesliga":      {"id": 35,  "url": "https://www.sofascore.com/tournament/football/germany/bundesliga/35"},
    "ChampionsLeague": {"id": 7,   "url": "https://www.sofascore.com/tournament/football/europe/uefa-champions-league/7"},
    "Eredivisie":      {"id": 37,  "url": "https://www.sofascore.com/tournament/football/netherlands/eredivisie/37"},
    "ProLeague":       {"id": 38,  "url": "https://www.sofascore.com/tournament/football/belgium/jupiler-pro-league/38"},
    "Championship":    {"id": 18,  "url": "https://www.sofascore.com/tournament/football/england/championship/18"},
}


def find_csv_for_player(league_dir: Path, team_dir: Path, player_name: str):
    """Cherche le CSV d'un joueur en ignorant les accents et casses."""
    target = unidecode(player_name).lower().replace(" ", "_").replace("-", "_")
    for csv in team_dir.glob("*.csv"):
        if unidecode(csv.stem).lower().replace("-", "_") == target:
            return csv
    return None


def patch_age_via_league_api():
    """
    Pour chaque ligue scrapée, récupère les équipes et leurs joueurs via l'API,
    puis injecte l'âge dans les CSV existants.
    """
    print(f"\n{'='*60}")
    print(f"  AthlytIQ — PATCH AGE (via API équipes Sofascore)")
    print(f"{'='*60}\n")

    print("Initialisation du navigateur stealth...")
    driver = creer_driver(headless=True)
    # On initialise les cookies Sofascore
    driver.get("https://www.sofascore.com")
    time.sleep(3)

    total_patched  = 0
    total_skipped  = 0
    total_notfound = 0
    total_failed   = 0

    try:
        for league_name, league_info in LEAGUES.items():
            league_dir = RAW_DIR / league_name

            # Dossier de la ligue peut avoir des variantes de nom
            if not league_dir.exists():
                # Chercher une variante
                possible = [d for d in RAW_DIR.iterdir() if d.is_dir() and unidecode(d.name).lower() == unidecode(league_name).lower()]
                if not possible:
                    print(f"\n[LIGUE] {league_name} → dossier introuvable, skip")
                    continue
                league_dir = possible[0]

            team_dirs = [d for d in league_dir.iterdir() if d.is_dir()]
            print(f"\n{'─'*60}")
            print(f"[LIGUE] {league_name} ({len(team_dirs)} équipes)")
            print(f"{'─'*60}")

            for team_dir in team_dirs:
                team_name_raw = team_dir.name
                # Chercher l'ID de l'équipe via l'API page de ligue
                # On utilise get_players_in_team avec une recherche sur le nom d'équipe
                # Pour obtenir les joueurs, on a besoin du team_id
                # STRATÉGIE : On récupère les équipes via la page HTML de la ligue
                pass  # Géré dans la boucle suivante

            # Récupérer toutes les équipes de la ligue via la page HTML (comme scraper_league.py)
            league_url = league_info['url']
            print(f"Récupération des équipes via : {league_url}")

            try:
                driver.get(league_url)
                time.sleep(5)

                from selenium.webdriver.common.by import By
                team_links_elems = driver.find_elements(By.CSS_SELECTOR, "a[href*='/team/']")
                team_links = []
                for t in team_links_elems:
                    href = t.get_attribute('href')
                    if href and '/team/' in href and href not in team_links:
                        team_links.append(href)

                if not team_links:
                    print(f"  Aucune équipe trouvée via HTML, tentative API JS...")
                    # Fallback : API JS directe
                    js = f"return await fetch('https://api.sofascore.com/api/v1/unique-tournament/{league_info['id']}/season/0/teams').then(r=>r.json()).catch(()=>({{}}));"
                    data = driver.execute_script(js)
                    api_teams = data.get('teams', [])
                    if api_teams:
                        team_links = [f"https://www.sofascore.com/team/football/x/{t['id']}" for t in api_teams]
                        print(f"  {len(team_links)} équipes via API")
                    else:
                        print(f"  Aucune équipe trouvée pour {league_name}")
                        continue

                print(f"  {len(team_links)} équipes trouvées")

                for team_url in team_links:
                    parts     = team_url.rstrip('/').split('/')
                    team_id   = parts[-1]
                    team_name = parts[-2].replace("-", " ").title()

                    # Trouver le dossier de l'équipe (fuzzy)
                    team_folder_search = unidecode(team_name).lower().replace(" ", "_")
                    matching_dirs = [d for d in league_dir.iterdir()
                                     if d.is_dir() and unidecode(d.name).lower().replace("-", "_") == team_folder_search]

                    if not matching_dirs:
                        continue  # Équipe non encore scrapée

                    team_dir = matching_dirs[0]
                    csv_files = list(team_dir.glob("*.csv"))
                    if not csv_files:
                        continue

                    print(f"\n  [{team_name}] — {len(csv_files)} joueurs à vérifier")

                    # Récupérer les joueurs via API
                    players = get_players_in_team(team_id, driver)
                    if not players:
                        print(f"    API: aucun joueur pour {team_name}")
                        continue

                    # Construire le mapping Nom → Age
                    for player in players:
                        p_name = player['name']
                        p_id   = player['id']

                        # Chercher le CSV correspondant
                        csv_path = find_csv_for_player(league_dir, team_dir, p_name)
                        if not csv_path:
                            continue

                        try:
                            df = pd.read_csv(csv_path)

                            # Déjà patché et valide ?
                            if 'Age' in df.columns and df['Age'].notna().all() and (df['Age'] > 0).all():
                                print(f"    {p_name} — deja fait ({int(df['Age'].iloc[0])} ans)")
                                total_skipped += 1
                                continue

                            # Récupérer l'âge
                            age = get_player_age(p_id, driver)
                            time.sleep(0.2)

                            if age is not None:
                                df['Age'] = age
                                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                                print(f"    {p_name} — OK ({age} ans)")
                                total_patched += 1
                            else:
                                print(f"    {p_name} — age introuvable")
                                total_notfound += 1

                        except Exception as e:
                            print(f"    Erreur {p_name}: {e}")
                            total_failed += 1

            except Exception as e:
                print(f"  Erreur ligue {league_name}: {e}")
                continue

    finally:
        driver.quit()

    print(f"\n{'='*60}")
    print(f"  PATCH TERMINE")
    print(f"{'='*60}")
    print(f"  Patches   : {total_patched}")
    print(f"  Deja fait : {total_skipped}")
    print(f"  Introuvables: {total_notfound}")
    print(f"  Erreurs   : {total_failed}")
    print(f"\n  Relance maintenant le pipeline :")
    print(f"  1. python DATA_PIPELINE/NETTOYAGE/scripts/data_cleaner.py")
    print(f"  2. python LM/models/feature_engineering.py")


if __name__ == "__main__":
    patch_age_via_league_api()
