import os
import sys
import time
import pandas as pd
from pathlib import Path

# Import du moteur avec les fonctions API
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from sofascore_match_scraper import extraire_matchs_joueur, creer_driver, get_players_in_team
except ImportError:
    print("❌ Erreur : Impossible d'importer le moteur sofascore_match_scraper")
    sys.exit(1)

# Chemins originaux
ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "raw" / "sofascore"

LEAGUES = {
    "1": {"id": 8, "name": "LaLiga", "country": "Espagne", "url": "https://www.sofascore.com/tournament/football/spain/laliga/8"},
    "2": {"id": 17, "name": "Premier", "country": "Angleterre", "url": "https://www.sofascore.com/tournament/football/england/premier-league/17"},
    "3": {"id": 34, "name": "Ligue 1", "country": "France", "url": "https://www.sofascore.com/tournament/football/france/ligue-1/34"},
    "4": {"id": 23, "name": "Serie A", "country": "Italie", "url": "https://www.sofascore.com/tournament/football/italy/serie-a/23"},
    "5": {"id": 35, "name": "Bundesliga", "country": "Allemagne", "url": "https://www.sofascore.com/tournament/football/germany/bundesliga/35"},
    "6": {"id": 7, "name": "Champions League", "country": "Europe", "url": "https://www.sofascore.com/tournament/football/europe/uefa-champions-league/7"},
}

def scrape_league(league_key):
    info = LEAGUES[league_key]
    print(f"✅ Sélection : {info['name']} ({info['country']}) (ID: {info['id']})")
    
    print("🚀 Initialisation du navigateur Stealth...")
    driver = creer_driver(headless=True)
    
    try:
        print(f"🔍 Récupération des équipes pour {info['name']}...")
        driver.get(info['url'])
        time.sleep(5)
        
        # On garde Selenium juste pour trouver les IDs des équipes sur la page de la ligue
        from selenium.webdriver.common.by import By
        teams_elems = driver.find_elements(By.CSS_SELECTOR, "a[href*='/team/']")
        team_links = []
        for t in teams_elems:
            href = t.get_attribute('href')
            if '/team/' in href and href not in team_links:
                team_links.append(href)
        
        if not team_links:
            print("❌ Aucune équipe trouvée.")
            return

        print(f"✅ {len(team_links)} équipes trouvées.")
        
        for i, team_url in enumerate(team_links, 1):
            team_id = team_url.split('/')[-1]
            team_name = team_url.split('/')[-2].replace("-", " ").title()
            print(f"\n🏙️  [{i}/{len(team_links)}] ÉQUIPE : {team_name} (ID: {team_id})")
            
            # --- OPTIMISATION : Utilisation de l'API pour les joueurs ---
            players = get_players_in_team(team_id, driver)
            
            if not players:
                print(f"      ⚠️ Aucun joueur trouvé via API pour {team_name}")
                continue
                
            print(f"      👥 {len(players)} joueurs identifiés.")
            
            for j, p in enumerate(players, 1):
                p_name = p['name']
                p_id = p['id']
                p_name_safe = p_name.replace(" ", "_")
                
                save_path = RAW_DIR / info['name'] / team_name.replace(" ", "_")
                file_path = save_path / f"{p_name_safe}.csv"
                
                if file_path.exists():
                    print(f"      ⏩ [{j}/{len(players)}] {p_name}... (Déjà fait)")
                    continue
                
                print(f"      🏃 [{j}/{len(players)}] {p_name} (ID: {p_id})...")
                
                # Scraping des matchs
                match_data = extraire_matchs_joueur(driver, p_id, p_name, nb_pages=2, total_match_limit=15)
                
                if match_data:
                    df = pd.DataFrame(match_data)
                    save_path.mkdir(parents=True, exist_ok=True)
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                    print(f"      🎯 {len(df)} matchs sauvés.")
                else:
                    print(f"      ⚠️ Pas de matchs trouvés.")
                    
    finally:
        driver.quit()

def main():
    print("\n============================================================")
    print("   AthlytIQ - SCRAPER DE LIGUE INTERACTIF")
    print("============================================================\n")
    print("🏆 CHOISISSEZ UNE LIGUE :")
    for k, v in LEAGUES.items():
        print(f"  [{k}] {v['name']} ({v['country']})")
    
    choix = input("\n👉 Votre choix : ").strip()
    if choix in LEAGUES:
        scrape_league(choix)

if __name__ == "__main__":
    main()
