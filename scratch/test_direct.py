import sys
from pathlib import Path

# Racine
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from DATA_PIPELINE.SCRAPPING.sofascore.engine import SofaScoreEngine
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
import pandas as pd

def run_direct_test():
    print("\n" + "="*60)
    print(" 🧪 TEST DIRECT : AS MONACO -> CONSOLIDATION")
    print("="*60)
    
    browser = SofaScoreBrowser(headless=True)
    engine = SofaScoreEngine(browser)
    
    try:
        # 1. Extraction directe via API (bypass navigation ligue)
        # ID AS Monaco = 1617
        print("🛰️  Récupération des joueurs de l'AS Monaco...")
        players = engine.get_players_in_team("1617")
        
        if not players:
            print("❌ Aucun joueur trouvé.")
            return

        # On prend les 3 premiers joueurs pour le test
        test_players = players[:3]
        
        raw_dir = Path("DATA_PIPELINE/SCRAPPING/raw/sofascore/Ligue_1/AS_Monaco")
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        for p in test_players:
            print(f"🔍 Scraping {p['name']}...")
            matches = engine.extract_player_matches(p['id'], p['name'], limit=5)
            if matches:
                df = pd.DataFrame(matches)
                # Injection de l'âge pour le test
                df['Age'] = engine.get_player_age(p['id'])
                file_path = raw_dir / f"{p['name'].replace(' ', '_')}.csv"
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"   ✅ {len(matches)} matchs sauvegardés.")

        print("\n🚀 Lancement du Nettoyage...")
        from DATA_PIPELINE.NETTOYAGE.scripts.data_cleaner import clean_and_merge_data
        clean_and_merge_data()
        
    finally:
        browser.stop()

if __name__ == "__main__":
    run_direct_test()
