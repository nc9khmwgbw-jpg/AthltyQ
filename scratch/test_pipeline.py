import sys
import os
from pathlib import Path

# Ajout de la racine du projet
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from DATA_PIPELINE.SCRAPPING.sofascore.scrapers.league_scraper import SofaScoreLeagueScraper
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("Test-Pipeline")

def run_test_cycle():
    print("\n" + "="*60)
    print(" 🧪 TEST ATHLYTIQ : CYCLE COMPLET (SCRAPE + CLEAN)")
    print("="*60)
    
    # 1. SCRAPPING LITE (3 joueurs)
    logger.info("--- ÉTAPE 1 : SCRAPPING LITE ---")
    scraper = SofaScoreLeagueScraper()
    try:
        # On scrape uniquement 3 joueurs de Ligue 1 pour le test
        scraper.scrape("Ligue 1", force_update=True, player_limit=3)
        print("\n✅ Scrapping terminé avec succès.")
    except Exception as e:
        print(f"\n❌ Erreur Scrapping : {e}")
        return

    # 2. NETTOYAGE & CONSOLIDATION
    logger.info("\n--- ÉTAPE 2 : NETTOYAGE & CONSOLIDATION ---")
    try:
        from DATA_PIPELINE.NETTOYAGE.scripts.data_cleaner import clean_and_merge_data
        clean_and_merge_data()
        print("\n✅ Nettoyage terminé avec succès.")
    except Exception as e:
        print(f"\n❌ Erreur Nettoyage : {e}")

if __name__ == "__main__":
    run_test_cycle()
