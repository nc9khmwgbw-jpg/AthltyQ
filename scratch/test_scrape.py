import sys
from pathlib import Path
from DATA_PIPELINE.SCRAPPING.sofascore.scrapers.league_scraper import SofaScoreLeagueScraper
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("Test-Pipeline")

def run_test_scrape():
    print("\n" + "="*60)
    print(" 🧪 TEST ATHLYTIQ : SCRAPPING LIGUE 1 (CIBLÉ)")
    print("="*60)
    
        # Pour le test, on va injecter une limite dans le scraper
        logger.info("Test rapide sur 3 joueurs uniquement...")
        scraper.scrape("Ligue 1", force_update=True)
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    run_test_scrape()
