import pandas as pd
from pathlib import Path
from DATA_PIPELINE.SCRAPPING.interfaces.base_scraper import BaseScraper
from DATA_PIPELINE.SCRAPPING.sofascore.engine import SofaScoreEngine
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("PlayerScraper")

class SofaScorePlayerScraper(BaseScraper):
    """
    Scraper spécialisé pour les joueurs SofaScore.
    Utilise le SofaScoreEngine pour une extraction hybride (Stats + Physique).
    """

    def __init__(self, browser=None):
        self.browser = browser or SofaScoreBrowser(headless=True)
        self.engine = SofaScoreEngine(self.browser)

    def scrape(self, player_id, player_name=None, existing_age=None, **kwargs):
        """Lance le cycle de scraping complet pour un joueur via le moteur unifié."""
        logger.info(f"🚀 Scraping unifié pour {player_name or player_id}")

        # Utilisation du moteur pour extraire les matchs (inclut stats + physique)
        matches = self.engine.extract_player_matches(
            player_id, 
            player_name or "Unknown", 
            nb_pages=kwargs.get('nb_pages', 2),
            limit=kwargs.get('limit', 15)
        )

        age = existing_age or self.engine.get_player_age(player_id)

        return {
            'player_id': player_id,
            'name': player_name,
            'age': age,
            'matches': matches
        }

    def save(self, data, path):
        """Sauvegarde les données au format CSV avec injection de l'âge."""
        if not data.get('matches'):
            logger.warning(f"Aucun match à sauvegarder pour {data.get('name')}")
            return

        df = pd.DataFrame(data['matches'])
        if data.get('age'):
            df['Age'] = data['age']

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        logger.info(f"✅ Fichier sauvegardé : {path}")
