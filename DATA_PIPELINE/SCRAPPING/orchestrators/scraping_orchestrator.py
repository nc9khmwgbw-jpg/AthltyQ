"""
Orchestrateur de scraping — Conservé pour compatibilité architecturale.

NOTE IMPORTANTE : Le scraping réel passe par league_scraper.py qui utilise
directement le moteur original (creer_driver + extraire_matchs_joueur).
Cet orchestrateur sera utilisé pour des opérations futures multi-sources
(ex: combiner SofaScore + Transfermarkt en un seul pipeline).
"""
from DATA_PIPELINE.SCRAPPING.common.config import LEAGUES
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("ScrapingOrchestrator")


class ScrapingOrchestrator:
    """
    Chef d'orchestre pour les opérations multi-sources futures.
    Actuellement, le scraping SofaScore est géré par SofaScoreLeagueScraper.
    """

    def __init__(self):
        self.leagues_config = LEAGUES

    def list_leagues(self):
        """Retourne la liste des ligues configurées."""
        return list(self.leagues_config.keys())

    def get_league_info(self, league_name):
        """Retourne les infos d'une ligue."""
        return self.leagues_config.get(league_name)
