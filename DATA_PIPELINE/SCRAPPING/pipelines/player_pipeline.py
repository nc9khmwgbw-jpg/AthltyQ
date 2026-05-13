"""
Pipeline joueur — Conservé pour compatibilité architecturale.

NOTE : Le scraping principal passe par league_scraper.py.
Ce pipeline sera utilisé pour les opérations futures
(ex: enrichissement Transfermarkt après un scraping SofaScore).
"""
from DATA_PIPELINE.SCRAPPING.sofascore.scrapers.player_scraper import SofaScorePlayerScraper
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger
from text_unidecode import unidecode
import pandas as pd
from pathlib import Path

logger = setup_logger("PlayerPipeline")


class PlayerPipeline:
    """Pipeline pour orchestrer le scraping multi-sources d'un joueur."""

    def __init__(self, browser=None):
        self.sofascore = SofaScorePlayerScraper(browser=browser)

    def run(self, player_id, player_name, league, team):
        """Exécute le pipeline complet avec gestion de l'historique."""
        logger.info(f"🚀 Lancement du pipeline pour {player_name}")

        # Normalisation des noms (Anti-doublons Unicode)
        league_safe = unidecode(league).replace(' ', '_')
        team_safe = unidecode(team).replace(' ', '_')
        player_safe = unidecode(player_name).replace(' ', '_')

        path = Path(f"DATA_PIPELINE/SCRAPPING/raw/sofascore/{league_safe}/{team_safe}/{player_safe}.csv")

        # Vérification de l'âge existant pour éviter Selenium inutile
        existing_age = None
        if path.exists():
            try:
                df_check = pd.read_csv(path, encoding='utf-8-sig')
                if 'Age' in df_check.columns and not df_check['Age'].isna().all():
                    existing_age = df_check['Age'].iloc[0]
                    logger.info(f"📅 Âge déjà connu ({existing_age} ans), bypass Selenium.")
            except Exception:
                pass

        sofa_data = self.sofascore.scrape(player_id, player_name, existing_age=existing_age)

        if not sofa_data.get('matches'):
            logger.warning(f"Aucune donnée récupérée pour {player_name}")
            return

        df_new = pd.DataFrame(sofa_data['matches'])
        df_new['Age'] = sofa_data.get('age')

        # Injection des métadonnées
        for key, value in sofa_data.get('metadata', {}).items():
            df_new[key.capitalize()] = value

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            try:
                df_old = pd.read_csv(path, encoding='utf-8-sig')
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates(
                    subset=['Match_Date', 'Home_Team', 'Away_Team'], keep='last'
                )
                df_combined = df_combined.sort_values('Match_Date', ascending=False)
                df_combined.to_csv(path, index=False, encoding='utf-8-sig')
                logger.info(f"✅ Historique mis à jour : {path}")
            except Exception as e:
                logger.error(f"Erreur lors de la fusion : {e}. Sauvegarde forcée.")
                df_new.to_csv(path, index=False, encoding='utf-8-sig')
        else:
            self.sofascore.save(sofa_data, path)

        logger.info(f"🏁 Fin du pipeline pour {player_name}")
