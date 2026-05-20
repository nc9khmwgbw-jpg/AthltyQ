import time
import pandas as pd
from pathlib import Path
from text_unidecode import unidecode
from typing import Optional

from selenium.common.exceptions import TimeoutException

from DATA_PIPELINE.SCRAPPING.sofascore.engine import SofaScoreEngine
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
from DATA_PIPELINE.SCRAPPING.common.config import LEAGUES
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("LeagueScraper")

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "data" / "raw" / "sofascore"


def find_existing_file(save_path: Path, p_name_safe: str) -> Path:
    """Cherche si un fichier similaire existe déjà (ignore les accents)."""
    target_clean = unidecode(p_name_safe).lower().replace('-', '_')
    if save_path.exists():
        for f in save_path.glob("*.csv"):
            if unidecode(f.stem).lower().replace('-', '_') == target_clean:
                return f
    return save_path / f"{p_name_safe}.csv"


class SofaScoreLeagueScraper:
    """
    Scraper de ligue moderne — utilise le SofaScoreEngine modulaire.
    """

    def __init__(self):
        self.browser = SofaScoreBrowser(headless=True)
        self.engine = SofaScoreEngine(self.browser)

    def scrape(self, league_name: str, force_update: bool = False, player_limit: Optional[int] = None):
        """Lance le scraping complet pour une ligue."""
        league_info = LEAGUES.get(league_name)
        if not league_info:
            logger.error(f"Ligue '{league_name}' inconnue.")
            return

        # --- OPTIMISATION ÉLITE : SKIP RAPIDE ---
        league_dir = RAW_DIR / league_name
        if league_dir.exists() and not force_update:
            existing_files = list(league_dir.rglob("*.csv"))
            if len(existing_files) > 100:  # Seuil arbitraire pour considérer une ligue comme "complète"
                logger.info(
                    f"⏩ Ligue '{league_name}' déjà présente ({len(existing_files)} joueurs). "
                    "Skip de la phase d'identification."
                )
                logger.info("💡 Utilisez le mode mise à jour (O) si vous voulez chercher de nouveaux joueurs.")
                return
        # ----------------------------------------

        logger.info(f"🚀 Scraping Ligue : {league_name} (ID: {league_info['id']})")

        try:
            self.browser.start()

            # Charger la page tournoi (ignorer timeout — on a juste besoin des cookies)
            logger.info(f"🌐 Chargement de la page tournoi : {league_info['url']}")
            try:
                self.browser.driver.set_page_load_timeout(15)
                self.browser.driver.get(league_info['url'])
            except TimeoutException:
                logger.warning("⚠️ Timeout page (normal) — continuation avec les cookies disponibles")

            time.sleep(3)

            logger.info("📡 Récupération de la liste des équipes via API...")
            teams = self.engine.get_teams_in_league(league_info['id'])

            if not teams:
                logger.error("❌ Impossible de récupérer les équipes. Vérifiez la connexion.")
                return

            logger.info(f"✅ {len(teams)} équipes identifiées.")

            player_count = 0
            for i, team in enumerate(teams, 1):
                if player_limit and player_count >= player_limit:
                    break

                team_id = team['id']
                team_name = team['name']
                logger.info(f"🏙️  [{i}/{len(teams)}] ÉQUIPE : {team_name}")

                players = self.engine.get_players_in_team(team_id)
                if not players:
                    continue

                for j, p in enumerate(players, 1):
                    if player_limit and player_count >= player_limit:
                        break

                    p_name, p_id = p['name'], p['id']
                    # Nettoyage des noms (suppression des accents pour les dossiers/fichiers)
                    p_name_safe = unidecode(p_name).replace(" ", "_")
                    team_name_safe = unidecode(team_name).replace(" ", "_")
                    
                    save_path = RAW_DIR / league_name / team_name_safe
                    file_path = find_existing_file(save_path, p_name_safe)

                    # Gestion incrémentale
                    last_date = None
                    is_update = file_path.exists() and force_update
                    if is_update:
                        try:
                            df_old = pd.read_csv(file_path)
                            if not df_old.empty:
                                last_date = df_old['Match_Date'].max()
                        except Exception:
                            pass

                    if file_path.exists() and not force_update:
                        player_count += 1
                        continue

                    logger.info(f"      🏃 [{j}/{len(players)}] {p_name}...")
                    match_data = self.engine.extract_player_matches(
                        p_id,
                        p_name,
                        nb_pages=1 if is_update else 2,
                        last_date=last_date,
                    )

                    if match_data:
                        df_new = pd.DataFrame(match_data)
                        save_path.mkdir(parents=True, exist_ok=True)
                        if is_update:
                            df_old = pd.read_csv(file_path)
                            df_merged = pd.concat([df_new, df_old]).drop_duplicates(subset=['Match_Date'])
                            df_merged.sort_values('Match_Date', ascending=True).to_csv(
                                file_path, index=False, encoding='utf-8-sig'
                            )
                        else:
                            df_new.to_csv(file_path, index=False, encoding='utf-8-sig')

                    player_count += 1

        finally:
            self.browser.stop()
