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
RAW_DIR  = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "data" / "raw" / "sofascore"
POS_FILE = ROOT / "data" / "player_positions.csv"   # Mapping centralisé Nom → Poste


# Mapping code SofaScore → poste granulaire AthlytIQ
# (même que dans POSTE_MAP du similarity engine)
SOFA_POS_MAP = {
    'CF': 'ATT', 'ST': 'ATT',
    'LW': 'AG',  'LF': 'AG',  'LM': 'AG',
    'RW': 'AD',  'RF': 'AD',  'RM': 'AD',
    'SS': 'ATT', 'F':  'ATT',
    'AM': 'MOF', 'CAM':'MOF',
    'CM': 'MC',  'M':  'MC',
    'DM': 'MDF', 'CDM':'MDF',
    'CB': 'CB',  'D':  'CB',
    'LB': 'LB',  'LWB':'LB',
    'RB': 'RB',  'RWB':'RB',  'WB': 'RB',
    'G':  'GK',  'GK': 'GK',
    # Pass-through
    'ATT':'ATT', 'AG':'AG', 'AD':'AD',
    'MOF':'MOF', 'MC':'MC', 'MDF':'MDF',
    'CB':'CB',   'LB':'LB', 'RB':'RB',
}



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

        positions_collected: dict = {}   # Nom → Poste_Cat (accumulateur de session)

        try:
            self.browser.start()

            # Charger la page tournoi (ignorer timeout — on a juste besoin des cookies)
            logger.info(f"🌐 Chargement de la page tournoi : {league_info['url']}")
            try:
                assert self.browser.driver is not None  # Pour rassurer le vérificateur de type (linter)
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

                    p_name     = p['name']
                    p_id       = p['id']
                    p_pos_raw  = p.get('position')          # Code SofaScore brut (ex: 'LW', 'CB')
                    p_pos_cat  = SOFA_POS_MAP.get(str(p_pos_raw).upper(), None) if p_pos_raw else None

                    # Log du poste récupéré
                    if p_pos_cat and p_pos_cat != 'GK':
                        pos_label = f"[{p_pos_raw}→{p_pos_cat}]"
                    elif p_pos_cat == 'GK':
                        pos_label = "[GK - ignoré]"
                        player_count += 1
                        continue   # On ne scrape pas les gardiens
                    else:
                        pos_label = "[pos:?]"

                    # Nettoyage des noms (suppression des accents pour les dossiers/fichiers)
                    p_name_safe    = unidecode(p_name).replace(" ", "_")
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
                        # Même si on skip le scraping, on met à jour le poste si on le connaît
                        if p_pos_cat:
                            positions_collected[p_name] = p_pos_cat
                        player_count += 1
                        continue

                    logger.info(f"      🏃 [{j}/{len(players)}] {p_name} {pos_label}...")
                    match_data = self.engine.extract_player_matches(
                        p_id,
                        p_name,
                        nb_pages=1 if is_update else 2,
                        last_date=last_date,
                    )

                    if match_data:
                        df_new = pd.DataFrame(match_data)
                        # ── Injection du poste dans le CSV ──
                        if p_pos_cat:
                            df_new['Position_SofaScore'] = p_pos_raw
                            df_new['Poste_Cat']          = p_pos_cat
                            positions_collected[p_name]  = p_pos_cat

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

            # ── Sauvegarde du mapping centralisé Nom → Poste ──
            if positions_collected:
                _update_positions_file(positions_collected)
                logger.info(f"📋 {len(positions_collected)} postes sauvegardés dans {POS_FILE}")


def _update_positions_file(new_positions: dict):
    """
    Met à jour data/player_positions.csv avec les nouveaux postes scrappés.
    Fusionne avec le fichier existant (les nouvelles entrées écrasent les anciennes).
    Exclut les gardiens (GK).
    """
    POS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Charger l'existant
    if POS_FILE.exists():
        df_old = pd.read_csv(POS_FILE)
        existing = dict(zip(df_old['Nom'], df_old['Poste_Cat']))
    else:
        existing = {}

    # Fusionner : les postes scrappés ont priorité
    merged = {**existing, **{k: v for k, v in new_positions.items() if v != 'GK'}}

    pd.DataFrame(
        [{'Nom': k, 'Poste_Cat': v} for k, v in merged.items()]
    ).to_csv(POS_FILE, index=False, encoding='utf-8-sig')
