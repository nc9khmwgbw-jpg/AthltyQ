"""
league_scraper_fixed.py
========================
Corrections vs l'original :

1. SUPPRESSION de la navigation préalable vers league_info['url'] avant get_teams_in_league().
   C'était le bug principal : les logs XHR de la navigation vers le tournoi étaient
   consommés AVANT que l'engine puisse les lire.

2. L'engine v2 gère lui-même toute la navigation et les appels API.

3. Timeout de chargement de page réduit (la page peut timeout, ce n'est pas grave).
"""

import time
import pandas as pd
from pathlib import Path
from text_unidecode import unidecode
from typing import Optional

from DATA_PIPELINE.SCRAPPING.sofascore.engine import SofaScoreEngine
from DATA_PIPELINE.SCRAPPING.common.config import LEAGUES
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("LeagueScraper")

ROOT    = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "data" / "raw" / "sofascore"
POS_FILE = ROOT / "data" / "player_positions.csv"

SOFA_POS_MAP = {
    'CF': 'ATT', 'ST': 'ATT',
    'LW': 'AG',  'LF': 'AG',  'LM': 'AG',
    'RW': 'AD',  'RF': 'AD',  'RM': 'AD',
    'SS': 'ATT', 'F':  'ATT',
    'AM': 'MOF', 'CAM': 'MOF',
    'CM': 'MC',  'M':  'MC',
    'DM': 'MDF', 'CDM': 'MDF',
    'CB': 'CB',  'D':  'CB',
    'LB': 'LB',  'LWB': 'LB',
    'RB': 'RB',  'RWB': 'RB', 'WB': 'RB',
    'G':  'GK',  'GK': 'GK',
    'ATT': 'ATT', 'AG': 'AG', 'AD': 'AD',
    'MOF': 'MOF', 'MC': 'MC', 'MDF': 'MDF',
    'CB': 'CB',   'LB': 'LB', 'RB': 'RB',
}


def find_existing_file(save_path: Path, p_name_safe: str) -> Path:
    target_clean = unidecode(p_name_safe).lower().replace('-', '_')
    if save_path.exists():
        for f in save_path.glob("*.csv"):
            if unidecode(f.stem).lower().replace('-', '_') == target_clean:
                return f
    return save_path / f"{p_name_safe}.csv"


class SofaScoreLeagueScraper:

    def __init__(self):
        self.engine  = SofaScoreEngine()

    def scrape(self, league_name: str, force_update: bool = False, player_limit: Optional[int] = None):
        league_info = LEAGUES.get(league_name)
        if not league_info:
            logger.error(f"Ligue '{league_name}' inconnue.")
            return

        # Skip rapide si déjà scrapé
        league_dir = RAW_DIR / league_name
        if league_dir.exists() and not force_update:
            existing_files = list(league_dir.rglob("*.csv"))
            if len(existing_files) > 100:
                logger.info(
                    f"⏩ Ligue '{league_name}' déjà présente ({len(existing_files)} joueurs). "
                    "Skip de la phase d'identification."
                )
                return

        logger.info(f"🚀 Scraping Ligue : {league_name} (ID: {league_info['id']})")
        positions_collected: dict = {}

        try:
            logger.info("📡 Récupération de la liste des équipes via API...")
            teams = self.engine.get_teams_in_league(league_info['id'])

            if not teams:
                logger.error("❌ Impossible de récupérer les équipes.")
                return

            logger.info(f"✅ {len(teams)} équipes identifiées.")

            player_count = 0
            for i, team in enumerate(teams, 1):
                if player_limit and player_count >= player_limit:
                    break

                team_id   = team['id']
                team_name = team['name']
                logger.info(f"🏙️  [{i}/{len(teams)}] ÉQUIPE : {team_name}")

                players = self.engine.get_players_in_team(team_id)
                if not players:
                    continue

                for j, p in enumerate(players, 1):
                    if player_limit and player_count >= player_limit:
                        break

                    p_name    = p['name']
                    p_id      = p['id']
                    p_pos_raw = p.get('position')
                    p_pos_cat = SOFA_POS_MAP.get(str(p_pos_raw).upper(), None) if p_pos_raw else None

                    if p_pos_cat == 'GK':
                        player_count += 1
                        continue

                    pos_label = f"[{p_pos_raw}→{p_pos_cat}]" if p_pos_cat else "[pos:?]"

                    p_name_safe    = unidecode(p_name).replace(" ", "_")
                    team_name_safe = unidecode(team_name).replace(" ", "_")

                    save_path = RAW_DIR / league_name / team_name_safe
                    file_path = find_existing_file(save_path, p_name_safe)

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
            if positions_collected:
                _update_positions_file(positions_collected)
                logger.info(f"📋 {len(positions_collected)} postes sauvegardés dans {POS_FILE}")


def _update_positions_file(new_positions: dict):
    POS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if POS_FILE.exists():
        df_old = pd.read_csv(POS_FILE)
        existing = dict(zip(df_old['Nom'], df_old['Poste_Cat']))
    merged = {**existing, **{k: v for k, v in new_positions.items() if v != 'GK'}}
    pd.DataFrame(
        [{'Nom': k, 'Poste_Cat': v} for k, v in merged.items()]
    ).to_csv(POS_FILE, index=False, encoding='utf-8-sig')
