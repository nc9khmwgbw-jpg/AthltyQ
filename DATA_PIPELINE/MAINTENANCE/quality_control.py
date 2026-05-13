import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from DATA_PIPELINE.SCRAPPING.fotmob.scrapers.match_scraper import FotMobMatchScraper
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("QualityControl")

class AthlytIQCerberus:
    """
    Système de contrôle qualité et de réparation (Arbitre Logique).
    Compare les données SofaScore avec FotMob et corrige les anomalies.
    """

    STATS_TO_CHECK = {
        'Goals':               '⚽ Buts',
        'Assists':             '🎯 Passes Décisives',
        'Minutes_Played':      '⏱️  Minutes',
        'Rating':              '⭐ Note',
        'Touches':             '🔘 Touches',
        'Expected_Goals':      '📊 xG',
        'Expected_Assists':    '📊 xA',
    }

    def __init__(self):
        self.root = Path(__file__).resolve().parents[2]
        self.raw_dir = self.root / "SCRAPPING" / "raw" / "sofascore"
        self.fotmob = FotMobMatchScraper(headless=True)

    def check_local_anomalies(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Détecte les impossibilités physiques (ex: but avec 0 min)."""
        for _, row in df.iterrows():
            try:
                mins = float(row.get('Minutes_Played', 0))
                goals = float(row.get('Goals', 0))
                if mins == 0 and goals > 0:
                    return True, f"But marqué avec 0 minute jouée le {row.get('Match_Date')}"
            except: continue
        return False, ""

    def repair_player(self, player_name: str) -> bool:
        """Tente de réparer un joueur via FotMob."""
        csv_name = f"{player_name.replace(' ', '_')}.csv"
        candidates = list(self.raw_dir.rglob(csv_name))
        if not candidates: return False

        file_path = candidates[0]
        try:
            df_local = pd.read_csv(file_path)
            
            logger.info(f"🕵️‍♂️ Analyse de vérité pour {player_name}...")
            fm_matches = self.fotmob.scrape_player(player_name)
            
            if not fm_matches:
                logger.warning(f"   ⚠️ Impossible de croiser les données avec FotMob.")
                return False

            changed = self._apply_corrections(file_path, df_local, fm_matches)
            return changed
        except Exception as e:
            logger.error(f"Erreur lors de la réparation de {player_name}: {e}")
            return False

    def _apply_corrections(self, path: Path, df: pd.DataFrame, fm_matches: List[Dict]) -> bool:
        df["Match_Date_DT"] = pd.to_datetime(df["Match_Date"], errors='coerce')
        changed = False

        for fm in fm_matches:
            try:
                fm_date = pd.to_datetime(fm["Match_Date"])
                mask = (df["Match_Date_DT"] >= fm_date - pd.Timedelta(days=1)) & \
                       (df["Match_Date_DT"] <= fm_date + pd.Timedelta(days=1))
                
                if not mask.any(): continue
                
                idx = df.index[mask][0]
                for col in ["Goals", "Assists", "Minutes_Played"]:
                    # Extraction sécurisée pour Pylance (zéro ligne rouge)
                    raw_l_val = df.at[idx, col] if col in df.columns else 0.0
                    l_val = float(pd.to_numeric(raw_l_val, errors='coerce') or 0.0)

                    raw_f_val = fm.get(col, 0.0)
                    f_val = float(pd.to_numeric(raw_f_val, errors='coerce') or 0.0)
                    
                    if abs(l_val - f_val) > 0.01:
                        df.at[idx, col] = f_val
                        logger.info(f"      ✅ Correction {col} : {l_val} -> {f_val}")
                        changed = True
            except: continue

        if changed:
            df.drop(columns=["Match_Date_DT"]).to_csv(path, index=False, encoding="utf-8-sig")
        return changed
