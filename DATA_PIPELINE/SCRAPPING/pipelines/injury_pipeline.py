import pandas as pd
from pathlib import Path
from typing import List, Set, Optional
from DATA_PIPELINE.SCRAPPING.transfermarkt.scrapers.injury_scraper import TransfermarktInjuryScraper
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("InjuryPipeline")

class InjuryPipeline:
    """Pipeline pour l'historique médical complet."""

    def __init__(self):
        self.root = Path(__file__).resolve().parents[3]
        self.input_csv = self.root / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"
        self.output_dir = self.root / "DATA_PIPELINE" / "SCRAPPING" / "data" / "raw" / "transfermarkt"
        self.output_csv = self.output_dir / "injury_history.csv"
        self.cache_file = self.output_dir / "players_processed.txt"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.browser = SofaScoreBrowser(headless=True)
        self.scraper = TransfermarktInjuryScraper(self.browser)

    def _load_players(self) -> List[str]:
        if not self.input_csv.exists():
            logger.error(f"Fichier d'entrée absent: {self.input_csv}")
            return []
        df = pd.read_csv(self.input_csv)
        return sorted(df["Nom"].dropna().unique().tolist())

    def _load_cache(self) -> Set[str]:
        if self.cache_file.exists():
            return set(self.cache_file.read_text(encoding="utf-8").strip().splitlines())
        return set()

    def _save_to_cache(self, player_name: str):
        with open(self.cache_file, "a", encoding="utf-8") as f:
            f.write(player_name + "\n")

    def run(self, limit: Optional[int] = None, force_update: bool = False, league_filter: Optional[str] = None):
        """Lance le cycle de mise à jour médicale."""
        if self.input_csv.exists():
            df = pd.read_csv(self.input_csv)
            if league_filter:
                # On cherche dans la colonne 'League' ou 'Tournament'
                col = 'League' if 'League' in df.columns else ('Tournament' if 'Tournament' in df.columns else None)
                if col:
                    df = df[df[col] == league_filter]
            players = set(df["Nom"].dropna().unique().tolist())
        else:
            players = set()

        # FALLBACK : On scanne aussi les dossiers RAW si la ligue est filtrée
        if league_filter:
            raw_sofa = self.root / "DATA_PIPELINE" / "SCRAPPING" / "data" / "raw" / "sofascore" / league_filter
            if raw_sofa.exists():
                logger.info(f"🔍 Scan des dossiers RAW pour {league_filter}...")
                for team_dir in raw_sofa.iterdir():
                    if team_dir.is_dir():
                        for p_file in team_dir.glob("*.csv"):
                            players.add(p_file.stem.replace("_", " "))
        
        players = sorted(list(players))
        if not players:
            logger.warning(f"Aucun joueur trouvé pour le filtre : {league_filter}")
            return

        cache = set() if force_update else self._load_cache()
        to_process = [p for p in players if p not in cache]
        
        if limit:
            to_process = to_process[:limit]

        logger.info(f"🔄 Mise à jour médicale : {len(to_process)} joueurs à traiter.")
        
        if not to_process:
            logger.info("✅ Historique médical déjà à jour.")
            return

        try:
            self.browser.start()
            if not self.browser.driver:
                logger.error("Impossible de démarrer le navigateur.")
                return

            # Initialisation cookies sur la home
            self.browser.driver.get("https://www.transfermarkt.com")
            import time
            time.sleep(3)

            for i, name in enumerate(to_process, 1):
                logger.info(f"[{i}/{len(to_process)}] 🔍 {name}...")
                
                player_info = self.scraper.search_player(name)
                if not player_info:
                    logger.warning(f"   ❌ Non trouvé sur TM : {name}")
                    self._save_to_cache(name)
                    continue

                injuries = self.scraper.scrape_player_injuries(player_info)
                if injuries:
                    self._save_to_csv(injuries)
                    logger.info(f"   ✅ {len(injuries)} lignes médicales ajoutées.")
                
                self._save_to_cache(name)

        finally:
            self.browser.stop()

    def _save_to_csv(self, rows: List[dict]):
        df_new = pd.DataFrame(rows)
        if self.output_csv.exists():
            df_new.to_csv(self.output_csv, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(self.output_csv, index=False, encoding="utf-8-sig")
