import re
import time
import random
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Optional, List, Dict, Any
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
from DATA_PIPELINE.SCRAPPING.transfermarkt.extractors.injury_extractor import TransfermarktInjuryExtractor
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("TM_Scraper")

class TransfermarktInjuryScraper:
    """Scraper pour l'historique des blessures sur Transfermarkt."""

    BASE_URL = "https://www.transfermarkt.com"
    SEARCH_URL = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"

    def __init__(self, browser: SofaScoreBrowser):
        self.browser = browser
        self.extractor = TransfermarktInjuryExtractor()

    def search_player(self, player_name: str) -> Optional[Dict[str, Any]]:
        """Recherche un joueur et retourne ses infos de base."""
        if not self.browser.driver:
            self.browser.start()
        
        if not self.browser.driver:
            return None
            
        search_url = f"{self.SEARCH_URL}?query={player_name.replace(' ', '+')}&Spieler_page=0"
        try:
            self.browser.driver.get(search_url)
            time.sleep(random.uniform(2, 4))
            
            html = self.browser.driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            
            tables = soup.select("div#yw1 table.items")
            if not tables:
                tables = soup.select("table.items")
            if not tables:
                return None

            for table in tables:
                rows = table.select("tbody tr")
                for row in rows:
                    link = row.select_one("td.hauptlink a[href*='/profil/spieler/']")
                    if not link: continue

                    href = str(link.get("href", ""))
                    match = re.search(r"/spieler/(\d+)", href)
                    if not match: continue

                    player_id = match.group(1)
                    found_name = link.get_text(strip=True)

                    team_cell = row.select_one("td.zentriert a[href*='/startseite/verein/']")
                    team_name = team_cell.get_text(strip=True) if team_cell else ""

                    cells = row.select("td")
                    position = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                    age = cells[6].get_text(strip=True) if len(cells) > 6 else ""

                    return {
                        "id": player_id,
                        "name": found_name,
                        "url": f"{self.BASE_URL}{href}",
                        "team": team_name,
                        "position": position,
                        "age": age
                    }
        except Exception as e:
            logger.error(f"Erreur recherche TM pour {player_name}: {e}")
        
        return None

    def scrape_player_injuries(self, player_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Récupère l'historique des blessures d'un joueur."""
        if not self.browser.driver:
            self.browser.start()

        player_id = player_info["id"]
        injury_url = f"{self.BASE_URL}/player/verletzungen/spieler/{player_id}"
        
        if not self.browser.driver:
            return []

        try:
            self.browser.driver.get(injury_url)
            time.sleep(random.uniform(1.5, 3))
            
            html = self.browser.driver.page_source
            return self.extractor.extract(html, player_info)
        except Exception as e:
            logger.error(f"Erreur extraction blessures TM pour {player_info['name']}: {e}")
            return []
