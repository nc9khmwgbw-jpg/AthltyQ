import re
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional

class TransfermarktInjuryExtractor:
    """Extracteur de données de blessures depuis le HTML de Transfermarkt."""

    INJURY_CATEGORIES = {
        "MUSCULAIRE": [
            "muscle", "muscular", "hamstring", "quadricep", "calf", "strain",
            "adductor", "groin", "thigh", "psoas", "ischios", "mollet",
        ],
        "LIGAMENT": [
            "ligament", "acl", "mcl", "pcl", "ankle sprain", "cruciate",
            "entorse", "cheville", "genou", "knee sprain",
        ],
        "TENDON": [
            "tendon", "achilles", "patellar", "tendinitis", "tendinopathy",
            "achille", "rotulien",
        ],
        "OS": [
            "fracture", "bone", "stress fracture", "broken", "metatarsal",
            "tibia", "fibula", "rib", "fractura",
        ],
        "GENOU": [
            "knee", "meniscus", "cartilage", "genou", "ménisque",
            "patella", "kneecap",
        ],
        "DOS_HANCHE": [
            "back", "hip", "spine", "lumbar", "dos", "hanche", "pubis",
            "hernia", "pubalgia",
        ],
        "TETE_COU": [
            "head", "neck", "concussion", "jaw", "shoulder", "collarbone",
            "tête", "cou", "commotion",
        ],
        "MALADIE": [
            "illness", "virus", "covid", "flu", "sick", "maladie",
            "appendix", "gastro",
        ],
        "AUTRE": [],
    }

    def classifier_blessure(self, injury_text: str) -> str:
        """Classifie une blessure en catégorie standardisée."""
        if not injury_text:
            return "AUTRE"
        text = injury_text.lower()
        for category, keywords in self.INJURY_CATEGORIES.items():
            if any(kw in text for kw in keywords):
                return category
        return "AUTRE"

    def parse_date(self, d_str: str) -> str:
        """Parse les formats de date variés de Transfermarkt."""
        for fmt in ("%b %d, %Y", "%d/%m/%Y", "%d.%m.%Y", "%B %d, %Y"):
            try:
                return datetime.strptime(d_str, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return d_str

    def extract(self, html: str, player_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrait l'historique de blessures depuis le HTML."""
        soup = BeautifulSoup(html, "html.parser")
        injuries = []
        
        player_name = player_info.get("name", "Unknown")
        player_id = player_info.get("id", "Unknown")
        team = player_info.get("team", "")
        age = player_info.get("age", "")
        position = player_info.get("position", "")

        table = soup.select_one("table.items")
        if not table:
            return [self._empty_row(player_name, team, player_id, age, position)]

        rows = table.select("tbody tr")
        for row in rows:
            cells = row.select("td")
            if len(cells) < 5:
                continue

            try:
                season      = cells[0].get_text(strip=True)
                injury_type = cells[1].get_text(strip=True)
                date_from   = cells[2].get_text(strip=True)
                date_to     = cells[3].get_text(strip=True)
                duration    = cells[4].get_text(strip=True)

                duration_days = 0
                dur_match = re.search(r"(\d+)", duration.replace(",", ""))
                if dur_match:
                    duration_days = int(dur_match.group(1))

                injuries.append({
                    "Nom":             player_name,
                    "Team":            team,
                    "Transfermarkt_ID": player_id,
                    "Season":          season,
                    "Injury_Type":     injury_type,
                    "Date_From":       self.parse_date(date_from),
                    "Date_To":         self.parse_date(date_to),
                    "Duration_Days":   duration_days,
                    "Cause_Category":  self.classifier_blessure(injury_type),
                    "Age":             age,
                    "Position":        position
                })
            except Exception:
                continue

        return injuries if injuries else [self._empty_row(player_name, team, player_id, age, position)]

    def _empty_row(self, name, team, tid, age, pos) -> Dict[str, Any]:
        return {
            "Nom":             name,
            "Team":            team,
            "Transfermarkt_ID": tid,
            "Season":          "N/A",
            "Injury_Type":     "NONE",
            "Date_From":       pd.NaT,
            "Date_To":         pd.NaT,
            "Duration_Days":   0,
            "Cause_Category":  "NONE",
            "Age":             age,
            "Position":        pos
        }
