import json
import re
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("SofaScoreParser")

class SofaScoreParser:
    """Analyseur de données SofaScore (HTML et JSON)."""

    @staticmethod
    def extract_json_from_html(html_content, pattern):
        """Extrait un bloc JSON du HTML en utilisant un pattern regex."""
        try:
            match = re.search(pattern, html_content)
            if match:
                return json.loads(match.group(1))
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction JSON : {e}")
        return None

    @staticmethod
    def parse_player_age(html_content):
        """Analyse le HTML pour trouver l'âge du joueur."""
        # Logique simplifiée : cherche le pattern de l'âge dans le HTML
        # (À affiner avec les vrais sélecteurs SofaScore)
        try:
            # Exemple de recherche : "26 years old"
            match = re.search(r'(\d{2})\s+years\s+old', html_content)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return None
