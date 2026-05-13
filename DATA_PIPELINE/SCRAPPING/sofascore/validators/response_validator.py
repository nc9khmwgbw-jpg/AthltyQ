from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("SofaScoreValidator")

class SofaScoreResponseValidator:
    """
    Contrôleur qualité pour les réponses de l'API SofaScore.
    Vérifie l'intégrité des données avant traitement.
    """

    @staticmethod
    def validate_player_data(data):
        """Vérifie si les données de base du joueur sont présentes."""
        if not data or 'player' not in data:
            logger.warning("Validation échouée : Données joueur manquantes.")
            return False
        return True

    @staticmethod
    def validate_match_stats(stats_json):
        """Vérifie si les stats de match sont exploitables (pas vides)."""
        if not stats_json or 'statistics' not in stats_json:
            logger.warning("Validation échouée : Statistiques de match manquantes.")
            return False
        
        # Vérification si le joueur a joué au moins 1 minute
        minutes = stats_json.get('statistics', {}).get('minutesPlayed', 0)
        if minutes == 0:
            logger.debug("Match ignoré : 0 minute jouée.")
            return False
            
        return True
