class SofaScorePlayerStatsExtractor:
    @staticmethod
    def extract(player_json):
        """Extrait les métadonnées de base du joueur."""
        p = player_json.get('player', {})
        return {
            'height': p.get('height'),
            'weight': p.get('weight'),
            'preferred_foot': p.get('preferredFoot'),
            'position': p.get('position'),
            'value': p.get('proposedMarketValue')
        }
