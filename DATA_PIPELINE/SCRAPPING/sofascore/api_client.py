from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("SofaScoreAPI")

class SofaScoreAPIClient:
    """Client pour interagir avec l'API SofaScore via le tunnel Navigateur."""
    
    BASE_URL = "https://api.sofascore.com/api/v1"

    def __init__(self, browser):
        self.browser = browser

    def get_json(self, endpoint):
        """Récupère du JSON via le fetch du navigateur."""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            # On utilise le tunnel Selenium pour éviter le 403
            return self.browser.fetch_json(url)
        except Exception as e:
            logger.error(f"Erreur API via Browser sur {url}: {e}")
            return None

    def get_current_season_id(self, tournament_id):
        """Récupère l'ID de la saison la plus récente."""
        data = self.get_json(f"tournament/{tournament_id}/seasons")
        if data and 'seasons' in data:
            return data['seasons'][0]['id']
        return None

    def get_teams_in_tournament(self, tournament_id, season_id):
        data = self.get_json(f"tournament/{tournament_id}/season/{season_id}/teams")
        return data.get('teams', []) if data else []

    def get_players_in_team(self, team_id, season_id):
        data = self.get_json(f"team/{team_id}/players")
        return data.get('players', []) if data else []

    def get_player_last_events(self, player_id):
        data = self.get_json(f"player/{player_id}/events/last/0")
        return data.get('events', []) if data else []

    def get_match_detailed_stats(self, event_id, player_id):
        """Récupère les stats détaillées d'un match précis."""
        return self.get_json(f"event/{event_id}/player/{player_id}/statistics")
