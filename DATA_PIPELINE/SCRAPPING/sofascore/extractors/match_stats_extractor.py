from datetime import datetime
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("MatchStatsExtractor")

class SofaScoreMatchStatsExtractor:
    """
    Spécialiste de la transformation des statistiques de match SofaScore.
    Traduit le JSON complexe en dictionnaire plat (flat dictionary) pour CSV/IA.
    """

    @staticmethod
    def _find_stat(stats_obj, key):
        """Cherche une stat dans l'objet principal ou dans les 'groups' de l'API SofaScore."""
        if not stats_obj: return 0
        if key in stats_obj: return stats_obj[key]
        if 'groups' in stats_obj:
            for group in stats_obj['groups']:
                for item in group.get('statisticsItems', []):
                    if item.get('key') == key:
                        return item.get('value')
        return 0

    @classmethod
    def extract_stats(cls, event_json, player_id):
        """
        Extrait les statistiques clés pour un match donné.
        """
        try:
            stats = event_json.get('statistics', {})
            
            # ── Bloc de base ──────────
            timestamp = event_json.get('startTimestamp')
            match_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d') if timestamp else None

            data = {
                'Match_Date': match_date,
                'Home_Team': event_json.get('homeTeam', {}).get('name'),
                'Away_Team': event_json.get('awayTeam', {}).get('name'),
                'Rating': stats.get('rating', 0),
                'Minutes_Played': stats.get('minutesPlayed', 0),
            }

            # ── Bloc Offensif (Utilisation de _find_stat pour la robustesse) ──
            data.update({
                'Goals': cls._find_stat(stats, 'goals'),
                'Assists': cls._find_stat(stats, 'assists'),
                'Expected_Goals': cls._find_stat(stats, 'expectedGoals'),
                'Expected_Assists': cls._find_stat(stats, 'expectedAssists'),
                'Total_Shots': cls._find_stat(stats, 'totalShots'),
                'Shots_On_Target': cls._find_stat(stats, 'shotsOnTarget'),
            })

            # ── Bloc Passes / Création ──
            data.update({
                'Accurate_Passes': cls._find_stat(stats, 'accuratePasses'),
                'Total_Passes': cls._find_stat(stats, 'totalPasses'),
                'Key_Passes': cls._find_stat(stats, 'keyPasses'),
                'Big_Chances_Created': cls._find_stat(stats, 'bigChancesCreated'),
            })

            # ── Bloc Défensif ──
            data.update({
                'Tackles': cls._find_stat(stats, 'tackles'),
                'Interceptions': cls._find_stat(stats, 'interceptions'),
                'Clearances': cls._find_stat(stats, 'clearances'),
                'Ball_Recovery': cls._find_stat(stats, 'ballRecovery'),
                'Blocked_Shots': cls._find_stat(stats, 'blockedShots'),
            })

            # ── Bloc Duels ──
            data.update({
                'Ground_Duels_Won': cls._find_stat(stats, 'groundDuelsWon'),
                'Ground_Duels_Total': cls._find_stat(stats, 'groundDuelsTotal'),
                'Aerial_Duels_Won': cls._find_stat(stats, 'aerialDuelsWon'),
                'Aerial_Duels_Total': cls._find_stat(stats, 'aerialDuelsTotal'),
                'Successful_Dribbles': cls._find_stat(stats, 'successfulDribbles'),
            })

            # ── Bloc Discipline / Fautes ──
            data.update({
                'Fouls': cls._find_stat(stats, 'fouls'),
                'Was_Fouled': cls._find_stat(stats, 'wasFouled'),
                'Yellow_Cards': 1 if stats.get('yellowCard') else 0,
                'Red_Cards': 1 if stats.get('redCard') else 0,
                'Touches': cls._find_stat(stats, 'touches'),
                'Possession_Lost': cls._find_stat(stats, 'possessionLost'),
            })

            return data
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des stats : {e}")
            return None
