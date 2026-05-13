import time
import math
from datetime import datetime
from typing import List, Dict, Any, Optional
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("SofaScoreEngine")

class SofaScoreEngine:
    """
    Le moteur central d'AthlytIQ pour SofaScore.
    Remplace l'ancien 'sofascore_match_scraper.py'.
    Gère les injections JS et les récupérations hybrides (Stats + Physique).
    """

    def __init__(self, browser: Optional[SofaScoreBrowser] = None):
        self.browser = browser or SofaScoreBrowser(headless=True)

    def get_players_in_team(self, team_id: str) -> List[Dict[str, Any]]:
        """Récupère la liste des joueurs d'une équipe via API."""
        script = f"return await fetch('https://api.sofascore.com/api/v1/team/{team_id}/players').then(r => r.json()).catch(() => ({{}}));"
        try:
            data = self.browser.execute_script(script)
            players = data.get('players', [])
            return [{'id': p['player']['id'], 'name': p['player']['name']} for p in players]
        except Exception as e:
            logger.error(f"Erreur get_players_in_team({team_id}): {e}")
            return []

    def get_player_age(self, player_id: str) -> Optional[int]:
        """Récupère l'âge d'un joueur via l'API interne."""
        script = f"return await fetch('https://api.sofascore.com/api/v1/player/{player_id}').then(r => r.json()).catch(() => ({{}}));"
        try:
            data = self.browser.execute_script(script)
            player_info = data.get('player', {})
            ts = player_info.get('dateOfBirthTimestamp')
            if ts:
                dob = datetime.fromtimestamp(ts)
                today = datetime.today()
                return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except: pass
        return None

    def _find_stat(self, obj: Any, key: str) -> Optional[Any]:
        """Cherche récursivement une stat dans l'objet API (Groupes/Items)."""
        if not obj: return None
        if isinstance(obj, dict):
            if key in obj: return obj[key]
            if 'groups' in obj:
                for group in obj['groups']:
                    for item in group.get('statisticsItems', []):
                        if item.get('key') == key:
                            return item.get('value')
        return None

    def extract_player_matches(self, player_id: str, player_name: str, nb_pages: int = 1, limit: int = 15, last_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extraction hybride haute performance : Stats Techniques + Stats Physiques (Opta).
        """
        all_events = []
        cutoff = datetime(2024, 8, 1)

        for page in range(nb_pages):
            script = f"return await fetch('https://api.sofascore.com/api/v1/player/{player_id}/events/last/{page}').then(r => r.json()).catch(() => ({{}}));"
            data = self.browser.execute_script(script)
            if not data or not isinstance(data, dict): break
            
            events = data.get('events', [])
            if not events: break
            
            for ev in events:
                if ev.get('status', {}).get('type') != 'finished': continue
                ts = ev.get('startTimestamp', 0)
                dt = datetime.fromtimestamp(ts)
                dt_str = dt.strftime('%Y-%m-%d')
                
                if dt < cutoff: continue
                if last_date and dt_str <= last_date: continue

                all_events.append({
                    'id': ev['id'], 'date': dt_str, 'dt_obj': dt,
                    'home': ev.get('homeTeam', {}).get('name', 'Unknown'),
                    'away': ev.get('awayTeam', {}).get('name', 'Unknown')
                })

        if not all_events: return []
        all_events.sort(key=lambda x: x['dt_obj'], reverse=True)
        targets = all_events[:limit]

        results = []
        age = self.get_player_age(player_id)

        for m in targets:
            eid = m['id']
            # 1. Stats Techniques
            s_script = f"return await fetch('https://api.sofascore.com/api/v1/event/{eid}/player/{player_id}/statistics').then(r => r.json()).catch(() => ({{}}));"
            stats_res = self.browser.execute_script(s_script)
            stats = stats_res.get('statistics', {}) if isinstance(stats_res, dict) else {}
            if not stats: continue
            
            # 2. Stats Physiques
            p_script = f"return await fetch('https://api.sofascore.com/api/v1/event/{eid}/player/{player_id}/performance').then(r => r.json()).catch(() => ({{}}));"
            perf_res = self.browser.execute_script(p_script)
            if not isinstance(perf_res, dict): perf_res = {}
            
            # Extraction multi-niveau (Opta vs Sofa)
            dist = float(perf_res.get('totalDistance') or self._find_stat(perf_res, 'distanceRun') or 0)
            sprints = float(self._find_stat(perf_res, 'sprints') or self._find_stat(perf_res, 'highIntensityRuns') or 0)
            
            mins = int(stats.get('minutesPlayed', 0))
            work_rate = round(dist / mins, 2) if (dist > 0 and mins > 0) else 0.0

            results.append({
                'Nom': player_name, 'Age': age, 'Match_Date': m['date'],
                'Home_Team': m['home'], 'Away_Team': m['away'],
                'Rating': round(float(stats.get('rating', 0)), 2) if stats.get('rating') else None,
                'Minutes_Played': mins,
                'distanceRun': dist if dist > 0 else None,
                'sprints': sprints if sprints > 0 else None,
                'kpi_work_rate': work_rate,
                'Goals': int(stats.get('goals', 0)), 
                'Assists': int(stats.get('assists', 0)),
                'Expected_Goals': round(float(stats.get('expectedGoals', 0)), 2),
                'Expected_Assists': round(float(stats.get('expectedAssists', 0)), 2),
                'Accurate_Passes': int(stats.get('accuratePass', 0)), 
                'Total_Passes': int(stats.get('totalPass', 0)),
                'Key_Passes': int(stats.get('keyPass', 0)), 
                'Tackles': int(stats.get('tackle', 0)),
                'Interceptions': int(stats.get('interceptionWon', 0)), 
                'Clearances': int(stats.get('clearance', 0)),
                'Ball_Recovery': int(stats.get('ballRecovery', 0)), 
                'Touches': int(stats.get('touches', 0))
            })
            m_date = m.get('date', 'Date?')
            opponent = m.get('away') if m.get('home') == player_name else m.get('home')
            logger.info(f"      ⚽ {m_date} vs {opponent} (Stats OK)")
            time.sleep(0.3)

        return results
