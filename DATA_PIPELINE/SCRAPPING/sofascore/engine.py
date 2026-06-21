from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
from httptools.parser import url_parser
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from selenium.common.exceptions import WebDriverException
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("SofaScoreEngine")

class SofaScoreEngine:

    def __init__(self, browser: Optional[SofaScoreBrowser] = None):
        self.browser = browser or SofaScoreBrowser(headless=True)

    def _api_get(self, url: str) -> dict:
        """Fetch API via JS dans le contexte sofascore.com (cookies déjà établis)."""
        for attempt in range(2):
            try:
                script = f"""
                    var callback = arguments[arguments.length - 1];
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 15000);
                    
                    fetch('{url}', {{
                        signal: controller.signal,
                        headers: {{
                            'Accept': 'application/json',
                            'X-Requested-With': 'XMLHttpRequest'
                        }}
                    }})
                    .then(r => r.json())
                    .then(data => {{
                        clearTimeout(timeoutId);
                        callback(data);
                    }})
                    .catch(e => {{
                        callback({{ error: "fetch_timeout", details: e.toString() }});
                    }});
                """
                if self.browser.driver is None:
                    self.browser.start()
                    self.browser.driver.get("https://www.sofascore.com")
                    import time
                    time.sleep(3)
                
                result = self.browser.driver.execute_async_script(script)
                if isinstance(result, dict):
                    if 'error' in result:
                        err_reason = str(result.get('error', ''))
                        is_cloudflare = "challenge" in err_reason or result['error'].get('code') in [403]
                        
                        if is_cloudflare:
                            logger.warning(f"⚠️ Cloudflare Block on {url}: {result['error']}")
                            if attempt == 0:
                                logger.error("🛡️ Bloqué par Cloudflare ! Basculement en mode visible pour vérification manuelle...")
                                self.browser.restart_visible()
                                continue
                        else:
                            # Simple 404 (Data missing)
                            pass # We don't need to loudly log 404s as errors if they just mean no stats
                            
                    return result
                return {}
            except Exception as e:
                logger.error(f"Erreur _api_get({url}): {e}")
                return {}
        return {}

    def get_current_season_id(self, tournament_id: int) -> Optional[int]:
        data = self._api_get(f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}/seasons")
        seasons = data.get('seasons', [])
        logger.info(f"DEBUG {len(seasons)} saisons trouvées pour tournoi {tournament_id}")

        best_id, best_count = None, 0
        for season in seasons[:5]:
            sid = season.get('id')
            sname = season.get('name', '?')
            test = self._api_get(f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{sid}/teams")
            count = len(test.get('teams', []))
            logger.info(f"DEBUG season {sid} ({sname}) → {count} équipes")
            if count > best_count:
                best_count, best_id = count, sid
            if count >= 14:
                break

        return best_id

    def get_teams_in_league(self, tournament_id: int) -> List[Dict[str, Any]]:
        season_id = self.get_current_season_id(tournament_id)
        if not season_id:
            logger.error(f"Impossible de trouver la saison pour le tournoi {tournament_id}")
            return []
        data = self._api_get(f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/teams")
        teams = data.get('teams', [])
        return [{'id': t['id'], 'name': t['name'], 'slug': t['slug']} for t in teams]

    def get_players_in_team(self, team_id: str) -> List[Dict[str, Any]]:
        data = self._api_get(f"https://api.sofascore.com/api/v1/team/{team_id}/players")
        players = data.get('players', [])
        return [{
            'id': p.get('player', {}).get('id'), 
            'name': p.get('player', {}).get('name'),
            'position': p.get('player', {}).get('position')
        } for p in players if p.get('player')]

    def get_player_age(self, player_id: str) -> Optional[int]:
        data = self._api_get(f"https://api.sofascore.com/api/v1/player/{player_id}")
        player_info = data.get('player', {})
        ts = player_info.get('dateOfBirthTimestamp')
        if ts:
            dob = datetime.fromtimestamp(ts)
            today = datetime.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return None

    def _find_stat(self, obj: Any, key: str) -> Optional[Any]:
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
        all_events = []
        cutoff = datetime(2024, 8, 1)

        for page in range(nb_pages):
            data = self._api_get(f"https://api.sofascore.com/api/v1/player/{player_id}/events/last/{page}")
            if not data: break
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
            stats_res = self._api_get(f"https://api.sofascore.com/api/v1/event/{eid}/player/{player_id}/statistics")
            stats = stats_res.get('statistics', {})
            if not stats: continue

            perf_res = self._api_get(f"https://api.sofascore.com/api/v1/event/{eid}/player/{player_id}/performance")

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
            logger.info(f"      ⚽ {m['date']} vs {m.get('away') if m.get('home') != player_name else m.get('home')} (Stats OK)")
            time.sleep(0.3)

        return results