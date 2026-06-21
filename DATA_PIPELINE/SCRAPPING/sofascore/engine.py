"""
engine_v3.py — SofaScoreEngine v3
===================================
Corrections vs v2 :

1. _goto_home() : timeout page ignoré proprement (TimeoutException est normale
   sur SofaScore — la page charge en JS mais Selenium timeout sur le DOM load)

2. _fetch() : utilise XHR synchrone comme fallback si fetch async échoue,
   et loggue le vrai message d'erreur

3. get_teams_in_league() : navigue d'abord vers la PAGE du tournoi pour
   avoir le bon Referer et les cookies avant les appels API

4. Délai après timeout réduit à 3s (pas besoin d'attendre 5s si timeout)
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from curl_cffi import requests
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("SofaScoreEngine")

API_BASE = "https://api.sofascore.com/api/v1"

class SofaScoreEngine:

    def __init__(self, browser=None):
        # We ignore the browser parameter if it's passed for backward compatibility
        self._season_cache: Dict[int, int] = {}
        
        # Initialize curl_cffi session with Dalvik UA to bypass Cloudflare
        self.session = requests.Session(impersonate="chrome110")
        self.session.headers.update({
            "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 12; Pixel 6 Build/SQ3A.220705.004)",
            "Accept-Encoding": "gzip",
            "Host": "api.sofascore.com",
            "Connection": "Keep-Alive"
        })
        logger.info("⚡ Moteur SofaScore initialisé avec bypass Dalvik Mobile")

    # ──────────────────────────────────────────────────────────────
    # FETCH — méthode principale
    # ──────────────────────────────────────────────────────────────

    def _fetch(self, endpoint: str) -> Optional[dict]:
        """
        Appel API direct via curl_cffi avec User-Agent Android.
        Contourne complètement Cloudflare Turnstile.
        """
        url = f"{API_BASE}{endpoint}"
        
        try:
            r = self.session.get(url, timeout=15)
            
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception as e:
                    logger.debug(f"JSON Decode Error sur {endpoint}: {e}")
                    return None
            elif r.status_code == 404:
                logger.debug(f"Endpoint non trouvé (404) : {endpoint}")
            else:
                logger.debug(f"Status {r.status_code} sur {endpoint}")
                
        except Exception as e:
            logger.debug(f"Request error sur {endpoint}: {type(e).__name__}: {str(e)[:80]}")

        return None

    # ──────────────────────────────────────────────────────────────
    # SAISONS
    # ──────────────────────────────────────────────────────────────

    def _get_season_id(self, tournament_id: int) -> Optional[int]:
        if tournament_id in self._season_cache:
            return self._season_cache[tournament_id]

        data = self._fetch(f"/unique-tournament/{tournament_id}/seasons")
        if not data:
            logger.error(f"❌ Impossible de récupérer les saisons pour tournoi {tournament_id}")
            return None

        seasons = data.get("seasons", [])
        if not seasons:
            logger.error(f"❌ Aucune saison dans la réponse pour tournoi {tournament_id}")
            return None

        season_id = seasons[0].get("id")
        season_name = seasons[0].get("name") or seasons[0].get("year", "?")
        logger.info(f"📅 Saison courante : {season_name} (ID: {season_id})")
        self._season_cache[tournament_id] = season_id
        return season_id

    # ──────────────────────────────────────────────────────────────
    # ÉQUIPES
    # ──────────────────────────────────────────────────────────────

    def get_teams_in_league(self, tournament_id: int) -> List[Dict[str, Any]]:
        """Récupère les équipes d'une ligue via l'API."""
        season_id = self._get_season_id(tournament_id)
        if not season_id:
            return []

        teams_raw = []

        # Option A : /teams direct
        data = self._fetch(f"/unique-tournament/{tournament_id}/season/{season_id}/teams")
        if data and "teams" in data:
            teams_raw = data["teams"]
            logger.info(f"✅ Équipes via /teams : {len(teams_raw)}")

        # Option B : standings
        if not teams_raw:
            logger.info("↳ Fallback via /standings/total...")
            data = self._fetch(f"/unique-tournament/{tournament_id}/season/{season_id}/standings/total")
            if data:
                for standing in data.get("standings", []):
                    for row in standing.get("rows", []):
                        team = row.get("team", {})
                        if team.get("id"):
                            teams_raw.append(team)
                if teams_raw:
                    logger.info(f"✅ Équipes via /standings : {len(teams_raw)}")

        # Option C : top-teams
        if not teams_raw:
            logger.info("↳ Fallback via /top-teams...")
            data = self._fetch(f"/unique-tournament/{tournament_id}/season/{season_id}/top-teams/overall")
            if data:
                for entry in data.get("topTeams", []):
                    team = entry.get("team", entry)
                    if team.get("id"):
                        teams_raw.append(team)
                if teams_raw:
                    logger.info(f"✅ Équipes via /top-teams : {len(teams_raw)}")

        if not teams_raw:
            logger.error(f"❌ Aucune équipe trouvée pour tournoi {tournament_id} / saison {season_id}")
            return []

        # Dédupliquer et normaliser
        seen, result = set(), []
        for t in teams_raw:
            tid = t.get("id")
            name = t.get("name") or t.get("shortName")
            if tid and name and tid not in seen:
                seen.add(tid)
                result.append({"id": tid, "name": name, "slug": t.get("slug", "")})

        logger.info(f"✅ {len(result)} équipes récupérées")
        return result

    # ──────────────────────────────────────────────────────────────
    # JOUEURS
    # ──────────────────────────────────────────────────────────────

    def get_players_in_team(self, team_id: int) -> List[Dict[str, Any]]:
        data = self._fetch(f"/team/{team_id}/players")
        if not data:
            logger.warning(f"Aucune donnée joueurs pour équipe {team_id}")
            return []

        players = []
        for entry in data.get("players", []):
            p = entry.get("player", entry)
            pid = p.get("id")
            name = p.get("name") or p.get("shortName")
            if pid and name:
                players.append({
                    "id": pid,
                    "name": name,
                    "position": p.get("position"),
                    "slug": p.get("slug", ""),
                })

        logger.info(f"✅ {len(players)} joueurs pour équipe {team_id}")
        return players

    # ──────────────────────────────────────────────────────────────
    # MATCHS
    # ──────────────────────────────────────────────────────────────

    def extract_player_matches(
        self,
        player_id: int,
        player_name: str,
        nb_pages: int = 1,
        limit: int = 15,
        last_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        cutoff = datetime(2024, 8, 1)
        all_events = []

        for page in range(nb_pages):
            data = self._fetch(f"/player/{player_id}/events/last/{page}")
            if not data:
                break
            events = data.get("events", [])
            if not events:
                break

            for ev in events:
                if ev.get("status", {}).get("type") != "finished":
                    continue
                ts = ev.get("startTimestamp", 0)
                dt = datetime.fromtimestamp(ts)
                dt_str = dt.strftime("%Y-%m-%d")
                if dt < cutoff:
                    continue
                if last_date and dt_str <= last_date:
                    continue
                all_events.append({
                    "id": ev["id"],
                    "date": dt_str,
                    "dt_obj": dt,
                    "home": ev.get("homeTeam", {}).get("name", "Unknown"),
                    "away": ev.get("awayTeam", {}).get("name", "Unknown"),
                    "tournament": ev.get("tournament", {}).get("name", ""),
                })
            if len(all_events) >= limit:
                break

        if not all_events:
            return []

        all_events.sort(key=lambda x: x["dt_obj"], reverse=True)
        targets = all_events[:limit]
        age = self._get_player_age(player_id)

        results = []
        for m in targets:
            stats = self._get_event_player_stats(m["id"], player_id)
            if stats:
                results.append({
                    "Nom": player_name,
                    "Age": age,
                    "Match_Date": m["date"],
                    "Home_Team": m["home"],
                    "Away_Team": m["away"],
                    "Tournament": m["tournament"],
                    **stats,
                })
                logger.info(f"      ⚽ {m['date']} {m['home']} vs {m['away']} ✓")
            time.sleep(0.05) # Délai très court pour éviter de surcharger

        return results

    def _get_event_player_stats(self, event_id: int, player_id: int) -> Optional[dict]:
        data = self._fetch(f"/event/{event_id}/player/{player_id}/statistics")
        if not data:
            return None
        stats = data.get("statistics", data)
        return self._normalize_stats(stats) if stats else None

    def _get_player_age(self, player_id: int) -> Optional[int]:
        data = self._fetch(f"/player/{player_id}")
        if not data:
            return None
        try:
            dob = data.get("player", data).get("dateOfBirthTimestamp")
            if dob:
                return (datetime.now() - datetime.fromtimestamp(dob)).days // 365
        except Exception:
            pass
        return None

    def get_player_age(self, player_id: int) -> Optional[int]:
        return self._get_player_age(player_id)

    # ──────────────────────────────────────────────────────────────
    # NORMALISATION
    # ──────────────────────────────────────────────────────────────

    def _normalize_stats(self, stats: dict) -> dict:
        def si(v): return int(v) if v else 0
        def sf(v): return float(v) if v else None
        return {
            "Rating":           round(float(stats["rating"]), 2) if stats.get("rating") else None,
            "Minutes_Played":   si(stats.get("minutesPlayed")),
            "distanceRun":      sf(stats.get("totalDistance") or stats.get("distanceRun")),
            "sprints":          si(stats.get("sprints") or stats.get("highIntensityRuns")),
            "kpi_work_rate":    self._calc_work_rate(stats),
            "Goals":            si(stats.get("goals")),
            "Assists":          si(stats.get("assists")),
            "Expected_Goals":   round(float(stats.get("expectedGoals") or 0), 2),
            "Expected_Assists": round(float(stats.get("expectedAssists") or 0), 2),
            "Accurate_Passes":  si(stats.get("accuratePass") or stats.get("accuratePasses")),
            "Total_Passes":     si(stats.get("totalPass") or stats.get("totalPasses")),
            "Key_Passes":       si(stats.get("keyPass") or stats.get("keyPasses")),
            "Tackles":          si(stats.get("tackle") or stats.get("tackles")),
            "Interceptions":    si(stats.get("interceptionWon") or stats.get("interceptions")),
            "Clearances":       si(stats.get("clearance") or stats.get("clearances")),
            "Ball_Recovery":    si(stats.get("ballRecovery") or stats.get("ballRecoveries")),
            "Touches":          si(stats.get("touches")),
        }

    def _calc_work_rate(self, stats: dict) -> float:
        dist = float(stats.get("totalDistance") or stats.get("distanceRun") or 0)
        mins = int(stats.get("minutesPlayed") or 0)
        return round(dist / mins, 2) if (dist > 0 and mins > 0) else 0.0

    def _find_stat(self, obj: Any, key: str) -> Optional[Any]:
        if not obj: return None
        if isinstance(obj, dict) and key in obj: return obj[key]
        return None

