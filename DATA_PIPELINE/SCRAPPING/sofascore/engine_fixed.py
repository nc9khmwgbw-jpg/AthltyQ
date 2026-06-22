"""
engine_fixed.py — SofaScoreEngine corrigé
==========================================
Stratégie : intercepter les vraies réponses XHR/Fetch via le CDP (Chrome DevTools Protocol).
Le navigateur résout Cloudflare lui-même → on récupère les données JSON réelles.

Avantages vs l'approche __NEXT_DATA__ :
  - SofaScore ne met PAS les données de joueurs/matchs dans __NEXT_DATA__ (SPA React)
  - Le CDP capture toutes les réponses réseau APRÈS que Cloudflare est passé
  - Aucun appel API direct = pas de ban IP
"""

import json
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# Imports relatifs — adapter selon ton projet
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("SofaScoreEngine")


# ─────────────────────────────────────────────────────────────────────────────
# Patterns d'URL SofaScore API réels (capturés via DevTools)
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = "https://api.sofascore.com/api/v1"

PATTERNS = {
    "teams":   re.compile(r"/api/v1/unique-tournament/\d+/season/\d+/teams"),
    "squad":   re.compile(r"/api/v1/team/\d+/players"),
    "events":  re.compile(r"/api/v1/player/\d+/events/last/\d+"),
    "stats":   re.compile(r"/api/v1/event/\d+/player/\d+/statistics"),
    "player":  re.compile(r"/api/v1/player/\d+$"),
}


class SofaScoreEngine:
    """
    Moteur de scraping SofaScore — interception XHR via CDP.
    """

    def __init__(self, browser: Optional[SofaScoreBrowser] = None):
        self.browser = browser or SofaScoreBrowser(headless=True)
        self._network_enabled = False

    # ──────────────────────────────────────────────
    # SETUP CDP
    # ──────────────────────────────────────────────

    def _enable_network_interception(self) -> None:
        """Active la capture réseau via Chrome DevTools Protocol."""
        if self._network_enabled:
            return
        driver = self.browser.driver
        driver.execute_cdp_cmd("Network.enable", {})
        self._network_enabled = True
        logger.debug("CDP Network interception activée")

    def _get_xhr_responses(self, pattern: re.Pattern, timeout: int = 10) -> Optional[dict]:
        """
        Attend et capture la première réponse réseau dont l'URL matche `pattern`.
        Utilise un polling sur les logs de performance du navigateur.
        """
        driver = self.browser.driver
        deadline = time.time() + timeout
        seen_request_ids = set()

        while time.time() < deadline:
            try:
                logs = driver.execute_cdp_cmd("Network.getResponseBody", {})
            except Exception:
                pass

            # Récupérer tous les logs de performance
            perf_logs = driver.get_log("performance")
            for entry in perf_logs:
                try:
                    msg = json.loads(entry["message"])["message"]
                    method = msg.get("method", "")

                    if method == "Network.responseReceived":
                        params = msg["params"]
                        url = params["response"]["url"]
                        request_id = params["requestId"]

                        if pattern.search(url) and request_id not in seen_request_ids:
                            seen_request_ids.add(request_id)
                            try:
                                body = driver.execute_cdp_cmd(
                                    "Network.getResponseBody",
                                    {"requestId": request_id}
                                )
                                if body and body.get("body"):
                                    return json.loads(body["body"])
                            except Exception as e:
                                logger.debug(f"Erreur lecture body {request_id}: {e}")
                except Exception:
                    continue

            time.sleep(0.3)

        logger.warning(f"Timeout : aucune réponse trouvée pour pattern {pattern.pattern}")
        return None

    def _navigate(self, url: str, wait: float = 4.0) -> None:
        """Navigation simple avec attente."""
        if self.browser.driver is None:
            self.browser.start()
        driver = self.browser.driver
        self._enable_network_interception()
        driver.get(url)
        time.sleep(wait)

    # ──────────────────────────────────────────────
    # API PUBLIQUE
    # ──────────────────────────────────────────────

    def get_teams_in_league(self, tournament_id: int, season_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Récupère les équipes d'une ligue.
        
        SofaScore nécessite un season_id. Si non fourni, on le récupère
        depuis la page du tournoi via interception réseau.
        
        Args:
            tournament_id: ID SofaScore du tournoi (ex: 17 pour La Liga)
            season_id: ID de la saison (ex: 61643). Si None, auto-détecté.
        """
        # Étape 1 : récupérer le season_id si non fourni
        if season_id is None:
            season_id = self._get_current_season_id(tournament_id)
            if season_id is None:
                logger.error(f"Impossible de récupérer le season_id pour tournoi {tournament_id}")
                return []

        # Étape 2 : naviguer vers la page standings (déclenche l'appel /teams)
        standings_url = (
            f"https://www.sofascore.com/tournament/football/spain/laliga/{tournament_id}/"
            f"#id:{season_id},tab:standings"
        )
        self._navigate(standings_url, wait=5)

        # Étape 3 : intercepter la réponse /teams
        data = self._get_xhr_responses(PATTERNS["teams"], timeout=12)

        if not data:
            # Fallback : essayer via l'URL standings qui déclenche /standings avec les équipes
            logger.warning("Fallback : extraction équipes depuis /standings")
            data = self._get_xhr_responses(
                re.compile(r"/api/v1/unique-tournament/\d+/season/\d+/standings"),
                timeout=8
            )

        if not data:
            return []

        # Parser la réponse — format SofaScore : {"teams": [...]}
        teams = []
        raw_teams = data.get("teams", [])
        for t in raw_teams:
            tid = t.get("id")
            name = t.get("name")
            slug = t.get("slug", "")
            if tid and name:
                teams.append({"id": tid, "name": name, "slug": slug})

        logger.info(f"✅ {len(teams)} équipes récupérées pour tournoi {tournament_id}")
        return teams

    def _get_current_season_id(self, tournament_id: int) -> Optional[int]:
        """Récupère le season_id courant depuis la page du tournoi."""
        url = f"https://www.sofascore.com/tournament/football/spain/laliga/{tournament_id}/"
        self._navigate(url, wait=5)

        pattern = re.compile(r"/api/v1/unique-tournament/\d+/seasons")
        data = self._get_xhr_responses(pattern, timeout=10)

        if not data:
            return None

        seasons = data.get("seasons", [])
        if seasons:
            # Première saison = la plus récente
            return seasons[0].get("id")
        return None

    def get_players_in_team(self, team_id: int) -> List[Dict[str, Any]]:
        """
        Récupère les joueurs d'une équipe.
        
        URL de navigation : https://www.sofascore.com/team/football/{slug}/{team_id}
        L'appel API intercepté : /api/v1/team/{team_id}/players
        
        Note: le slug n'est pas obligatoire pour la navigation (redirige automatiquement).
        """
        # Naviguer vers la page équipe
        team_url = f"https://www.sofascore.com/team/football/team/{team_id}"
        self._navigate(team_url, wait=5)

        # Intercepter /api/v1/team/{team_id}/players
        data = self._get_xhr_responses(PATTERNS["squad"], timeout=10)

        if not data:
            logger.warning(f"Aucun joueur trouvé pour équipe {team_id}")
            return []

        players = []
        # Format : {"players": [{"player": {...}, "team": {...}}, ...]}
        for entry in data.get("players", []):
            p = entry.get("player", entry)
            pid = p.get("id")
            name = p.get("name") or p.get("shortName")
            pos = p.get("position")
            if pid and name:
                players.append({
                    "id": pid,
                    "name": name,
                    "position": pos,
                    "slug": p.get("slug", ""),
                })

        logger.info(f"✅ {len(players)} joueurs trouvés pour équipe {team_id}")
        return players

    def extract_player_matches(
        self,
        player_id: int,
        player_name: str,
        nb_pages: int = 1,
        limit: int = 15,
        last_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extrait les derniers matchs d'un joueur avec ses statistiques.
        
        SofaScore pagine les matchs par tranches de 10 (page 0, 1, 2...).
        On navigue vers la page joueur qui déclenche les appels :
          GET /api/v1/player/{id}/events/last/0
          GET /api/v1/player/{id}/events/last/1  (si scroll / pagination)
        """
        cutoff = datetime(2024, 8, 1)
        all_events = []

        # Construire le slug depuis le nom
        slug = player_name.lower().replace(" ", "-")
        player_url = f"https://www.sofascore.com/player/{slug}/{player_id}"

        for page in range(nb_pages):
            if page == 0:
                self._navigate(player_url, wait=5)
            else:
                # Déclencher la pagination via JS (scroll ou click "voir plus")
                self._trigger_next_page(page)

            # Intercepter les événements de la page courante
            page_pattern = re.compile(
                rf"/api/v1/player/{player_id}/events/last/{page}"
            )
            data = self._get_xhr_responses(page_pattern, timeout=10)

            if not data:
                logger.debug(f"Pas de données page {page} pour joueur {player_id}")
                break

            events = data.get("events", [])
            if not events:
                break

            for ev in events:
                if len(all_events) >= limit:
                    break

                # Ne garder que les matchs terminés
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

        # Récupérer les stats pour chaque match
        results = []
        age = self._get_player_age_from_page(player_id)

        for m in targets:
            stats = self._get_match_player_stats(m["id"], player_id, m["home"], m["away"])
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
            time.sleep(0.8)  # Pause polie entre les matchs

        return results

    def _trigger_next_page(self, page: int) -> None:
        """
        Simule un scroll vers le bas pour déclencher le chargement de la page suivante.
        SofaScore charge les matchs suivants au scroll.
        """
        driver = self.browser.driver
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            # Double scroll pour forcer le chargement
            driver.execute_script("window.scrollBy(0, -200);")
            time.sleep(0.5)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        except Exception as e:
            logger.debug(f"Erreur scroll page {page}: {e}")

    def _get_match_player_stats(
        self,
        event_id: int,
        player_id: int,
        home: str,
        away: str,
    ) -> Optional[dict]:
        """
        Récupère les stats d'un joueur pour un match donné.
        Navigation vers la page du match → interception /player/{id}/statistics
        """
        home_slug = home.lower().replace(" ", "-")
        away_slug = away.lower().replace(" ", "-")
        match_url = (
            f"https://www.sofascore.com/{home_slug}-{away_slug}/{event_id}/"
            f"#id:{event_id},tab:lineups"
        )
        self._navigate(match_url, wait=4)

        pattern = re.compile(rf"/api/v1/event/{event_id}/player/{player_id}/statistics")
        data = self._get_xhr_responses(pattern, timeout=8)

        if not data:
            # Fallback : intercepter les stats générales du match
            all_stats_pattern = re.compile(rf"/api/v1/event/{event_id}/lineups")
            data = self._get_xhr_responses(all_stats_pattern, timeout=6)
            if data:
                return self._extract_player_from_lineups(data, player_id)
            return None

        stats_raw = data.get("statistics", data)
        return self._normalize_stats(stats_raw)

    def _extract_player_from_lineups(self, lineups_data: dict, player_id: int) -> Optional[dict]:
        """Extrait les stats d'un joueur depuis les lineups du match."""
        try:
            for side in ("home", "away"):
                players = lineups_data.get(side, {}).get("players", [])
                for entry in players:
                    p = entry.get("player", {})
                    if p.get("id") == player_id:
                        stats = entry.get("statistics", {})
                        return self._normalize_stats(stats) if stats else None
        except Exception as e:
            logger.debug(f"Erreur extraction lineups: {e}")
        return None

    def _get_player_age_from_page(self, player_id: int) -> Optional[int]:
        """Récupère l'âge depuis les données joueur interceptées."""
        pattern = re.compile(rf"/api/v1/player/{player_id}$")
        data = self._get_xhr_responses(pattern, timeout=5)
        if not data:
            return None
        try:
            dob = data.get("player", {}).get("dateOfBirthTimestamp")
            if dob:
                birth = datetime.fromtimestamp(dob)
                return (datetime.now() - birth).days // 365
        except Exception:
            pass
        return None

    def _normalize_stats(self, stats: dict) -> dict:
        """Normalise les statistiques au format AthlytIQ."""
        return {
            "Rating":           round(float(stats.get("rating", 0)), 2) if stats.get("rating") else None,
            "Minutes_Played":   int(stats.get("minutesPlayed", 0)),
            "distanceRun":      float(stats.get("totalDistance", stats.get("distanceRun", 0)) or 0) or None,
            "sprints":          int(stats.get("sprints", stats.get("highIntensityRuns", 0)) or 0) or None,
            "kpi_work_rate":    self._calc_work_rate(stats),
            "Goals":            int(stats.get("goals", 0)),
            "Assists":          int(stats.get("assists", 0)),
            "Expected_Goals":   round(float(stats.get("expectedGoals", 0) or 0), 2),
            "Expected_Assists": round(float(stats.get("expectedAssists", 0) or 0), 2),
            "Accurate_Passes":  int(stats.get("accuratePass", stats.get("accuratePasses", 0)) or 0),
            "Total_Passes":     int(stats.get("totalPass", stats.get("totalPasses", 0)) or 0),
            "Key_Passes":       int(stats.get("keyPass", stats.get("keyPasses", 0)) or 0),
            "Tackles":          int(stats.get("tackle", stats.get("tackles", 0)) or 0),
            "Interceptions":    int(stats.get("interceptionWon", stats.get("interceptions", 0)) or 0),
            "Clearances":       int(stats.get("clearance", stats.get("clearances", 0)) or 0),
            "Ball_Recovery":    int(stats.get("ballRecovery", stats.get("ballRecoveries", 0)) or 0),
            "Touches":          int(stats.get("touches", 0)),
        }

    def _calc_work_rate(self, stats: dict) -> float:
        dist = float(stats.get("totalDistance", stats.get("distanceRun", 0)) or 0)
        mins = int(stats.get("minutesPlayed", 0))
        return round(dist / mins, 2) if (dist > 0 and mins > 0) else 0.0

    def get_player_age(self, player_id: int) -> Optional[int]:
        """API publique pour récupérer l'âge d'un joueur."""
        return self._get_player_age_from_page(player_id)
