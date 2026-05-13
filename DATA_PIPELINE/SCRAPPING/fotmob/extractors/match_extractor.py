from typing import List, Dict, Any

class FotMobMatchExtractor:
    """Extracteur de données de matchs depuis le JSON de FotMob."""

    def extract(self, player_data: dict) -> List[Dict[str, Any]]:
        """Extrait l'historique des matchs depuis le JSON intercepté."""
        matches = []

        raw_matches = (
            player_data.get("recentMatches")
            or player_data.get("lastMatches")
            or player_data.get("stats", {}).get("matchStats", [])
            or []
        )

        for m in raw_matches[:15]:
            try:
                date_raw = (
                    m.get("matchDate", {}).get("utcTime", "")
                    or m.get("date", "")
                    or m.get("matchTms", "")
                )
                date = date_raw.split("T")[0] if "T" in str(date_raw) else str(date_raw)[:10]

                # Stats de base
                goals   = int(m.get("goals", m.get("g", 0)) or 0)
                assists = int(m.get("assists", m.get("a", 0)) or 0)
                minutes = int(m.get("minutesPlayed", m.get("minsPlayed", m.get("mp", 0))) or 0)
                rating  = float(m.get("ratingProps", {}).get("num", m.get("rating", 0.0)) or 0.0)
                shots   = int(m.get("shots", m.get("totalShots", 0)) or 0)
                touches = int(m.get("touches", 0) or 0)
                
                # Passes
                key_passes = int(m.get("keyPasses", m.get("keypasses", 0)) or 0)
                total_passes = int(m.get("totalPasses", m.get("passes", {}).get("total", 0)) or 0)
                acc_passes = int(m.get("accuratePasses", m.get("passes", {}).get("accurate", 0)) or 0)

                # Stats Avancées
                xg = float(m.get("expectedGoals", m.get("xg", 0.0)) or 0.0)
                xa = float(m.get("expectedAssists", m.get("xa", 0.0)) or 0.0)
                tackles = int(m.get("tackles", m.get("tackle", 0)) or 0)
                interceptions = int(m.get("interceptions", 0) or 0)
                clearances = int(m.get("clearances", 0) or 0)
                recoveries = int(m.get("ballRecovery", m.get("recoveries", 0)) or 0)

                home_team = m.get("homeTeam", {}).get("name", "") if isinstance(m.get("homeTeam"), dict) else m.get("homeTeamName", "")
                away_team = m.get("awayTeam", {}).get("name", "") if isinstance(m.get("awayTeam"), dict) else m.get("awayTeamName", "")

                matches.append({
                    "Match_Date":       date,
                    "Home_Team":        home_team,
                    "Away_Team":        away_team,
                    "Goals":            goals,
                    "Assists":          assists,
                    "Minutes_Played":   minutes,
                    "Rating":           rating,
                    "Shots":            shots,
                    "Touches":          touches,
                    "Key_Passes":       key_passes,
                    "Total_Passes":     total_passes,
                    "Accurate_Passes":  acc_passes,
                    "Expected_Goals":   xg,
                    "Expected_Assists": xa,
                    "Tackles":          tackles,
                    "Interceptions":    interceptions,
                    "Clearances":       clearances,
                    "Ball_Recovery":    recoveries
                })
            except Exception:
                continue

        return matches
