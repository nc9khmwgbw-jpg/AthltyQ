# common/config.py
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "raw"
LOG_DIR = ROOT_DIR / "logs"

# Configuration des 16 Ligues AthlytIQ
LEAGUES = {
    "LaLiga": {"id": 8, "url": "https://www.sofascore.com/tournament/football/spain/laliga/8"},
    "Premier": {"id": 17, "url": "https://www.sofascore.com/tournament/football/england/premier-league/17"},
    "Ligue 1": {"id": 34, "url": "https://www.sofascore.com/tournament/football/france/ligue-1/34"},
    "SerieA": {"id": 23, "url": "https://www.sofascore.com/tournament/football/italy/serie-a/23"},
    "Bundesliga": {"id": 35, "url": "https://www.sofascore.com/tournament/football/germany/bundesliga/35"},
    "ChampionsLeague": {"id": 7, "url": "https://www.sofascore.com/tournament/football/europe/uefa-champions-league/7"},
    "Eredivisie": {"id": 37, "url": "https://www.sofascore.com/tournament/football/netherlands/eredivisie/37"},
    "ProLeague": {"id": 38, "url": "https://www.sofascore.com/tournament/football/belgium/jupiler-pro-league/38"},
    "SaudiProLeague": {"id": 955, "url": "https://www.sofascore.com/tournament/football/saudi-arabia/saudi-professional-league/955"},
    "Championship": {"id": 18, "url": "https://www.sofascore.com/tournament/football/england/championship/18"},
    "SuperLeagueGreece": {"id": 670, "url": "https://www.sofascore.com/tournament/football/greece/super-league/670"},
    "SuperLigTurquie": {"id": 52, "url": "https://www.sofascore.com/tournament/football/turkey/super-lig/52"},
    "ScottishPrem": {"id": 36, "url": "https://www.sofascore.com/tournament/football/scotland/premiership/36"},
    "LigaPortugal": {"id": 238, "url": "https://www.sofascore.com/tournament/football/portugal/liga-portugal/238"},
    "MLS": {"id": 242, "url": "https://www.sofascore.com/tournament/football/usa/mls/242"},
    "UAEProLeague": {"id": 1322, "url": "https://www.sofascore.com/tournament/football/united-arab-emirates/uae-pro-league/1322"},
}

# Délais par défaut
DEFAULT_DELAY = 1.5
TIMEOUT = 15
