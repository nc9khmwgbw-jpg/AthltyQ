import pandas as pd
import numpy as np
import sys
import os
import re
from pathlib import Path

# Import config
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

# Import FotMob
sys.path.append(str(config.SCRAPPING_DIR))
try:
    from fotmob import repair_player_with_fotmob
except ImportError:
    repair_player_with_fotmob = None

def verifier_et_reparer_avec_fotmob(df, player_name, team_name):
    """Logique de réparation FotMob systématique."""
    if not repair_player_with_fotmob: return df
    print(f"      🔄 Vérification FotMob pour {player_name}...")
    # ... (La logique de comparaison match par match et correction Goals/Assists)
    return df

def run_cleaner():
    print("🧹 Lancement du nettoyage global...")
    # ... (Parcours des dossiers SCRAPPING/data et création du MERGED_DATASET)
    
if __name__ == "__main__":
    run_cleaner()
