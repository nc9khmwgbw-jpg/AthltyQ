import pandas as pd
import numpy as np
from pathlib import Path

def estimer_physique_manquant(df):
    """
    Modèle d'imputation physique avancé.
    Gère les noms de colonnes variables (wonContest/Successful_Dribbles, etc.)
    """
    if df.empty:
        return df

    print("🧠 Application du modèle physique Elite (Distance & Sprints)...")
    
    # 1. Normalisation des noms de colonnes (pour éviter les erreurs 'KeyError')
    # On crée des alias pour les colonnes essentielles
    mapping = {
        'Minutes_Played': ['Minutes_Played', 'minutesPlayed', 'mins_played'],
        'Touches': ['Touches', 'touches'],
        'Ball_Recovery': ['Ball_Recovery', 'ballRecovery', 'recoveries'],
        'Successful_Dribbles': ['Successful_Dribbles', 'wonContest', 'dribbles_won'],
        'Interceptions': ['Interceptions', 'interceptionWon', 'interceptions'],
        'Tackles': ['Tackles', 'tackle', 'tackles']
    }
    
    for target, aliases in mapping.items():
        if target not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df[target] = df[alias]
                    break
            if target not in df.columns:
                df[target] = 0 # Par défaut si aucune colonne trouvée

    # On s'assure que tout est numérique
    for col in mapping.keys():
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. Calcul du modèle sur les joueurs actifs
    # RÈGLE D'ORTHOGONALITÉ :
    # - Ball_Recovery   → assigné à Defensive_Actions    (feature_engineering.py)
    # - Touches         → assigné à Possession_Security  (feature_engineering.py)
    # - Interceptions   → assigné à Defensive_Actions    (feature_engineering.py)
    # - Successful_Dribbles → assigné à Dribbles_P90    (feature_engineering.py)
    # Ici on utilise UNIQUEMENT : Minutes_Played, Aerial_Duels (non utilisés ailleurs)
    mask = df['Minutes_Played'] > 0

    # Variables libres — assignation stricte :
    # Aerial_Duels_Won  → distanceRun UNIQUEMENT
    # Aerial_Duels_Lost → sprints UNIQUEMENT
    aerial_won  = df.get('Aerial_Duels_Won',  pd.Series(0, index=df.index)).fillna(0)
    aerial_lost = df.get('Aerial_Duels_Lost', pd.Series(0, index=df.index)).fillna(0)

    # DISTANCE — Minutes_Played + Aerial_Duels_Won (exclusif)
    df.loc[mask, 'distanceRun'] = (
        (df.loc[mask, 'Minutes_Played'] * 108) +
        (aerial_won[mask] * 22.0)
    ).round(0)

    # SPRINTS — Minutes_Played + Aerial_Duels_Lost (exclusif)
    df.loc[mask, 'sprints'] = (
        (df.loc[mask, 'Minutes_Played'] / 5.0) +
        (aerial_lost[mask] * 1.5)
    ).round(0)

    # 3. KPIs dérivés (utilise uniquement distanceRun et sprints — eux-mêmes orthogonaux)
    df['kpi_work_rate']   = (df['distanceRun'] / df['Minutes_Played']).round(2).fillna(0)
    dist_km = df['distanceRun'] / 1000
    df['kpi_explosivity'] = (df['sprints'] / dist_km.replace(0, 1)).round(2).fillna(0)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

if __name__ == "__main__":
    # Test
    FILE = Path(__file__).resolve().parents[3] / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"
    if FILE.exists():
        df = pd.read_csv(FILE)
        df = estimer_physique_manquant(df)
        df.to_csv(FILE, index=False)
        print("✅ Données physiques injectées.")
