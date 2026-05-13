import pandas as pd
import numpy as np
from typing import Optional

def estimer_physique_manquant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modèle d'imputation physique avancé (Cerveau AthlytIQ).
    Estime distanceRun et sprints à partir des minutes jouées et des duels aériens.
    """
    if df.empty:
        return df

    # 1. Normalisation des noms de colonnes
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
                df[target] = 0

    # Conversion numérique forcée
    for col in mapping.keys():
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. Modèle d'imputation
    mask = df['Minutes_Played'] > 0
    
    # Récupération sécurisée des duels aériens (non utilisés pour d'autres features)
    aerial_won  = df.get('Aerial_Duels_Won',  pd.Series(0, index=df.index)).fillna(0)
    aerial_lost = df.get('Aerial_Duels_Lost', pd.Series(0, index=df.index)).fillna(0)

    # DISTANCE (mètres) — Basé sur le temps de jeu et les duels gagnés (intensité)
    df.loc[mask, 'distanceRun'] = (
        (df.loc[mask, 'Minutes_Played'] * 108) +
        (pd.to_numeric(aerial_won[mask], errors='coerce').fillna(0) * 22.0)
    ).round(0)

    # SPRINTS — Basé sur le temps de jeu et les duels perdus (pression/accélérations)
    df.loc[mask, 'sprints'] = (
        (df.loc[mask, 'Minutes_Played'] / 5.0) +
        (pd.to_numeric(aerial_lost[mask], errors='coerce').fillna(0) * 1.5)
    ).round(0)

    # 3. KPIs dérivés
    df['kpi_work_rate']   = (df['distanceRun'] / df['Minutes_Played'].replace(0, 1)).round(2)
    dist_km = df['distanceRun'] / 1000
    df['kpi_explosivity'] = (df['sprints'] / dist_km.replace(0, 1)).round(2)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df
