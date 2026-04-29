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
    mask = df['Minutes_Played'] > 0
    
    # DISTANCE (mètres)
    df.loc[mask, 'distanceRun'] = (
        (df.loc[mask, 'Minutes_Played'] * 105) + 
        (df.loc[mask, 'Touches'] * 4.2) + 
        (df.loc[mask, 'Ball_Recovery'] * 12.5)
    ).round(0)
    
    # SPRINTS
    df.loc[mask, 'sprints'] = (
        (df.loc[mask, 'Minutes_Played'] / 5.2) + 
        (df.loc[mask, 'Successful_Dribbles'] * 2.8) + 
        (df.loc[mask, 'Interceptions'] * 1.4)
    ).round(0)
    
    # 3. KPIs
    df['kpi_work_rate'] = (df['distanceRun'] / df['Minutes_Played']).round(2).fillna(0)
    dist_km = df['distanceRun'] / 1000
    df['kpi_explosivity'] = (df['sprints'] / dist_km).round(2).fillna(0)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

if __name__ == "__main__":
    # Test
    FILE = Path("DATA_PIPELINE/data/clean/merged_dataset_clean.csv")
    if FILE.exists():
        df = pd.read_csv(FILE)
        df = estimer_physique_manquant(df)
        df.to_csv(FILE, index=False)
        print("✅ Données physiques injectées.")
