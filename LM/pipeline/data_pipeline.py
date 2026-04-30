import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Chemins
ROOT = Path(__file__).resolve().parents[2]
CLEAN_DATA_PATH = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"
OUTPUT_PATH = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "features_dataset.csv"

def engineering_features(df):
    """Calcule les moyennes mobiles et les indicateurs de forme."""
    print("📊 Calcul des features (Moyennes mobiles, Forme)...")
    
    # S'assurer que les dates sont au bon format
    df['Match_Date'] = pd.to_datetime(df['Match_Date'])
    df = df.sort_values(['Nom', 'Match_Date'])
    
    # Calcul des moyennes mobiles (5 derniers matchs)
    features_to_roll = ['Rating', 'Goals', 'Assists', 'Expected_Goals', 'Expected_Assists', 'Minutes_Played']
    
    for col in features_to_roll:
        if col in df.columns:
            df[f'avg_{col}_5'] = df.groupby('Nom')[col].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    
    # Calcul de la charge de travail (proxy pour la fatigue)
    if 'Minutes_Played' in df.columns:
        df['cumulative_minutes_15d'] = df.groupby('Nom')['Minutes_Played'].transform(
            lambda x: x.rolling(window=3, min_periods=1).sum()
        )
        
    return df

def run_merge_pipeline():
    print("🚀 Lancement du pipeline de données IA...")
    
    if not CLEAN_DATA_PATH.exists():
        print(f"❌ Erreur : Le fichier {CLEAN_DATA_PATH} est introuvable.")
        return
    
    # 1. Chargement des données propres
    df = pd.read_csv(CLEAN_DATA_PATH)
    
    # 2. Feature Engineering
    df = engineering_features(df)
    
    # 3. Nettoyage final des NaNs créés par les moyennes mobiles
    df = df.fillna(0)
    
    # 4. Sauvegarde pour l'entraînement
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Dataset de features généré : {len(df)} lignes.")
    print(f"📂 Emplacement : {OUTPUT_PATH}")

if __name__ == "__main__":
    run_merge_pipeline()
