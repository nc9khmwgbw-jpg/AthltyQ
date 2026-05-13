import sys
import os
from pathlib import Path

# Ajout du chemin racine pour éviter les erreurs d'import
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))

import pandas as pd
import numpy as np
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("Reconciler-Elite")

class DataReconciler:
    """Audit global de haute précision pour le dataset AthlytIQ."""

    def __init__(self):
        self.root = Path(__file__).resolve().parents[2]
        self.clean_csv = self.root / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"

    def run_audit(self):
        if not self.clean_csv.exists():
            logger.error(f"❌ Erreur critique : {self.clean_csv} est introuvable.")
            return

        logger.info("⚙️  Chargement du dataset en mode haute performance...")
        # low_memory=False pour éviter les erreurs de type sur les gros fichiers
        df = pd.read_csv(self.clean_csv, low_memory=False)
        
        # Correction immédiate : On s'assure que les dates sont des objets datetime
        df['Match_Date'] = pd.to_datetime(df['Match_Date'], errors='coerce')
        
        total_rows = len(df)
        # On utilise 'Nom' comme identifiant unique (plus stable)
        unique_players = df['Nom'].nunique()
        
        logger.info(f"📊 Analyse en cours sur {total_rows} lignes...")

        # 1. Audit des Stats Physiques (Le cœur de l'IA)
        # On ne compte comme "manquant" que si le joueur a joué (Minutes > 0) 
        # mais qu'on n'a pas de stats (distance <= 0)
        mask_phys_missing = (df['Minutes_Played'] > 0) & ((df['distanceRun'].isna()) | (df['distanceRun'] <= 0))
        missing_phys_count = mask_phys_missing.sum()
        
        # 2. Audit de l'Historique (Minimum 10 matchs DISTINCTS)
        # On groupe par nom et on compte les dates uniques
        history_counts = df.groupby('Nom')['Match_Date'].nunique()
        incomplete_players = history_counts[history_counts < 10]
        
        # 3. Détection des Doublons (Le bug silencieux le plus dangereux)
        # Un doublon = même joueur + même date
        duplicates = df.duplicated(subset=['Nom', 'Match_Date'], keep='first')
        dup_count = duplicates.sum()

        # 4. Audit des Ligues (Répartition)
        league_dist = df['League'].value_counts() if 'League' in df.columns else "N/A"

        # --- AFFICHAGE DU RAPPORT ÉLITE ---
        print("\n" + "═"*60)
        print(" 🏥  ATHLYTIQ DATA HEALTH TERMINAL (v2.1)")
        print("═"*60)
        print(f"  ● VOLUME TOTAL      : {total_rows} matchs")
        print(f"  ● JOUEURS UNIQUES   : {unique_players}")
        print("─"*60)
        
        # Section Physique
        phys_pct = (missing_phys_count / total_rows) * 100
        color_phys = "✅" if phys_pct < 5 else "⚠️" if phys_pct < 20 else "❌"
        print(f"  {color_phys} STATS PHYSIQUES  : {missing_phys_count} manquants ({phys_pct:.1f}%)")
        
        # Section Historique
        hist_pct = (len(incomplete_players) / unique_players) * 100
        color_hist = "✅" if hist_pct < 10 else "⚠️"
        print(f"  {color_hist} HISTORIQUE (min 10) : {len(incomplete_players)} joueurs incomplets ({hist_pct:.1f}%)")
        
        # Section Doublons
        color_dup = "✅" if dup_count == 0 else "🚫"
        print(f"  {color_dup} DOUBLONS RÉSEAU   : {dup_count} détectés")
        
        print("─"*60)
        print("  📍 RÉPARTITION PAR LIGUE :")
        if isinstance(league_dist, pd.Series):
            for l, v in league_dist.items():
                print(f"    - {l:<15} : {v} matchs")
        
        print("═"*60)
        
        if dup_count > 0:
            print("🚨 ALERTE : Supprimez les doublons avant l'entraînement !")
        elif phys_pct > 15:
            print("💡 CONSEIL : Lancez le script d'imputation physique pour boucher les trous.")
        else:
            print("🚀 ÉTAT : Dataset validé pour entraînement LightGBM.")
        print("═"*60 + "\n")

if __name__ == "__main__":
    reconciler = DataReconciler()
    reconciler.run_audit()
