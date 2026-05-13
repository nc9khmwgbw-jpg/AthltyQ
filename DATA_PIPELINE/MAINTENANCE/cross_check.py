import sys
import pandas as pd
import random
from pathlib import Path

# Ajout de la RACINE du projet au PATH
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from DATA_PIPELINE.SCRAPPING.fotmob.scrapers.match_scraper import FotMobMatchScraper
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("CrossCheck-Elite")

class DataVerifier:
    """Vérificateur de vérité externe : SofaScore vs FotMob."""

    def __init__(self, sample_size: int = 5):
        self.root = Path(__file__).resolve().parents[2]
        self.clean_csv = self.root / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"
        self.sample_size = sample_size
        self.fotmob = FotMobMatchScraper(headless=True)

    def run(self):
        if not self.clean_csv.exists():
            logger.error("❌ Fichier de données absent.")
            return

        df = pd.read_csv(self.clean_csv, low_memory=False)
        df['Match_Date'] = pd.to_datetime(df['Match_Date'], errors='coerce')
        
        all_players = df['Nom'].unique().tolist()
        sample_players = random.sample(all_players, min(len(all_players), self.sample_size))
        
        print("\n" + "═"*60)
        print(f" 🕵️‍♂️  INSPECTION DE VÉRITÉ EXTERNE (Échantillon : {len(sample_players)} joueurs)")
        print("═"*60)

        results = []

        for name in sample_players:
            print(f"🔍 Vérification de {name} sur FotMob...")
            fm_matches = self.fotmob.scrape_player(name)
            
            if not fm_matches:
                print(f"  ⚠️  Impossible de trouver {name} sur FotMob. Skip.")
                continue
            
            # On prend les 5 derniers matchs du joueur en local pour comparer
            p_df = df[df['Nom'] == name].sort_values('Match_Date', ascending=False).head(5)
            
            p_errors = 0
            p_checks = 0
            
            for _, local_match in p_df.iterrows():
                l_date = local_match['Match_Date']
                # Trouver le match correspondant sur FotMob (à +/- 1 jour près pour le fuseau horaire)
                match_found = False
                for fm in fm_matches:
                    fm_date = pd.to_datetime(fm['Match_Date'])
                    if abs((l_date - fm_date).days) <= 1:
                        match_found = True
                        # Comparaison des stats clés
                        for col in ['Goals', 'Assists', 'Minutes_Played']:
                            l_val = float(local_match.get(col, 0))
                            f_val = float(fm.get(col, 0))
                            
                            p_checks += 1
                            if abs(l_val - f_val) > 0.1:
                                p_errors += 1
                                logger.warning(f"  ❌ Écart détecté pour {name} ({l_date.date()}) : {col} {l_val} (Sofa) vs {f_val} (FotMob)")
                        break
            
            reliability = 100 * (1 - (p_errors / p_checks)) if p_checks > 0 else 0
            results.append(reliability)
            print(f"  ✅ Fiabilité {name} : {reliability:.1f}%")

        if results:
            avg_rel = sum(results) / len(results)
            print("═"*60)
            status = "🟢 EXCELLENT" if avg_rel > 95 else "🟡 CORRECT" if avg_rel > 85 else "🔴 ALERTE"
            print(f" 🏁  SCORE DE FIABILITÉ GLOBAL : {avg_rel:.1f}% [{status}]")
            print("═"*60 + "\n")
        else:
            print("❌ Échec de l'audit : Aucune donnée de comparaison récupérée.")

if __name__ == "__main__":
    # On teste 5 joueurs par défaut pour la vitesse
    verifier = DataVerifier(sample_size=5)
    verifier.run()
