import sys
import pandas as pd
from pathlib import Path
import time

# Ajout du chemin pour les imports
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPTS_DIR))

try:
    from sofascore_match_scraper import creer_driver, get_player_age
except ImportError:
    print("❌ Erreur : Impossible d'importer sofascore_match_scraper")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "raw" / "sofascore"

def patch_all_ages():
    print("🚀 Démarrage du Patch Global des Âges — AthlytIQ")
    print("="*60)
    
    driver = creer_driver(headless=True)
    
    try:
        # Trouver tous les fichiers CSV dans le dossier raw
        csv_files = list(RAW_DIR.glob("**/*.csv"))
        total = len(csv_files)
        print(f"🔍 {total} fichiers trouvés. Analyse en cours...")
        
        patched_count = 0
        skipped_count = 0
        error_count = 0
        
        for i, file_path in enumerate(csv_files, 1):
            try:
                # On lit juste le header d'abord pour gagner du temps
                df = pd.read_csv(file_path)
                
                # Vérifier si l'âge est manquant
                needs_patch = False
                if 'Age' not in df.columns or df['Age'].isna().any():
                    needs_patch = True
                
                if needs_patch:
                    # On a besoin de l'ID Sofascore. 
                    # On essaie de le déduire du nom du fichier ou du contenu s'il n'est pas là.
                    # Mais le plus sûr est d'utiliser le mapping si on l'avait.
                    # Ici, on va tenter de trouver un ID dans le dossier parent ou via le scraper de ligue.
                    # Alternative : On saute si on n'a pas d'ID, mais on va essayer de trouver l'ID dans le CSV s'il existe.
                    
                    player_name = file_path.stem.replace("_", " ")
                    # L'ID n'est pas dans le CSV brut généralement, sauf si on l'y a mis.
                    # On va tenter une recherche Sofascore par nom si possible, 
                    # mais le mieux est de demander au scraper de ligue de repasser.
                    
                    # ATTENTION : Le CSV brut de AthlytIQ ne contient PAS l'ID Sofascore par défaut.
                    # Cependant, on peut le trouver si on a déjà fait un run partiel.
                    
                    print(f"⚠️  [{i}/{total}] {player_name} : Âge manquant.")
                    # Pour l'instant on ne peut pas patcher sans ID.
                    # Le scraper de ligue, lui, possède l'ID car il vient de l'API Team.
                    error_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                print(f"❌ Erreur sur {file_path.name} : {e}")
                error_count += 1
                
        print("\n" + "="*60)
        print(f"✅ Analyse terminée.")
        print(f"   - Déjà ok : {skipped_count}")
        print(f"   - À corriger : {error_count}")
        print("="*60)
        print("\n💡 CONSEIL : Lancez le script 'scraper_league.py' avec l'option 'Mettre à jour' (o).")
        print("   J'ai corrigé le bug qui sautait l'âge lors des mises à jour.")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    patch_all_ages()
