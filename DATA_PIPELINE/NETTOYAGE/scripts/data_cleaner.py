import os
import sys
import pandas as pd
import glob
from pathlib import Path

# Ajout des chemins
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "SCRAPPING" / "scripts"
sys.path.append(str(SCRIPTS_DIR))

try:
    from prediction_physique import estimer_physique_manquant
    print("✅ Module physique chargé.")
except ImportError:
    estimer_physique_manquant = None

RAW_DIR = Path(__file__).resolve().parents[2] / "SCRAPPING" / "raw" / "sofascore"
CLEAN_DIR = Path(__file__).resolve().parents[1] / "data"

def clean_and_merge_data():
    print("\n" + "="*60)
    print(" 🧹 ATHLYTIQ — CONSOLIDATION & CALCULS PHYSIQUES")
    print("="*60)
    
    all_player_files = glob.glob(str(RAW_DIR / "**/*.csv"), recursive=True)
    final_dfs = []

    for i, file_path in enumerate(all_player_files, 1):
        p_path = Path(file_path)
        player_name = p_path.stem.replace("_", " ")
        team_name = p_path.parent.name.replace("_", " ")
        league_name = p_path.parent.parent.name
        
        print(f"[{i}/{len(all_player_files)}] Consolidation : {player_name} ({team_name})...", end="\r")

        try:
            df = pd.read_csv(file_path)
            if df.empty: continue

            # 1. CALCUL PHYSIQUE (IA)
            if estimer_physique_manquant:
                df = estimer_physique_manquant(df)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')

            # 2. NETTOYAGE & CONTEXTE
            df['Player_Name'] = player_name
            df['Team'] = team_name
            df['League'] = league_name
            
            # Normalisation numérique
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            final_dfs.append(df)
        except Exception as e:
            print(f"\n❌ Erreur sur {player_name}: {e}")

    # 3. FUSION FINALE
    if final_dfs:
        master_df = pd.concat(final_dfs, ignore_index=True)
        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        master_df.to_csv(CLEAN_DIR / "merged_dataset_clean.csv", index=False, encoding='utf-8-sig')
        print(f"\n✅ PIPELINE TERMINÉE. Dataset créé avec succès.")

if __name__ == "__main__":
    clean_and_merge_data()
