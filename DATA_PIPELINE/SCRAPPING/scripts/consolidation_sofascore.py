import pandas as pd
import glob
from pathlib import Path

RAW_DIR   = Path(__file__).resolve().parents[2] / "SCRAPPING" / "raw" / "sofascore"
OUT_DIR   = Path(__file__).resolve().parents[2] / "NETTOYAGE" / "data"
OUT_FILE  = OUT_DIR / "merged_dataset_clean.csv"

# Colonnes attendues dans l'ordre final
EXPECTED_COLS = [
    "Nom", "Match_Date", "Home_Team", "Away_Team", "Rating",
    "Minutes_Played", "distanceRun", "sprints", "kpi_work_rate",
    "Goals", "Assists", "Expected_Goals", "Expected_Assists",
    "Accurate_Passes", "Total_Passes", "Key_Passes",
    "Tackles", "Interceptions", "Clearances", "Ball_Recovery",
    "Touches", "Successful_Dribbles", "kpi_explosivity"
]

def run_consolidation():
    print("\n" + "="*60)
    print("  🔗  ATHLYTIQ — CONSOLIDATION SOFASCORE")
    print("="*60)

    all_files = glob.glob(str(RAW_DIR / "**/*.csv"), recursive=True)
    print(f"📂 {len(all_files)} fichiers trouvés.\n")

    frames = []
    skipped = 0

    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)

            # Ignorer les fichiers vides ou sans colonne minimale
            if df.empty or "Match_Date" not in df.columns:
                skipped += 1
                continue

            # Ignorer les joueurs sans aucune minute jouée (gardiens inactifs)
            if "Minutes_Played" in df.columns and df["Minutes_Played"].sum() == 0:
                skipped += 1
                continue

            # Ajouter les colonnes manquantes avec valeur par défaut 0
            for col in EXPECTED_COLS:
                if col not in df.columns:
                    df[col] = 0

            # Forcer le bon ordre et les bons types
            df = df[EXPECTED_COLS]
            df["Match_Date"] = pd.to_datetime(df["Match_Date"], errors="coerce")
            df["Minutes_Played"] = pd.to_numeric(df["Minutes_Played"], errors="coerce").fillna(0)
            df["Goals"] = pd.to_numeric(df["Goals"], errors="coerce").fillna(0).astype(int)
            df["Assists"] = pd.to_numeric(df["Assists"], errors="coerce").fillna(0).astype(int)
            df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

            frames.append(df)

        except Exception as e:
            print(f"  ⚠️ Ignoré : {Path(file_path).name} ({e})")
            skipped += 1

    if not frames:
        print("❌ Aucun fichier valide trouvé.")
        return

    # Fusion
    merged = pd.concat(frames, ignore_index=True)

    # Tri par joueur puis par date
    merged.sort_values(by=["Nom", "Match_Date"], ascending=[True, False], inplace=True)

    # Suppression des doublons éventuels
    merged.drop_duplicates(subset=["Nom", "Match_Date"], keep="first", inplace=True)

    # Sauvegarde
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print(f"✅ Dataset fusionné sauvegardé :")
    print(f"   📄 Fichier : {OUT_FILE}")
    print(f"   👤 Joueurs uniques : {merged['Nom'].nunique()}")
    print(f"   📊 Matchs totaux  : {len(merged)}")
    print(f"   ⏭️  Fichiers ignorés : {skipped}")
    print("="*60)

if __name__ == "__main__":
    run_consolidation()
