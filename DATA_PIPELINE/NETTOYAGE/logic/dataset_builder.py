import pandas as pd
import glob
from pathlib import Path
from typing import List, Optional
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("DatasetBuilder")

class AthlytIQDatasetBuilder:
    """
    Moteur de consolidation et de construction du dataset maître.
    Fusionne les fichiers joueurs individuels en un dataset unique prêt pour l'IA.
    """

    EXPECTED_COLS = [
        "Nom", "Match_Date", "Home_Team", "Away_Team", "Rating",
        "Minutes_Played", "distanceRun", "sprints", "kpi_work_rate",
        "Goals", "Assists", "Expected_Goals", "Expected_Assists",
        "Accurate_Passes", "Total_Passes", "Key_Passes",
        "Tackles", "Interceptions", "Clearances", "Ball_Recovery",
        "Touches", "Successful_Dribbles", "kpi_explosivity", "Age", "League"
    ]

    def __init__(self):
        self.root = Path(__file__).resolve().parents[2]
        self.raw_dir = self.root / "SCRAPPING" / "raw" / "sofascore"
        self.output_dir = self.root / "data"
        self.output_file = self.output_dir / "merged_dataset_clean.csv"

    def build(self, force_rebuild: bool = True) -> Optional[pd.DataFrame]:
        """Exécute la consolidation globale."""
        logger.info(f"🚀 Début de la consolidation dans {self.raw_dir}")
        
        all_files = glob.glob(str(self.raw_dir / "**/*.csv"), recursive=True)
        if not all_files:
            logger.error("❌ Aucun fichier source trouvé dans raw/sofascore")
            return None

        frames = []
        skipped = 0

        for file_path_str in all_files:
            file_path = Path(file_path_str)
            try:
                df = pd.read_csv(file_path)
                if df.empty or "Match_Date" not in df.columns:
                    skipped += 1
                    continue

                # Extraction Métadonnées depuis le chemin (League/Team)
                # Structure: .../raw/sofascore/League/Team/Player.csv
                league = file_path.parents[1].name
                team = file_path.parent.name
                
                # Injection dans le DF
                df["League"] = league
                df["Team"] = team

                # Ajout des colonnes manquantes
                for col in self.EXPECTED_COLS:
                    if col not in df.columns:
                        df[col] = 0 if col not in ["Age", "League", "Team"] else "Unknown"

                # Nettoyage et typage
                df = df[[c for c in self.EXPECTED_COLS if c in df.columns or c == "Team"]]
                df["Match_Date"] = pd.to_datetime(df["Match_Date"], errors="coerce")
                df["Minutes_Played"] = pd.to_numeric(df["Minutes_Played"], errors="coerce").fillna(0)
                
                frames.append(df)

            except Exception as e:
                logger.warning(f"⚠️ Erreur sur {Path(file_path).name}: {e}")
                skipped += 1

        if not frames:
            logger.error("❌ Aucun dataframe valide à fusionner.")
            return None

        merged = pd.concat(frames, ignore_index=True)
        
        # Tri et dédoublonnage
        merged.sort_values(by=["Nom", "Match_Date"], ascending=[True, False], inplace=True)
        merged.drop_duplicates(subset=["Nom", "Match_Date"], keep="first", inplace=True)

        # Sauvegarde
        self.output_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(self.output_file, index=False, encoding="utf-8-sig")

        logger.info(f"✅ Dataset construit : {len(merged)} lignes, {merged['Nom'].nunique()} joueurs.")
        logger.info(f"⏭️  Fichiers ignorés : {skipped}")
        
        return merged
