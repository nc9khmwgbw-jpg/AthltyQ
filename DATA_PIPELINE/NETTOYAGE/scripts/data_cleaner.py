import sys
import re
import pandas as pd
import numpy as np
import glob
from pathlib import Path

# Ajout de la racine du projet au PYTHONPATH
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# Importations modulaires
try:
    from DATA_PIPELINE.NETTOYAGE.logic.physical_engine import estimer_physique_manquant
except ImportError:
    estimer_physique_manquant = None

if estimer_physique_manquant is None:
    print("❌ Erreur : Moteur physique AthlytIQ introuvable.")
    sys.exit(1)
else:
    print("✅ Moteur physique AthlytIQ chargé.")

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "data" / "raw" / "sofascore"
CLEAN_DIR = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data"

RATING_GLOBAL_DEFAULT = 6.8  # Moyenne ligue européenne

# Mots-clés pour détecter les matchs de réserve / jeunes (dans les noms d'équipes)
YOUTH_KEYWORDS = [
    'U19', 'U21', 'U20', 'U18', 'U17', 'U16', 'U23',
    'B squad', ' B', 'Castilla', 'Reserves', 'Reserve',
    'Youth', 'Academy', 'Jong ', 'II', 'Under-',
]

# Mots-clés pour détecter les gardiens de but (dans les noms de poste si disponibles)
GOALKEEPER_KEYWORDS = ['goalkeeper', 'gardien', 'portero', 'portiere', 'torwart']


def imputer_rating_manquant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige les Rating = 0.0 pour les joueurs qui ont joué (Minutes_Played > 30).
    """
    df = df.copy()
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

    mask_a_corriger = (
        (df['Minutes_Played'] > 30) &
        ((df['Rating'].isna()) | (df['Rating'] == 0.0))
    )

    nb_a_corriger = mask_a_corriger.sum()
    if nb_a_corriger == 0:
        df['Rating_Imputed'] = False
        return df

    df['Rating_Imputed'] = False

    # Pré-calcul : moyenne de rating valide par joueur
    ratings_valides = df[df['Rating'] > 0][['Nom', 'Rating']]
    moyenne_par_joueur = ratings_valides.groupby('Nom')['Rating'].mean()

    if 'Match_Date' in df.columns and 'Home_Team' in df.columns:
        df['_match_key'] = df['Home_Team'].astype(str).str.cat(df['Match_Date'].astype(str), sep="_")
        ratings_match = df[df['Rating'] > 0].groupby('_match_key')['Rating'].mean()
    else:
        ratings_match = pd.Series(dtype=float)

    corrections = {'niveau1': 0, 'niveau2': 0, 'niveau3': 0}

    for idx in df[mask_a_corriger].index:
        nom = df.at[idx, 'Nom']
        if nom in moyenne_par_joueur and not pd.isna(moyenne_par_joueur[nom]):
            df.at[idx, 'Rating'] = round(moyenne_par_joueur[nom], 2)
            df.at[idx, 'Rating_Imputed'] = True
            corrections['niveau1'] += 1
            continue
        if '_match_key' in df.columns:
            match_key = df.at[idx, '_match_key']
            if match_key in ratings_match and not pd.isna(ratings_match[match_key]):
                df.at[idx, 'Rating'] = round(ratings_match[match_key], 2)
                df.at[idx, 'Rating_Imputed'] = True
                corrections['niveau2'] += 1
                continue
        df.at[idx, 'Rating'] = RATING_GLOBAL_DEFAULT
        df.at[idx, 'Rating_Imputed'] = True
        corrections['niveau3'] += 1

    if '_match_key' in df.columns:
        df.drop(columns=['_match_key'], inplace=True)

    print(f"\n   ✅ Rating imputed : {nb_a_corriger} matchs corrigés")
    return df


def is_youth_match(home_team: str, away_team: str) -> bool:
    combined = f"{home_team} {away_team}".lower()
    for kw in YOUTH_KEYWORDS:
        if kw.lower() in combined:
            if kw == ' B':
                if re.search(r' b(\s|,|$)', combined):
                    return True
            else:
                return True
    return False


def is_goalkeeper(player_name: str, df: pd.DataFrame) -> bool:
    if 'Position' in df.columns:
        positions = df['Position'].dropna().astype(str).str.upper().unique()
        for pos in positions:
            if pos in ('G', 'GK', 'GOALKEEPER', 'GARDIEN', 'PORTERO'):
                return True
    if 'Expected_Goals' in df.columns and 'Goals' in df.columns:
        rows_played = df[df['Minutes_Played'] > 30].copy() if 'Minutes_Played' in df.columns else df.copy()
        if len(rows_played) >= 3:
            all_xg_zero        = (rows_played['Expected_Goals'].fillna(0) == 0).all()
            all_goals_zero     = (rows_played['Goals'].fillna(0) == 0).all()
            if all_xg_zero and all_goals_zero:
                return True
    return False


def clean_and_merge_data():
    print("\n" + "="*60)
    print(" 🧹 ATHLYTIQ — CONSOLIDATION & CALCULS PHYSIQUES")
    print("="*60)

    all_player_files = glob.glob(str(RAW_DIR / "**/*.csv"), recursive=True)
    final_dfs = []

    for i, file_path in enumerate(all_player_files, 1):
        p_path      = Path(file_path)
        player_name = p_path.stem.replace("_", " ")
        team_name   = p_path.parent.name.replace("_", " ")
        league_name = p_path.parent.parent.name

        if i % 200 == 0 or i == len(all_player_files):
            print(f"[{i}/{len(all_player_files)}] Consolidation en cours... (Dernier: {player_name} - {team_name})")

        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            if df.empty: continue

            if 'Home_Team' in df.columns and 'Away_Team' in df.columns:
                df = df[~df.apply(lambda row: is_youth_match(str(row.get('Home_Team', '')), str(row.get('Away_Team', ''))), axis=1)].copy()

            if df.empty: continue
            if is_goalkeeper(player_name, df): continue
            
            if estimer_physique_manquant:
                df = estimer_physique_manquant(df)

            df['Player_Name'] = player_name
            df['Team']        = team_name
            df['League']      = league_name

            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            final_dfs.append(df)

        except Exception as e:
            print(f"\n❌ Erreur sur {player_name}: {e}")

    if not final_dfs:
        print("\n❌ Aucune donnée trouvée.")
        return

    master_df = pd.concat(final_dfs, ignore_index=True)
    
    # DÉDOUBLONNAGE STRICT (Nom + Date de match)
    col_nom = 'Nom' if 'Nom' in master_df.columns else 'Player_Name'
    if col_nom in master_df.columns and 'Match_Date' in master_df.columns:
        avant = len(master_df)
        master_df = master_df.drop_duplicates(subset=[col_nom, 'Match_Date'], keep='first')
        apres = len(master_df)
        if avant > apres:
            print(f"   🧹 Doublons supprimés : {avant - apres}")

    master_df = imputer_rating_manquant(master_df)

    cols_numeriques = master_df.select_dtypes(include='number').columns.tolist()
    final_standardized_dfs = []
    col_nom = 'Nom' if 'Nom' in master_df.columns else 'Player_Name'
    
    for nom, group in master_df.groupby(col_nom):
        # On ne compte que les matchs où le joueur a réellement joué (Minutes > 0)
        nb_matchs_actifs = len(group[group['Minutes_Played'] > 0])
        
        # Seuil de rigueur stricte : 12 matchs RÉELLEMENT JOUÉS minimum
        if nb_matchs_actifs < 12: continue
        
        nb_matchs = len(group)
        if 'Match_Date' in group.columns:
            group['Match_Date'] = pd.to_datetime(group['Match_Date'], errors='coerce')
            group = group.sort_values('Match_Date').reset_index(drop=True)
        if nb_matchs > 15:
            group = group.tail(15).reset_index(drop=True)
        elif nb_matchs < 15:
            nb_a_ajouter = 15 - nb_matchs
            moyenne_individuelle = group[cols_numeriques].mean()
            lignes_manquantes = []
            base_date = group['Match_Date'].min() if 'Match_Date' in group.columns else pd.Timestamp.now()
            
            for k in range(1, nb_a_ajouter + 1):
                nouvelle_ligne = group.iloc[0].copy()
                for col in cols_numeriques:
                    nouvelle_ligne[col] = moyenne_individuelle[col]
                # On recule d'un jour par ligne ajoutée pour éviter les doublons de date
                if 'Match_Date' in group.columns:
                    nouvelle_ligne['Match_Date'] = base_date - pd.Timedelta(days=k)
                lignes_manquantes.append(nouvelle_ligne)
            group = pd.concat([pd.DataFrame(lignes_manquantes), group], ignore_index=True)
        final_standardized_dfs.append(group)

    if not final_standardized_dfs: return
    master_df = pd.concat(final_standardized_dfs, ignore_index=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(CLEAN_DIR / "merged_dataset_clean.csv", index=False, encoding='utf-8-sig')
    print(f"\n✅ Dataset final sauvegardé : {CLEAN_DIR / 'merged_dataset_clean.csv'}")
    
    # --- METRICS POUR L'ADMIN PANEL ---
    nb_raw_players = len(final_dfs)
    nb_clean_players = len(final_standardized_dfs)
    nb_excluded = nb_raw_players - nb_clean_players
    
    print(f"[METRIC:RAW_PLAYERS:{nb_raw_players}]")
    print(f"[METRIC:CLEANED_PLAYERS:{nb_clean_players}]")
    print(f"[METRIC:EXCLUDED_PLAYERS:{nb_excluded}]")


if __name__ == "__main__":
    clean_and_merge_data()
