import os
import sys
import pandas as pd
import numpy as np
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

RAW_DIR   = Path(__file__).resolve().parents[2] / "SCRAPPING" / "raw" / "sofascore"
CLEAN_DIR = Path(__file__).resolve().parents[1] / "data"

RATING_GLOBAL_DEFAULT = 6.8  # Moyenne ligue européenne


# ══════════════════════════════════════════════════════════════════════
# IMPUTATION DU RATING MANQUANT — 3 NIVEAUX
# ══════════════════════════════════════════════════════════════════════

def imputer_rating_manquant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige les Rating = 0.0 pour les joueurs qui ont joué (Minutes_Played > 30).

    3 niveaux de fallback :
      Niveau 1 : Moyenne des ratings valides du même joueur
      Niveau 2 : Moyenne des ratings de son équipe lors de ce match
      Niveau 3 : Moyenne globale de la ligue (6.8 par défaut)

    Returns:
        DataFrame avec la colonne Rating corrigée + colonne Rating_Imputed (bool)
    """
    df = df.copy()
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

    # Identifier les lignes à corriger :
    # joueur qui a joué > 30 min mais rating manquant ou = 0
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

    # Pré-calcul : moyenne de rating par match (Home_Team + Away_Team + Match_Date)
    # → représente la performance moyenne de l'équipe lors de ce match
    if 'Match_Date' in df.columns and 'Home_Team' in df.columns:
        df['_match_key'] = df['Home_Team'].astype(str) + "_" + df['Match_Date'].astype(str)
        ratings_match = df[df['Rating'] > 0].groupby('_match_key')['Rating'].mean()
    else:
        ratings_match = pd.Series(dtype=float)

    corrections = {'niveau1': 0, 'niveau2': 0, 'niveau3': 0}

    for idx in df[mask_a_corriger].index:
        nom = df.at[idx, 'Nom']

        # ── Niveau 1 : historique du joueur ──
        if nom in moyenne_par_joueur and not pd.isna(moyenne_par_joueur[nom]):
            df.at[idx, 'Rating'] = round(moyenne_par_joueur[nom], 2)
            df.at[idx, 'Rating_Imputed'] = True
            corrections['niveau1'] += 1
            continue

        # ── Niveau 2 : moyenne de l'équipe lors de ce match ──
        if '_match_key' in df.columns:
            match_key = df.at[idx, '_match_key']
            if match_key in ratings_match and not pd.isna(ratings_match[match_key]):
                df.at[idx, 'Rating'] = round(ratings_match[match_key], 2)
                df.at[idx, 'Rating_Imputed'] = True
                corrections['niveau2'] += 1
                continue

        # ── Niveau 3 : moyenne globale ligue ──
        df.at[idx, 'Rating'] = RATING_GLOBAL_DEFAULT
        df.at[idx, 'Rating_Imputed'] = True
        corrections['niveau3'] += 1

    # Nettoyage colonne temporaire
    if '_match_key' in df.columns:
        df.drop(columns=['_match_key'], inplace=True)

    print(f"\n   ✅ Rating imputed : {nb_a_corriger} matchs corrigés")
    print(f"      Niveau 1 (joueur)  : {corrections['niveau1']}")
    print(f"      Niveau 2 (équipe)  : {corrections['niveau2']}")
    print(f"      Niveau 3 (défaut)  : {corrections['niveau3']}")

    return df


# ══════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

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

        print(f"[{i}/{len(all_player_files)}] Consolidation : {player_name} ({team_name})...", end="\r")

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue

            # 1. CALCUL PHYSIQUE (IA)
            if estimer_physique_manquant:
                df = estimer_physique_manquant(df)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')

            # 2. NETTOYAGE & CONTEXTE
            df['Player_Name'] = player_name
            df['Team']        = team_name
            df['League']      = league_name

            # Normalisation numérique
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

            final_dfs.append(df)

        except Exception as e:
            print(f"\n❌ Erreur sur {player_name}: {e}")

    # 3. FUSION FINALE
    if not final_dfs:
        print("\n❌ Aucune donnée trouvée.")
        return

    print("\n\n🔗 Fusion de tous les datasets...")
    master_df = pd.concat(final_dfs, ignore_index=True)

    # 4. IMPUTATION DU RATING MANQUANT (3 niveaux)
    print("\n📊 Correction des Rating manquants...")
    master_df = imputer_rating_manquant(master_df)

    # 5. SAUVEGARDE
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(CLEAN_DIR / "merged_dataset_clean.csv", index=False, encoding='utf-8-sig')

    # Rapport final
    total    = len(master_df)
    imputed  = master_df['Rating_Imputed'].sum() if 'Rating_Imputed' in master_df.columns else 0
    restants = ((master_df['Rating'] == 0.0) & (master_df['Minutes_Played'] > 30)).sum()

    print(f"\n{'='*60}")
    print(f" ✅ PIPELINE TERMINÉE")
    print(f"{'='*60}")
    print(f"   Lignes totales     : {total:,}")
    print(f"   Ratings corrigés   : {imputed:,}")
    print(f"   Rating = 0 restants: {restants} (joueurs < 30 min)")
    print(f"   Fichier            : {CLEAN_DIR / 'merged_dataset_clean.csv'}")


if __name__ == "__main__":
    clean_and_merge_data()
