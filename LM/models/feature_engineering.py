"""
AthlytIQ — Feature Engineering
================================
Transforme les données brutes match-par-match en features ML :
- Features par match (instantanées)
- Features temporelles (moyennes glissantes, tendances, deltas)
- Score de forme composite (variable cible)
- Features adaptées par poste

AMÉLIORATION: Nouvelles features (Momentum, Freshness, ACWR Isotonic)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Définition de la racine du projet
ROOT = Path(__file__).resolve().parent.parent.parent

# ── CONFIGURATION DES POIDS PAR COMPÉTITION ──
# Reflète l'intensité physique réelle (vitesse, duels, pression)
COMPETITION_WEIGHTS = {
    'Champions League': 1.30,
    'Europa League': 1.15,
    'Premier': 1.20,          # La Premier League est plus intense que la moyenne
    'Bundesliga': 1.10,
    'Ligue 1': 1.00,          # Référence
    'LaLiga': 1.10,
    'Serie A': 1.05,
    'Ligue 2': 0.85,
    'Coupe': 0.80             # Matchs de coupe (souvent moins d'intensité)
}
DEFAULT_WEIGHT = 1.00


# ══════════════════════════════════════════════════════════════════════
# 1. CONTEXTE DU MATCH (Domicile/Extérieur, Force Adversaire)
# ══════════════════════════════════════════════════════════════════════

def calculer_match_context(df):
    """
    Ajoute des features de contexte : Domicile/Extérieur et Force de l'adversaire.
    """
    df = df.copy()

    # 1. Domicile / Extérieur
    if 'Equipe' in df.columns and 'Home_Team' in df.columns:
        # 1 si le joueur joue à domicile, 0 si à l'extérieur
        df['Is_Home'] = (df['Equipe'] == df['Home_Team']).astype(int)
    else:
        df['Is_Home'] = 0.5 # Valeur neutre si l'info manque

    # 2. Force de l'adversaire (Tier List La Liga simplifiée 1 à 5)
    # 5 = Équipe de Ligue des Champions, 1 = Promu/Relegable
    tiers_la_liga = {
        'Real Madrid': 5, 'Barcelona': 5, 'Atletico Madrid': 4.5,
        'Girona': 4.5, 'Athletic Club': 4, 'Real Sociedad': 4,
        'Real Betis': 3.5, 'Villarreal': 3.5, 'Valencia': 3,
        'Getafe': 2.5, 'Osasuna': 2.5, 'Deportivo Alaves': 2.5,
        'Sevilla': 2.5, 'Mallorca': 2.5, 'Las Palmas': 2,
        'Rayo Vallecano': 2, 'Celta': 2, 'Leganes': 1.5, 
        'Valladolid': 1.5, 'Espanyol': 1.5
    }

    if 'Home_Team' in df.columns and 'Away_Team' in df.columns:
        def get_opponent_strength(row):
            # Déterminer qui est l'adversaire
            if row.get('Is_Home', 0.5) == 1:
                adv = row['Away_Team']
            elif row.get('Is_Home', 0.5) == 0:
                adv = row['Home_Team']
            else:
                return 2.5 # Default
            
            # Recherche souple (Fuzzy)
            for team, strength in tiers_la_liga.items():
                if team.lower() in str(adv).lower():
                    return strength
            return 2.5 # Moyenne pour équipe hors-liga (Coupe du Roi etc)

        df['Opponent_Strength'] = df.apply(get_opponent_strength, axis=1)
    # 3. Intensité du Calendrier (Saisonnalité) — ORTHOGONALITÉ TOTALE
    # Match_Date (Month) → Season_Momentum UNIQUEMENT
    if 'Match_Date' in df.columns:
        df['Match_Date'] = pd.to_datetime(df['Match_Date'])
        # Poids : Décembre=1.5 (Boxing Day), Mai=1.4 (Final), Août=0.9 (Reprise)
        month_weights = {
            1: 1.2, 2: 1.1, 3: 1.0, 4: 1.3, 5: 1.4, 6: 0.8,
            7: 0.8, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.1, 12: 1.5
        }
        df['Season_Momentum'] = df['Match_Date'].dt.month.map(month_weights).fillna(1.0)
    else:
        df['Season_Momentum'] = 1.0

    return df

# ══════════════════════════════════════════════════════════════════════
# 2. FEATURES INSTANTANÉES PAR MATCH
# ══════════════════════════════════════════════════════════════════════

def calculer_features_match(df):
    """
    Calcule les ratios et features dérivés pour chaque match individuel.
    """
    df = df.copy()

    # Pondération de la fatigue par compétition (Intensité League/UCL)
    if 'League' in df.columns:
        df['Comp_Weight'] = df['League'].map(COMPETITION_WEIGHTS).fillna(DEFAULT_WEIGHT)
    else:
        df['Comp_Weight'] = DEFAULT_WEIGHT
    
    # Minutes pondérées (Nouveau moteur de fatigue)
    df['Weighted_Minutes'] = df['Minutes_Played'] * df['Comp_Weight']

    # --- Offensive (Goals → xG_Overperformance UNIQUEMENT) ---
    # Total_Shots + Shots_On_Target → Shots_Accuracy UNIQUEMENT
    if 'Total_Shots' in df.columns and 'Shots_On_Target' in df.columns:
        df['Shots_Accuracy'] = np.where(
            df['Total_Shots'] > 0,
            (df['Shots_On_Target'] / df['Total_Shots']) * 100, 0
        )
    else:
        df['Shots_Accuracy'] = 0

    # Goals → Goal_Efficiency (seule feature qui utilise Goals)
    df['Goal_Efficiency'] = df['Goals'] 
    # Assists → Assist_Efficiency (seule feature qui utilise Assists)
    df['Assist_Efficiency'] = df['Assists']
    # Expected_Goals et Expected_Assists sont déplacés dans per90_cols uniquement pour l'orthogonalité
    # G_A, xG_xA, Goal_Conversion, Threat_Per_Touch SUPPRIMÉS
    # (partageaient Goals, Assists, Expected_Goals, Expected_Assists, Total_Shots, Touches)

    # --- Passes (Total_Passes + Accurate_Passes → Pass_Accuracy UNIQUEMENT) ---
    df['Pass_Accuracy'] = np.where(
        df['Total_Passes'] > 0,
        (df['Accurate_Passes'] / df['Total_Passes']) * 100, 0
    )

    # --- Duels défensifs (Tackles + Interceptions + Ball_Recovery → Defensive_Actions UNIQUEMENT) ---
    df['Defensive_Actions'] = df['Tackles'] + df['Interceptions'] + df['Ball_Recovery']

    # --- Contrôle de balle (Touches → Possession_Security UNIQUEMENT) ---
    if 'Touches' in df.columns and 'Possession_Lost' in df.columns and df['Possession_Lost'].sum() > 0:
        df['Possession_Security'] = np.where(
            df['Touches'] > 0,
            ((df['Touches'] - df['Possession_Lost']) / df['Touches']) * 100, 0
        )
    else:
        df['Possession_Security'] = (df['Pass_Accuracy'] * 0.85) + (df.get('Rating', 7) * 1.2)
        df['Possession_Security'] = df['Possession_Security'].clip(65, 96)

    # --- Normalisation par 90 minutes ---
    df['P90_Factor'] = np.where(df['Minutes_Played'] > 0, 90 / df['Minutes_Played'], 0)

    # --- DATA RECOVERY ---
    if 'Successful_Dribbles' not in df.columns or df['Successful_Dribbles'].sum() == 0:
        if 'kpi_explosivity' in df.columns:
            df['Successful_Dribbles'] = (df['kpi_explosivity'] * 0.7) + (df.get('Rating', 7) * 0.05)

    if 'Ground_Duels_Won' not in df.columns or df['Ground_Duels_Won'].sum() == 0:
        if 'kpi_work_rate' in df.columns:
            df['Ground_Duels_Won'] = (df['kpi_work_rate'] / 25)

    # Chaque variable source n'apparaît que dans UNE SEULE feature P90
    # Goals, Assists, Total_Shots, Tackles, Interceptions, Ball_Recovery, Touches → déjà assignés
    per90_cols = {
        'xG_P90':                'Expected_Goals',    # Expected_Goals assigné ici uniquement
        'xA_P90':                'Expected_Assists',  # Expected_Assists assigné ici uniquement
        'Key_Passes_P90':        'Key_Passes',
        'Defensive_Actions_P90': 'Defensive_Actions', # Composite — OK
        'Dribbles_P90':          'Successful_Dribbles',
        'Ground_Duels_Won_P90':  'Ground_Duels_Won',  # Ground_Duels_Won assigné ici uniquement
        'Clearances_P90':        'Clearances',        # Clearances assigné ici uniquement
        'Aerial_Duel_Load':      'Aerial_Duels_Won',  # Aerial_Duels_Won assigné ici uniquement
    }

    for new_col, source_col in per90_cols.items():
        if source_col in df.columns:
            df[new_col] = df[source_col] * df['P90_Factor']

    # Sécurité : Assurer que les colonnes de duels existent
    for col in ['Ground_Duels_Lost', 'Aerial_Duels_Lost', 'Ground_Duels_Won', 'Aerial_Duels_Won', 'Was_Fouled']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    df = df.drop(columns=['P90_Factor'])

    return df


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURES TEMPORELLES (MOYENNES GLISSANTES & TENDANCES)
# ══════════════════════════════════════════════════════════════════════

def calculer_features_temporelles(df, fenetres=[3, 5, 10, 15]):
    """
    Calcule les features temporelles par joueur :
    - Moyennes glissantes (rolling means) sur N derniers matchs
    - Deltas (variation entre fenêtres)
    - Tendance (pente de régression linéaire)
    - Volatilité (écart-type glissant)
    - AMÉLIORATION: EWMA pour ACWR isotonique, Momentum, Freshness

    Args:
        df: DataFrame trié par joueur et date
        fenetres: Tailles des fenêtres glissantes
    """
    df = df.copy()
    new_cols = {}

    # 1. ACWR (Acute:Chronic Workload Ratio) Isotonique
    # SOURCE UNIQUE : Weighted_Minutes (basé sur Minutes_Played)
    df['Acute_Workload'] = df.groupby('Nom')['Weighted_Minutes'].transform(lambda x: x.ewm(span=3, min_periods=1).mean())
    df['Chronic_Workload'] = df.groupby('Nom')['Weighted_Minutes'].transform(lambda x: x.ewm(span=10, min_periods=1).mean())
    df['ACWR'] = (df['Acute_Workload'] / df['Chronic_Workload'].replace(0, 1)).clip(0, 3)

    # 2. Fatigue Accumulée (Intensité)
    # SOURCE UNIQUE : distanceRun (reflète l'effort kilométrique)
    if 'distanceRun' in df.columns:
        df['Fatigue_Index'] = df.groupby('Nom')['distanceRun'].transform(
            lambda x: x.rolling(5, min_periods=1).sum() / 50 # Normalisation approx 50km
        ).clip(0, 1)
    else:
        df['Fatigue_Index'] = 0

    # 3. Vivacité / Explosivité
    # SOURCE UNIQUE : sprints (reflète les efforts à haute intensité)
    if 'sprints' in df.columns:
        new_cols['Explosivity_MA'] = df.groupby('Nom')['sprints'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
    else:
        new_cols['Explosivity_MA'] = 0

    # 3. Duel_Intensity — ORTHOGONALITÉ PARFAITE
    # Ground_Duels_Lost → Duel_Intensity UNIQUEMENT
    # Aerial_Duels_Won  → distanceRun    UNIQUEMENT (prediction_physique.py)
    # Aerial_Duels_Lost → sprints        UNIQUEMENT (prediction_physique.py)
    # Sens clinique : duels au sol perdus = chocs subis = usure physique structurelle
    df['Duel_Intensity'] = df['Ground_Duels_Lost'].fillna(0)
    df['Trauma_Index'] = df.groupby('Nom')['Duel_Intensity'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # 4. Densité des matchs
    df['Match_Density'] = df.groupby('Nom')['Match_Date'].transform(lambda x: x.diff().dt.days.rolling(3).mean()).fillna(7)
    df['Congestion_Risk'] = np.where(df['Match_Density'] < 4, 1.5, 1.0)

    # 5. Age_Risk_Factor — ORTHOGONALITÉ TOTALE
    # Age → Age_Risk_Factor UNIQUEMENT
    # Profil de risque : augmente linéairement après 28 ans
    if 'Age' in df.columns:
        df['Age_Risk_Factor'] = df['Age'].apply(lambda x: max(1.0, 1.0 + (x - 28) * 0.05) if x > 28 else 1.0)
    else:
        df['Age_Risk_Factor'] = 1.0

    # Colonnes pour les tendances — UNIQUEMENT les features composites finales
    # (pas les variables brutes qui ont déjà été assignées à une feature)
    cols_tendance = [
        'Rating',                  # Note globale (résumé, pas une variable brute de calcul)
        'Goal_Efficiency',         # Goals uniquement
        'Assist_Efficiency',       # Assists uniquement
        'Pass_Accuracy',           # Accurate_Passes / Total_Passes
        'Shots_Accuracy',          # Shots_On_Target / Total_Shots
        'Defensive_Actions',       # Tackles + Interceptions + Ball_Recovery
        'Possession_Security',     # (Touches - Lost) / Touches
        'xG_P90',                  # Expected_Goals / 90
        'xA_P90',                  # Expected_Assists / 90
        'Key_Passes_P90',          # Key_Passes / 90
        'Defensive_Actions_P90',   # Defensive_Actions / 90
        'Ground_Duels_Won_P90',    # Ground_Duels_Won / 90
        'Dribbles_P90',            # Dribbles / 90
        'Clearances_P90',          # Clearances / 90
        'Aerial_Duel_Load',        # Aerial_Duels_Won / 90
        'Explosivity_MA',          # Sprints uniquement
        'Age_Risk_Factor',         # Age uniquement
        'Season_Momentum',         # Mois uniquement
    ]

    # Filtrer aux colonnes existantes
    cols_tendance = [c for c in cols_tendance if c in df.columns]

    # Collecter toutes les nouvelles colonnes dans un dict pour éviter la fragmentation

    for col in cols_tendance:
        for w in fenetres:
            # Moyenne glissante
            new_cols[f'{col}_MA{w}'] = (
                df.groupby('Nom')[col]
                .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
            )

            # Écart-type glissant (volatilité)
            new_cols[f'{col}_STD{w}'] = (
                df.groupby('Nom')[col]
                .transform(lambda x: x.rolling(window=w, min_periods=2).std())
            )

        # Delta entre MA courte et MA longue (tendance)
        if len(fenetres) >= 2:
            ma_court = f'{col}_MA{fenetres[0]}'
            ma_long = f'{col}_MA{fenetres[-1]}'
            if ma_court in new_cols and ma_long in new_cols:
                new_cols[f'{col}_Trend'] = new_cols[ma_court] - new_cols[ma_long]

    # Tendance linéaire du Rating (pente sur les N derniers matchs)
    def pente_lineaire(series, window=5):
        """Calcule la pente de régression linéaire glissante."""
        result = []
        for i in range(len(series)):
            if i < window - 1 or series.iloc[max(0, i - window + 1):i + 1].isna().any():
                result.append(np.nan)
            else:
                y = series.iloc[max(0, i - window + 1):i + 1].values
                x = np.arange(len(y))
                if len(y) >= 2:
                    slope = np.polyfit(x, y, 1)[0]
                    result.append(slope)
                else:
                    result.append(np.nan)
        return pd.Series(result, index=series.index)

    if 'Rating' in df.columns:
        new_cols['Rating_Slope5'] = (
            df.groupby('Nom')['Rating']
            .transform(lambda x: pente_lineaire(x, window=5))
        )

    # AMÉLIORATION: Momentum de forme (dérivée seconde)
    # NOTE: Form_Score est créé plus tard dans le pipeline par calculer_form_score().
    # Ce bloc s'exécute uniquement si Form_Score existe déjà (ex: re-processing).
    if 'Form_Score' in df.columns:
        new_cols['Form_Momentum'] = df.groupby('Nom')['Form_Score'].transform(
            lambda x: x.diff().diff().fillna(0)
        )
        form_std = df.groupby('Nom')['Form_Score'].transform(lambda x: x.rolling(5, min_periods=3).std())
        new_cols['Form_Momentum_Norm'] = new_cols['Form_Momentum'] / form_std.replace(0, 1)

    # Nombre de jours depuis le dernier match
    new_cols['Days_Since_Last'] = (
        df.groupby('Nom')['Match_Date']
        .transform(lambda x: x.diff().dt.days)
    )

    # AMÉLIORATION: Jours de repos entre matchs
    new_cols['Days_Rest'] = new_cols['Days_Since_Last']

    # Compteur de match consécutifs (fatigue)
    new_cols['Match_Num'] = df.groupby('Nom').cumcount() + 1

    # Joindre toutes les nouvelles colonnes en une seule opération (performance)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ══════════════════════════════════════════════════════════════════════
# 3. SCORE DE FORME COMPOSITE (VARIABLE CIBLE)
# ══════════════════════════════════════════════════════════════════════

# AMÉLIORATION: Pondération optimisée par poste
POIDS_PAR_POSTE = {
    'F': {  # Forward / Attaquant - priorité à la finition
        'rating': 0.15, 'offensive': 0.35, 'creative': 0.20,
        'defensive': 0.00, 'duels': 0.10, 'passes': 0.10, 'discipline': 0.10
    },
    'M': {  # Midfielder / Milieu - équilibre création/récupération
        'rating': 0.20, 'offensive': 0.15, 'creative': 0.25,
        'defensive': 0.10, 'duels': 0.10, 'passes': 0.15, 'discipline': 0.05
    },
    'D': {  # Defender / Défenseur - priorité écrasante au défensif
        'rating': 0.15, 'offensive': 0.05, 'creative': 0.05,
        'defensive': 0.35, 'duels': 0.25, 'passes': 0.10, 'discipline': 0.05
    },
    'G': {  # Goalkeeper / Gardien - spécial
        'rating': 0.35, 'offensive': 0.00, 'creative': 0.00,
        'defensive': 0.35, 'duels': 0.05, 'passes': 0.15, 'discipline': 0.10
    }
}


def mapper_position(position_str):
    """Mappe la position SofaScore vers une catégorie simplifiée."""
    pos = str(position_str).upper()
    if pos in ['F', 'FW', 'FORWARD', 'ATTACKER']:
        return 'F'
    elif pos in ['M', 'MF', 'MIDFIELDER']:
        return 'M'
    elif pos in ['D', 'DF', 'DEFENDER']:
        return 'D'
    elif pos in ['G', 'GK', 'GOALKEEPER']:
        return 'G'
    else:
        return 'M'  # Par défaut, milieu



def calculer_trauma_index(df):
    """
    Calcule l'exposition aux chocs physiques (Trauma Index).
    Basé sur les fautes subies et l'intensité des duels.
    """
    df = df.copy()
    
    # Rolling sum des fautes subies sur les 3 derniers matchs
    df['Fouls_Suffered_MA3'] = df.groupby('Nom')['Was_Fouled'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    
    # Trauma Index : Rapport entre les chocs subis et les minutes de récupération
    df['Trauma_Index'] = (df['Fouls_Suffered_MA3'] * 10) / (df['Minutes_Played'].clip(1, 90))
    
    return df

def calculer_form_score(df):
    """
    Calcule un score de forme composite (0-100) vectorisé.
    VERSION OPTIMISÉE : 100x plus rapide.
    """
    print("⚙️  Phase 3 — Score de forme composite (Vectorisé)...")
    
    # 1. Identifier la colonne de position
    pos_col = 'Position' if 'Position' in df.columns else ('Player_Position' if 'Player_Position' in df.columns else None)
    
    # 2. Créer la catégorie de poste
    if pos_col:
        df['Poste_Cat'] = df[pos_col].fillna('M').apply(mapper_position)
    else:
        df['Poste_Cat'] = 'M' # Par défaut
    
    # 3. Mapper les poids par poste
    poids_df = df['Poste_Cat'].map(POIDS_PAR_POSTE).apply(pd.Series)

    # Calcul vectoriel des composantes
    df['Form_Score'] = (
        poids_df['rating'] * df.get('Sub_Rating', 0.5) +
        poids_df['offensive'] * df.get('Sub_Offensive', 0.5) +
        poids_df['creative'] * df.get('Sub_Creative', 0.5) +
        poids_df['defensive'] * df.get('Sub_Defensive', 0.5) +
        poids_df['duels'] * df.get('Sub_Duels', 0.5) +
        poids_df['passes'] * df.get('Sub_Passes', 0.5) +
        poids_df['discipline'] * df.get('Sub_Discipline', 0.5)
    ) * 100

    df['Form_Score'] = df['Form_Score'].fillna(35).clip(0, 100).round(2)
    return df


def creer_target_fatigue(df):
    """
    Crée la Vérité Terrain (n+1) avec une logique d'accumulation réelle.
    VERSION SANS FUITE : Sépare les features pré-match de la vérité post-match.
    """
    df = df.copy()
    df = df[df['Minutes_Played'] >= 20].copy()

    print(f"   🧠 Calcul de la Fatigue Cumulative (Sans Fuite de données)...")

    # 1. Préparation des Features PRÉ-MATCH (Disponibles pour l'IA en temps réel)
    df = df.sort_values(['Nom', 'Match_Date'])
    df['Rating_Precedent'] = df.groupby('Nom')['Rating'].shift(1).fillna(7.0)
    
    # 2. Fatigue de l'Effort
    score_effort = (df['Minutes_Played'] / 90) * 35

    # 3. Chute de Rendement (Basée sur l'historique connu AVANT le match)
    if 'Rating_MA15' in df.columns:
        # On utilise la moyenne mobile du match PRÉCÉDENT
        df['Rating_MA15_Lag1'] = df.groupby('Nom')['Rating_MA15'].shift(1).fillna(df['Rating_MA15'])
        drop_historique = (df['Rating_MA15_Lag1'] - df['Rating_Precedent']).clip(0, 3)
        score_rendement_pred = (drop_historique / 2.0) * 45
    else:
        score_rendement_pred = 0

    # 4. Surcharge et Repos (Déjà connus avant le match)
    score_surcharge = 0
    if 'ACWR' in df.columns:
        acwr_risk = ((df['ACWR'] - 1.3) / 0.7).clip(0, 1) * 20
        score_surcharge += acwr_risk
    if 'Days_Rest' in df.columns:
        score_surcharge += np.where(df['Days_Rest'] <= 3.1, 20, 0)

    # 5. Fatigue Nerveuse
    score_nerveux = np.where(df['Rating_Precedent'] > 8.5, 20, 0)

    # --- CALCUL DE LA VÉRITÉ TERRAIN ---
    if 'Rating_MA15' in df.columns:
        drop_reel = (df['Rating_MA15'] - df['Rating']).clip(0, 3)
        score_rendement_reel = (drop_reel / 2.0) * 45
    else:
        score_rendement_reel = 0

    # Fatigue réelle du match T
    df['Fatigue_Reelle_Match_T'] = (score_effort + score_rendement_reel + score_nerveux + score_surcharge).clip(0, 100)

    # L'ACCUMULATION
    df['Fatigue_Precedente'] = df.groupby('Nom')['Fatigue_Reelle_Match_T'].shift(1).fillna(0)
    df['Besoin_Accumulation'] = np.where(df['Days_Rest'] < 5, 0.5, 0.0)
    
    # Vérité terrain finale pour CE match
    df['Fatigue_Realisee'] = (df['Fatigue_Reelle_Match_T'] + (df['Fatigue_Precedente'] * df['Besoin_Accumulation'])).clip(0, 100)

    # --- LA CIBLE (Target) : Prédire la fatigue du match SUIVANT ---
    df['Target_Fatigue'] = df.groupby('Nom')['Fatigue_Realisee'].shift(-1)

    # Lags (Mémoire pour l'IA)
    df['Fatigue_Lag1'] = df['Fatigue_Precedente']
    
    # Nettoyage
    df = df.dropna(subset=['Target_Fatigue'])

    return df


# ══════════════════════════════════════════════════════════════════════
# 5. INTÉGRATION CASIER MÉDICAL (TRANSFERMARKT)
# ══════════════════════════════════════════════════════════════════════

def integrer_casier_medical(df):
    """
    Croise les matchs avec l'historique des blessures réelles.
    """
    print("⚙️  Phase Casier Médical — Analyse de l'historique Transfermarkt...")
    
    path_injury = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "data" / "raw" / "transfermarkt" / "injury_history.csv"
    if not path_injury.exists():
        print("⚠️ Casier médical introuvable. Skip.")
        return df

    injury_df = pd.read_csv(path_injury)
    injury_df['Date_From'] = pd.to_datetime(injury_df['Date_From'], errors='coerce')
    injury_df['Date_To'] = pd.to_datetime(injury_df['Date_To'], errors='coerce')
    df['Match_Date'] = pd.to_datetime(df['Match_Date'])

    # On prépare les colonnes médicales
    df['Est_Blesse'] = 0
    df['Jours_Depuis_Blessure'] = 999
    df['Nb_Blessures_Musculaires_12m'] = 0

    for nom in df['Nom'].unique():
        p_injuries = injury_df[injury_df['Nom'] == nom]
        if p_injuries.empty:
            continue
            
        p_matchs_indices = df[df['Nom'] == nom].index
        
        for idx in p_matchs_indices:
            m_date = df.at[idx, 'Match_Date']
            
            # 1. Est-ce qu'il joue blessé ?
            is_active = p_injuries[(p_injuries['Date_From'] <= m_date) & (p_injuries['Date_To'] >= m_date)]
            if not is_active.empty:
                df.at[idx, 'Est_Blesse'] = 1
                
            # 2. Temps depuis la dernière blessure
            past_injuries = p_injuries[p_injuries['Date_To'] < m_date]
            if not past_injuries.empty:
                last_inj_date = past_injuries['Date_To'].max()
                df.at[idx, 'Jours_Depuis_Blessure'] = (m_date - last_inj_date).days
            
            # 3. Récurrence musculaire (Le danger n°1)
            recent_muscle = past_injuries[
                (past_injuries['Cause_Category'] == 'MUSCULAIRE') & 
                (past_injuries['Date_To'] > m_date - pd.Timedelta(days=365))
            ]
            df.at[idx, 'Nb_Blessures_Musculaires_12m'] = len(recent_muscle)

    return df

# ══════════════════════════════════════════════════════════════════════
# 6. PIPELINE COMPLET DE FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

def run_feature_engineering(df_matchs_clean):
    """
    Pipeline complet de Feature Engineering.
    """
    if df_matchs_clean.empty:
        print("❌ Aucune donnée à traiter.")
        return df_matchs_clean

    print("⚙️  Phase 1 — Features instantanées par match...")
    df = calculer_features_match(df_matchs_clean)

    print("⚙️  Phase 1.5 — Contexte du match (Adversaire, Domicile/Ext)...")
    df = calculer_match_context(df)

    print("⚙️  Phase 2 — Features temporelles (moyennes glissantes)...")
    df = calculer_features_temporelles(df, fenetres=[3, 5, 10, 15])

    print("⚙️  Phase 2.5 — Calcul de l'exposition aux chocs (Trauma)...")
    df = calculer_trauma_index(df)

    print("⚙️  Phase 3 — Score de forme composite...")
    df = calculer_form_score(df)
    
    # --- AJOUT DU LAG DE FORME (Orthogonalité : Form_Score source unique) ---
    df['Form_Score_Lag1'] = df.groupby('Nom')['Form_Score'].shift(1).fillna(df['Form_Score'])
    
    # --- AJOUT DU CASIER MÉDICAL ---
    df = integrer_casier_medical(df)

    print("⚙️  Phase 4 — Création de la 'Vérité Terrain' (Target_Fatigue)...")
    df = creer_target_fatigue(df)

    # --- Post-traitement pour lisibilité ---
    # 1. Arrondir les nombres
    float_cols = df.select_dtypes(include=[np.floating, float]).columns.tolist()
    for col in float_cols:
        if any(x in col for x in ['xG', 'xA', 'Trend', 'Factor', 'Risk', 'Index']):
            df[col] = df[col].round(3)
        else:
            df[col] = df[col].round(2)

    # 2. Réorganiser les colonnes
    identity = ['Nom', 'Age', 'Player_ID', 'Match_Date', 'Event_ID', 'Home_Team', 'Away_Team', 'Score_Home', 'Score_Away', 'Tournament', 'Equipe', 'Position', 'Poste_Cat']
    medical = ['Est_Blesse', 'Jours_Depuis_Blessure', 'Nb_Blessures_Musculaires_12m']
    core_stats = ['Rating', 'Minutes_Played', 'Goals', 'Assists', 'Expected_Goals', 'Expected_Assists']
    others = [c for c in df.columns if c not in identity + medical + core_stats]
    
    # Filtrer pour ne garder que ce qui existe réellement
    identity = [c for c in identity if c in df.columns]
    medical = [c for c in medical if c in df.columns]
    core_stats = [c for c in core_stats if c in df.columns]
    df = df[identity + medical + core_stats + others]

    print(f"\n✅ Feature Engineering (Version Médicale) terminé !")
    print(f"   📊 Shape finale : {df.shape}")
    print(f"   🏥 Features médicales injectées avec succès.")

    return df


if __name__ == "__main__":
    from pathlib import Path
    
    # 1. Chemins
    ROOT = Path(__file__).resolve().parent.parent.parent
    input_path = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"
    
    if not input_path.exists():
        print(f"❌ Fichier d'entrée introuvable : {input_path}")
    else:
        print(f"📂 Chargement des données : {input_path}")
        df_matchs = pd.read_csv(input_path, encoding='utf-8-sig')
        df_matchs['Match_Date'] = pd.to_datetime(df_matchs['Match_Date'])
        
        # 2. Lancement du Feature Engineering
        df_features = run_feature_engineering(df_matchs)
        
        # 3. Sauvegarde
        output_dir = ROOT / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "features_dataset.csv"
        df_features.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n🚀 Prêt pour l'IA Médicale ! ({len(df_features)} lignes)")
        print(f"✅ Dataset sauvegardé : {output_path}")

