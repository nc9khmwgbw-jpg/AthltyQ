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
    else:
        df['Opponent_Strength'] = 2.5

    return df

# ══════════════════════════════════════════════════════════════════════
# 2. FEATURES INSTANTANÉES PAR MATCH
# ══════════════════════════════════════════════════════════════════════

def calculer_features_match(df):
    """
    Calcule les ratios et features dérivés pour chaque match individuel.
    """
    df = df.copy()

    # --- Offensive ---
    df['Shots_Accuracy'] = np.where(
        df['Total_Shots'] > 0,
        (df['Shots_On_Target'] / df['Total_Shots']) * 100, 0
    )
    df['Goal_Conversion'] = np.where(
        df['Total_Shots'] > 0,
        (df['Goals'] / df['Total_Shots']) * 100, 0
    )
    df['xG_Overperformance'] = df['Goals'] - df['Expected_Goals']
    df['xA_Overperformance'] = df['Assists'] - df['Expected_Assists']
    df['G_A'] = df['Goals'] + df['Assists']
    df['xG_xA'] = df['Expected_Goals'] + df['Expected_Assists']

    # --- Passes ---
    df['Pass_Accuracy'] = np.where(
        df['Total_Passes'] > 0,
        (df['Accurate_Passes'] / df['Total_Passes']) * 100, 0
    )

    # --- Duels ---
    df['Defensive_Actions'] = df['Tackles'] + df['Interceptions'] + df['Ball_Recovery']

    # --- Ball Control ---
    df['Possession_Security'] = np.where(
        df['Touches'] > 0,
        ((df['Touches'] - df['Possession_Lost']) / df['Touches']) * 100, 0
    )
    df['Threat_Per_Touch'] = np.where(
        df['Touches'] > 0,
        (df['Expected_Goals'] + df['Expected_Assists']) / df['Touches'], 0
    )

    # --- Normalisation par 90 minutes ---
    df['P90_Factor'] = np.where(df['Minutes_Played'] > 0, 90 / df['Minutes_Played'], 0)

    per90_cols = {
        'Goals_P90': 'Goals',
        'Assists_P90': 'Assists',
        'xG_P90': 'Expected_Goals',
        'xA_P90': 'Expected_Assists',
        'Key_Passes_P90': 'Key_Passes',
        'Shots_P90': 'Total_Shots',
        'Tackles_P90': 'Tackles',
        'Interceptions_P90': 'Interceptions',
        'Recoveries_P90': 'Ball_Recovery',
        'Defensive_Actions_P90': 'Defensive_Actions',
        'Touches_P90': 'Touches',
        'Dribbles_P90': 'Successful_Dribbles',
    }

    for new_col, source_col in per90_cols.items():
        if source_col in df.columns:
            df[new_col] = df[source_col] * df['P90_Factor']

    # Sécurité : Assurer que les colonnes de duels existent (parfois absentes selon le scraper)
    for col in ['Ground_Duels_Lost', 'Aerial_Duels_Lost', 'Ground_Duels_Won', 'Aerial_Duels_Won', 'Was_Fouled']:
        if col not in df.columns:
            df[col] = 0

    df = df.drop(columns=['P90_Factor'])

    return df


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURES TEMPORELLES (MOYENNES GLISSANTES & TENDANCES)
# ══════════════════════════════════════════════════════════════════════

def calculer_features_temporelles(df, fenetres=[3, 5, 10]):
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

    # ── Paramètres Médicaux (Fatigue & ACWR) ──
    # Cœur de la prédiction de blessures
    workload_col = 'Minutes_Played'
    if 'Distance_Covered_km' in df.columns:
        workload_col = 'Distance_Covered_km'

    # 1. ACWR (Acute:Chronic Workload Ratio) Isotonique
    # Ration entre charge récente (court terme) et charge chronique (long terme)
    # Zone de sécurité (Goldilocks) : 0.8 - 1.3
    df['Acute_Workload'] = df.groupby('Nom')[workload_col].transform(lambda x: x.ewm(span=3, min_periods=1).mean())
    df['Chronic_Workload'] = df.groupby('Nom')[workload_col].transform(lambda x: x.ewm(span=10, min_periods=1).mean())
    df['ACWR'] = (df['Acute_Workload'] / df['Chronic_Workload'].replace(0, 1)).clip(0, 3)

    # 2. Fatigue Accumulée (Rolling sum 21 jours)
    df['Cumulative_Minutes_21d'] = df.groupby('Nom')['Minutes_Played'].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df['Fatigue_Index'] = (df['Cumulative_Minutes_21d'] / 450).clip(0, 1) # Normalisé sur ~5 matchs complets

    # 3. Intensité des Duels & Trauma (Chocs physiques)
    df['Duel_Intensity'] = (df['Ground_Duels_Won'] + df['Ground_Duels_Lost'] + df['Was_Fouled'])
    df['Trauma_Index'] = df.groupby('Nom')['Duel_Intensity'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # 4. Stress Cardiovasculaire (estimation par minutes jouées consécutives)
    df['Match_Density'] = df.groupby('Nom')['Match_Date'].transform(lambda x: x.diff().dt.days.rolling(3).mean()).fillna(7)
    df['Congestion_Risk'] = np.where(df['Match_Density'] < 4, 1.5, 1.0) # Risque boosté si < 4 jours de repos moyen

    # Colonnes sur lesquelles calculer les tendances
    cols_tendance = [
        'Rating', 'Goals', 'Assists', 'Expected_Goals', 'Expected_Assists',
        'xG_Overperformance', 'G_A', 'xG_xA',
        'Pass_Accuracy', 'Shots_Accuracy', 'Goal_Conversion',
        'Touches', 'Defensive_Actions', 'Possession_Security',
        'Ground_Duels_Won_Pct', 'Aerial_Duels_Won_Pct',
        'Goals_P90', 'xG_P90', 'xA_P90', 'Key_Passes_P90',
        'Tackles_P90', 'Interceptions_P90', 'Threat_Per_Touch',
        'Was_Fouled_P90'
    ]

    # Filtrer aux colonnes existantes
    cols_tendance = [c for c in cols_tendance if c in df.columns]

    # Collecter toutes les nouvelles colonnes dans un dict pour éviter la fragmentation
    new_cols = {}

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


def integrer_historique_medical(df):
    """
    Intègre l'historique médical scrapé (Transfermarkt) dans les features.
    Crée un profil de risque par joueur.
    """
    df = df.copy()
    history_path = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "raw" / "transfermarkt" / "injury_history.csv"
    
    if not history_path.exists():
        print("   ⚠️ Historique médical introuvable. Features médicales ignorées.")
        df['Injury_Prone_Index'] = 0.0
        df['Dominant_Injury_Cause'] = 'NONE'
        return df

    history_df = pd.read_csv(history_path)
    
    # Agrégation par joueur
    stats_med = history_df.groupby('Nom').agg({
        'Duration_Days': ['sum', 'count', 'mean'],
        'Cause_Category': lambda x: x.value_counts().index[0] if not x.empty else 'UNKNOWN'
    })
    stats_med.columns = ['Total_Injury_Days', 'Injury_Count', 'Avg_Injury_Duration', 'Dominant_Injury_Cause']
    
    # Score de fragilité (normalisé 0-1)
    max_days = stats_med['Total_Injury_Days'].max() if not stats_med.empty else 1
    stats_med['Injury_Prone_Index'] = (stats_med['Total_Injury_Days'] / max_days).clip(0, 1)
    
    # Merge
    df = df.merge(stats_med[['Injury_Prone_Index', 'Dominant_Injury_Cause', 'Injury_Count']], on='Nom', how='left')
    df['Injury_Prone_Index'] = df['Injury_Prone_Index'].fillna(0)
    df['Dominant_Injury_Cause'] = df['Dominant_Injury_Cause'].fillna('NONE')
    
    return df

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
    Calcule un score de forme composite (0–100) pour chaque match.
    Les poids varient selon le poste du joueur.
    AMÉLIORATION: Pondération par poste optimisée.
    """
    df = df.copy()
    # Dictionnaire des postes réels (FC Barcelone 2024/2025)
    postes_kwnown = {
        'Robert Lewandowski': 'F', 'Lamine Yamal': 'F', 'Raphinha': 'F',
        'Ferran Torres': 'F', 'Ansu Fati': 'F', 'Pau Víctor': 'F',
        'Pedri': 'M', 'Frenkie de Jong': 'M', 'Gavi': 'M', 'Fermín López': 'M',
        'Marc Casadó': 'M', 'Dani Olmo': 'M', 'Marc Bernal': 'M', 'Pablo Torre': 'M',
        'Ronald Araújo': 'D', 'Jules Koundé': 'D', 'Pau Cubarsí': 'D',
        'Alejandro Balde': 'D', 'Andreas Christensen': 'D', 'Eric García': 'D',
        'Iñigo Martínez': 'D', 'João Cancelo': 'D', 'Gerard Martín': 'D', 'Héctor Fort': 'D',
        'Marc-André ter Stegen': 'G', 'Iñaki Peña': 'G', 'Wojciech Szczęsny': 'G'
    }

    if 'Position' not in df.columns:
        df['Position'] = df['Nom'].map(postes_kwnown).fillna('M')

    df['Poste_Cat'] = df.get('Position', pd.Series('M', index=df.index)).apply(mapper_position)

    # Sous-scores normalisés (0–1)
    def norm_0_1(series):
        """Min-max normalization robuste."""
        s_min = series.quantile(0.05)
        s_max = series.quantile(0.95)
        if s_max == s_min:
            return pd.Series(0.5, index=series.index)
        return ((series - s_min) / (s_max - s_min)).clip(0, 1)

    # Rating SofaScore (déjà sur 0-10, normaliser en 0-1)
    if 'Rating' in df.columns:
        df['Sub_Rating'] = ((df['Rating'] - 5.0) / 5.0).clip(0, 1)
    else:
        df['Sub_Rating'] = 0.5

    # Score offensif
    df['Sub_Offensive'] = norm_0_1(
        df.get('Goals_P90', 0) * 0.4 +
        df.get('xG_P90', 0) * 0.3 +
        df.get('Shots_Accuracy', 0) / 100 * 0.3
    )

    # Score créatif
    df['Sub_Creative'] = norm_0_1(
        df.get('Assists_P90', pd.Series(0, index=df.index)) * 0.3 +
        df.get('xA_P90', pd.Series(0, index=df.index)) * 0.3 +
        df.get('Key_Passes_P90', pd.Series(0, index=df.index)) * 0.4
    )

    # Score défensif
    df['Sub_Defensive'] = norm_0_1(
        df.get('Tackles_P90', pd.Series(0, index=df.index)) * 0.3 +
        df.get('Interceptions_P90', pd.Series(0, index=df.index)) * 0.3 +
        df.get('Recoveries_P90', pd.Series(0, index=df.index)) * 0.4
    )

    # Score duels
    df['Sub_Duels'] = norm_0_1(
        df.get('Ground_Duels_Won_Pct', pd.Series(50, index=df.index)) / 100 * 0.5 +
        df.get('Aerial_Duels_Won_Pct', pd.Series(50, index=df.index)) / 100 * 0.5
    )

    # Score passes
    df['Sub_Passes'] = norm_0_1(
        df.get('Pass_Accuracy', pd.Series(70, index=df.index)) / 100 * 0.6 +
        df.get('Possession_Security', pd.Series(70, index=df.index)) / 100 * 0.4
    )

    # Score discipline (inversé : moins de cartons/fautes = mieux)
    df['Sub_Discipline'] = 1.0 - norm_0_1(
        df.get('Fouls_Committed', pd.Series(0, index=df.index)) * 0.5 +
        df.get('Yellow_Cards', pd.Series(0, index=df.index)) * 3.0 +
        df.get('Red_Cards', pd.Series(0, index=df.index)) * 10.0
    )

    # Calcul du Form Score pondéré par poste
    form_scores = []
    for _, row in df.iterrows():
        poste = row['Poste_Cat']
        poids = POIDS_PAR_POSTE.get(poste, POIDS_PAR_POSTE['M'])

        score = (
            poids['rating'] * row.get('Sub_Rating', 0.5) +
            poids['offensive'] * row.get('Sub_Offensive', 0.5) +
            poids['creative'] * row.get('Sub_Creative', 0.5) +
            poids['defensive'] * row.get('Sub_Defensive', 0.5) +
            poids['duels'] * row.get('Sub_Duels', 0.5) +
            poids['passes'] * row.get('Sub_Passes', 0.5) +
            poids['discipline'] * row.get('Sub_Discipline', 0.5)
        ) * 100

        form_scores.append(round(score, 2))

    df['Form_Score'] = form_scores

    return df


# ══════════════════════════════════════════════════════════════════════
# 4. CRÉATION DE LA CIBLE (TARGET) POUR LE ML
# ══════════════════════════════════════════════════════════════════════

def creer_targets(df, horizons=[1, 2, 4]):
    """
    Crée les variables cibles (Form Score futur) pour chaque horizon.
    horizons: [1, 2, 4] → prochain match, dans 2 matchs, dans 4 matchs

    Équivalent conceptuel de J+7, J+14, J+30 (environ 1 match/semaine).

    AMÉLIORATION: Target de blessure basée sur des règles plus réalistes.
    """
    df = df.copy()

    for h in horizons:
        col_name = f'Target_Form_{h}m'
        df[col_name] = (
            df.groupby('Nom')['Form_Score']
            .transform(lambda x: x.shift(-h))
        )

        # Moyenne du form score sur les h prochains matchs (plus lisse)
        col_avg = f'Target_Form_Avg_{h}m'
        df[col_avg] = (
            df.groupby('Nom')['Form_Score']
            .transform(lambda x: x.shift(-1).rolling(window=h, min_periods=1).mean())
        )

        # AJOUT : Moyenne des notes SofaScore (Rating) sur les h prochains matchs
        if h <= 2 and 'Rating' in df.columns:
            col_rating_avg = f'Target_Rating_Avg_{h}m'
            df[col_rating_avg] = (
                df.groupby('Nom')['Rating']
                .transform(lambda x: x.shift(-1).rolling(window=h, min_periods=1).mean())
            )

    # ── Indice de Fatigue (Cœur du projet — Fatigue Focus v1.0) ──
    if 'ACWR' in df.columns:
        # 1. Facteur d'Exposition (Exposure/Usage Factor)
        # Un joueur qui ne joue pas ne peut pas statistiquement subir une blessure d'usure de 50%.
        # On base l'exposition sur les minutes cumulées (Ex: 270 min = 3 matchs pleins = 1.0 exposure)
        df['Usage_Factor'] = (df['Cumulative_Minutes_21d'] / 270.0).clip(0.05, 1.2)
        
        # 2. Seuils cliniques réels
        # ACWR : 0.8-1.3 est la zone "Goldilocks", >1.5 est surcharge, <0.7 est sous-charge (désentrainement)
        condition_surcharge = (df['ACWR'] > 1.5) | (df['ACWR'] < 0.7)
        condition_epuisement = (df['Fatigue_Index'] > 0.8)
        
        # 3. Calcul de l'Indice de Fatigue Brut (Basé sur la physiologie)
        raw_risk = (
            (condition_surcharge.astype(int) * 0.45) + 
            (condition_epuisement.astype(int) * 0.45) + 
            ((df['Congestion_Risk'] - 1.0).clip(0, 1) * 0.1)
        ).clip(0, 1)
        
        # 4. Application du Usage_Factor (Scaling Dynamique)
        # Si un joueur n'a presque pas joué, son risque de fatigue chute drastiquement.
        df['Medical_Risk_Score'] = (raw_risk * df['Usage_Factor']).clip(0, 1)

        # 5. Target de Surcharge (pour l'entraînement ML)
        # On retire le random shock pour plus de cohérence scientifique dans l'API
        df['Target_Injury_Occurred'] = (df['Medical_Risk_Score'] > 0.65).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════
# 5. PIPELINE COMPLET DE FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

def run_feature_engineering(df_matchs_clean):
    """
    Pipeline complet de Feature Engineering.

    Args:
        df_matchs_clean: DataFrame nettoyé (sortie de data_cleaner)

    Returns:
        DataFrame enrichi avec toutes les features et targets
    """
    if df_matchs_clean.empty:
        print("❌ Aucune donnée à traiter.")
        return df_matchs_clean

    print("⚙️  Phase 1 — Features instantanées par match...")
    df = calculer_features_match(df_matchs_clean)

    print("⚙️  Phase 1.5 — Contexte du match (Adversaire, Domicile/Ext)...")
    df = calculer_match_context(df)

    print("⚙️  Phase 2 — Features temporelles (moyennes glissantes)...")
    df = calculer_features_temporelles(df, fenetres=[3, 5, 10])

    print("⚙️  Phase 2.5 — Intégration Médicale & Trauma...")
    df = integrer_historique_medical(df)
    df = calculer_trauma_index(df)

    print("⚙️  Phase 3 — Score de forme composite...")
    df = calculer_form_score(df)

    print("⚙️  Phase 4 — Création des cibles ML...")
    df = creer_targets(df, horizons=[1, 2, 4])

    # --- Post-traitement pour lisibilité ---
    # 1. Arrondir les nombres
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        if any(x in col for x in ['xG', 'xA', 'Trend', 'Factor', 'Risk', 'Index']):
            df[col] = df[col].round(3)
        else:
            df[col] = df[col].round(2)

    # 2. Réorganiser les colonnes
    identity = ['Nom', 'Player_ID', 'Match_Date', 'Event_ID', 'Home_Team', 'Away_Team', 'Score_Home', 'Score_Away', 'Tournament', 'Equipe', 'Position', 'Poste_Cat']
    core_stats = ['Rating', 'Minutes_Played', 'Goals', 'Assists', 'Expected_Goals', 'Expected_Assists', 'G_A', 'xG_xA']
    others = [c for c in df.columns if c not in identity + core_stats]
    
    # Filtrer pour ne garder que ce qui existe réellement
    identity = [c for c in identity if c in df.columns]
    core_stats = [c for c in core_stats if c in df.columns]
    df = df[identity + core_stats + others]

    # 3. Sauvegarder le dataset enrichi
    output_path = Path("data/processed/features_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ Feature Engineering terminé !")
    print(f"   📊 Shape finale : {df.shape}")
    print(f"   📁 Sauvegardé dans : {output_path}")
    print(f"   🎯 Form Score moyen : {df['Form_Score'].mean():.1f} / 100")
    print(f"   📈 Features temporelles : {len([c for c in df.columns if 'MA' in c or 'Trend' in c])}")
    print(f"   🎯 Targets créées : {len([c for c in df.columns if 'Target' in c])}")

    return df


if __name__ == "__main__":
    from pathlib import Path
    
    # 1. Chemins
    ROOT = Path(__file__).resolve().parent.parent.parent
    input_path = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"
    
    if not input_path.exists():
        print(f"❌ Fichier d'entrée introuvable : {input_path}")
        print("💡 Veuillez d'abord exécuter data_cleaner.py")
    else:
        print(f"📂 Chargement des données : {input_path}")
        df_matchs = pd.read_csv(input_path)
        df_matchs['Match_Date'] = pd.to_datetime(df_matchs['Match_Date'])
        
        # 2. Lancement du Feature Engineering
        df_features = run_feature_engineering(df_matchs)
        
        print(f"\n🚀 Prêt pour l'IA ! ({len(df_features)} lignes, {len(df_features.columns)} colonnes)")

