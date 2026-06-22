"""
AthlytIQ - Injury Predictor V2
==============================
Predicts the official real-world target:
Target_Injury_Next_30D

The target is generated upstream from Transfermarkt injury starts, not from
Medical_Risk_Score. This module only trains and scores the injury model.
"""

import sys
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# Configuration des chemins
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SAVE_DIR = ROOT / "LM" / "models" / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "dataset_v2_injury.csv"
RAW_MATCH_DATA_PATH = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"
INJURY_HISTORY_PATH = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "data" / "raw" / "transfermarkt" / "injury_history.csv"
TARGET = "Target_Injury_Next_30D"

# Silencing DtypeWarnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# ══════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION DES POIDS (INTENSITÉ)
# ══════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════
# 1. DEFINITION DES FEATURES
# ══════════════════════════════════════════════════════════════════════

INJURY_FEATURES = [
    'ACWR',                   # Charge récente / charge chronique
    'Fatigue_Index',          # Minutes cumulées normalisées
    'Congestion_Risk',        # Risque de congestion calendrier
    'Trauma_Index',           # Intensité des chocs physiques
    'Usage_Factor',           # Taux d'utilisation du joueur
    'Cumulative_Minutes_21d', # Minutes cumulées sur 21 jours
    'Days_Since_Last',        # Jours de repos depuis dernier match
    'Days_Rest',
    'Match_Density',
    'Form_Score',             # État de forme global
    'Fatigue_Score',
    'Age_Risk_Factor',
    'Injury_History_Available',
    'Ambiguous_Player_Name',
    'Injury_Count_Career_Before_T',
    'Injury_Count_12M_Before_T',
    'Days_Since_Last_Injury',
    'Total_Injury_Days_12M',
    'Muscle_Injury_Count_12M',
    'Recurring_Same_Category_12M',
    'Last_Injury_Duration_Days',
]

LEAKAGE_COLUMNS = {
    TARGET,
    'Target_Injury_Occurred',
    'Medical_Risk_Score',
    'Injury_Risk',
    'Risk_Level',
    'Risk_Emoji',
    'Risk_Message',
    'Current_Injury',
    'Injury_Type_Text',
    'Est_Blesse',
    'Next_Injury_Date',
    'Next_Injury_Type',
    'Next_Injury_Category',
    'Next_Injury_Duration_Days',
    'Days_To_Next_Injury',
    'Target_Fatigue',
    'Fatigue_Realisee',
    'Fatigue_Reelle_Match_T',
}

# ══════════════════════════════════════════════════════════════════════
# 2. CALCUL DU SCORE DE FATIGUE (0-100)
# ══════════════════════════════════════════════════════════════════════

def calculer_fatigue_score(df):
    """Calcule un score de fatigue composite (0-100) pondéré par l'intensité de la compétition."""
    df = df.copy()
    
    # Composante 1 : ACWR (40% du score)
    if 'ACWR' in df.columns:
        acwr = df['ACWR'].fillna(1.0).clip(0, 3)
        acwr_risk = np.where(acwr < 0.8, (0.8 - acwr) / 0.8,
                    np.where(acwr > 1.3, (acwr - 1.3) / 1.7, 0))
        comp_acwr = acwr_risk.clip(0, 1) * 40
    else: comp_acwr = 20

    # Composante 2 : Fatigue accumulée (35% du score)
    comp_fatigue = df['Fatigue_Index'].clip(0, 1) * 35 if 'Fatigue_Index' in df.columns else 0

    # Composante 3 : Congestion (15% du score)
    comp_congestion = ((df['Congestion_Risk'].fillna(1.0) - 1.0) / 0.5).clip(0, 1) * 15 if 'Congestion_Risk' in df.columns else 0

    # Composante 4 : Trauma (10% du score)
    comp_trauma = (df['Trauma_Index'] / 10).clip(0, 1) * 10 if 'Trauma_Index' in df.columns else 0

    # Calcul de base
    base_score = comp_acwr + comp_fatigue + comp_congestion + comp_trauma

    # Multiplicateur de Compétition
    if 'League' in df.columns:
        weights = df['League'].map(COMPETITION_WEIGHTS).fillna(1.0)
        base_score = base_score * weights

    df['Fatigue_Score'] = base_score.round(1).clip(0, 100)
    return df

# ══════════════════════════════════════════════════════════════════════
# 3. DETECTION DES BLESSURES REELLES (TRANSFERMARKT)
# ══════════════════════════════════════════════════════════════════════

def _detecter_blessures_actuelles(df):
    """Récupère l'état de blessure actuel et le type depuis l'historique scrapé."""
    history_path = INJURY_HISTORY_PATH
    df['Current_Injury'] = 0
    df['Injury_Type_Text'] = ""
    
    if not history_path.exists():
        return df

    try:
        history = pd.read_csv(history_path, low_memory=False)
        injury_type = history['Injury_Type'].astype(str).str.upper().str.strip()
        if 'Cause_Category' in history.columns:
            category = history['Cause_Category'].astype(str).str.upper().str.strip()
        else:
            category = pd.Series('', index=history.index)
        mask_valid = ~injury_type.isin(['NONE', 'N/A', '', 'NAN', 'REST', 'FITNESS'])
        mask_valid &= ~category.isin(['NONE', 'MALADIE'])
        mask_date = history['Date_To'].isna() | (history['Date_To'].astype(str).str.strip() == "")
        blesses = history[mask_valid & mask_date].copy()
        
        type_map = blesses.groupby('Nom')['Injury_Type'].last().to_dict()
        df['Current_Injury'] = df['Nom'].apply(lambda x: 1 if x in type_map else 0)
        df['Injury_Type_Text'] = df['Nom'].map(type_map).fillna("")
    except:
        pass
    return df


def _preparer_features_injury(df):
    """Adds pre-match feature engineering needed by the injury model."""
    df = df.copy()
    if 'Match_Date' in df.columns:
        df['Match_Date'] = pd.to_datetime(df['Match_Date'], errors='coerce')
        df = df.sort_values(['Nom', 'Match_Date']).reset_index(drop=True)

    from LM.models.feature_engineering import (
        calculer_features_match,
        calculer_features_temporelles,
        calculer_form_score,
        calculer_match_context,
        calculer_trauma_index,
    )

    # Only pre-match / row-current transformations. Do not create Target_Fatigue here.
    df = calculer_features_match(df)
    df = calculer_match_context(df)
    df = calculer_features_temporelles(df, fenetres=[3, 5, 10, 15])
    df = calculer_trauma_index(df)
    df = calculer_form_score(df)
    df['Form_Score_Lag1'] = df.groupby('Nom')['Form_Score'].shift(1).fillna(df['Form_Score'])
    df = calculer_fatigue_score(df)
    return df


def _selectionner_features(df):
    """Returns numeric model features, excluding target/future/leakage columns."""
    preferred = [c for c in INJURY_FEATURES if c in df.columns and c not in LEAKAGE_COLUMNS]
    if preferred:
        return preferred

    identity_cols = {
        'Nom', 'Player_Name', 'Player_ID', 'Transfermarkt_ID', 'Match_Date',
        'Event_ID', 'Home_Team', 'Away_Team', 'Score_Home', 'Score_Away',
        'Tournament', 'Equipe', 'Position', 'Poste_Cat', 'Team', 'League',
        'Last_Injury_Category',
    }
    excluded = identity_cols | LEAKAGE_COLUMNS
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    return [c for c in numeric_cols if c not in excluded]


def _split_temporel(df_clean, features, target):
    """Chronological split to respect future-prediction semantics."""
    if 'Match_Date' not in df_clean.columns:
        return train_test_split(
            df_clean[features],
            df_clean[target],
            test_size=0.2,
            random_state=42,
            stratify=df_clean[target] if df_clean[target].nunique() == 2 else None,
        )

    data = df_clean.sort_values('Match_Date').reset_index(drop=True)
    unique_dates = data['Match_Date'].dropna().sort_values().unique()
    if len(unique_dates) < 5:
        return train_test_split(
            data[features],
            data[target],
            test_size=0.2,
            random_state=42,
            stratify=data[target] if data[target].nunique() == 2 else None,
        )

    cutoff = unique_dates[int(len(unique_dates) * 0.8)]
    train_mask = data['Match_Date'] < cutoff
    test_mask = data['Match_Date'] >= cutoff

    if train_mask.sum() < 10 or test_mask.sum() < 10:
        return train_test_split(
            data[features],
            data[target],
            test_size=0.2,
            random_state=42,
            stratify=data[target] if data[target].nunique() == 2 else None,
        )

    return (
        data.loc[train_mask, features],
        data.loc[test_mask, features],
        data.loc[train_mask, target],
        data.loc[test_mask, target],
    )


def _sample_weights(y):
    classes = np.array(sorted(y.unique()))
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    weight_map = dict(zip(classes, weights))
    return y.map(weight_map).astype(float)


def _print_eval(y_test, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    print("\n   Évaluation temporelle :")
    if y_test.nunique() == 2:
        print(f"   ROC-AUC     : {roc_auc_score(y_test, proba):.3f}")
        print(f"   PR-AUC      : {average_precision_score(y_test, proba):.3f}")
        print(f"   Brier Score : {brier_score_loss(y_test, proba):.3f}")
    else:
        print("   ROC/PR-AUC non calculables : une seule classe dans le test.")
    print(f"   Precision@0.50 : {precision_score(y_test, pred, zero_division=0):.3f}")
    print(f"   Recall@0.50    : {recall_score(y_test, pred, zero_division=0):.3f}")
    print(classification_report(y_test, pred, zero_division=0))

# ══════════════════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT DU MODÈLE ENSEMBLE
# ══════════════════════════════════════════════════════════════════════

def entrainer_injury_model(df):
    """Entraîne un ensemble (GB + RF) sur Target_Injury_Next_30D."""
    print("\n" + "─" * 65)
    print("   🏥 ENTRAÎNEMENT — Injury V2 Target_Injury_Next_30D")
    print("─" * 65)

    if TARGET not in df.columns:
        raise ValueError(
            f"{TARGET} absent. Génère d'abord Dataset V2 avec "
            "DATA_PIPELINE/NETTOYAGE/scripts/build_dataset_v2.py"
        )

    df = _preparer_features_injury(df)
    features = _selectionner_features(df)
    keep_cols = features + [TARGET]
    if 'Match_Date' in df.columns:
        keep_cols.append('Match_Date')
    df_clean = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    df_clean[features] = df_clean[features].fillna(0)
    df_clean[TARGET] = df_clean[TARGET].astype(int)
    
    if len(df_clean) < 10:
        print("   ⚠️  Données insuffisantes pour l'entraînement.")
        return None
    if df_clean[TARGET].nunique() < 2:
        print("   ⚠️  La cible ne contient qu'une seule classe.")
        return None

    X = df_clean[features]
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = _split_temporel(df_clean, features, TARGET)

    # Scaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Modèles
    model_gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    model_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    sample_weight = _sample_weights(y_train)
    model_gb.fit(X_train_sc, y_train, sample_weight=sample_weight)
    model_rf.fit(X_train_sc, y_train)

    prob_test = (
        model_gb.predict_proba(X_test_sc)[:, 1] * 0.6
        + model_rf.predict_proba(X_test_sc)[:, 1] * 0.4
    )
    _print_eval(y_test, prob_test)

    # Sauvegarde
    joblib.dump(model_gb, SAVE_DIR / "injury_model_gb.joblib")
    joblib.dump(model_rf, SAVE_DIR / "injury_model_rf.joblib")
    joblib.dump(scaler, SAVE_DIR / "injury_scaler.joblib")
    joblib.dump(features, SAVE_DIR / "injury_features.joblib")

    print(f"   📊 Dataset : {len(X)} lignes | Blessures 30j : {y.sum()} ({y.mean():.1%})")
    print(f"   🔎 Features utilisées : {len(features)}")
    print(f"   ✅ Modèles sauvegardés dans {SAVE_DIR.name}/")
    return True

# ══════════════════════════════════════════════════════════════════════
# 5. PRÉDICTION HYBRIDE ET CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════

def predire_risque_blessure(df):
    """Prédit la probabilité de blessure réelle dans les 30 prochains jours."""
    df = _preparer_features_injury(df)
    df = _detecter_blessures_actuelles(df)

    # 1. Probabilités ML (Ensemble)
    try:
        model_gb = joblib.load(SAVE_DIR / "injury_model_gb.joblib")
        model_rf = joblib.load(SAVE_DIR / "injury_model_rf.joblib")
        scaler = joblib.load(SAVE_DIR / "injury_scaler.joblib")
        features = joblib.load(SAVE_DIR / "injury_features.joblib")
        
        X = df[features].fillna(0).replace([np.inf, -np.inf], 0)
        X_sc = scaler.transform(X)
        prob_ml = (model_gb.predict_proba(X_sc)[:, 1] * 0.6 + model_rf.predict_proba(X_sc)[:, 1] * 0.4)
    except Exception:
        prob_ml = df['Fatigue_Score'] / 100

    # Probabilité ML directe sur Target_Injury_Next_30D.
    df['Injury_Risk'] = pd.Series(prob_ml, index=df.index).clip(0, 1)

    # 4. Force 100% si déjà blessé
    df.loc[df['Current_Injury'] == 1, 'Injury_Risk'] = 1.0

    # 5. Classification métier
    df = _classifier_niveaux(df)
    
    # 6. Dilation visuelle (Désactivée ou très légère pour garder la cohérence fatigue/risque)
    # On reste sur les probabilités réelles pour éviter les incohérences visuelles.
    # mask_danger = (df['Injury_Risk'] > 0.60) & (df['Current_Injury'] == 0)
    # df.loc[mask_danger, 'Injury_Risk'] = (0.60 + (df.loc[mask_danger, 'Injury_Risk'] - 0.60) * 1.1).clip(0, 0.95)

    # 7. Résumé final (dernier match par joueur)
    result = df.sort_values(['Nom', 'Match_Date'], ascending=[True, False]).groupby('Nom').first().reset_index()
    
    _afficher_rapport(result)
    return result

def _classifier_niveaux(df):
    """Classifie les joueurs selon les seuils fixes demandés."""
    risk = df['Injury_Risk']
    
    conditions = [
        (df['Current_Injury'] == 1) | (risk >= 0.60),  # 🔴 ÉLEVÉ
        (risk >= 0.16)                                  # 🟠 MODÉRÉ
    ]
    choices = ['🔴 ÉLEVÉ', '🟠 MODÉRÉ']
    df['Risk_Level'] = np.select(conditions, choices, default='🟢 FAIBLE')
    
    return df

def _identifier_facteur_majeur(row):
    """Identifie la cause principale du risque élevé avec détails et explications cliniques."""
    if row.get('Current_Injury', 0) == 1:
        return None, None
    
    factors = []
    # 1. Historique Médical (Fragilité structurelle)
    if row.get('Injury_Count_Career_Before_T', 0) > 0:
        jours = int(row.get('Total_Injury_Days_12M', 0))
        count = int(row.get('Injury_Count_Career_Before_T', 0))
        title = f"HISTORIQUE MÉDICAL ({count} blessures passées, {jours} jours d'absence)"
        explanation = "Le joueur présente une fragilité structurelle récurrente. Ses antécédents suggèrent une vulnérabilité accrue aux rechutes lors des pics de charge, particulièrement sur les zones déjà touchées."
        factors.append(((title, explanation), min(count / 5, 1.0)))
    
    if row.get('Muscle_Injury_Count_12M', 0) > 0:
        count = int(row.get('Muscle_Injury_Count_12M', 0))
        title = f"RÉCURRENCE MUSCULAIRE ({count} blessure(s) musculaire(s) sur 12 mois)"
        explanation = "Les antécédents musculaires récents augmentent le risque de rechute, surtout lorsque la charge de match remonte rapidement."
        factors.append(((title, explanation), min(count / 3, 1.0)))
    
    # 2. Fatigue (Score cumulé et neuromusculaire)
    fatigue = row.get('Fatigue_Score', 0)
    if fatigue > 55:
        title = f"FATIGUE ACCUMULÉE (Score de fatigue élevé : {fatigue:.1f}/100)"
        cum_min = row.get('Cumulative_Minutes_21d', 0)
        days_rest = row.get('Days_Since_Last', 7)
        if cum_min > 270:
            explanation = f"Le joueur a accumulé {int(cum_min)} minutes en 21 jours. Ce volume excessif sature ses capacités de régénération et dégrade sa qualité d'appui (risque de lésion non-contact)."
        elif days_rest < 4:
            explanation = f"Temps de récupération insuffisant ({int(days_rest)} jours) entre les sollicitations à haute intensité. La fatigue neuromusculaire résiduelle n'a pas été éliminée."
        else:
            explanation = "La somme des contraintes physiques (minutes, intensité, chocs) atteint un seuil critique. Le système nerveux central est en état de surcharge, réduisant la réactivité musculaire."
        factors.append(((title, explanation), fatigue / 100))
        
    # 3. ACWR (Déséquilibre de charge)
    acwr = row.get('ACWR', 1.0)
    if acwr > 1.3:
        title = f"INTENSITÉ ACWR (Surcharge brutale : {acwr:.2f}x la charge normale)"
        explanation = "Pic de charge (spike) détecté : le volume de travail hebdomadaire dépasse de loin la préparation chronique du joueur. Les tissus (tendons/muscles) subissent une contrainte à laquelle ils ne sont pas adaptés."
        factors.append(((title, explanation), acwr - 1.0))
    elif acwr < 0.8:
        title = f"INTENSITÉ ACWR (Sous-charge / Reprise : {acwr:.2f}x)"
        explanation = "Déficit de charge chronique : le joueur manque de rythme de compétition. Cette 'sous-préparation' le rend paradoxalement plus vulnérable aux intensités maximales d'un match réel."
        factors.append(((title, explanation), 0.8 - acwr))
        
    # 4. Congestion (Calendrier)
    if row.get('Congestion_Risk', 1.0) > 1.0:
        title = "CALENDRIER SURCHARGÉ (Fixture Congestion)"
        explanation = "Enchaînement de matchs sans cycle de décharge (deload). Sans rotation, l'épuisement des réserves de glycogène et la fatigue mentale augmentent la probabilité d'erreur technique et de blessure."
        factors.append(((title, explanation), 0.5))

    if not factors:
        return None, None
    
    # Prendre le facteur avec le poids le plus élevé
    factors.sort(key=lambda x: x[1], reverse=True)
    return factors[0][0]

def _afficher_rapport(result, max_players_per_level=25):
    """Rapport premium AthlytIQ — Affichage exhaustif et explicatif détaillé."""
    print("\n" + "═"*65)
    print("   🏥 RAPPORT DE RISQUE DE BLESSURE — AthlytIQ")
    print("═"*65 + "\n")

    for niveau in ['🔴 ÉLEVÉ', '🟠 MODÉRÉ', '🟢 FAIBLE']:
        groupe_all = result[result['Risk_Level'] == niveau].sort_values('Injury_Risk', ascending=False)
        groupe = groupe_all.head(max_players_per_level)
        count = len(groupe)
        print(f"{niveau} ({len(groupe_all)} joueurs, top {count} affichés) :")
        print("   " + "─"*54)
        
        for _, row in groupe.iterrows():
            risk = row['Injury_Risk']
            fatigue = row.get('Fatigue_Score', 0)
            
            # Identification du facteur explicatif
            facteur_titre, facteur_expl = _identifier_facteur_majeur(row) if risk >= 0.16 else (None, None)
            
            # Ligne principale
            line = f"   {row['Nom'] : <46} Fatigue: {fatigue: >5.1f}/100  Risque: {int(risk*100)}%"
            print(line)
            
            # Détails et État Actuel
            if row.get('Current_Injury', 0) == 1:
                injury_type = row.get('Injury_Type_Text', 'Blessure')
                print(f"                                                   -> État Actuel : 🚑 ACTUELLEMENT BLESSÉ ({injury_type})")
                print(f"                                                   -> ✅ PRÉDICTION CORRECTE (Forcée)")
            else:
                # Affichage du facteur clé si présent
                if facteur_titre:
                    print(f"                                                   -> Facteur Clé : ⚠️ {facteur_titre}")
                    if facteur_expl:
                        print(f"                                                      {facteur_expl}")
                
                # État de forme / Aptitude
                if fatigue > 75:
                    print(f"                                                   -> État Actuel : 🟠 SURCHARGE DÉTECTÉE")
                elif risk >= 0.60:
                    print(f"                                                   -> État Actuel : 🟢 APTE (Risque Imminent)")
                elif risk >= 0.16:
                    print(f"                                                   -> État Actuel : 🟢 APTE (Temps réduit)")
                else:
                    print(f"                                                   -> État Actuel : 🟢 APTE (Plein temps)")
            
            print("")

# ══════════════════════════════════════════════════════════════════════
# 6. INTERFACE API (CLASSE)
# ══════════════════════════════════════════════════════════════════════

class InjuryPredictor:
    """Interface simple pour AthlytIQ."""
    def __init__(self):
        self.load_models()

    def load_models(self):
        try:
            self.gb = joblib.load(SAVE_DIR / "injury_model_gb.joblib")
            self.rf = joblib.load(SAVE_DIR / "injury_model_rf.joblib")
            self.scaler = joblib.load(SAVE_DIR / "injury_scaler.joblib")
            self.feats = joblib.load(SAVE_DIR / "injury_features.joblib")
        except:
            self.gb = None

    def predict(self, data):
        return predire_risque_blessure(data)

# ══════════════════════════════════════════════════════════════════════
# 7. EXECUTION
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"⚙️ Dataset V2 absent, génération depuis les données sources...")
        from DATA_PIPELINE.NETTOYAGE.logic.injury_target_builder import build_dataset_v2, print_report

        if not RAW_MATCH_DATA_PATH.exists() or not INJURY_HISTORY_PATH.exists():
            print(f"❌ Sources introuvables : {RAW_MATCH_DATA_PATH} / {INJURY_HISTORY_PATH}")
            sys.exit(1)
        _, report = build_dataset_v2(
            match_path=RAW_MATCH_DATA_PATH,
            injury_path=INJURY_HISTORY_PATH,
            output_path=DATA_PATH,
        )
        print_report(report, output_path=DATA_PATH)

    print(f"📂 Chargement des données : {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH, encoding='utf-8-sig', low_memory=False)
    df_raw['Match_Date'] = pd.to_datetime(df_raw['Match_Date'], errors='coerce')

    # Entraînement et Prédiction
    entrainer_injury_model(df_raw)
    predire_risque_blessure(df_raw)
