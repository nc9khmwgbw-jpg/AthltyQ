"""
AthlytIQ — Injury Predictor (Fusion Hybride ML & Physiologie)
=============================================================
Pipeline avancée combinant :
  1. Ensemble de modèles ML (Gradient Boosting & Random Forest)
  2. Logique Physiologique (Medical Risk Score & Fatigue Index)
  3. Détection temps réel des blessures (Transfermarkt)
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
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Configuration des chemins
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SAVE_DIR = ROOT / "LM" / "models" / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"

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

FATIGUE_FEATURES = [
    'ACWR',                   # Charge récente / charge chronique
    'Fatigue_Index',          # Minutes cumulées normalisées
    'Congestion_Risk',        # Risque de congestion calendrier
    'Trauma_Index',           # Intensité des chocs physiques
    'Medical_Risk_Score',     # Score de risque médical global
    'Usage_Factor',           # Taux d'utilisation du joueur
    'Cumulative_Minutes_21d', # Minutes cumulées sur 21 jours
    'Days_Since_Last',        # Jours de repos depuis dernier match
    'Form_Score'              # État de forme global
]

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
    history_path = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "raw" / "transfermarkt" / "injury_history.csv"
    df['Current_Injury'] = 0
    df['Injury_Type_Text'] = ""
    
    if not history_path.exists():
        return df

    try:
        history = pd.read_csv(history_path, low_memory=False)
        mask_valid = ~history['Injury_Type'].astype(str).str.upper().isin(['NONE', 'N/A', '', 'NAN'])
        mask_date = history['Date_To'].isna() | (history['Date_To'].astype(str).str.strip() == "")
        blesses = history[mask_valid & mask_date].copy()
        
        type_map = blesses.groupby('Nom')['Injury_Type'].last().to_dict()
        df['Current_Injury'] = df['Nom'].apply(lambda x: 1 if x in type_map else 0)
        df['Injury_Type_Text'] = df['Nom'].map(type_map).fillna("")
    except:
        pass
    return df

# ══════════════════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT DU MODÈLE ENSEMBLE
# ══════════════════════════════════════════════════════════════════════

def entrainer_injury_model(df):
    """Entraîne un ensemble (GB + RF) sur les risques de blessure."""
    print("\n" + "─" * 65)
    print("   🏥 ENTRAÎNEMENT — Ensemble ML AthlytIQ (GB + RF)")
    print("─" * 65)

    df = calculer_fatigue_score(df)
    target = 'Target_Injury_Occurred'
    
    if target not in df.columns:
        # Seuil calibré à 0.55 pour la distribution cible
        df[target] = (df['Medical_Risk_Score'] > 0.55).astype(int)

    features = [f for f in FATIGUE_FEATURES + ['Fatigue_Score'] if f in df.columns]
    df_clean = df[features + [target]].dropna()
    
    if len(df_clean) < 10:
        print("   ⚠️  Données insuffisantes pour l'entraînement.")
        return None

    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Modèles
    model_gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    model_rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)

    model_gb.fit(X_train_sc, y_train)
    model_rf.fit(X_train_sc, y_train)

    # Sauvegarde
    joblib.dump(model_gb, SAVE_DIR / "injury_model_gb.joblib")
    joblib.dump(model_rf, SAVE_DIR / "injury_model_rf.joblib")
    joblib.dump(scaler, SAVE_DIR / "injury_scaler.joblib")
    joblib.dump(features, SAVE_DIR / "injury_features.joblib")

    print(f"   📊 Dataset : {len(X)} lignes | Cas à risque : {y.sum()} ({y.mean():.1%})")
    print(f"   ✅ Modèles sauvegardés dans {SAVE_DIR.name}/")
    return True

# ══════════════════════════════════════════════════════════════════════
# 5. PRÉDICTION HYBRIDE ET CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════

def predire_risque_blessure(df):
    """Pipeline de prédiction hybride : 30% ML / 70% Physiologie."""
    df = calculer_fatigue_score(df)
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
    except:
        prob_ml = df['Fatigue_Score'] / 100

    # 2. Score Physiologique (Direct)
    prob_physio = df['Medical_Risk_Score'] if 'Medical_Risk_Score' in df.columns else (df['Fatigue_Score'] / 100)

    # 3. Mélange Hybride (10/90) pour privilégier la stabilité physiologique
    df['Injury_Risk'] = (prob_ml * 0.10 + prob_physio * 0.90).clip(0, 1)

    # 4. Force 100% si déjà blessé
    df.loc[df['Current_Injury'] == 1, 'Injury_Risk'] = 1.0

    # 5. Classification (Basée sur les seuils physiques bruts)
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
    """Identifie la cause principale du risque élevé avec détails et explications."""
    if row.get('Current_Injury', 0) == 1:
        return None, None
    
    factors = []
    # 1. Historique Médical
    if row.get('Injury_Prone_Index', 0) > 0.20:
        jours = int(row.get('Total_Injury_Days', 0))
        count = int(row.get('Injury_Count', 0))
        title = f"HISTORIQUE MÉDICAL ({count} blessures passées, {jours} jours d'absence)"
        explanation = "Le joueur présente une fragilité structurelle récurrente. Ses antécédents suggèrent une vulnérabilité aux rechutes lors des pics de charge."
        factors.append(((title, explanation), row['Injury_Prone_Index'] * 1.5))
    
    # 2. Fatigue (Score cumulé)
    fatigue = row.get('Fatigue_Score', 0)
    if fatigue > 55:
        title = f"FATIGUE ACCUMULÉE (Score de fatigue élevé : {fatigue:.1f}/100)"
        cum_min = row.get('Cumulative_Minutes_21d', 0)
        days_rest = row.get('Days_Since_Last', 7)
        if cum_min > 270:
            explanation = f"Le joueur a dépassé 270 minutes de jeu ({int(cum_min)} min) en 21 jours. Le volume accumulé excède ses capacités de récupération."
        elif days_rest < 4:
            explanation = f"Le temps de repos entre les derniers matchs est insuffisant ({int(days_rest)} jours). La régénération musculaire n'est pas complète."
        else:
            explanation = "Somme des contraintes physiques (minutes, chocs) atteignant un seuil critique pour son profil."
        factors.append(((title, explanation), fatigue / 100))
        
    # 3. ACWR (Intensité de charge)
    acwr = row.get('ACWR', 1.0)
    if acwr > 1.3:
        title = f"INTENSITÉ ACWR (Surcharge brutale : {acwr:.2f}x la charge normale)"
        explanation = "Le volume de travail cette semaine est largement supérieur à sa moyenne habituelle. Cette hausse soudaine est une cause majeure de risque."
        factors.append(((title, explanation), acwr - 1.0))
    elif acwr < 0.8:
        title = f"INTENSITÉ ACWR (Sous-charge / Reprise : {acwr:.2f}x)"
        explanation = "Le joueur est en phase de reprise. Son déficit de charge chronique le rend vulnérable aux intensités de match (Manque de rythme)."
        factors.append(((title, explanation), 0.8 - acwr))
        
    # 4. Congestion (Calendrier)
    if row.get('Congestion_Risk', 1.0) > 1.0:
        title = "CALENDRIER SURCHARGÉ (Repos insuffisant entre les matchs)"
        explanation = "Enchaînement de matchs à haute intensité sans cycle de décharge. Une rotation est recommandée pour éviter la blessure de fatigue."
        factors.append(((title, explanation), 0.5))

    if not factors:
        return None, None
    
    # Prendre le facteur avec le poids le plus élevé
    factors.sort(key=lambda x: x[1], reverse=True)
    return factors[0][0]

def _afficher_rapport(result):
    """Rapport premium AthlytIQ — Affichage exhaustif et explicatif détaillé."""
    print("\n" + "═"*65)
    print("   🏥 RAPPORT DE RISQUE DE BLESSURE — AthlytIQ")
    print("═"*65 + "\n")

    for niveau in ['🔴 ÉLEVÉ', '🟠 MODÉRÉ', '🟢 FAIBLE']:
        groupe = result[result['Risk_Level'] == niveau].sort_values('Injury_Risk', ascending=False)
        count = len(groupe)
        print(f"{niveau} ({count} joueurs) :")
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
        print(f"❌ Fichier introuvable : {DATA_PATH}")
        sys.exit(1)

    print(f"📂 Chargement des données : {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    df_raw['Match_Date'] = pd.to_datetime(df_raw['Match_Date'], errors='coerce')

    # Feature engineering dynamique
    from LM.models.feature_engineering import run_feature_engineering
    df_features = run_feature_engineering(df_raw)

    # Entraînement et Prédiction
    entrainer_injury_model(df_features)
    predire_risque_blessure(df_features)
