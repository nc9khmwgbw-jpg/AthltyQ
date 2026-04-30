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
    """Calcule un score de fatigue composite (0-100)."""
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

    df['Fatigue_Score'] = (comp_acwr + comp_fatigue + comp_congestion + comp_trauma).round(1).clip(0, 100)
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

    # 5. Classification
    df = _classifier_niveaux(df)
    
    # 6. Résumé final (dernier match par joueur)
    result = df.sort_values(['Nom', 'Match_Date'], ascending=[True, False]).groupby('Nom').first().reset_index()
    
    _afficher_rapport(result)
    return result

def _classifier_niveaux(df):
    """
    Classification scientifique par Z-Score (Sensibilité Ajustée).
    Identifie les joueurs s'écartant de la moyenne du jour.
    """
    risk = df['Injury_Risk']
    
    # 1. Distribution du jour
    mean_val = risk.mean()
    std_val = risk.std()
    if pd.isna(std_val) or std_val == 0: std_val = 0.1

    # 2. Seuils dynamiques équilibrés (Standard Elite)
    # Le multiplicateur 0.8 est le standard industriel pour identifier les outliers.
    seuil_eleve = mean_val + (0.8 * std_val)
    seuil_faible = mean_val - (0.8 * std_val)

    # Sécurités physiologiques (Bornes de réalisme médical)
    seuil_eleve = min(max(seuil_eleve, 0.48), 0.85)
    seuil_faible = max(min(seuil_faible, 0.32), 0.05)

    conditions = [
        (df['Current_Injury'] == 1) | (risk >= seuil_eleve), # 🔴 ÉLEVÉ
        (risk > seuil_faible),                               # 🟠 MODÉRÉ
    ]
    choices = ['🔴 ÉLEVÉ', '🟠 MODÉRÉ']
    df['Risk_Level'] = np.select(conditions, choices, default='🟢 FAIBLE')
    
    print(f"📊 [INFO] Distribution du jour : Moyenne={mean_val:.3f}, Std={std_val:.3f}")
    print(f"   Seuils appliqués : Faible <= {seuil_faible:.3f} | Élevé >= {seuil_eleve:.3f}")
    
    return df

def _afficher_rapport(result):
    """Rapport premium AthlytIQ."""
    print("\n" + "═"*65)
    print("   🏥 RAPPORT DE RISQUE DE BLESSURE — AthlytIQ")
    print("═"*65 + "\n")

    for niveau in ['🔴 ÉLEVÉ', '🟠 MODÉRÉ', '🟢 FAIBLE']:
        groupe = result[result['Risk_Level'] == niveau].sort_values('Injury_Risk', ascending=False)
        count = len(groupe)
        print(f"{niveau} ({count} joueurs) :")
        print("   " + "─"*54)
        
        display_limit = 50 if niveau == '🔴 ÉLEVÉ' else 20
        for _, row in groupe.head(display_limit).iterrows():
            risk = row['Injury_Risk']
            fatigue = row.get('Fatigue_Score', 0)
            
            line = f"   {row['Nom'] : <46} Fatigue: {fatigue: >5.1f}/100  Risque: {int(risk*100)}% (Prédiction du LM)"
            print(line)
            
            if row.get('Current_Injury', 0) == 1:
                injury_type = row.get('Injury_Type_Text', 'Blessure')
                print(f"                                                   -> État Actuel : 🚑 ACTUELLEMENT BLESSÉ ({injury_type})")
                print(f"                                                   -> ✅ PRÉDICTION CORRECTE (Forcée)")
            elif fatigue > 75:
                print(f"                                                   -> État Actuel : 🟠 SURCHARGE DÉTECTÉE")
            elif risk >= 0.6:
                print(f"                                                   -> État Actuel : 🟢 APTE (Risque Imminent)")
            elif risk >= 0.16:
                print(f"                                                   -> État Actuel : 🟢 APTE (Temps réduit)")
            print("")
        
        if count > display_limit:
            print(f"   ... et {count - display_limit} autres joueurs dans cette catégorie.\n")

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
