"""
AthlytIQ — Injury Predictor (Basé sur la Fatigue)
===================================================
Pipeline en 2 étapes :
  1. Prédiction de la FATIGUE  (ACWR, minutes cumulées, densité des matchs...)
  2. Prédiction du RISQUE DE BLESSURE (basé sur la fatigue + historique physique)

Features utilisées :
  - ACWR                    → ratio charge récente / charge chronique
  - Fatigue_Index           → minutes cumulées sur 5 matchs (normalisé 0-1)
  - Congestion_Risk         → risque si < 4 jours de repos entre matchs
  - Trauma_Index            → intensité des duels sur 3 matchs glissants
  - Medical_Risk_Score      → score de risque combiné (déjà calculé)
  - Usage_Factor            → exposition réelle du joueur
  - Days_Since_Last         → jours de récupération depuis le dernier match
  - Match_Num               → numéro de match (fatigue accumulée en saison)

Sortie :
  - Fatigue_Score           → 0-100 (score de fatigue estimé)
  - Injury_Risk             → 0-1   (probabilité de blessure)
  - Risk_Level              → 🟢 FAIBLE / 🟠 MODÉRÉ / 🔴 ÉLEVÉ
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SAVE_DIR = ROOT / "LM" / "models" / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 1. FEATURES DE FATIGUE
# ══════════════════════════════════════════════════════════════════════

FATIGUE_FEATURES = [
    'ACWR',                   # Charge récente / charge chronique
    'Fatigue_Index',          # Minutes cumulées normalisées
    'Congestion_Risk',        # Risque de congestion calendrier
    'Trauma_Index',           # Intensité des chocs physiques
    'Medical_Risk_Score',     # Score de risque médical global
    'Usage_Factor',           # Taux d'utilisation du joueur
    'Cumulative_Minutes_21d', # Minutes cumulées sur 5 derniers matchs
    'Match_Density',          # Densité des matchs (jours entre matchs)
    'Days_Since_Last',        # Jours de repos depuis dernier match
    'Match_Num',              # Numéro de match dans la saison
    'Minutes_Played',         # Minutes jouées ce match
    'Duel_Intensity',         # Intensité des duels ce match
    'Age',                    # Âge du joueur
    'Age_Factor',             # Multiplicateur de risque selon l'âge
]


# ══════════════════════════════════════════════════════════════════════
# 2. CALCUL DU SCORE DE FATIGUE (0-100)
# ══════════════════════════════════════════════════════════════════════

def calculer_fatigue_score(df):
    """
    Calcule un score de fatigue composite (0-100) pour chaque ligne.

    Logique :
    - ACWR hors zone goldilocks (0.8-1.3) → surcharge ou sous-charge
    - Fatigue_Index élevé → trop de minutes accumulées
    - Congestion_Risk → matchs trop rapprochés
    - Trauma_Index → trop de chocs physiques
    """
    df = df.copy()

    # Composante 1 : ACWR (40% du score)
    # Zone idéale : 0.8-1.3 → score 0. Hors zone → score monte
    if 'ACWR' in df.columns:
        acwr = df['ACWR'].clip(0, 3)
        # Distance par rapport à la zone goldilocks
        acwr_risk = np.where(
            acwr < 0.8, (0.8 - acwr) / 0.8,           # Sous-charge
            np.where(acwr > 1.3, (acwr - 1.3) / 1.7,  # Surcharge
            0)
        )
        composante_acwr = acwr_risk.clip(0, 1) * 40
    else:
        composante_acwr = 20  # Valeur neutre

    # Composante 2 : Fatigue accumulée (35% du score)
    if 'Fatigue_Index' in df.columns:
        composante_fatigue = df['Fatigue_Index'].clip(0, 1) * 35
    else:
        composante_fatigue = 0

    # Composante 3 : Congestion calendrier (15% du score)
    if 'Congestion_Risk' in df.columns:
        # Congestion_Risk = 1.0 (normal) ou 1.5 (danger)
        composante_congestion = ((df['Congestion_Risk'] - 1.0) / 0.5).clip(0, 1) * 15
    else:
        composante_congestion = 0

    # Composante 4 : Trauma physique (10% du score)
    if 'Trauma_Index' in df.columns:
        trauma_norm = (df['Trauma_Index'] / df['Trauma_Index'].quantile(0.95).clip(1)).clip(0, 1)
        composante_trauma = trauma_norm * 10
    else:
        composante_trauma = 0

    # Composante 5 : Âge (10% du score)
    if 'Age_Factor' in df.columns:
        composante_age = ((df['Age_Factor'] - 1.0) / 0.3).clip(0, 1) * 10
    else:
        composante_age = 0

    df['Fatigue_Score'] = (
        composante_acwr + composante_fatigue + composante_congestion + composante_trauma + composante_age
    ).round(1).clip(0, 100)

    return df


# ══════════════════════════════════════════════════════════════════════
# 3. ENTRAÎNEMENT DU MODÈLE DE BLESSURE
# ══════════════════════════════════════════════════════════════════════

def entrainer_injury_model(df):
    """
    Entraîne un modèle de prédiction de blessure basé sur les features de fatigue.

    Target : Target_Injury_Occurred (1 = risque de blessure, 0 = normal)

    Returns:
        dict avec le modèle, le scaler et les features utilisées
    """
    print("\n" + "─" * 60)
    print("   🏥 ENTRAÎNEMENT — Modèle de Prédiction de Blessure")
    print("─" * 60)

    # 1. Calcul du Fatigue Score
    df = calculer_fatigue_score(df)

    # 2. Sélection des features disponibles
    features_disponibles = [f for f in FATIGUE_FEATURES + ['Fatigue_Score']
                            if f in df.columns]

    if 'Target_Injury_Occurred' not in df.columns:
        print("   ⚠️  Target 'Target_Injury_Occurred' manquant.")
        print("   💡 Exécutez d'abord feature_engineering.py")
        return None

    # 3. Préparation des données
    df_clean = df[features_disponibles + ['Target_Injury_Occurred']].dropna()

    if len(df_clean) < 50:
        print(f"   ⚠️  Pas assez de données ({len(df_clean)} lignes). Minimum 50 requis.")
        return None

    X = df_clean[features_disponibles]
    y = df_clean['Target_Injury_Occurred']

    print(f"   📊 Dataset : {len(X)} lignes | {y.sum()} cas à risque ({y.mean()*100:.1f}%)")

    # 4. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Gestion du déséquilibre de classes (blessures rares)
    classes = np.array([0, 1])
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}

    # 6. Ensemble de modèles
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Gradient Boosting (principal)
    model_gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model_gb.fit(X_train_sc, y_train)

    # Random Forest (secondaire)
    model_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    model_rf.fit(X_train_sc, y_train)

    # Prédictions ensemblées (moyenne des probabilités)
    prob_gb = model_gb.predict_proba(X_test_sc)[:, 1]
    prob_rf = model_rf.predict_proba(X_test_sc)[:, 1]
    prob_ensemble = (prob_gb * 0.6 + prob_rf * 0.4)
    y_pred = (prob_ensemble >= 0.5).astype(int)

    # 7. Évaluation
    auc = roc_auc_score(y_test, prob_ensemble)
    print(f"\n   📈 Performance :")
    print(f"      AUC-ROC : {auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'À Risque'])}")

    # 8. Importance des features
    importances = model_gb.feature_importances_
    feat_imp = sorted(zip(features_disponibles, importances), key=lambda x: x[1], reverse=True)
    print("   🔑 Top 5 features :")
    for feat, imp in feat_imp[:5]:
        bar = "█" * int(imp * 30)
        print(f"      {feat:<30} {bar} {imp:.3f}")

    # 9. Sauvegarde
    joblib.dump(model_gb, SAVE_DIR / "injury_model_gb.joblib")
    joblib.dump(model_rf, SAVE_DIR / "injury_model_rf.joblib")
    joblib.dump(scaler, SAVE_DIR / "injury_scaler.joblib")
    joblib.dump(features_disponibles, SAVE_DIR / "injury_features.joblib")

    print(f"\n   💾 Modèles sauvegardés dans {SAVE_DIR}")

    return {
        'model_gb': model_gb,
        'model_rf': model_rf,
        'scaler': scaler,
        'features': features_disponibles,
        'auc': auc
    }


# ══════════════════════════════════════════════════════════════════════
# 4. PRÉDICTION DU RISQUE DE BLESSURE
# ══════════════════════════════════════════════════════════════════════

def predire_risque_blessure(df):
    """
    Prédit le risque de blessure pour chaque joueur.

    Pipeline :
      1. Calcule le Fatigue_Score
      2. Charge les modèles entraînés
      3. Prédit la probabilité de blessure
      4. Classe en 3 niveaux : FAIBLE / MODÉRÉ / ÉLEVÉ

    Returns:
        DataFrame avec colonnes :
          - Fatigue_Score      (0-100)
          - Injury_Risk        (0-1)
          - Risk_Level         (FAIBLE / MODÉRÉ / ÉLEVÉ)
          - Risk_Emoji         (🟢 / 🟠 / 🔴)
          - Risk_Message       (message explicatif)
    """
    # 1. Calcul du score de fatigue
    df = calculer_fatigue_score(df)

    # 2. Chargement des modèles
    try:
        model_gb = joblib.load(SAVE_DIR / "injury_model_gb.joblib")
        model_rf = joblib.load(SAVE_DIR / "injury_model_rf.joblib")
        scaler   = joblib.load(SAVE_DIR / "injury_scaler.joblib")
        features = joblib.load(SAVE_DIR / "injury_features.joblib")
    except FileNotFoundError:
        print("   ⚠️  Modèles non trouvés. Entraînement automatique...")
        result = entrainer_injury_model(df)
        if result is None:
            return _predire_par_regles(df)
        model_gb = result['model_gb']
        model_rf = result['model_rf']
        scaler   = result['scaler']
        features = result['features']

    # 3. Préparation des features
    features_dispo = [f for f in features if f in df.columns]
    X = df[features_dispo].fillna(0).replace([np.inf, -np.inf], 0)
    X_sc = scaler.transform(X)

    # 4. Prédiction ensemblée
    prob_gb = model_gb.predict_proba(X_sc)[:, 1]
    prob_rf = model_rf.predict_proba(X_sc)[:, 1]
    df['Injury_Risk'] = (prob_gb * 0.6 + prob_rf * 0.4).round(3)

    # 5. Classification des niveaux de risque
    df = _classifier_niveaux(df)

    # 6. Résumé par joueur (dernier match connu)
    result = _resumer_par_joueur(df)

    _afficher_rapport(result)
    return result


def _predire_par_regles(df):
    """Fallback basé sur les règles (si pas de modèle ML)."""
    df = df.copy()
    if 'Medical_Risk_Score' in df.columns:
        df['Injury_Risk'] = df['Medical_Risk_Score']
    elif 'Fatigue_Score' in df.columns:
        df['Injury_Risk'] = (df['Fatigue_Score'] / 100).clip(0, 1)
    else:
        df['Injury_Risk'] = 0.2
    return _classifier_niveaux(df)


def _classifier_niveaux(df):
    """Classifie le risque en 3 niveaux."""
    df = df.copy()

    conditions = [
        df['Injury_Risk'] >= 0.65,
        df['Injury_Risk'] >= 0.35,
    ]

    df['Risk_Level'] = np.select(
        conditions,
        ['ÉLEVÉ', 'MODÉRÉ'],
        default='FAIBLE'
    )
    df['Risk_Emoji'] = np.select(
        conditions,
        ['🔴', '🟠'],
        default='🟢'
    )
    df['Risk_Message'] = np.select(
        conditions,
        [
            'Risque élevé de blessure — repos recommandé',
            'Surveiller la charge d\'entraînement',
        ],
        default='Joueur en bonne condition physique'
    )

    return df


def _resumer_par_joueur(df):
    """Garde uniquement le dernier match par joueur."""
    if 'Match_Date' in df.columns:
        df['Match_Date'] = pd.to_datetime(df['Match_Date'])
        result = (
            df.sort_values('Match_Date', ascending=False)
            .drop_duplicates(subset=['Nom'], keep='first')
        )
    else:
        result = df.drop_duplicates(subset=['Nom'], keep='last')

    cols_sortie = ['Nom']
    for col in ['Equipe', 'Match_Date', 'Fatigue_Score', 'ACWR',
                'Fatigue_Index', 'Medical_Risk_Score', 'Injury_Risk',
                'Risk_Level', 'Risk_Emoji', 'Risk_Message']:
        if col in result.columns:
            cols_sortie.append(col)

    return result[cols_sortie].sort_values('Injury_Risk', ascending=False)


def _afficher_rapport(df):
    """Affiche un rapport lisible dans le terminal."""
    print("\n" + "═" * 65)
    print("   🏥 RAPPORT DE RISQUE DE BLESSURE — AthlytIQ")
    print("═" * 65)

    for niveau, emoji in [('ÉLEVÉ', '🔴'), ('MODÉRÉ', '🟠'), ('FAIBLE', '🟢')]:
        groupe = df[df['Risk_Level'] == niveau]
        if groupe.empty:
            continue
        print(f"\n{emoji} {niveau} ({len(groupe)} joueurs) :")
        print("   " + "─" * 55)
        for _, row in groupe.head(10).iterrows():
            fatigue = row.get('Fatigue_Score', '?')
            risk = row.get('Injury_Risk', 0)
            equipe = row.get('Equipe', '')
            print(f"   {row['Nom']:<25} {equipe:<20} "
                  f"Fatigue: {fatigue:>5.1f}/100  Risque: {risk:.0%}")


# ══════════════════════════════════════════════════════════════════════
# 5. CLASSE PRINCIPALE (API)
# ══════════════════════════════════════════════════════════════════════

class InjuryPredictor:
    """Interface simple pour utiliser le modèle depuis d'autres modules."""

    def __init__(self):
        self.model_gb = None
        self.model_rf = None
        self.scaler   = None
        self.features  = None
        self._charger()

    def _charger(self):
        try:
            self.model_gb = joblib.load(SAVE_DIR / "injury_model_gb.joblib")
            self.model_rf = joblib.load(SAVE_DIR / "injury_model_rf.joblib")
            self.scaler   = joblib.load(SAVE_DIR / "injury_scaler.joblib")
            self.features  = joblib.load(SAVE_DIR / "injury_features.joblib")
        except FileNotFoundError:
            pass

    def predict(self, player_features: dict) -> dict:
        """
        Prédit le risque pour un joueur donné.

        Args:
            player_features: dict avec les stats du joueur

        Returns:
            dict {fatigue_score, injury_risk, risk_level, message}
        """
        df = pd.DataFrame([player_features])
        df = calculer_fatigue_score(df)

        if self.model_gb is None:
            risk = float(df['Fatigue_Score'].iloc[0]) / 100
        else:
            feats = [f for f in self.features if f in df.columns]
            X = df[feats].fillna(0)
            X_sc = self.scaler.transform(X)
            p_gb = self.model_gb.predict_proba(X_sc)[0, 1]
            p_rf = self.model_rf.predict_proba(X_sc)[0, 1]
            risk = float(p_gb * 0.6 + p_rf * 0.4)

        fatigue_score = float(df['Fatigue_Score'].iloc[0])

        if risk >= 0.65:
            level, emoji, msg = 'ÉLEVÉ', '🔴', 'Repos recommandé'
        elif risk >= 0.35:
            level, emoji, msg = 'MODÉRÉ', '🟠', 'Surveiller la charge'
        else:
            level, emoji, msg = 'FAIBLE', '🟢', 'Bonne condition physique'

        return {
            'fatigue_score': round(fatigue_score, 1),
            'injury_risk':   round(risk, 3),
            'risk_level':    level,
            'risk_emoji':    emoji,
            'message':       msg
        }


# ══════════════════════════════════════════════════════════════════════
# 6. TEST STANDALONE
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_path = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"

    if not data_path.exists():
        print(f"❌ Fichier introuvable : {data_path}")
        print("💡 Exécutez d'abord : python DATA_PIPELINE/NETTOYAGE/scripts/data_cleaner.py")
        sys.exit(1)

    print(f"📂 Chargement des données : {data_path}")
    df_raw = pd.read_csv(data_path)
    df_raw['Match_Date'] = pd.to_datetime(df_raw['Match_Date'], errors='coerce')

    # Feature engineering
    from LM.models.feature_engineering import run_feature_engineering
    df_features = run_feature_engineering(df_raw)

    # Entraînement
    entrainer_injury_model(df_features)

    # Prédictions
    predictions = predire_risque_blessure(df_features)
    predictions.to_csv(ROOT / "data" / "predictions_blessures.csv", index=False)
    print(f"\n✅ Prédictions sauvegardées.")
