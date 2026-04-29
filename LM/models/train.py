"""
AthlytIQ — Script d'Entraînement Global
==========================================
Entraîne les 3 modèles du Module 1 :
1. XGBoost + Random Forest (prédiction tabulaire)
2. LSTM (séries temporelles)
3. Isolation Forest (détection d'anomalies)
"""

import sys
from pathlib import Path
# Configuration du chemin racine (3 niveaux au-dessus : models/ <- ml/ <- root/)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd


def entrainer_tous_les_modeles(df_features):
    """
    Entraîne les 3 modèles de prédiction de performance.

    Args:
        df_features: DataFrame enrichi (sortie du feature engineering)
    """
    print("\n" + "═" * 70)
    print("   🤖 ENTRAÎNEMENT DES MODÈLES — MODULE 1")
    print("═" * 70)

    # ── 1. Isolation Forest ──
    # Détecte les comportements anormaux qui précèdent souvent une blessure
    print("\n" + "─" * 50)
    print("   🚨 MODÈLE 1 : Isolation Forest (Détection d'Anomalies Physiques)")
    print("─" * 50)

    from LM.models.anomaly_detector import entrainer_anomaly_detector
    anomaly_result = entrainer_anomaly_detector(df_features)

    # ── 2. Modèle de Risque de Blessure ──
    # Le cœur du projet : Prédit la probabilité de blessure par match
    print("\n" + "─" * 50)
    print("   🚑 MODÈLE 2 : Ensemble Classifier (XGB + LGBM + SMOTE)")
    print("─" * 50)

    from LM.models.injury_predictor import entrainer_injury_model
    injury_result = entrainer_injury_model(df_features)

    # ── Résumé ──
    print("\n" + "═" * 70)
    print("   ✅ ENTRAÎNEMENT MÉDICAL TERMINÉ")
    print("═" * 70)
    print("   Modèles sauvegardés dans : LM/models/")
    print("   Pour lancer les prédictions : python -m LM.pipeline.data_pipeline")


if __name__ == "__main__":
    features_path = ROOT / "data" / "processed" / "features_dataset.csv"

    if not features_path.exists():
        print("❌ Dataset de features non trouvé.")
        print("   Exécutez d'abord le pipeline de données :")
        print("   python -m LM.pipeline.data_pipeline")
        sys.exit(1)

    df = pd.read_csv(features_path)
    print(f"📊 Dataset chargé : {df.shape}")

    entrainer_tous_les_modeles(df)
