"""
AthlytIQ — Anomaly Detector (Isolation Forest)
================================================
Détecte les baisses et hausses anormales de performance.

Utilise Isolation Forest pour identifier les matchs où la
performance d'un joueur dévie significativement de sa norme.

Sorties :
- Alertes de baisse de performance
- Alertes de hausse exceptionnelle
- Score d'anomalie (-1 = anomalie, 1 = normal)
"""

import sys
from pathlib import Path
# Configuration du chemin racine (3 niveaux au-dessus : models/ <- ml/ <- root/)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# Features clés pour la détection d'anomalies
ANOMALY_FEATURES = [
    'Rating', 'Form_Score', 'Goals', 'Assists',
    'Expected_Goals', 'Expected_Assists',
    'xG_Overperformance', 'xA_Overperformance',
    'Pass_Accuracy', 'Shots_Accuracy',
    'Ground_Duels_Won_Pct', 'Touches',
    'Defensive_Actions', 'Possession_Security',
    'Minutes_Played'
]

# Features de tendance pour détecter les changements de direction
TREND_FEATURES = [
    'Rating_Trend', 'Form_Score_MA3', 'Form_Score_MA5',
    'Rating_MA3', 'Rating_MA5', 'Rating_Slope5',
    'Goals_P90_Trend', 'xG_P90_Trend'
]


# ══════════════════════════════════════════════════════════════════════
# 2. ENTRAÎNEMENT DE L'ISOLATION FOREST
# ══════════════════════════════════════════════════════════════════════

def entrainer_anomaly_detector(df, contamination=0.1):
    """
    Entraîne un Isolation Forest sur l'ensemble du dataset.

    Args:
        df: DataFrame avec features
        contamination: Proportion attendue d'anomalies (0.1 = 10%)

    Returns:
        dict avec le modèle et le scaler
    """
    # Sélection des features disponibles
    available = [f for f in ANOMALY_FEATURES + TREND_FEATURES if f in df.columns]

    if len(available) < 3:
        print("   ⚠️ Pas assez de features pour la détection d'anomalies")
        return None

    X = df[available].copy()
    X = X.fillna(X.median()).replace([np.inf, -np.inf], 0)

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest - AMÉLIORÉ: contamination basée sur la volatilité historique
    # Calcule automatiquement une contamination plus précise basée sur l'écart-type
    form_std = df['Form_Score'].std() if 'Form_Score' in df.columns else 15
    auto_contamination = min(0.1, max(0.02, form_std / 200))  # Entre 2% et 10%

    model = IsolationForest(
        n_estimators=200,
        contamination=auto_contamination,  # AMÉLIORÉ: contamination adaptative
        max_features=min(1.0, 10 / len(available)),
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled)

    # Sauvegarde
    save_dir = ROOT / "LM" / "models" / "saved"
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, save_dir / "anomaly_detector.joblib")
    joblib.dump(scaler, save_dir / "anomaly_scaler.joblib")
    joblib.dump(available, save_dir / "anomaly_features.joblib")

    # Stats
    scores = model.decision_function(X_scaled)
    labels = model.predict(X_scaled)
    n_anomalies = (labels == -1).sum()

    print(f"\n   🚨 Isolation Forest entraîné :")
    print(f"      Features     : {len(available)}")
    print(f"      Échantillons : {len(X)}")
    print(f"      Anomalies    : {n_anomalies} ({n_anomalies / len(X) * 100:.1f}%)")
    print(f"      Contamination adaptative: {auto_contamination:.1%}")
    print(f"      💾 Sauvegardé dans {save_dir}")

    return {
        'model': model,
        'scaler': scaler,
        'features': available
    }


# ══════════════════════════════════════════════════════════════════════
# 3. DÉTECTION D'ANOMALIES
# ══════════════════════════════════════════════════════════════════════

def detecter_anomalies(df):
    """
    Détecte les anomalies de performance dans les données.

    Combine :
    1. Isolation Forest (anomalies statistiques globales)
    2. Détection par seuil (variation brutale du Form Score)

    AMÉLIORATION: Logique "Match Manqué" plus robuste pour distinguer
    blessure vs repos/rotation.
    """
    save_dir = ROOT / "LM" / "models" / "saved"

    try:
        model = joblib.load(save_dir / "anomaly_detector.joblib")
        scaler = joblib.load(save_dir / "anomaly_scaler.joblib")
        features = joblib.load(save_dir / "anomaly_features.joblib")
    except FileNotFoundError:
        print("   ⚠️ Modèle d'anomalies non trouvé. Entraînement automatique...")
        result = entrainer_anomaly_detector(df)
        if result is None:
            return pd.DataFrame()
        model = result['model']
        scaler = result['scaler']
        features = result['features']

    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    X = X.fillna(X.median()).replace([np.inf, -np.inf], 0)
    X_scaled = scaler.transform(X)

    # Scores d'anomalie
    df = df.copy()

    if 'Form_Score_MA5' not in df.columns and 'Nom' in df.columns:
        df = df.sort_values(['Nom', 'Match_Date'])
        df['Form_Score_MA5'] = df.groupby('Nom')['Form_Score'].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df['Anomaly_Score'] = model.decision_function(X_scaled)
    df['Is_Anomaly'] = model.predict(X_scaled)  # -1 = anomalie

    # ── Analyse des anomalies ──
    alertes = []

    for _, row in df[df['Is_Anomaly'] == -1].iterrows():
        # Déterminer le type d'anomalie en utilisant la volatilité historique du joueur
        nom = row.get('Nom', 'Unknown')
        player_data = df[df['Nom'] == nom]
        player_std = player_data['Form_Score'].std() if 'Form_Score' in player_data.columns else 10
        player_mean = player_data['Form_Score'].mean() if 'Form_Score' in player_data.columns else 50

        # Seuil adaptatif: 1.5 écart-type comme limite de normalité
        seuil_critique = player_mean - 1.5 * player_std
        seuil_modere = player_mean - 1.0 * player_std

        form = row.get('Form_Score', 50)
        form_ma5 = row.get('Form_Score_MA5', form)
        rating = row.get('Rating', 6.5)
        rating_ma5 = row.get('Rating_MA5', rating)

        if form < seuil_critique:  # Seuil adaptatif
            type_anomalie = 'BAISSE_CRITIQUE'
            severite = 'HAUTE'
            emoji = '🔴'
            message = f"Baisse critique de forme ({form:.1f} vs moy. {player_mean:.1f})"
        elif form < seuil_modere:
            type_anomalie = 'BAISSE_MODEREE'
            severite = 'MOYENNE'
            emoji = '🟠'
            message = f"Baisse modérée ({form:.1f} vs moy. {player_mean:.1f})"
        elif form > player_mean + 1.5 * player_std:
            type_anomalie = 'HAUSSE_EXCEPTIONNELLE'
            severite = 'INFO'
            emoji = '🟢'
            message = f"Performance exceptionnelle ({form:.1f} vs moy. {player_mean:.1f})"
        else:
            continue

        # Mapping CSS: le DASHBOARD utilise .alert-item.high/.medium/.low
        css_severity_map = {'HAUTE': 'high', 'MOYENNE': 'medium', 'BASSE': 'low', 'INFO': 'info'}

        alertes.append({
            'Nom': nom,
            'Match_Date': row.get('Match_Date', ''),
            'Type': type_anomalie,
            'Severité': severite,
            'Severite_CSS': css_severity_map.get(severite, 'info'),
            'Emoji': emoji,
            'Message': message,
            'Form_Score': form,
            'Form_Score_MA5': form_ma5,
            'Rating': rating,
            'Anomaly_Score': row['Anomaly_Score'],
            'Adversaire': f"{row.get('Home_Team', '')} vs {row.get('Away_Team', '')}"
        })

    df_alertes = pd.DataFrame(alertes)

    if not df_alertes.empty:
        # 1. Ne garder que les alertes récentes (45 derniers jours max)
        df_alertes['Match_Date'] = pd.to_datetime(df_alertes['Match_Date'])
        if not df_alertes.empty:
            max_date = df_alertes['Match_Date'].max()
            df_alertes = df_alertes[df_alertes['Match_Date'] >= (max_date - pd.Timedelta(days=45))]

            # 2. Garder uniquement l'alerte LA PLUS RÉCENTE par joueur
            df_alertes = df_alertes.sort_values('Match_Date', ascending=False)
            df_alertes = df_alertes.drop_duplicates(subset=['Nom'], keep='first')

            # 3. Trier par sévérité
            severity_order = {'HAUTE': 0, 'MOYENNE': 1, 'BASSE': 2, 'INFO': 3}
            df_alertes['_order'] = df_alertes['Severité'].map(severity_order)
            df_alertes = df_alertes.sort_values(['_order', 'Match_Date'], ascending=[True, False])
            df_alertes = df_alertes.drop(columns=['_order'])

            # Re-convertir Match_Date
            df_alertes['Match_Date'] = df_alertes['Match_Date'].dt.strftime('%Y-%m-%d')

            # --- AMÉLIORATION: Correction Inactivité ---
            # Distingue blessure (absence prolongée) vs repos (rotation)
            global_max_date = pd.to_datetime(df['Match_Date']).max()
            last_matches = pd.to_datetime(df.groupby('Nom')['Match_Date'].max())

            # Vérifier l'historique de titularisation du joueur
            for idx, alerte in df_alertes.iterrows():
                nom = alerte['Nom']
                last_m_player = last_matches[nom]
                days_inactive = (global_max_date - last_m_player).days

                # AMÉLIORATION: Seuil plus précis et distinction repos/blessure
                if days_inactive > 0:
                    # Calculer les matchs manqués vs matchs joués récemment
                    player_recent = df[df['Nom'] == nom].sort_values('Match_Date').tail(5)
                    avg_minutes_recent = player_recent['Minutes_Played'].mean() if len(player_recent) > 0 else 90

                    if days_inactive >= 14:  # Blessure probable (>14 jours)
                        df_alertes.at[idx, 'Type'] = 'ALERTE_SANS_TEMPS_DE_JEU'
                        df_alertes.at[idx, 'Severité'] = 'HAUTE'
                        df_alertes.at[idx, 'Emoji'] = '🏥'
                        df_alertes.at[idx, 'Message'] = f"Blessure prolongée ({days_inactive} jours d'absence)"
                    elif days_inactive >= 3 and avg_minutes_recent >= 60:
                        # Joueur régulièrement titularisé absent = possible blessure
                        df_alertes.at[idx, 'Type'] = 'ALERTE_BLESSURE_PROBABLE'
                        df_alertes.at[idx, 'Severité'] = 'MOYENNE'
                        df_alertes.at[idx, 'Emoji'] = '⚠️'
                        df_alertes.at[idx, 'Message'] = f"Absence inhabituelle ({days_inactive} jours)"
                    else:
                        # Joueur avec peu de minutes = repos/rotation, pas alerte critique
                        df_alertes.at[idx, 'Type'] = 'REPOS_ROTATION'
                        df_alertes.at[idx, 'Severité'] = 'BASSE'
                        df_alertes.at[idx, 'Emoji'] = '🟡'
                        df_alertes.at[idx, 'Message'] = f"Repos/rotation (titularisations récentes: {avg_minutes_recent:.0f} min avg)"

            print(f"\n   🚨 {len(df_alertes)} alertes récentes détectées :")
            print("   " + "─" * 65)
            for _, alerte in df_alertes.head(15).iterrows():
                print(f"   {alerte['Emoji']} [{alerte['Severité']:<7}] "
                      f"{alerte['Nom']:<25} {alerte['Message']}")

    return df_alertes


# ══════════════════════════════════════════════════════════════════════
# 4. ANALYSE DE TENDANCE PAR JOUEUR
# ══════════════════════════════════════════════════════════════════════

def analyser_tendance_joueur(df, nom_joueur):
    """
    Analyse approfondie de la tendance de forme d'un joueur spécifique.

    Returns:
        dict avec la tendance, le statut et les recommandations
    """
    player_data = df[df['Nom'] == nom_joueur].sort_values('Match_Date')

    if player_data.empty:
        return None

    last = player_data.iloc[-1]

    # Pente de la courbe de forme
    if len(player_data) >= 5:
        y = player_data['Form_Score'].tail(5).values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0

    # Volatilité récente
    if len(player_data) >= 3:
        volatilite = player_data['Form_Score'].tail(5).std()
    else:
        volatilite = 0

    # Statut basé sur la pente avec seuils adaptatifs
    if slope > 2:
        statut = 'EN_PROGRESSION'
        emoji = '🚀'
    elif slope > 0.5:
        statut = 'FORME_ASCENDANTE'
        emoji = '📈'
    elif slope > -0.5:
        statut = 'STABLE'
        emoji = '➡️'
    elif slope > -2:
        statut = 'LEGERE_BAISSE'
        emoji = '📉'
    else:
        statut = 'DECLINE'
        emoji = '⚠️'

    return {
        'Nom': nom_joueur,
        'Statut': statut,
        'Emoji': emoji,
        'Form_Actuel': last.get('Form_Score', 0),
        'Form_MA5': last.get('Form_Score_MA5', 0),
        'Pente': round(slope, 3),
        'Volatilite': round(volatilite, 2),
        'Nb_Matchs': len(player_data),
        'Rating_Actuel': last.get('Rating', 0),
    }


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data/processed/features_dataset.csv")
    result = entrainer_anomaly_detector(df)
    alertes = detecter_anomalies(df)
