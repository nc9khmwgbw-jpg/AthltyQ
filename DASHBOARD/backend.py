import sys
import os
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import time
import joblib
import unicodedata
import re

def normalize_team_name(name: str) -> str:
    n = ''.join(c for c in unicodedata.normalize('NFD', str(name)) if unicodedata.category(c) != 'Mn')
    n = re.sub(r'[^a-zA-Z0-9]', '', n)
    return n.lower()

from fastapi.responses import HTMLResponse, FileResponse

# Add root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LM.models.fatigue_predictor import FatiguePredictor
from LM.models.feature_engineering import run_feature_engineering
from LM2.similarity_engine import SimilarityEngine

app = FastAPI(title="AthltyQ API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Tuple, Optional, Dict, Any, cast

CACHE_MODELS: Dict[str, Dict[str, Any]] = {
    "poly": {"results": None, "features": None, "last_update": 0.0},
    "rf": {"results": None, "features": None, "last_update": 0.0},
    "lgbm": {"results": None, "features": None, "last_update": 0.0},
}
CACHE_SIM_ENGINE = None
CACHE_TTL = 300
CACHE_INJURY_DF = None

predictor = FatiguePredictor()

# Mock data for physiological markers
MOCK_PHYSIO = {
    "avg_sleep_hours": 7.2,
    "avg_hrv": 62,
    "hrv_rmssd": 58,
    "sleep_duration": 7.5,
    "sleep_quality": 85
}

def get_data(model_type: str = 'lgbm') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Fetch and process data with caching to prevent OOM/Killed errors."""
    global CACHE_MODELS, CACHE_SIM_ENGINE
    now = time.time()
    
    PROCESSED_PATH = ROOT / "data" / "processed" / "features_dataset.csv"
    RAW_PATH = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"
    
    # LOGIQUE AUTO-UPDATE : On vérifie si le fichier RAW est plus récent que le PROCESSED
    should_reprocess = False
    if RAW_PATH.exists():
        if not PROCESSED_PATH.exists():
            should_reprocess = True
        else:
            raw_mtime = os.path.getmtime(RAW_PATH)
            proc_mtime = os.path.getmtime(PROCESSED_PATH)
            if raw_mtime > proc_mtime:
                print(f"🔄 Dashboard: Nouvelles données détectées ({RAW_PATH}). Mise à jour du cache...")
                should_reprocess = True

    # Vérification du cache en mémoire : pas de limite de temps (TTL infini)
    if CACHE_MODELS[model_type]["results"] is not None and not should_reprocess:
        # On vérifie seulement que le fichier PROCESSED n'a pas été modifié depuis le dernier chargement en mémoire
        if PROCESSED_PATH.exists() and os.path.getmtime(PROCESSED_PATH) <= CACHE_MODELS[model_type]["last_update"]:
            return CACHE_MODELS[model_type]["results"], CACHE_MODELS[model_type]["features"]

    print(f"Dashboard: Tentative de chargement des données...")
    
    if PROCESSED_PATH.exists() and not should_reprocess:
        print(f"Dashboard: Chargement des features pré-traitées ({PROCESSED_PATH})")
        df_features = pd.read_csv(PROCESSED_PATH, low_memory=False)
        df_features['Match_Date'] = pd.to_datetime(df_features['Match_Date'], errors='coerce')
        
        # 1. Calcul de la Fatigue IA (0-100)
        df_features = df_features.copy()
        predictions = predictor.predict(df_features, model_type=model_type)
        if predictions is None:
            df_features['Fatigue_IA'] = 0.0
        else:
            df_features['Fatigue_IA'] = pd.Series(predictions).fillna(0).values
        
        # Cast explicite et conversion en liste pour satisfaire l'IDE
        fatigue_vals = np.array(df_features['Fatigue_IA'], dtype=float)
        fatigue_final = np.clip(fatigue_vals, 0, 100).tolist()
        df_features['Fatigue_IA'] = fatigue_final
        df_features['Fatigue_Score'] = fatigue_final
        
        # 2. Calcul du Trauma Index (Historique médical)
        trauma_cols = ['Nb_Blessures_Musculaires_12m', 'Trauma_Index']
        present_trauma_cols = [c for c in trauma_cols if c in df_features.columns]
        
        if 'Jours_Depuis_Blessure' in df_features.columns:
            days_since = pd.to_numeric(df_features['Jours_Depuis_Blessure'], errors='coerce').fillna(365)
            # Cast float pour np.exp
            days_vals = np.array(days_since, dtype=float)
            injury_recency_score = 100 * np.exp(-days_vals / 30.0)
        else:
            injury_recency_score = 0.0

        if present_trauma_cols:
            trauma_sum = df_features[present_trauma_cols].sum(axis=1).fillna(0)
            trauma_vals = np.array(trauma_sum, dtype=float)
            trauma_score = (trauma_vals * 20) + injury_recency_score
        else:
            # Remplacement de pd.Series par np.full pour satisfaire le linter
            trauma_score = np.full(len(df_features), injury_recency_score, dtype=float)

        # 3. Calcul de l'Anomalie ACWR
        if 'ACWR' in df_features.columns:
            acwr_val = pd.to_numeric(df_features['ACWR'], errors='coerce').fillna(1.0)
        else:
            acwr_val = np.full(len(df_features), 1.0, dtype=float)
        
        acwr_vals = np.array(acwr_val, dtype=float)
        acwr_stress = np.clip((np.abs(acwr_vals - 1.0) * 50.0), 0, 30)

        # 4. Score de Risque Médical Composite
        fatigue_part = np.array(df_features['Fatigue_IA'], dtype=float) * 0.50
        trauma_part  = np.array(trauma_score, dtype=float) * 0.30
        acwr_part    = np.array(acwr_stress, dtype=float) * 0.20

        # Normalisation finale
        medical_risk = (fatigue_part + trauma_part + acwr_part) / 100.0
        df_features['Medical_Risk_Score'] = np.clip(medical_risk, 0, 1).tolist()
        df_features['Injury_Risk'] = df_features['Medical_Risk_Score']

        # 5. Classification des niveaux et Statuts (25% / 60% / 85%)
        df_features['Risk_Level'] = 'FAIBLE'
        df_features['Status'] = 'FULL TRAINING'
        
        risk_vals = np.array(df_features['Medical_Risk_Score'], dtype=float)
        
        # Conversion en listes pour satisfaire le linter de .loc
        # Nouveaux seuils calibrés sur la distribution réelle (Max: ~0.62)
        mask_mod  = ((risk_vals > 0.15) & (risk_vals <= 0.35)).tolist()
        mask_high = ((risk_vals > 0.35) & (risk_vals <= 0.60)).tolist()
        mask_crit = (risk_vals > 0.60).tolist()
        
        df_features.loc[mask_mod, 'Risk_Level'] = 'MODÉRÉ'
        df_features.loc[mask_mod, 'Status'] = 'VIGILANCE'
        
        df_features.loc[mask_high, 'Risk_Level'] = 'ÉLEVÉ'
        df_features.loc[mask_high, 'Status'] = 'REST FORCE'
        
        df_features.loc[mask_crit, 'Risk_Level'] = 'ÉLEVÉ'
        df_features.loc[mask_crit, 'Status'] = 'ALERTE BLESSURE'

        # Pour le dashboard, on groupe par joueur (dernier match)
        results = df_features.sort_values(['Nom', 'Match_Date'], ascending=[True, False]).groupby('Nom').first().reset_index()
        
    elif RAW_PATH.exists():
        print(f"Dashboard: Fichier pré-traité absent. Lancement du Feature Engineering sur {RAW_PATH}")
        df = pd.read_csv(RAW_PATH, low_memory=False)
        df['Match_Date'] = pd.to_datetime(df['Match_Date'], errors='coerce')
        # Feature Engineering (Version Médicale)
        df_features = run_feature_engineering(df)
        
        # Sauvegarde au chemin ABSOLU défini par ROOT
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(PROCESSED_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ Dashboard: Dataset mis à jour et sauvegardé à : {PROCESSED_PATH}")
        
        # 1. Calcul de la Fatigue IA (0-100)
        predictions = predictor.predict(df_features, model_type=model_type)
        df_features = df_features.copy()
        if predictions is None:
            df_features['Fatigue_IA'] = 0.0
        else:
            df_features['Fatigue_IA'] = pd.Series(predictions).fillna(0).values
        
        fatigue_vals_upd = np.array(df_features['Fatigue_IA'], dtype=float)
        fatigue_final_upd = np.clip(fatigue_vals_upd, 0, 100).tolist()
        df_features['Fatigue_IA'] = fatigue_final_upd
        df_features['Fatigue_Score'] = fatigue_final_upd

        # 2. Calcul du Trauma Index (Historique médical)
        trauma_cols = ['Nb_Blessures_Musculaires_12m', 'Trauma_Index']
        present_trauma_cols = [c for c in trauma_cols if c in df_features.columns]
        
        if 'Jours_Depuis_Blessure' in df_features.columns:
            days_since = pd.to_numeric(df_features['Jours_Depuis_Blessure'], errors='coerce').fillna(365)
            days_vals_upd = np.array(days_since, dtype=float)
            injury_recency_score = 100 * np.exp(-days_vals_upd / 30.0)
        else:
            injury_recency_score = 0.0

        if present_trauma_cols:
            trauma_sum_upd = df_features[present_trauma_cols].sum(axis=1).fillna(0)
            trauma_vals_upd = np.array(trauma_sum_upd, dtype=float)
            trauma_score = (trauma_vals_upd * 20) + injury_recency_score
        else:
            # Remplacement de pd.Series par np.full pour satisfaire le linter
            trauma_score = np.full(len(df_features), injury_recency_score, dtype=float)

        # 3. Calcul de l'Anomalie ACWR
        if 'ACWR' in df_features.columns:
            acwr_val = pd.to_numeric(df_features['ACWR'], errors='coerce').fillna(1.0)
        else:
            acwr_val = np.full(len(df_features), 1.0, dtype=float)
        
        acwr_vals_upd = np.array(acwr_val, dtype=float)
        acwr_stress = np.clip((np.abs(acwr_vals_upd - 1.0) * 50.0), 0, 30)

        # 4. Score de Risque Médical Composite
        fatigue_part_upd = np.array(df_features['Fatigue_IA'], dtype=float) * 0.50
        trauma_part_upd  = np.array(trauma_score, dtype=float) * 0.30
        acwr_part_upd    = np.array(acwr_stress, dtype=float) * 0.20

        medical_risk_upd = (fatigue_part_upd + trauma_part_upd + acwr_part_upd) / 100.0
        df_features['Medical_Risk_Score'] = np.clip(medical_risk_upd, 0, 1).tolist()
        df_features['Injury_Risk'] = df_features['Medical_Risk_Score']

        # 5. Classification des niveaux et Statuts
        df_features['Risk_Level'] = 'FAIBLE'
        df_features['Status'] = 'FULL TRAINING'
        
        risk_vals_upd = np.array(df_features['Medical_Risk_Score'], dtype=float)
        
        mask_mod_upd  = ((risk_vals_upd > 0.25) & (risk_vals_upd <= 0.60)).tolist()
        mask_high_upd = ((risk_vals_upd > 0.60) & (risk_vals_upd <= 0.85)).tolist()
        mask_crit_upd = (risk_vals_upd > 0.85).tolist()
        
        df_features.loc[mask_mod_upd, 'Risk_Level'] = 'MODÉRÉ'
        df_features.loc[mask_mod_upd, 'Status'] = 'VIGILANCE'
        
        df_features.loc[mask_high_upd, 'Risk_Level'] = 'ÉLEVÉ'
        df_features.loc[mask_high_upd, 'Status'] = 'REST FORCE'
        
        df_features.loc[mask_crit_upd, 'Risk_Level'] = 'ÉLEVÉ'
        df_features.loc[mask_crit_upd, 'Status'] = 'ALERTE BLESSURE'
        
        results = df_features.sort_values(['Nom', 'Match_Date'], ascending=[True, False]).groupby('Nom').first().reset_index()
    else:
        print(f"Dashboard: ERREUR - Aucun fichier de données trouvé !")
        return None, None
    
    # 3. Traitement global des données pour éviter les calculs répétitifs
    # On remplace les NaN par None pour la sécurité JSON
    df_features = df_features.replace({np.nan: None})
    results = results.replace({np.nan: None})
    
    # 4. Initialisation du Similarity Engine (Caché globalement)
    try:
        CACHE_SIM_ENGINE = SimilarityEngine(df_features)
        print("✅ Dashboard: Moteur de similarité initialisé.")
    except Exception as e:
        print(f"⚠️ Dashboard: Erreur moteur similarité : {e}")

    # Cache it
    CACHE_MODELS[model_type]["results"] = results
    CACHE_MODELS[model_type]["features"] = df_features
    CACHE_MODELS[model_type]["last_update"] = float(now)
    
    print(f"Dashboard: Données chargées avec succès. {len(results)} joueurs identifiés.")
    return results, df_features

# Serve static files from the current directory
# 1. On récupère le chemin EXACT et absolu du dossier où se trouve backend.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. On crée le chemin exact vers le dossier "static"
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Ajoutez ce bloc juste AVANT @app.get("/")
@app.get("/radio.jpg")
def serve_radio_image():
    # On crée le chemin exact vers l'image
    image_path = os.path.join(BASE_DIR, "static", "radio.jpg")
    
    # Sécurité : Si Python ne trouve pas l'image, il nous dira exactement où il a cherché !
    if not os.path.exists(image_path):
        return {"error": f"Impossible de trouver l'image. Python cherche ici : {image_path}"}
    return FileResponse(image_path)
@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    index_path = Path(__file__).parent / "index.html"
    return FileResponse(index_path)

@app.get("/api/dashboard-data")
def get_dashboard_data(model: str = 'lgbm'):
    _res, _feat = get_data(model_type=model)
    if _res is None or _feat is None:
        return {"error": "Data not found or corrupted"}
    
    results = cast(pd.DataFrame, _res)
    features = cast(pd.DataFrame, _feat)
    
    # 1. Team Summary
    total_players = len(results)
    
    # Sécurité si les données médicales sont absentes
    if 'Current_Injury' in results.columns:
        available_players = len(results[results['Current_Injury'] == 0])
        injured_count = total_players - available_players
    else:
        available_players = total_players
        injured_count = 0
        
    # Agrégation des risques pour le graphique du haut (3 catégories)
    raw_dist = results['Risk_Level'].value_counts().to_dict()
    risk_dist = {
        "ÉLEVÉ": raw_dist.get('ÉLEVÉ', 0) + raw_dist.get('CRITIQUE', 0),
        "MODÉRÉ": raw_dist.get('MODÉRÉ', 0),
        "FAIBLE": raw_dist.get('FAIBLE', 0)
    }
    
    # 2. Top At-Risk Players (Critiques)
    # On prend les joueurs avec la plus haute fatigue prédite par l'IA
    at_risk = results[results['Fatigue_IA'] >= 60]
    at_risk = at_risk.sort_values('Fatigue_IA', ascending=False).head(5)
    
    top_risks = []
    for _, row in at_risk.iterrows():
        fatigue = row['Fatigue_IA']
        if fatigue >= 80:
            title = "SURCHARGE CRITIQUE"
            expl = "Le modèle détecte un épuisement sévère. Risque de rupture neuromusculaire immédiat."
        elif fatigue >= 65:
            title = "ALERTE FATIGUE"
            expl = "Accumulation de charge excessive détectée par l'IA. Rotation fortement conseillée."
        else:
            title = "VIGILANCE"
            expl = "Signaux faibles de baisse de performance couplés à une charge élevée."

        top_risks.append({
            "name": row['Nom'],
            "player_id": row['Nom'].lower().replace(" ", "_"),
            "risk_score": int(fatigue),
            "position": row.get('Position', 'M'),
            "recommendation_title": title,
            "recommendation_details": expl
        })

    # 3. Training Load Trend (Last 28 days)
    features['Match_Date'] = pd.to_datetime(features['Match_Date'])
    last_date = features['Match_Date'].max()
    recent_data = features[features['Match_Date'] > (last_date - pd.Timedelta(days=28))]
    load_trend = recent_data.groupby('Match_Date')['Fatigue_Score'].mean().reset_index()
    load_trend = load_trend.sort_values('Match_Date')
    
    chart_labels = [d.strftime('%d %b') for d in load_trend['Match_Date']]
    chart_values = [round(v, 1) for v in load_trend['Fatigue_Score']]

    # Summary stats
    summary = {
        "total_players": total_players,
        "available_players": available_players,
        "injured_players": injured_count,
        "avg_acr": round(results['ACWR'].mean(), 2),
        "avg_sleep_hours": round(8.2 - (results['Fatigue_Score'].mean() / 50), 1),
        "avg_hrv": int(72 - (results['Fatigue_Score'].mean() / 4)),
        "risk_distribution": {
            "high": int(risk_dist.get('ÉLEVÉ', 0)),
            "medium": int(risk_dist.get('MODÉRÉ', 0)),
            "low": int(risk_dist.get('FAIBLE', 0))
        }
    }

    return {
        "summary": summary,
        "top_risks": top_risks,
        "load_chart": {
            "labels": chart_labels,
            "values": chart_values
        }
    }

@app.get("/api/player-data")
def get_player_data(model: str = 'lgbm'):
    _res, _feat = get_data(model_type=model)
    if _res is None or _feat is None:
        return {"error": "Data not found or corrupted"}
        
    results = cast(pd.DataFrame, _res)
    features = cast(pd.DataFrame, _feat)
    
    # NOUVEAU : On extrait les 15 derniers matchs individuels de chaque joueur (100% RÉEL)
    features['Match_Date'] = pd.to_datetime(features['Match_Date'])
    features = features.sort_values(by=['Nom', 'Match_Date'])
    
    # On prend les 15 derniers matchs pour chaque joueur
    recent_features = features.groupby('Nom').tail(15)
    player_hist_dict = {}
    
    for _, row in recent_features.iterrows():
        name = row['Nom']
        if name not in player_hist_dict:
            player_hist_dict[name] = {}
            
        m_date = row['Match_Date']
        iso_date = m_date.strftime('%Y-%m-%d')
        
        try:
            mins = float(row.get('Minutes_Played', 0))
            rating = float(row.get('Rating', 5.0))
            # Calcul de l'intensité sur 100%
            intensity_raw = (mins / 90.0) * (rating / 10.0) * 100 * 1.2
            intensity = min(100.0, max(0.0, round(intensity_raw, 1)))
            volume = float(row.get('Distance_Covered_km', mins / 10.0)) # approx volume
            
            club_team = str(row.get('Team', ''))
            home = str(row.get('Home_Team', ''))
            away = str(row.get('Away_Team', ''))
            
            c_norm = normalize_team_name(club_team)
            h_norm = normalize_team_name(home)
            a_norm = normalize_team_name(away)
            # Détection d'un match international (le club n'est ni à domicile ni à l'extérieur)
            # On vérifie dans les deux sens après normalisation pour gérer "1 Fc Koln" vs "1. FC Köln"
            is_club_match = (c_norm in h_norm) or (c_norm in a_norm) or (h_norm in c_norm) or (a_norm in c_norm)
            
            if is_club_match:
                actual_match_team = home if club_team.lower() in home.lower() else away
                opponent = away if actual_match_team == home else home
                is_international = False
            else:
                # Match international : On ne sait pas exactement pour quel pays il joue sans info extra,
                # mais on sait que ce n'est pas le club.
                opponent = f"{home} (Int)" if away.lower() == "inconnu" else away # Par défaut on montre l'extérieur
                # Si le pays domicile semble être son équipe nationale, l'adversaire est l'extérieur
                is_international = True

        except:
            intensity = 50.0
            volume = 10.0
            opponent = "Inconnu"
            is_international = False
            
        player_hist_dict[name][iso_date] = {
            "intensite": intensity,
            "volume": volume,
            "date": m_date.strftime('%d %b'),
            "opponent": opponent,
            "is_international": is_international
        }

    players_list = []
    def generate_clinical_insight(row):
        """Génère un diagnostic médical et sportif 100% dynamique et personnalisé."""
        acr = row.get('ACWR', 1.0)
        fatigue = row.get('Fatigue_IA', 0)
        risk_score = row.get('Medical_Risk_Score', 0) # Définition de la variable manquante
        if risk_score > 0.85:
            return f"URGENCE MÉDICALE : Le score de risque combiné est critique ({risk_score*100:.0f}%). Risque de blessure musculaire imminente. Mise au repos totale indispensable."

        if fatigue >= 80:
            return f"DIAGNOSTIC CRITIQUE : L'IA détecte un épuisement total ({fatigue}%). La chute de rendement sur le dernier match corrélée à la charge ACWR ({acr:.2f}) impose un repos immédiat."

        if fatigue >= 60:
            if acr > 1.4:
                return f"ALERTE SURCHARGE : Pic de charge détecté. Le système neuromusculaire est en état de stress. Réduire le volume d'entraînement de 40%."
            return f"FATIGUE IMPORTANTE : Score IA de {fatigue}%. Baisse de lucidité détectée dans les duels et la précision. Risque de blessure indirecte accru."

        if acr >= 0.8 and acr <= 1.25:
            return f"ÉTAT OPTIMAL : Équilibre parfait entre charge et récupération. Le joueur est dans sa zone de performance maximale (Fatigue IA: {fatigue}%)."
            
        return "STABILITÉ PHYSIOLOGIQUE : Aucun signal d'alerte majeur. Le joueur répond normalement aux sollicitations."

    for _, row in results.iterrows():
        insight = generate_clinical_insight(row)
        risk_level = row['Risk_Level'].replace('🔴 ', '').replace('🟠 ', '').replace('🟢 ', '')
        
        # Sécurité sur les noms de colonnes
        team_val = row.get('Team') or row.get('Equipe') or 'AthlytIQ FC'
        league_val = row.get('League') or row.get('Tournament') or 'Inconnue'
        name_val = row['Nom']

        player = {
            "player_id": name_val.lower().replace(" ", "_"),
            "name": name_val,
            "position": row.get('Position', 'M'),
            "team": team_val,
            "league": league_val,
            "acr_ratio": round(row['ACWR'], 2),
            "fatigue_score": int(row.get('Fatigue_Score', 0)),
            "hrv_rmssd": int(75 - (row.get('Fatigue_Score', 0) / 3)),
            "rating": round(float(row.get('Rating') if pd.notna(row.get('Rating')) else 0.0), 1),
            "distance": float(row.get('distanceRun') if pd.notna(row.get('distanceRun')) else 0.0),
            "sprints": int(float(row.get('sprints') if pd.notna(row.get('sprints')) else 0)),
            "age": int(float(row.get('Age') if pd.notna(row.get('Age')) else 0)),
            "injury_risk_level": risk_level,
            "injury_risk_score": int(row['Injury_Risk'] * 100),
        }
        
        player.update({
            "status": row['Status'],
            "current_injury": int(row.get('Current_Injury', 0)),
            "injury_type": row.get('Injury_Type_Text', ''),
            "dominant_cause": row.get('Dominant_Injury_Cause', 'NONE'),
            "medical_history": {
                "recent_muscle_injuries": int(row.get('Nb_Blessures_Musculaires_12m', 0)),
                "days_since_last": int(row.get('Jours_Depuis_Blessure', 999))
            },
            "recommendation_title": "Diagnostic Clinique",
            "recommendation_details": insight,
            "historique_jours": player_hist_dict.get(name_val, {})
        })
        players_list.append(player)
    
    if players_list:
        print(f"Dashboard: Premier joueur généré: {players_list[0]['name']} | Ligue: {players_list[0]['league']} | Equipe: {players_list[0]['team']}")

    return {"version": "1.3", "players": players_list}

@app.get("/api/player-history/{player_name}")
def get_player_history(player_name: str, model: str = 'lgbm'):
    _, _feat = get_data(model_type=model)
    if _feat is None:
        return {"error": "Data not found"}
        
    features = cast(pd.DataFrame, _feat)
    
    # Simple name matching (case insensitive)
    # The frontend sends slugified IDs, we need to match them or use the original name
    # Let's try to find the player in the results
    history = features[features['Nom'].str.lower().str.replace(" ", "_") == player_name.lower()].copy()
    
    if history.empty:
        # Fallback to direct name match if slugified fails
        history = features[features['Nom'].str.lower() == player_name.lower()].copy()

    if history.empty:
        return {"error": "Player not found"}
    
    # Sort by date and take last 15
    history['Match_Date'] = pd.to_datetime(history['Match_Date'])
    history = history.sort_values('Match_Date', ascending=False).head(15)
    
    history_list = []
    # Pre-calculate team averages for these specific dates to avoid repeated filtering
    dates = history['Match_Date'].unique()
    team_averages = features[features['Match_Date'].isin(dates)].groupby('Match_Date')['Fatigue_Score'].mean().to_dict()
    
    for _, row in history.iterrows():
        # Determine opponent and result
        current_team = str(row.get('Team', '')).lower()
        h_team = str(row.get('Home_Team', '')).lower()
        
        is_home = current_team in h_team or h_team in current_team
        opponent = row['Away_Team'] if is_home else row['Home_Team']
        
        m_date = row['Match_Date']
        
        # Calcul d'intelligence supplémentaire avec sécurité sur les types
        try:
            mins = float(row.get('Minutes_Played', 0))
            rating = float(row.get('Rating', 5.0))
            
            # Heuristic for result since scores are missing
            # Rating > 7.3 -> Win (V), Rating < 6.2 -> Loss (D), Else -> Draw (N)
            res_code = 'V'
            if rating > 7.3: res_code = 'V'
            elif rating < 6.2: res_code = 'D'
            else: res_code = 'N'
            fatigue = float(row.get('Fatigue_Score', 50.0))
            # Simulate HRV if not present, but use real data if available
            hrv = float(row.get('HRV_RMSSD', 75 - (fatigue / 3)))
            
            intensity = round((mins / 90.0) * rating * 1.2, 1)
            recovery = round(100.0 - fatigue * 0.8 + (hrv / 10.0), 1)
            
            # Physical metrics avec sécurité sur les nuls
            dist_val = row.get('distanceRun')
            if pd.isna(dist_val): dist_val = row.get('Distance_Covered_km', 10.2)
            distance = float(dist_val if pd.notnull(dist_val) else 10.2)
            if distance > 30: distance /= 1000.0 # Convert meters to km (threshold lowered for accuracy)
            
            s_val = row.get('sprints')
            sprints = int(s_val if pd.notnull(s_val) else 18)
            
            t_val = row.get('Trauma_Index')
            trauma = float(t_val if pd.notnull(t_val) else 0.5)
            
            # Per-match risk based on multiple factors for variation
            # Use intensity and sprints if fatigue/trauma are flat
            base_risk = fatigue * 0.4 + trauma * 25
            performance_stress = (sprints * 0.8) + (intensity * 0.2)
            match_risk = min(98, max(8, int(base_risk + performance_stress - 15)))
        except:
            res_code = 'V'
            intensity = 50.0
            recovery = 70.0
            mins = 0
            rating = 5.0
            fatigue = 50.0
            distance = 10.2
            sprints = 18
            hrv = 60.0
            trauma = 0.5
            match_risk = 25 + (int(time.time()) % 15) # Add slight deterministic variation if failing
        
        history_list.append({
            "iso_date": m_date.strftime('%Y-%m-%d'),
            "date": m_date.strftime('%d %b %y'), # Added Year
            "rating": rating,
            "minutes": int(mins),
            "fatigue": fatigue,
            "hrv": int(hrv),
            "team_avg": round(team_averages.get(m_date, 50.0), 1),
            "intensity": min(intensity, 100.0),
            "recovery": min(recovery, 100.0),
            "opponent": str(opponent),
            "result": res_code, # Added result
            "goals": int(row.get('Goals', 0)),
            "assists": int(row.get('Assists', 0)),
            "distance": round(distance, 1),
            "sprints": sprints,
            "trauma": round(trauma, 2),
            "risk": match_risk, # Added per-match risk
            "is_home": 1 if is_home else 0
        })
    
    # 4. Analyse Clinique IA des tendances
    avg_intensity = sum(h['intensity'] for h in history_list) / len(history_list) if history_list else 0
    avg_recovery = sum(h['recovery'] for h in history_list) / len(history_list) if history_list else 0
    
    trend = "STABLE"
    if avg_intensity > 75 and avg_recovery < 65:
        trend = "SURCHARGE DÉTECTÉE : L'intensité dépasse la capacité de régénération."
    elif avg_intensity < 40:
        trend = "SOUS-ENTRAÎNEMENT : Charge insuffisante pour maintenir le rythme de compétition."
    else:
        trend = "ZONE OPTIMALE : Équilibre parfait entre effort et récupération."

    insight = f"Tendance : {trend} Le joueur est à {round(avg_recovery)}% de sa capacité de récupération moyenne."

    # 5. Extraction de l'historique Médical (Transfermarkt)
    path_injury = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "data" / "raw" / "transfermarkt" / "injury_history.csv"
    medical_history_list = []
    
    global CACHE_INJURY_DF
    if path_injury.exists():
        if CACHE_INJURY_DF is None:
            CACHE_INJURY_DF = pd.read_csv(path_injury)
        injury_df = CACHE_INJURY_DF
        p_injuries = injury_df[injury_df['Nom'].str.lower() == player_name.lower()]
        if p_injuries.empty:
            p_injuries = injury_df[injury_df['Nom'].str.lower().str.replace(" ", "_") == player_name.lower()]
        
        if not p_injuries.empty:
            p_injuries = p_injuries.sort_values('Date_From', ascending=False)
            for _, i_row in p_injuries.iterrows():
                if str(i_row['Injury_Type']).upper() == 'NONE': continue
                medical_history_list.append({
                    "type": i_row['Injury_Type'],
                    "date": str(i_row['Date_From']),
                    "duration": int(i_row['Duration_Days']),
                    "category": i_row['Cause_Category']
                })

    return {
        "history": history_list[::-1],
        "squad_avg": [round(team_averages.get(d, 50.0), 1) for d in history['Match_Date'].tolist()][::-1],
        "clinical_insight": insight,
        "medical_history": medical_history_list # Nouvelles données envoyées au Dashboard
    }

@app.get("/api/predict-future")
def predict_future(player_name: str, rest_days: float = 4.0, model: str = 'lgbm'):
    _res, _feat = get_data(model_type=model)
    if _feat is None:
        return {"error": "Data not found"}
        
    features = cast(pd.DataFrame, _feat)
    
    # 1. Trouver le dernier match du joueur
    player_data = features[features['Nom'].str.lower() == player_name.lower()].copy()
    if player_data.empty:
        # Essayer avec le slug
        player_data = features[features['Nom'].str.lower().str.replace(" ", "_") == player_name.lower()].copy()
        
    if player_data.empty:
        return {"error": "Player not found"}
        
    last_match = player_data.sort_values('Match_Date', ascending=False).iloc[0].copy()
    
    # 2. Préparer le scénario futur
    # LOGIQUE CLINIQUE : Courbe Sigmoïde de Récupération Biologique
    current_pred = last_match.get('Fatigue_IA', 50.0)
    current_acr = float(last_match.get('ACWR', 1.0))
    
    # Equation Sigmoïde : Centrée sur 3 jours (le pivot du foot pro)
    # k=0.8 définit la vitesse de récupération
    k = 0.8
    center = 3.0
    # Le facteur va de ~1.5 (surcharge à 0j) à ~0.3 (repos total à 10j)
    recovery_factor = 1.4 / (1 + np.exp(k * (float(rest_days) - center))) + 0.25
    
    # Application de la logique de surcharge cumulée
    # Si rest_days < 3, adjusted_lag sera > current_pred (Accumulation de stress)
    adjusted_lag = current_pred * recovery_factor
    adjusted_acr = 0.8 + (current_acr - 0.8) * recovery_factor
    
    future_scenario = last_match.copy()
    future_scenario['Days_Rest'] = float(rest_days)
    future_scenario['Rating'] = 7.0 
    future_scenario['Minutes_Played'] = 90
    future_scenario['Fatigue_Lag1'] = adjusted_lag
    future_scenario['ACWR'] = adjusted_acr
    
    # 3. Inférence IA avec sécurité sur le retour
    prediction_result = predictor.predict(pd.DataFrame([future_scenario]), model_type=model)
    if prediction_result is not None and len(prediction_result) > 0:
        raw_pred = prediction_result[0]
    else:
        # Fallback sur la fatigue actuelle si l'IA échoue
        raw_pred = current_pred
    
    # 4. Calibration de sortie pour cohérence clinique
    # On assure que si recovery_factor > 1, le risque est obligatoirement CRITIQUE
    future_pred = raw_pred * recovery_factor
    
    # Sécurités biostats
    if float(rest_days) < 2.0:
        future_pred = max(future_pred, 82.0) # Zone de danger immédiat
    elif float(rest_days) > 7.0:
        future_pred = min(future_pred, 30.0) # Zone de fraîcheur totale
    
    # --- NOUVELLE LOGIQUE CLINIQUE ---
    
    # 1. Fatigue de Base (Celle qui descend avec le repos)
    # Modèle conservateur : ~8% de baisse le premier jour, accélération ensuite
    # Coefficient -0.12 au lieu de -0.20
    base_fatigue = current_pred * np.exp(-0.12 * float(rest_days))
    # Sécurité : la fatigue ne tombe pas à 0 instantanément
    base_fatigue = max(15.0, min(95.0, base_fatigue))

    # 2. Risque de Match (L'effort supplémentaire de 90 mins)
    # Jouer un match rajoute une charge de stress fixe + un multiplicateur de fatigue
    match_load = 30.0 + (base_fatigue * 0.1)
    match_risk = min(99.9, base_fatigue + match_load)

    # 3. Ajustement si repos trop court (< 48h)
    # Si le joueur n'a qu'un jour de repos, le risque de match explose
    if float(rest_days) < 2.0:
        match_risk = max(match_risk, 85.0)

    return {
        "player": player_name,
        "rest_days": float(rest_days),
        "future_fatigue": round(base_fatigue, 1), # Jauge BLEUE
        "post_match_fatigue": round(match_risk, 1), # Jauge ROUGE
        "risk_level": "CRITIQUE" if match_risk > 80 else ("MODÉRÉ" if match_risk > 50 else "OPTIMAL"),
        "recommendation": "DANGER : Risque de blessure élevé" if match_risk > 80 else "Apte pour 90 minutes"
    }

@app.get("/api/similarity")
def get_player_similarity(player_name: str, alpha: float = 0.5, model: str = 'lgbm'):
    _res, _feat = get_data(model_type=model)
    if _feat is None:
        return {"error": "Data not found"}
        
    features = cast(pd.DataFrame, _feat)
    
    global CACHE_SIM_ENGINE
    engine = CACHE_SIM_ENGINE
    if engine is None:
        engine = SimilarityEngine(features)
        CACHE_SIM_ENGINE = engine
        
    results = engine.get_similar_players(player_name, alpha=alpha)
    
    if isinstance(results, dict) and "error" in results:
        return {
            "target_player": player_name,
            "error": results["error"],
            "candidates": []
        }

    return {
        "target_player": player_name,
        "alpha": alpha,
        "candidates": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
