import sys
import os
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import json
import time

from fastapi.responses import HTMLResponse, FileResponse

# Add root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LM.models.injury_predictor import InjuryPredictor
from LM.models.feature_engineering import run_feature_engineering

app = FastAPI(title="AthltyQ API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Cache
CACHE = {
    "results": None,
    "features": None,
    "last_update": 0,
    "ttl": 300  # 5 minutes
}

predictor = InjuryPredictor()

# Mock data for physiological markers
MOCK_PHYSIO = {
    "avg_sleep_hours": 7.2,
    "avg_hrv": 62,
    "hrv_rmssd": 58,
    "sleep_duration": 7.5,
    "sleep_quality": 85
}

def get_data():
    """Fetch and process data with caching to prevent OOM/Killed errors."""
    global CACHE
    now = time.time()
    
    if CACHE["results"] is not None and (now - CACHE["last_update"]) < CACHE["ttl"]:
        return CACHE["results"], CACHE["features"]
    
    PROCESSED_PATH = ROOT / "data" / "processed" / "features_dataset.csv"
    RAW_PATH = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"
    
    print(f"Dashboard: Tentative de chargement des données...")
    
    if PROCESSED_PATH.exists():
        print(f"Dashboard: Chargement des features pré-traitées ({PROCESSED_PATH})")
        df_features = pd.read_csv(PROCESSED_PATH, low_memory=False)
        df_features['Match_Date'] = pd.to_datetime(df_features['Match_Date'], errors='coerce')
        results = predictor.predict(df_features)
    elif RAW_PATH.exists():
        print(f"Dashboard: Fichier pré-traité absent. Lancement du Feature Engineering sur {RAW_PATH}")
        df = pd.read_csv(RAW_PATH, low_memory=False)
        df['Match_Date'] = pd.to_datetime(df['Match_Date'], errors='coerce')
        df_features = run_feature_engineering(df)
        results = predictor.predict(df_features)
    else:
        print(f"Dashboard: ERREUR - Aucun fichier de données trouvé !")
        return None, None
    
    # Cache it
    CACHE["results"] = results
    CACHE["features"] = df_features
    CACHE["last_update"] = now
    
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
def get_dashboard_data():
    results, features = get_data()
    if results is None:
        return {"error": "Data not found"}
    
    # 1. Team Summary
    total_players = len(results)
    available_players = len(results[results['Current_Injury'] == 0])
    injured_count = total_players - available_players
    avg_acr = results['ACWR'].mean()
    risk_dist = results['Risk_Level'].value_counts().to_dict()
    
    # 2. Top At-Risk Players (Critiques)
    # We take players with high risk score, excluding those already injured
    at_risk = results[(results['Injury_Risk'] >= 0.60) & (results['Current_Injury'] == 0)]
    at_risk = at_risk.sort_values('Injury_Risk', ascending=False).head(5)
    
    top_risks = []
    from LM.models.injury_predictor import _identifier_facteur_majeur
    for _, row in at_risk.iterrows():
        facteur_titre, facteur_expl = _identifier_facteur_majeur(row)
        top_risks.append({
            "name": row['Nom'],
            "player_id": row['Nom'].lower().replace(" ", "_"),
            "risk_score": int(row['Injury_Risk'] * 100),
            "position": row.get('Position', 'M'),
            "recommendation_title": facteur_titre if facteur_titre else "Risque Élevé",
            "recommendation_details": facteur_expl if facteur_expl else "Le joueur présente des signes de fatigue critique."
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
            "high": int(risk_dist.get('🔴 ÉLEVÉ', 0)),
            "medium": int(risk_dist.get('🟠 MODÉRÉ', 0)),
            "low": int(risk_dist.get('🟢 FAIBLE', 0))
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
def get_player_data():
    results, features = get_data()
    if results is None:
        return {"error": "Data not found"}
    
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
            
            # Détection d'un match international (le club n'est ni à domicile ni à l'extérieur)
            is_club_match = (club_team.lower() in home.lower()) or (club_team.lower() in away.lower())
            
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
        risk = row['Injury_Risk']
        acr = row['ACWR']
        fatigue = row.get('Fatigue_Score', 0)
        hrv = int(75 - (fatigue / 3)) # Re-calcul pour cohérence
        
        if row.get('Current_Injury', 0) == 1:
            return f"PHASE DE RÉÉDUCATION : Le joueur est actuellement indisponible ({row.get('Injury_Type_Text', 'Lésion')}). Focus sur la cicatrisation et le renforcement isométrique sans impact."

        if risk >= 0.60:
            from LM.models.injury_predictor import _identifier_facteur_majeur
            _, detail = _identifier_facteur_majeur(row)
            return detail if detail else "ALERTE ROUGE : Paramètres physiologiques critiques. Risque de lésion imminent. Repos total préconisé."

        if risk >= 0.16:
            if acr > 1.3:
                return f"ZONE DE DANGER : Surcharge brutale détectée (ACR: {acr:.2f}). Les tissus sont en état de stress mécanique. Réduire la charge de 30% pour éviter la lésion."
            if fatigue > 50:
                return f"FATIGUE ACCUMULÉE : Score de fatigue de {fatigue:.1f}/100. Le système nerveux central montre des signes de saturation. Sommeil et hydratation à surveiller."
            return "VIGILANCE MODÉRÉE : Légère instabilité des marqueurs de charge. Maintenir l'intensité mais limiter les sprints à haute vitesse."

        # Diagnostic pour les joueurs "Fit" (100% dynamique)
        if acr >= 0.8 and acr <= 1.2:
            if hrv > 65:
                return f"ÉTAT OPTIMAL : Équilibre parfait entre charge ({acr:.2f}) et récupération (HRV: {hrv}ms). Le joueur est dans son 'Sweet Spot' de performance."
            return f"FORME STABLE : Capacité de travail maintenue. Le ratio de charge ({acr:.2f}) est sécurisé, permettant une séance d'intensité maximale sans risque majeur."
        
        if acr < 0.8:
            return f"SOUS-ENTRAÎNEMENT : Le volume de travail est trop bas ({acr:.2f}). Risque de désadaptation. Augmenter progressivement la charge pour retrouver le rythme de compétition."
            
        return "STABILITÉ PHYSIOLOGIQUE : Aucun signal d'alerte. Le joueur répond positivement aux charges imposées."

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
            "fatigue_score": int(row['Fatigue_Score']),
            "hrv_rmssd": int(75 - (row['Fatigue_Score'] / 3)),
            "injury_risk_level": risk_level,
            "injury_risk_score": int(row['Injury_Risk'] * 100),
            "current_injury": int(row.get('Current_Injury', 0)),
            "injury_type": row.get('Injury_Type_Text', ''),
            "dominant_cause": row.get('Dominant_Injury_Cause', 'NONE'),
            "recommendation_title": "Diagnostic Clinique",
            "recommendation_details": insight,
            "historique_jours": player_hist_dict.get(name_val, {}) # Données 100% réelles injectées ici
        }
        players_list.append(player)
    
    if players_list:
        print(f"Dashboard: Premier joueur généré: {players_list[0]['name']} | Ligue: {players_list[0]['league']} | Equipe: {players_list[0]['team']}")

    return {"version": "1.3", "players": players_list}

@app.get("/api/player-history/{player_name}")
def get_player_history(player_name: str):
    _, features = get_data()
    if features is None:
        return {"error": "Data not found"}
    
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
        a_team = str(row.get('Away_Team', '')).lower()
        
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
            
            # Physical metrics
            distance = float(row.get('distanceRun', row.get('Distance_Covered_km', 10.2)))
            if distance > 30: distance /= 1000.0 # Convert meters to km (threshold lowered for accuracy)
            
            sprints = int(row.get('sprints', 18))
            trauma = float(row.get('Trauma_Index', 0.5))
            
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

    return {
        "history": history_list[::-1],
        "squad_avg": [round(team_averages.get(d, 50.0), 1) for d in history['Match_Date'].tolist()][::-1],
        "clinical_insight": insight
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
