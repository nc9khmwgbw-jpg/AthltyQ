import sys
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
    
    print(f"Backend: Tentative de chargement des données...")
    
    if PROCESSED_PATH.exists():
        print(f"Backend: Chargement des features pré-traitées ({PROCESSED_PATH})")
        df_features = pd.read_csv(PROCESSED_PATH, low_memory=False)
        df_features['Match_Date'] = pd.to_datetime(df_features['Match_Date'], errors='coerce')
        results = predictor.predict(df_features)
    elif RAW_PATH.exists():
        print(f"Backend: Fichier pré-traité absent. Lancement du Feature Engineering sur {RAW_PATH}")
        df = pd.read_csv(RAW_PATH, low_memory=False)
        df['Match_Date'] = pd.to_datetime(df['Match_Date'], errors='coerce')
        df_features = run_feature_engineering(df)
        results = predictor.predict(df_features)
    else:
        print(f"Backend: ERREUR - Aucun fichier de données trouvé !")
        return None, None
    
    # Cache it
    CACHE["results"] = results
    CACHE["features"] = df_features
    CACHE["last_update"] = now
    
    print(f"Backend: Données chargées avec succès. {len(results)} joueurs identifiés.")
    return results, df_features

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
    results, _ = get_data()
    if results is None:
        return {"error": "Data not found"}
    
    players_list = []
    from LM.models.injury_predictor import _identifier_facteur_majeur
    
    for _, row in results.iterrows():
        facteur_titre, facteur_expl = _identifier_facteur_majeur(row) if row['Injury_Risk'] >= 0.16 else (None, None)
        risk_level = row['Risk_Level'].replace('🔴 ', '').replace('🟠 ', '').replace('🟢 ', '')
        
        player = {
            "player_id": row['Nom'].lower().replace(" ", "_"),
            "name": row['Nom'],
            "position": row.get('Position', 'M'),
            "team": row.get('Team', 'AthlytIQ FC'),
            "age": int(row.get('Age', 25)) if not pd.isna(row.get('Age')) else 25,
            "acr_ratio": round(row['ACWR'], 2),
            "fatigue_score": int(row['Fatigue_Score']),
            "hrv_rmssd": int(75 - (row['Fatigue_Score'] / 3)),
            "injury_risk_level": risk_level,
            "injury_risk_score": int(row['Injury_Risk'] * 100),
            "current_injury": int(row.get('Current_Injury', 0)),
            "recommendation_title": facteur_titre if facteur_titre else "État Optimal",
            "recommendation_details": facteur_expl if facteur_expl else "Aucune anomalie détectée. Le joueur peut suivre le programme d'entraînement standard à 100%."
        }
        players_list.append(player)
    
    return {"version": "1.1", "players": players_list}

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
        # Determine opponent
        opponent = row['Away_Team'] if row['Team'] == row['Home_Team'] else row['Home_Team']
        m_date = row['Match_Date']
        
        history_list.append({
            "date": m_date.strftime('%d %b'),
            "rating": round(row['Rating'], 1),
            "minutes": int(row['Minutes_Played']),
            "fatigue": round(row.get('Fatigue_Score', 0), 1),
            "team_avg": round(team_averages.get(m_date, 50.0), 1),
            "opponent": str(opponent),
            "goals": int(row.get('Goals', 0)),
            "assists": int(row.get('Assists', 0)),
            "is_home": int(row.get('Is_Home', 1))
        })
    
    return {"history": history_list[::-1]} # Return chronological for the charts

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
