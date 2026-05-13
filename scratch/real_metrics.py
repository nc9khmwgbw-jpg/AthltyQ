import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

ROOT = Path("/Users/fahamayoub/Desktop/AthlytIQ")
DATA_PATH = ROOT / "data" / "processed" / "features_dataset.csv"

def get_real_values():
    df = pd.read_csv(DATA_PATH)
    # On prend un échantillon de 5000 pour la vitesse
    df_sample = df.sample(min(5000, len(df)), random_state=42)
    
    exclure = ['Nom', 'Player_ID', 'Match_Date', 'Match_Num', 'Event_ID', 'Home_Team', 'Away_Team', 
               'Tournament', 'Equipe', 'Position', 'Poste_Cat', 'League', 'Target_Fatigue', 
               'Risk_Category', 'Target_Injury_Occurred', 'Current_Injury']
    
    X = df_sample.drop(columns=[c for c in exclure if c in df_sample.columns])
    X = X.select_dtypes(include='number').fillna(0)
    y = df_sample['Target_Fatigue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"REAL_MAE:{mae:.4f}")
    print(f"REAL_R2:{r2:.4f}")

if __name__ == "__main__":
    get_real_values()
