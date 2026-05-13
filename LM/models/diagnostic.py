import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

# Chemins
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "features_dataset.csv"

def diagnostic_expert() -> None:
    print("🔍 DIAGNOSTIC HAUTE PRÉCISION — AthlytIQ Brain")
    
    if not DATA_PATH.exists():
        print(f"❌ Dataset introuvable : {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Préparation
    exclure = ['Nom', 'Player_ID', 'Match_Date', 'Match_Num', 'Event_ID', 'Home_Team', 'Away_Team', 
               'Tournament', 'Equipe', 'Position', 'Poste_Cat', 'League', 'Target_Fatigue', 
               'Risk_Category', 'Target_Injury_Occurred', 'Current_Injury', 
               'Fatigue_Realisee', 'Fatigue_Reelle_Match_T']
    
    X = df.drop(columns=[c for c in exclure if c in df.columns])
    X = X.select_dtypes(include='number').fillna(0)
    y = df['Target_Fatigue']

    # 1. Test de Fuite (Correlation)
    # Ignorer les colonnes avec une variance nulle pour éviter le RuntimeWarning
    std = X.std()
    X_var = X.loc[:, std > 0]
    correlations = X_var.corrwith(y).sort_values(ascending=False)
    print("\n⚠️  TOP 5 DES VARIABLES LES PLUS CORRÉLÉES (Risque de fuite) :")
    print(correlations.head(5))

    # 2. Entraînement avec Cross-Validation (Robustesse)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    
    print(f"\n📊 FIABILITÉ SUR 5 TESTS DIFFÉRENTS (CV R²) : {scores.mean():.4f} (+/- {scores.std():.4f})")

    # 3. Importance des Features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.fit(X_train, y_train)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n🏆 VARIABLES QUI DÉCIDENT DE LA FATIGUE (Top 10) :")
    print(importances.head(10))

    # 4. Analyse des Erreurs
    y_pred = rf.predict(X_test)
    errors = np.abs(y_test - y_pred)
    print(f"\n🎯 ERREUR MAXIMALE CONSTATÉE : {errors.max():.2f}%")
    print(f"🎯 ERREUR MOYENNE (MAE) : {mean_absolute_error(y_test, y_pred):.4f}%")

if __name__ == "__main__":
    diagnostic_expert()
