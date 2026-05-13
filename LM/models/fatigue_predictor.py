"""
AthlytIQ — Fatigue Predictor (Moteur de Régression Pure)
========================================================
Nouveau cerveau centralisé remplaçant l'ancienne classification.
Utilise un RandomForestRegressor pour estimer une jauge de fatigue (0-100).
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration des chemins
ROOT = Path(__file__).resolve().parent.parent.parent
SAVE_DIR = ROOT / "LM" / "models" / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

class FatiguePredictor:
    """
    Interface unique pour le calcul de la fatigue AthlytIQ.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.load_model()

    def load_model(self):
        """Charge le modèle entraîné."""
        try:
            self.model = joblib.load(SAVE_DIR / "fatigue_model.joblib")
            self.scaler = joblib.load(SAVE_DIR / "fatigue_scaler.joblib")
            self.features = joblib.load(SAVE_DIR / "fatigue_features.joblib")
            print("   ✅ Cerveau AthlytIQ (RandomForest) chargé avec succès.")
        except Exception as e:
            print(f"   ⚠️ Impossible de charger le modèle : {e}")
            self.model = None

    def _prepare_data(self, df):
        """Nettoyage strict : uniquement les colonnes numériques, pas de triche."""
        # On exclut les colonnes d'identité et les cibles
        exclure = [
            'Nom', 'Player_Name', 'Player_ID', 'Match_Date', 'Match_Num',
            'Event_ID', 'Home_Team', 'Away_Team', 'Tournament', 'Equipe', 
            'Position', 'Poste_Cat', 'League', 'Target_Fatigue', 
            'Risk_Category', 'Target_Injury_Occurred', 'Current_Injury'
        ]
        
        # On ne garde que les colonnes numériques existantes dans le DataFrame
        X = df.drop(columns=[c for c in exclure if c in df.columns])
        X = X.select_dtypes(include='number')
        
        # Gestion des valeurs infinies ou manquantes
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        return X

    def train(self, df):
        """
        Entraîne le RandomForestRegressor sur la 'Vérité Terrain'.
        Optimisé pour puce Apple Silicon (n_jobs=-1).
        """
        print("\n" + "─" * 65)
        print("   🧠 ENTRAÎNEMENT DU NOUVEAU CERVEAU — AthlytIQ (Régression)")
        print("─" * 65)

        if 'Target_Fatigue' not in df.columns:
            print("   ❌ Erreur : Colonne 'Target_Fatigue' introuvable.")
            return None

        # 1. Préparation X, y
        X = self._prepare_data(df)
        y = df['Target_Fatigue']
        
        # Sauvegarde des noms de colonnes pour la production
        self.features = X.columns.tolist()

        # 2. Split 80/20 (Le Coffre-Fort)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Normalisation
        self.scaler = StandardScaler()
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)

        # 4. Le Modèle (Ultra-Apprentissage)
        # On augmente la capacité pour capturer les patterns complexes (n+1)
        self.model = RandomForestRegressor(
            n_estimators=500,        # Plus d'arbres pour plus de stabilité
            max_depth=15,            # Plus de profondeur pour apprendre les lois complexes
            min_samples_split=10,    # Évite le par cœur (overfitting)
            min_samples_leaf=4,
            n_jobs=-1,               # Utilise tous les cœurs M1
            random_state=42
        )

        print(f"   📊 Apprentissage sur {len(X_train)} matchs...")
        self.model.fit(X_train_sc, y_train)

        # 5. Examen Final (Évaluation)
        y_pred = self.model.predict(X_test_sc)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"   📈 RÉSULTATS DE L'EXAMEN :")
        print(f"      - Marge d'erreur (MAE) : {mae:.2f} %")
        print(f"      - Score de fiabilité (R²) : {r2:.2f}")

        # 6. Sauvegarde
        joblib.dump(self.model, SAVE_DIR / "fatigue_model.joblib")
        joblib.dump(self.scaler, SAVE_DIR / "fatigue_scaler.joblib")
        joblib.dump(self.features, SAVE_DIR / "fatigue_features.joblib")
        joblib.dump(self.features, SAVE_DIR / "model_columns.joblib")
        
        print(f"   💾 Modèle sauvegardé dans {SAVE_DIR.name}/")
        return mae

    def predict(self, df):
        """
        Prédit la fatigue pour un ou plusieurs matchs.
        VERSION SÉCURISÉE : Scaling + Gestion des Infinis + Alignement Colonnes.
        """
        if self.model is None:
            print("⚠️ Modèle non chargé.")
            return None

        # 1. Chargement et vérification des colonnes attendues
        try:
            model_cols_path = SAVE_DIR / "model_columns.joblib"
            if not model_cols_path.exists():
                print(f"❌ Fichier des colonnes introuvable : {model_cols_path}")
                return None
            expected_cols = joblib.load(model_cols_path)
            
            if hasattr(self.model, 'feature_names_in_'):
                model_cols = set(self.model.feature_names_in_)
                file_cols = set(expected_cols)
                if model_cols != file_cols:
                    print(f"⚠️ Mismatch colonnes détecté : {model_cols.symmetric_difference(file_cols)}")
                    expected_cols = self.model.feature_names_in_.tolist()
        except Exception as e:
            if self.model is not None and hasattr(self.model, 'feature_names_in_'):
                expected_cols = self.model.feature_names_in_.tolist()
            else:
                print(f"❌ Impossible de déterminer les colonnes du modèle : {e}")
                return None

        # 2. Préparation des données
        X = df.copy()
        
        # S'assurer que toutes les colonnes attendues sont présentes
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        
        # Sélection et ORDRE STRICT des colonnes
        X = X[expected_cols]
        
        # Nettoyage (Infinis et NaN)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        # 3. Normalisation
        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                print(f"⚠️ Erreur de scaling : {e}")
                X_scaled = X
        else:
            X_scaled = X

        # 4. Prédiction
        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0, 100).round(1)

    def prepare_for_prediction(self, df_upcoming, df_history):
        """
        Prépare un match à venir (planning) en utilisant l'historique 
        comme proxy pour les données non encore connues (Rating, etc.).
        """
        df = df_upcoming.copy()
        
        for nom in df['Nom'].unique():
            hist = df_history[df_history['Nom'] == nom]
            if hist.empty: continue
            
            # Utiliser la dernière MA15 connue pour le Rating
            if 'Rating_MA15' in hist.columns:
                df.loc[df['Nom'] == nom, 'Rating'] = hist['Rating_MA15'].iloc[-1]
            
        return df

if __name__ == "__main__":
    # Test rapide si lancé en direct
    DATA_PATH = ROOT / "data" / "processed" / "features_dataset.csv"
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        predictor = FatiguePredictor()
        predictor.train(df)
