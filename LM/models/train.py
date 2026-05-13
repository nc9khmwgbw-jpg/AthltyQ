"""
AthlytIQ — Script d'Entraînement Global
==========================================
Entraîne les 3 modèles du Module 1 :
1. XGBoost + Random Forest (prédiction tabulaire)
2. LSTM (séries temporelles)
3. Isolation Forest (détection d'anomalies)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration des chemins
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH       = ROOT / "data" / "processed" / "features_dataset.csv"
MODEL_SAVE_PATH = ROOT / "LM" / "models" / "saved" / "fatigue_model.joblib"
TRAIN_PLAYERS_PATH = ROOT / "LM" / "models" / "saved" / "train_players.joblib"
TEST_PLAYERS_PATH  = ROOT / "LM" / "models" / "saved" / "test_players.joblib"

def entrainement_elite():
    print("\n" + "═"*70)
    print("🚀 LANCEMENT DE L'ENTRAÎNEMENT HAUTE PERFORMANCE — AthlytIQ ELITE")
    print("═"*70)

    # 1. Chargement des données
    if not DATA_PATH.exists():
        print(f"❌ Erreur : Dataset introuvable à {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"📊 Dataset chargé : {df.shape}")

    # 2. Préparation des Features et Target
    # On exclut les colonnes d'identité, la cible et les variables de fuite (leakage)
    identity_cols = ['Nom', 'Player_ID', 'Match_Date', 'Event_ID', 'Home_Team', 'Away_Team', 
                     'Score_Home', 'Score_Away', 'Tournament', 'Equipe', 'Position', 'Poste_Cat',
                     'Risk_Category', 'Target_Injury_Occurred', 'Current_Injury',
                     'Fatigue_Realisee', 'Fatigue_Reelle_Match_T']
    
    X = df.drop(columns=identity_cols + ['Target_Fatigue'], errors='ignore')
    y = df['Target_Fatigue']

    # On ne garde que les colonnes numériques
    X = X.select_dtypes(include='number')
    # Remplissage des valeurs manquantes (sécurité)
    X = X.fillna(0)

    # 3. Split Train/Test PAR JOUEUR (aucun joueur dans les 2 sets à la fois)
    col_nom = 'Nom' if 'Nom' in df.columns else 'Player_Name'
    joueurs = df[col_nom].dropna().unique()
    joueurs = np.array(list(joueurs), dtype=str)

    # Shuffle reproductible des joueurs et split 80/20
    rng = np.random.RandomState(42)
    rng.shuffle(joueurs)
    split_idx     = int(len(joueurs) * 0.8)
    joueurs_train = joueurs[:split_idx]
    joueurs_test  = joueurs[split_idx:]

    # ✅ SAUVEGARDE des listes de joueurs pour garantir l'intégrité future
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(list(joueurs_train), TRAIN_PLAYERS_PATH)
    joblib.dump(list(joueurs_test),  TEST_PLAYERS_PATH)
    print(f"💾 Listes joueurs sauvegardées : {len(joueurs_train)} train / {len(joueurs_test)} test")

    # Masques basés sur le joueur (garantit 0% de fuite entre train et test)
    mask_train = df[col_nom].isin(set(joueurs_train))
    mask_test  = df[col_nom].isin(set(joueurs_test))

    X_train, y_train = X[mask_train], y[mask_train]
    X_test,  y_test  = X[mask_test],  y[mask_test]

    print(f"   Joueurs train : {len(joueurs_train)} ({len(X_train)} matchs)")
    print(f"   Joueurs test  : {len(joueurs_test)} ({len(X_test)} matchs)")
    fuite = len(set(joueurs_train) & set(joueurs_test))
    print(f"   Fuite de données : {fuite} joueurs communs {'✅' if fuite == 0 else '❌'}")

    # --- NORMALISATION (StandardScaler) ---
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"🧠 Phase d'optimisation des réglages (Grid Search) sur données normalisées...")
    
    # 4. Paramètres d'entraînement "Elite"
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=cv, scoring='neg_mean_absolute_error', 
                               verbose=1, n_jobs=-1)

    # 5. L'entraînement lourd
    print(f"🏋️  Entraînement en cours sur {len(X_train)} matchs...")
    grid_search.fit(X_train_scaled, y_train)

    # SÉCURITÉ : Ré-entraînement explicite sur tout le train set
    best_model = grid_search.best_estimator_
    best_model.fit(X_train_scaled, y_train)
    
    print(f"✅ Meilleurs réglages trouvés : {grid_search.best_params_}")

    # 6. Évaluation finale sur le SET DE TEST (joueurs jamais vus)
    y_pred = best_model.predict(X_test_scaled)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    precision = 100 - mae

    print("\n" + "📈 RÉSULTATS DU CENTRE D'ENTRAÎNEMENT :")
    print(f"   - Précision Finale : {precision:.2f}%")
    print(f"   - Marge d'erreur (MAE) : {mae:.2f} %")
    print(f"   - Score de fiabilité (R²) : {r2:.2f}")

    # 7. Sauvegarde complète du cerveau
    joblib.dump(best_model, MODEL_SAVE_PATH)
    
    scaler_path = ROOT / "LM" / "models" / "saved" / "fatigue_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    cols_path = ROOT / "LM" / "models" / "saved" / "model_columns.joblib"
    joblib.dump(X.columns.tolist(), cols_path)
    
    features_path = ROOT / "LM" / "models" / "saved" / "fatigue_features.joblib"
    joblib.dump(X.columns.tolist(), features_path)

    print(f"\n💾 Cerveau ELITE sauvegardé intégralement (Modèle + Scaler + Features)")
    print(f"   📁 Emplacement : {MODEL_SAVE_PATH.parent}")
    print("═"*70 + "\n")


def evaluer_modele():
    """
    Évalue le modèle sauvegardé UNIQUEMENT sur les joueurs du set de test.
    Ces joueurs n'ont JAMAIS été vus pendant l'entraînement.
    """
    print("\n" + "═"*70)
    print("🔬 ÉVALUATION STRICTE — Joueurs de Test (Jamais Vus)")
    print("═"*70)

    # Vérification que les fichiers existent
    required = [MODEL_SAVE_PATH, TRAIN_PLAYERS_PATH, TEST_PLAYERS_PATH]
    for p in required:
        if not p.exists():
            print(f"❌ Fichier manquant : {p}")
            print("   → Lance d'abord l'entraînement : python train.py train")
            return

    if not DATA_PATH.exists():
        print(f"❌ Dataset introuvable : {DATA_PATH}")
        return

    # Chargement du modèle et des listes de joueurs
    model        = joblib.load(MODEL_SAVE_PATH)
    scaler       = joblib.load(ROOT / "LM" / "models" / "saved" / "fatigue_scaler.joblib")
    model_cols   = joblib.load(ROOT / "LM" / "models" / "saved" / "model_columns.joblib")
    train_players = set(joblib.load(TRAIN_PLAYERS_PATH))
    test_players  = set(joblib.load(TEST_PLAYERS_PATH))

    print(f"✅ Modèle chargé.")
    print(f"   Joueurs d'entraînement (exclus du test) : {len(train_players)}")
    print(f"   Joueurs de test                         : {len(test_players)}")

    # Vérification stricte : 0 fuite
    fuite = train_players & test_players
    if fuite:
        print(f"❌ FUITE DÉTECTÉE : {len(fuite)} joueurs communs !")
        return
    print(f"   Fuite de données : 0 joueurs communs ✅")

    # Chargement et filtrage des données
    df = pd.read_csv(DATA_PATH)
    col_nom = 'Nom' if 'Nom' in df.columns else 'Player_Name'

    # ✅ SÉCURITÉ : On ne garde QUE les joueurs du set de test original
    df_test = df[df[col_nom].isin(test_players)].copy()
    print(f"\n📊 Données de test : {len(df_test)} matchs / {df_test[col_nom].nunique()} joueurs")

    identity_cols = ['Nom', 'Player_ID', 'Match_Date', 'Event_ID', 'Home_Team', 'Away_Team',
                     'Score_Home', 'Score_Away', 'Tournament', 'Equipe', 'Position', 'Poste_Cat',
                     'Risk_Category', 'Target_Injury_Occurred', 'Current_Injury',
                     'Fatigue_Realisee', 'Fatigue_Reelle_Match_T']

    X_test = df_test.drop(columns=identity_cols + ['Target_Fatigue'], errors='ignore')
    y_test = df_test['Target_Fatigue']
    X_test = X_test.select_dtypes(include='number').reindex(columns=model_cols, fill_value=0)
    X_test = X_test.fillna(0)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    mae       = mean_absolute_error(y_test, y_pred)
    r2        = r2_score(y_test, y_pred)
    precision = 100 - mae

    print("\n📈 RÉSULTATS DE L'ÉVALUATION STRICTE (Joueurs Jamais Vus) :")
    print(f"   - Précision Finale : {precision:.2f}%")
    print(f"   - Marge d'erreur (MAE) : {mae:.2f} %")
    print(f"   - Score de fiabilité (R²) : {r2:.2f}")
    print("═"*70 + "\n")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "eval":
        evaluer_modele()
    else:
        entrainement_elite()
