"""
AthlytIQ — Pipeline Orchestrateur
===================================
Orchestre l'ensemble du pipeline Module 1 :
Scraping → Nettoyage → Feature Engineering → Entraînement → Prédiction
"""

import sys
from pathlib import Path
import pandas as pd

# Ajouter le répertoire racine au path (3 niveaux au-dessus : pipeline/ <- ml/ <- root/)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

def run_full_pipeline(nb_pages=10, retrain=False, saison_debut='2024-08-01'):
    """
    Exécute le pipeline complet pour TOUTES les équipes détectées.
    """
    # Détection dynamique des équipes basées sur les fichiers brut_*.csv
    data_dir = ROOT / "DATA_PIPELINE/SCRAPPING/data"
    raw_files = list(data_dir.glob("brut_*.csv"))
    # Extraire le nom de l'équipe (gère brut_equipe_Fc_Barcelona.csv ou brut_Fc_Barcelona.csv)
    teams = []
    for f in raw_files:
        name = f.name.replace("brut_", "").replace("equipe_", "").replace(".csv", "").replace("_", " ")
        if name not in teams:
            teams.append(name)
    
    if not teams:
        # Fallback si rien n'est trouvé
        teams = ["Fc Barcelona"]

    print("\n" + "═" * 70)
    print("   AthlytIQ — MODULE 1 : PRÉDICTION DE PERFORMANCE (DYNAMIQUE)")
    print(f"   Saison depuis : {saison_debut}")
    print(f"   Équipes détectées : {', '.join(teams)}")
    print("═" * 70)

    from DATA_PIPELINE.SCRAPPING.scripts.sofascore_match_scraper import scraper_equipe_match_par_match
    from DATA_PIPELINE.NETTOYAGE.scripts.data_cleaner import run_cleaning_pipeline
    from LM.pipeline.feature_engineering import run_feature_engineering
    from LM.models.train import entrainer_tous_les_modeles
    from LM.models.anomaly_detector import detecter_anomalies
    from LM.models.injury_predictor import predire_risque_blessure

    all_cleaned_matchs = []
    for equipe_nom in teams:
        print(f"\n🔄 TRAITEMENT DE L'ÉQUIPE : {equipe_nom.upper()}")
        print("-" * 50)
        
        # Chercher le fichier (avec ou sans préfixe 'equipe_')
        matchs_csv = data_dir / f"brut_{equipe_nom.replace(' ', '_')}.csv"
        if not matchs_csv.exists():
            matchs_csv = data_dir / f"brut_equipe_{equipe_nom.replace(' ', '_')}.csv"
        
        # ── Étape 1 : Scraping ──
        if not matchs_csv.exists():
            print(f"\n📥 [1/2] Scraping match par match pour {equipe_nom}...")
            df_raw = scraper_equipe_match_par_match(
                equipe_nom=equipe_nom, nb_pages=nb_pages, saison_debut=saison_debut
            )
            if df_raw is None or df_raw.empty:
                print(f"❌ Aucune donnée pour {equipe_nom}. On passe à la suivante.")
                continue
        else:
            print(f"\n✅ [1/2] Données scrapées trouvées : {matchs_csv.name}")

        # ── Étape 2 : Nettoyage ──
        print(f"🧹 [2/2] Nettoyage contextuel pour {equipe_nom}...")
        df_matchs = run_cleaning_pipeline(
            matchs_csv_path=str(matchs_csv)
        )

        if not df_matchs.empty:
            # Assurer que la colonne Equipe est présente
            if 'Equipe' not in df_matchs.columns:
                df_matchs['Equipe'] = equipe_nom
            all_cleaned_matchs.append(df_matchs)
        else:
            print(f"❌ Aucune donnée propre pour {equipe_nom}.")

    if not all_cleaned_matchs:
        print("\n❌ ERREUR CRITIQUE : Aucune donnée propre disponible pour aucune équipe.")
        return

    # Fusion de toutes les données d'équipe
    print("\n🔗 FUSION DES DONNÉES DE TOUTE LA LIGUE...")
    df_combined = pd.concat(all_cleaned_matchs, ignore_index=True)
    print(f"   📊 Shape combinée globale : {df_combined.shape}")

    # ── Étape 3 : Feature Engineering Global ──
    print("\n⚙️  ÉTAPE 3 — Feature Engineering sur le dataset global...")
    df_features = run_feature_engineering(df_combined)

    if df_features.empty:
        print("❌ Aucune feature générée.")
        return
        
    # On sauvegarde au cas où
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_dir / "features_dataset.csv", index=False)

    # ── Étape 4 : Entraînement des modèles Universels ──
    models_dir = ROOT / "LM" / "models" / "saved"
    model_exists = (models_dir / "xgb_predictor_1m.joblib").exists()

    if retrain or not model_exists:
        print("\n🤖 ÉTAPE 4 — Entraînement des modèles ML sur la ligue entière...")
        entrainer_tous_les_modeles(df_features)
    else:
        print(f"\n✅ ÉTAPE 4 — Modèles existants trouvés dans {models_dir}")

    # ── Étape 5 : Prédictions & Alertes (Multi-équipes) ──
    print("\n🔮 ÉTAPE 5 — Génération des prédictions et alertes globales...")

    df_final = predire_risque_blessure(df_features)
    
    # Rajouter l'équipe aux prédictions pour le frontend
    if df_final is not None and 'Equipe' not in df_final.columns:
        # On va chercher la dernière équipe connue pour chaque joueur dans df_features
        equipes_map = dict(zip(df_features['Nom'], df_features['Equipe']))
        df_final['Equipe'] = df_final['Nom'].map(equipes_map)

    df_anomalies = detecter_anomalies(df_features)
    if df_anomalies is not None and 'Equipe' not in df_anomalies.columns and not df_anomalies.empty:
        df_anomalies['Equipe'] = df_anomalies['Nom'].map(equipes_map)

    if df_final is not None and not df_final.empty:
        pred_path = output_dir / "predictions_all.csv"
        df_final.to_csv(pred_path, index=False, encoding='utf-8-sig')
        print(f"   💾 Préditions sauvegardées : {pred_path}")

    if df_anomalies is not None and not df_anomalies.empty:
        anom_path = output_dir / "anomalies_all.csv"
        df_anomalies.to_csv(anom_path, index=False, encoding='utf-8-sig')
        print(f"   💾 Anomalies sauvegardées : {anom_path}")

    print("\n" + "═" * 70)
    print("   🚀 PIPELINE TERMINÉ — MODE MULTI-ÉQUIPES OPÉRATIONNEL")
    print("═" * 70)
    print(f"\n   📊 Joueurs analysés : {df_features['Nom'].nunique()} (à travers {len(teams)} équipes)")
    print(f"   📈 Matchs traités  : {len(df_features)}")
    if df_final is not None:
        print(f"   🔮 Prédictions     : {len(df_final)}")
    if df_anomalies is not None:
        print(f"   🚨 Alertes         : {len(df_anomalies)}")
    print(f"\n   Lancez le DASHBOARD : python -m DASHBOARD.api.main")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AthlytIQ — Pipeline complet multi-équipes")
    parser.add_argument('--retrain', action='store_true', help="Ré-entraîner les modèles ML sur toutes les données")
    parser.add_argument('--pages', type=int, default=10, help="Pages max de matchs par joueur")
    parser.add_argument('--saison', type=str, default='2024-08-01', help="Début de la saison (YYYY-MM-DD)")
    args = parser.parse_args()

    run_full_pipeline(
        nb_pages=args.pages,
        retrain=args.retrain,
        saison_debut=args.saison
    )
