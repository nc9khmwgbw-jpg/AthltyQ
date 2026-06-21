#!/bin/bash

# Configuration du chemin Python pour éviter les erreurs de module
export PYTHONPATH=$PYTHONPATH:.

# ==========================================================
# 🚀 ATHLYTIQ ENTERPRISE - PIPELINE COMPLET (v2.0)
# ==========================================================

echo "----------------------------------------------------------"
echo " 🏁 DÉMARRAGE DU PIPELINE AUTOMATISÉ"
echo "----------------------------------------------------------"

# 1. SCRAPPING SOFASCORE (Auto-détection des manques)
echo "[1/7] Mise à jour des données SofaScore (Bypass Cloudflare activé)..."
.venv/bin/python DATA_PIPELINE/SCRAPPING/main.py --mode auto

# 2. SCRAPPING TRANSFERMARKT (Blessures)
echo "[2/7] Mise à jour des données Transfermarkt..."
.venv/bin/python DATA_PIPELINE/SCRAPPING/main.py --source 2

# 3. NETTOYAGE ET CONSOLIDATION
echo "[3/6] Nettoyage et fusion des données (Master Dataset)..."
.venv/bin/python DATA_PIPELINE/NETTOYAGE/scripts/data_cleaner.py

# 3.5 RÉCONCILIATION ET AUDIT DE SANTÉ
echo "[3.5/6] Audit de santé du dataset..."
.venv/bin/python DATA_PIPELINE/MAINTENANCE/reconciler.py

# 4. FEATURE ENGINEERING (Indicateurs IA)
echo "[4/6] Génération des indicateurs de fatigue et risques..."
.venv/bin/python LM/models/feature_engineering.py

# 5. GÉNÉRATION DES 9 POSTES (Inférence IA + Mapping)
echo "[5/11] Classification des joueurs en 9 postes granulaires..."
.venv/bin/python LM2/build_positions.py

# 6. PRÉPARATION DU BENCHMARK SCOUTING (Profils agrégés)
echo "[6/11] Préparation des profils scouting (EWM 15 matchs)..."
.venv/bin/python LM2/benchmark/setup_scouting_data.py

# 7. APPRENTISSAGE DES POIDS DE SIMILARITÉ (Ridge + Optuna par poste)
echo "[7/11] Apprentissage des poids de similarité par poste..."
.venv/bin/python LM2/benchmark/train_weights.py

# 8. CLUSTERING DES ARCHÉTYPES (K-Means + Silhouette)
echo "[8/11] Découverte des archétypes de joueurs..."
.venv/bin/python LM2/benchmark/train_clusters.py

# 9. PRÉPARATION DU BENCHMARK IA (Player-wise split 80/20)
echo "[9/11] Définition de la règle Train/Test stricte..."
.venv/bin/python LM/models/benchmark/setup_data.py

# 10. ENTRAÎNEMENT DES 3 MODÈLES (Ordre optimisé : RF en dernier car plus lent)
echo "[10/11] Entraînement des algorithmes (Poly, LGBM, RF)..."
.venv/bin/python LM/models/benchmark/Polynomial_Regression/train_poly.py > reports/training_poly.txt 2>&1
.venv/bin/python LM/models/benchmark/LightGBM/train_lgbm.py > reports/training_lgbm.txt 2>&1
.venv/bin/python LM/models/benchmark/Random_Forest/train_rf.py > reports/training_rf.txt 2>&1

# 11. TEST DE PERFORMANCE ET COMPARAISON (Sur les 20% isolés)
echo "[11/11] Évaluation stricte et génération des graphiques comparatifs..."
.venv/bin/python LM/models/benchmark/generate_visuals.py
.venv/bin/python LM2/benchmark/generate_scouting_visuals.py

echo "----------------------------------------------------------"
echo " ✅ PIPELINE ET BENCHMARK TERMINÉS AVEC SUCCÈS !"
echo " 📈 Les graphiques d'analyse sont dans LM/models/benchmark/plots/"
echo " 🖥️  Le Dashboard est maintenant à jour et prêt à l'emploi avec les 3 IA."
echo "----------------------------------------------------------"
