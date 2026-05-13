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
# Le script vérifie quelles ligues manquent sur les 16 et les scrappe
echo "[1/5] Vérification et récupération des ligues manquantes (SofaScore)..."
.venv/bin/python DATA_PIPELINE/SCRAPPING/main.py --mode auto

# 2. SCRAPPING TRANSFERMARKT (Blessures)
echo "[2/5] Récupération de l'historique médical Transfermarkt..."
.venv/bin/python DATA_PIPELINE/SCRAPPING/main.py --source 2

# 3. NETTOYAGE ET CONSOLIDATION
echo "[3/5] Nettoyage et fusion des données (Master Dataset)..."
.venv/bin/python DATA_PIPELINE/NETTOYAGE/scripts/data_cleaner.py

# 3.5 RÉCONCILIATION ET AUDIT DE SANTÉ
echo "[3.5/5] Audit de santé du dataset..."
.venv/bin/python DATA_PIPELINE/MAINTENANCE/reconciler.py

# 4. FEATURE ENGINEERING (Indicateurs IA)
echo "[4/5] Génération des indicateurs de fatigue et risques..."
.venv/bin/python LM/models/feature_engineering.py

# 5. ENTRAÎNEMENT DU MODÈLE (Player-wise split 80/20)
echo "[5/5] Ré-entraînement du cerveau AthlytIQ (Split : 80% Train / 20% Test)..."
.venv/bin/python LM/models/train.py > reports/training_report.txt 2>&1

# 6. TEST DE PERFORMANCE (Sur les 20% de joueurs isolés)
echo "[6/5] Évaluation stricte sur joueurs jamais vus (20% de réserve)..."
.venv/bin/python LM/models/test.py

echo "----------------------------------------------------------"
echo " ✅ PIPELINE TERMINÉ AVEC SUCCÈS !"
echo " 📊 Le rapport d'entraînement est dans reports/training_report.txt"
echo " 🖥️  Le Dashboard est maintenant à jour."
echo "----------------------------------------------------------"
