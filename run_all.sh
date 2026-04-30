#!/bin/bash

echo "=========================================================="
echo " 🚀 DÉMARRAGE DU PIPELINE COMPLET ATHLYTIQ (NIGHT RUN)"
echo "=========================================================="

# 1. Scraping complet Transfermarkt (1671 joueurs)
echo "[1/3] Scraping Transfermarkt en cours (cela prendra ~2 heures)..."
.venv/bin/python DATA_PIPELINE/SCRAPPING/scripts/transfermarkt_injury_scraper.py --resume

# 2. Nettoyage et Fusion
echo "[2/3] Nettoyage et application de l'Age et de la Position..."
.venv/bin/python DATA_PIPELINE/NETTOYAGE/scripts/data_cleaner.py

# 3. Entraînement du modèle
echo "[3/3] Entraînement du modèle Machine Learning..."
.venv/bin/python LM/models/injury_predictor.py > rapport_nuit.txt 2>&1

echo "=========================================================="
echo " ✅ PIPELINE TERMINÉ ! Le rapport est dans rapport_nuit.txt"
echo "=========================================================="
