# AthlytIQ — MedSport Intelligence Platform

AthlytIQ est une plateforme d'intelligence médicale de pointe conçue pour le football professionnel. Elle utilise des modèles de Machine Learning (RandomForest) pour prédire la fatigue neuromusculaire et évaluer les risques de blessures des joueurs.

## 🚀 Installation & Lancement

### 1. Environnement
Assurez-vous d'utiliser le virtualenv configuré :
```bash
source .venv/bin/activate
```

### 2. Pipeline de Données (Auto-Scraping & Training)
Pour rafraîchir l'intégralité des données (SofaScore + Transfermarkt + Nettoyage + IA) :
```bash
bash run_all.sh
```

### 3. Lancement du Dashboard Medical
```bash
python DASHBOARD/backend.py
```

## 🖥️ Accès à l'Interface (IMPORTANT)

> [!CAUTION]
> **Ne pas ouvrir index.html directement avec votre navigateur (file://).** 
> Les appels API seront bloqués par les restrictions de sécurité du navigateur (CORS).

Pour accéder au dashboard, utilisez l'URL servie par le backend FastAPI :
👉 **[http://localhost:8081](http://localhost:8081)**

## 🏥 Logique de Risque Médical v3.0

Le système calcule un **Medical Risk Score** composite basé sur trois piliers orthogonaux :
1. **Fatigue IA (50%)** : Prédiction de l'épuisement neuromusculaire via RandomForest.
2. **Historique Trauma (30%)** : Basé sur le nombre de blessures musculaires récentes et l'index de traumatisme.
3. **Stress de Charge (20%)** : Basé sur l'ACWR (Acute:Chronic Workload Ratio).

### Seuils d'Alerte :
- 🔴 **> 50%** : Risque ÉLEVÉ (Alerte Blessure / Repos Forcé)
- 🟠 **> 15%** : Risque MODÉRÉ (Vigilance / Entraînement Adapté)
- 🟢 **<= 15%** : Risque FAIBLE (Full Training)

## 🛠️ Architecture du Projet
- `DASHBOARD/` : Frontend HTML/JS et Backend FastAPI.
- `DATA_PIPELINE/` : Scrapers (SofaScore/Transfermarkt) et scripts de nettoyage.
- `LM/` : Modèles d'IA et logique de feature engineering.
- `data/processed/` : Dataset final enrichi (`features_dataset.csv`).

---
*Développé pour la haute performance athlétique.*
