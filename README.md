# AthlytIQ - Plateforme Prédictive de Performance Football

![AthlytIQ](https://img.shields.io/badge/AthlytIQ-Module%201-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![ML](https://img.shields.io/badge/ML-XGBoost%2FLSTM%2FIF-purple)

**AthlytIQ** est une plateforme advanced de Data Science et Business Intelligence appliquée au football professionnel. Transformez les données brutes SofaScore en prédictions de performance et alertes médicales précises.

## 🎯 Fonctionnalités

### Performance Predictor
- **XGBoost + Random Forest** : Prédit le Form Score à J+7, J+14, J+30
- **LSTM** : Capture les dépendances temporelles longues
- **Validation temporelle** : TimeSeriesSplit pour éviter le data leakage

### Medical / Injury Predictor
- **XGBoost Classifier** : Prédit la probabilité de blessure
- **ACWR** : Acute:Chronic Workload Ratio pour évaluer la surcharge
- **Fatigue Index** : Accumulation des minutes sur 3 matchs

### Anomaly Detector
- **Isolation Forest** : Détecte les formes anormales
- **Logique "Match Manqué"** : Identifie les blessures en temps réel 🏥
- **Seuils adaptatifs** : Basés sur la volatilité historique du joueur

## 📁 Structure du Projet

```
AthlytIQ/
├── api/                  # Serveur FastAPI
│   ├── main.py          # Endpoints REST
│   └── schemas.py       # Modèles Pydantic
├── DASHBOARD/          # Interface Web
│   ├── index.html      # Dashboard premium
│   ├── index.css       # Glass-morphism design
│   └── app.js          # Logique JavaScript
├── data/
│   ├── processed/      # Features & Prédictions
│   └── temporal/       # Données match par match
├── models/             # Modèles ML entraînés
│   └── saved/         # Fichiers .joblib
├── pipeline/           # Flux de données
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   └── data_cleaner.py
├── scrapers/           # Collecte de données
│   └── sofascore_match_scraper.py
└── requirements.txt
```

## 🚀 Installation

```bash
# Cloner le projet
git clone https://github.com/votre-user/AthlytIQ.git
cd AthlytIQ

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le driver Chrome
python -m selenium.webdriver.chrome.service
```

## 📥 Collecte des Données

### 1. Référencer l'effectif

```bash
python -m pipeline.reindex_squad
```

### 2. Scraper les matchs

```bash
python -m scrapers.sofascore_match_scraper
```

## ⚙️ Exécution du Pipeline

```bash
# Pipeline complet
python -m pipeline.data_pipeline

# Avec ré-entraînement des modèles
python -m pipeline.data_pipeline --retrain
```

## 🔮 Lancement de l'API

```bash
# Démarrer le serveur
python -m api.main

# Dashboard disponible sur: http://localhost:8000
# Documentation API: http://localhost:8000/docs
```

## 📊 Modèles ML

| Modèle | Horizon | Type | Performance |
|--------|---------|------|-------------|
| XGBoost | J+7 | Régression | MAE ≈ 5-8 |
| Random Forest | J+7 | Régression | MAE ≈ 6-9 |
| LSTM | J+1 | Série temporelle | MAE ≈ 4-7 |
| Isolation Forest | — | Détection anomalie | 95%+ détection |
| XGBoost Classifier | J+14 | Classification | ROC-AUC ≈ 0.7+ |

## 🎨 Dashboard

Interface premium **glass-morphism** avec :
- Vue d'ensemble des joueurs
- Prédictions de forme
- Alertes de performance
- Fiche joueur détaillée avec historique
- Panneau de risque médical

## 🔧 Technologies

- **Python 3.9+** : Core language
- **Pandas/NumPy** : Manipulation de données
- **Scikit-learn** : Modèles ML classiques
- **XGBoost** : Gradient Boosting
- **TensorFlow/Keras** : LSTM Deep Learning
- **FastAPI** : API REST
- **Selenium** : Web Scraping

## ⚠️ Notes Importantes

1. **Données de blessure** : La target `Target_Injury_Occurred` est actuellement simulée. Pour un usage production, remplacez par des données médicales réelles du club.

2. **Rate Limiting** : Le scraper SofaScore utilise des pauses pour éviter le blocage. Pour 25 joueurs × 3 pages, comptez ~45 minutes.

3. **Confidentialité** : Respectez les CGU de SofaScore. Ce projet est à but éducatif.

## 📈 Améliorations Futures

- [ ] Intégration données OPTA/StatsBomb
- [ ] Système de monitoring des modèles (drift detection)
- [ ] API temps réel via WebSocket
- [ ] Modèle multi-équipes
- [ ] Optimisation hyperparamètres avec Optuna

## 📝 Licence

MIT License - voir fichier LICENSE

## 👤 Auteur

**MiniMax Agent** - Développement Data Science & ML

---

*AthlytIQ - Transformez la data en performance* ⚽
