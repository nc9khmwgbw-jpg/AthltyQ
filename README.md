# ⚽ AthlytIQ — Plateforme de Prédiction de Fatigue & Blessures

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/ML-GradientBoosting%2FRandomForest-purple)
![Data](https://img.shields.io/badge/Data-SofaScore%20%2B%20Transfermarkt-orange)
![Status](https://img.shields.io/badge/Status-En%20développement-yellow)

**AthlytIQ** est une plateforme Data Science appliquée au football professionnel.  
Elle prédit la **fatigue des joueurs** et le **risque de blessure** à partir des statistiques match par match.

---

## 🎯 Objectif

```
Stats joueur (matchs, minutes, passes, duels...)
            ↓
    PRÉDICTION DE FATIGUE
  (ACWR, charge accumulée, densité des matchs)
            ↓
  PRÉDICTION DE BLESSURE
  (basée sur la fatigue + historique médical)
            ↓
  🟢 FAIBLE  🟠 MODÉRÉ  🔴 ÉLEVÉ
```

---

## 📁 Structure du Projet

```
AthlytIQ/
│
├── DATA_PIPELINE/
│   ├── SCRAPPING/
│   │   ├── scripts/
│   │   │   ├── scraper_league.py              ← Scrape tous les joueurs d'une ligue
│   │   │   ├── sofascore_match_scraper.py     ← Moteur de scraping SofaScore
│   │   │   ├── transfermarkt_injury_scraper.py ← Scrape l'historique de blessures
│   │   │   ├── fotmob_match_scraper.py        ← Réparation via FotMob
│   │   │   ├── orchestrate_repair.py          ← Pipeline de correction automatique
│   │   │   ├── prediction_physique.py         ← Estime distance/sprints manquants
│   │   │   ├── calcul_tracking_brut.py        ← Calculs tracking avancés
│   │   │   └── consolidation_sofascore.py     ← Consolidation des données
│   │   └── raw/
│   │       ├── sofascore/                     ← CSVs bruts par joueur
│   │       │   ├── Premier/[Equipe]/[Joueur].csv
│   │       │   ├── LaLiga/[Equipe]/[Joueur].csv
│   │       │   └── Ligue 1/[Equipe]/[Joueur].csv
│   │       └── transfermarkt/
│   │           └── injury_history.csv         ← Historique blessures (Transfermarkt)
│   │
│   ├── NETTOYAGE/
│   │   ├── scripts/
│   │   │   └── data_cleaner.py               ← Fusionne tous les CSVs → dataset propre
│   │   └── data/
│   │       └── merged_dataset_clean.csv       ← Dataset principal (2.3 MB)
│   │
│   └── features/                             ← Calculs de features avancés
│
├── LM/                                       ← Module Machine Learning
│   ├── models/
│   │   ├── feature_engineering.py            ← Calcule ACWR, Fatigue, Form Score...
│   │   ├── injury_predictor.py               ← Modèle de prédiction de blessure
│   │   ├── anomaly_detector.py               ← Détection d'anomalies de performance
│   │   ├── train.py                          ← Entraînement de tous les modèles
│   │   └── saved/                            ← Modèles .joblib entraînés
│   └── pipeline/
│       └── data_pipeline.py                  ← Orchestrateur du pipeline ML complet
│
├── opendata/                                 ← Données SkillCorner (tracking GPS)
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

```bash
# 1. Cloner le projet
git clone https://github.com/nc9khmwgbw-jpg/AthltyQ.git
cd AthltyQ

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 📋 Pipeline Complet — Étape par Étape

### ÉTAPE 1 — Scraper les matchs (SofaScore)

Scrape les statistiques match par match pour tous les joueurs d'une ligue.

```bash
python DATA_PIPELINE/SCRAPPING/scripts/scraper_league.py
```

> Choisir la ligue dans le menu interactif (Premier League, LaLiga, Ligue 1...)  
> Durée : ~2-3h pour une ligue complète  
> Output : `DATA_PIPELINE/SCRAPPING/raw/sofascore/[Ligue]/[Equipe]/[Joueur].csv`

---

### ÉTAPE 2 — Nettoyer et fusionner les données

Fusionne tous les CSVs des joueurs en un seul dataset propre.  
Estime automatiquement la **distance parcourue** et les **sprints** manquants via un modèle physique.

```bash
python DATA_PIPELINE/NETTOYAGE/scripts/data_cleaner.py
```

> Output : `DATA_PIPELINE/NETTOYAGE/data/merged_dataset_clean.csv`

---

### ÉTAPE 3 — Scraper l'historique de blessures (Transfermarkt)

Scrape l'historique complet de blessures pour chaque joueur du dataset.  
À faire **une seule fois** (ou relancer avec `--resume` pour mettre à jour).

```bash
# Test sur 10 joueurs d'abord
python DATA_PIPELINE/SCRAPPING/scripts/transfermarkt_injury_scraper.py --limit 10

# Tous les joueurs (~1671 — environ 2-3 heures)
python DATA_PIPELINE/SCRAPPING/scripts/transfermarkt_injury_scraper.py

# Reprendre si interrompu
python DATA_PIPELINE/SCRAPPING/scripts/transfermarkt_injury_scraper.py --resume

# Repartir de zéro
python DATA_PIPELINE/SCRAPPING/scripts/transfermarkt_injury_scraper.py --reset
```

> Output : `DATA_PIPELINE/SCRAPPING/raw/transfermarkt/injury_history.csv`  
> Colonnes : `Nom, Season, Injury_Type, Date_From, Date_To, Duration_Days, Cause_Category`

---

### ÉTAPE 4 — Feature Engineering (Fusion + Calculs)

Fusionne les stats SofaScore + l'historique Transfermarkt.  
Calcule toutes les features de fatigue et de forme.

Ce script s'exécute **automatiquement** via `injury_predictor.py` (étape 5).  
Il peut aussi être lancé manuellement :

```bash
python LM/models/feature_engineering.py
```

**Features calculées :**

| Feature | Description |
|---------|-------------|
| `ACWR` | Ratio charge récente / charge chronique (zone idéale : 0.8–1.3) |
| `Fatigue_Index` | Minutes cumulées sur 5 matchs, normalisé 0–1 |
| `Congestion_Risk` | Risque si moins de 4 jours entre matchs |
| `Trauma_Index` | Intensité des duels sur 3 matchs glissants |
| `Medical_Risk_Score` | Score de risque médical combiné |
| `Injury_Prone_Index` | Fragilité du joueur (basé sur historique Transfermarkt) |
| `Form_Score` | Score de forme composite 0–100 |

---

### ÉTAPE 5 — Prédiction Fatigue & Blessure

Lance l'entraînement du modèle ML et génère les prédictions pour tous les joueurs.

```bash
python LM/models/injury_predictor.py
```

**Ce que fait ce script :**
1. Calcule le **Fatigue Score** (0–100) pour chaque match
2. Entraîne le modèle **Gradient Boosting + Random Forest** sur les données historiques
3. Prédit le **risque de blessure** (0–1) pour chaque joueur
4. Affiche un rapport classé par niveau de risque

**Résultat :**
```
🔴 ÉLEVÉ   → Repos recommandé
🟠 MODÉRÉ  → Surveiller la charge d'entraînement
🟢 FAIBLE  → Bonne condition physique
```

---

## 📊 Modèles ML

| Modèle | Rôle | Performance estimée |
|--------|------|-------------------|
| Gradient Boosting | Prédiction risque de blessure | ~75-80% AUC-ROC |
| Random Forest | Prédiction risque de blessure (ensemble) | ~73-78% AUC-ROC |
| Isolation Forest | Détection d'anomalies de performance | 95%+ détection |

> ⚠️ Pour atteindre 80%+ : scraper l'historique Transfermarkt sur tous les joueurs (Étape 3)

---

## 🔁 Workflow Quotidien (Maintien du Dataset)

```bash
# Corriger les données manquantes/erronées via FotMob
python DATA_PIPELINE/SCRAPPING/scripts/orchestrate_repair.py

# Refaire le dataset propre
python DATA_PIPELINE/NETTOYAGE/scripts/data_cleaner.py

# Relancer les prédictions
python LM/models/injury_predictor.py
```

---

## 🔧 Réparation des Données (FotMob)

Si des statistiques sont manquantes ou incorrectes dans les CSVs SofaScore :

```bash
python DATA_PIPELINE/SCRAPPING/scripts/orchestrate_repair.py
```

Ce script compare automatiquement les données SofaScore avec FotMob et corrige les valeurs incorrectes.

---

## 💻 Technologies

| Outil | Usage |
|-------|-------|
| Python 3.10+ | Langage principal |
| Pandas / NumPy | Manipulation des données |
| Scikit-learn | Modèles ML |
| XGBoost | Gradient Boosting |
| Selenium | Scraping SofaScore (anti-bot) |
| BeautifulSoup | Scraping Transfermarkt |
| Joblib | Sauvegarde des modèles |

---

## ⚠️ Notes Importantes

1. **Rate Limiting** : Les scrapers utilisent des pauses aléatoires entre les requêtes pour éviter le blocage. Ne pas réduire les délais.

2. **Reprise automatique** : Les scrapers sauvegardent leur progression. Si interrompus, ils reprennent là où ils s'étaient arrêtés (`--resume`).

3. **Respect des CGU** : Ce projet est à but éducatif. Respectez les conditions d'utilisation de SofaScore et Transfermarkt.

4. **Qualité des données** : Plus le dataset Transfermarkt est complet (beaucoup de joueurs scrapés), plus le modèle de blessure sera précis.

---

## 👥 Auteurs

Projet AthlytIQ — Data Science & Football Analytics

---

*AthlytIQ — De la data à la performance* ⚽



Ce que tu as (précision estimée : ~68-72%)
✅ Minutes jouées, dates, ACWR, Fatigue_Index
✅ Historique blessures Transfermarkt (une fois scrapé)
⚠️ distanceRun et sprints → estimés, pas réels
⚠️ Beaucoup de Rating = 0.0 dans tes CSV (données manquantes)

Ce qui manque pour passer à 80%+
🔴 Priorité 1 — Corriger les données manquantes (gain : +4-5%)
Regarde tes CSV → beaucoup de lignes ont Rating = 0.0 alors que le joueur a joué. C'est du bruit qui affaiblit le modèle. Il faut réparer ces valeurs via FotMob (orchestrate_repair.py).

🟠 Priorité 2 — Ajouter l'âge du joueur (gain : +3-4%)
Un joueur de 34 ans se fatigue différemment qu'un joueur de 21 ans. C'est disponible sur Transfermarkt — tu le scrapes déjà, il suffit d'ajouter la date de naissance.

🟠 Priorité 3 — Ajouter le poste (gain : +3-4%)
Un ailier fait 11km/match, un défenseur central 9km. Sans le poste, le modèle compare des profils incomparables.

🟡 Priorité 4 — Augmenter le nombre de matchs par joueur (gain : +3-5%)
Tu limites à 15 matchs par joueur. Passer à 30-40 matchs (2 saisons) donnerait beaucoup plus de données temporelles à apprendre.

🟡 Priorité 5 — Ajouter les matchs en sélection nationale (gain : +2-3%)
Un joueur qui part en sélection et revient 3 jours avant un match de ligue est beaucoup plus à risque. Ces matchs ne sont pas capturés dans tes CSV de ligue.