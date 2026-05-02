# AthltyQ Dashboard — Documentation

## 🚀 Démarrage Rapide

### 1. Prérequis
- Python 3.10+
- Dépendances : `fastapi`, `uvicorn`, `pandas`, `joblib`, `scikit-learn`

### 2. Lancement du Backend (API)
Le backend traite les données du dossier `DATA_PIPELINE` et expose les analyses via une API REST.

```bash
# Depuis la racine du projet
.venv/bin/python backend/main.py
```
L'API sera disponible sur : `http://localhost:8000`

### 3. Accès au Dashboard
Ouvrez simplement le fichier `backend/index.html` dans votre navigateur.
Il se connectera automatiquement à l'API locale pour afficher les données réelles.

---

## 📊 Endpoints API

- **GET `/api/team-summary`** : Retourne les métriques globales de l'équipe (disponibilité, ACR moyen, distribution des risques).
- **GET `/api/player-data`** : Retourne la liste complète des joueurs avec leurs scores de fatigue, risques de blessure et recommandations détaillées.

---

## 🛠️ Configuration & Seuils

Les seuils de risque sont calculés dynamiquement dans `LM/models/injury_predictor.py` basés sur :
- **ACWR (Acute:Chronic Workload Ratio)** : Optimal entre 0.8 et 1.3.
- **Fatigue Score** : Basé sur les minutes pondérées par l'intensité de la ligue.
- **Historique Médical** : Intègre les jours d'absence scrapés sur Transfermarkt.

---

## 🆙 Déploiement / Mise à jour

Pour mettre à jour les données du dashboard après une nouvelle session de match :
1. Exécuter le pipeline de collecte : `./run_all.sh`
2. Le backend rechargera automatiquement les fichiers CSV mis à jour lors de la prochaine requête.
