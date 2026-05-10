import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import unicodedata

def normalize_name(name):
    if not isinstance(name, str): return ""
    return "".join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
    ).lower()

# ══════════════════════════════════════════════════════════════════════
# 1. MATRICE DE COMPATIBILITÉ RIGOUREUSE
# ══════════════════════════════════════════════════════════════════════
# On définit la capacité d'un joueur de poste B à remplir le rôle de poste A
POSITION_FIT = {
    "ATT": {"ATT": 1.00, "MOF": 0.60, "MDF": 0.10, "DEF": 0.02, "GK": 0.00},
    "MOF": {"MOF": 1.00, "ATT": 0.70, "MDF": 0.50, "DEF": 0.10, "GK": 0.00},
    "MDF": {"MDF": 1.00, "MOF": 0.60, "DEF": 0.60, "ATT": 0.10, "GK": 0.00},
    "DEF": {"DEF": 1.00, "MDF": 0.50, "MOF": 0.10, "ATT": 0.00, "GK": 0.00},
    "GK": {"GK": 1.00}
}

POSITION_WEIGHTS = {
    "ATT": {
        "Rating_MA10": 1.0, "xG_P90": 1.8, "xA_P90": 1.2, "Pass_Accuracy": 0.5,
        "Tackles_P90": 0.0, "distanceRun": 0.7, "Possession_Security": 0.8,
        "Ground_Duels_Won": 0.7, "Successful_Dribbles": 1.5
    },
    "MOF": {
        "Rating_MA10": 1.0, "xG_P90": 0.9, "xA_P90": 1.8, "Pass_Accuracy": 1.3,
        "Tackles_P90": 0.3, "distanceRun": 1.0, "Possession_Security": 1.2,
        "Ground_Duels_Won": 0.8, "Successful_Dribbles": 1.4
    },
    "MDF": {
        "Rating_MA10": 1.0, "xG_P90": 0.2, "xA_P90": 0.8, "Pass_Accuracy": 1.5,
        "Tackles_P90": 1.4, "distanceRun": 1.2, "Possession_Security": 1.3,
        "Ground_Duels_Won": 1.2, "Successful_Dribbles": 0.4
    },
    "DEF": {
        "Rating_MA10": 1.0, "xG_P90": 0.0, "xA_P90": 0.2, "Pass_Accuracy": 1.2,
        "Tackles_P90": 1.8, "distanceRun": 0.8, "Possession_Security": 1.0,
        "Ground_Duels_Won": 1.5, "Successful_Dribbles": 0.2
    }
}

class SimilarityEngine:
    def __init__(self, df_features):
        self.df = df_features.copy()
        self.features_list = [
            "Rating_MA10", "xG_P90", "xA_P90", "Pass_Accuracy", 
            "Tackles_P90", "distanceRun", "Possession_Security", 
            "Ground_Duels_Won_P90", "Dribbles_P90"
        ]
        
        # 1. On récupère le dernier état connu (Poste, Age, Equipe, etc.)
        self.df_raw = self.df.sort_values('Match_Date').groupby('Nom').tail(1).copy()
        
        # 2. On calcule le profil MOYEN (lissée) pour chaque joueur sur son historique
        # Cela permet d'avoir des stats représentatives et non basées sur un seul match
        df_means = self.df.groupby('Nom')[self.features_list].mean()
        
        # 3. On injecte ces moyennes dans df_raw pour l'affichage et le calcul
        for feat in self.features_list:
            self.df_raw[feat] = self.df_raw['Nom'].map(df_means[feat])
        
        # Sécurité: On remplit les éventuels NaNs par 0 pour le front-end
        self.df_raw[self.features_list] = self.df_raw[self.features_list].fillna(0)

        self.df_norm = self.df_raw.copy()
        self.scaler = StandardScaler()
        if not self.df_norm.empty:
            self.df_norm[self.features_list] = self.scaler.fit_transform(self.df_norm[self.features_list])

    def get_similar_players(self, target_player_name, alpha=0.5, top_n=10):
        # 1. Trouver le joueur cible (A) avec normalisation (accent-insensitive)
        target_norm = normalize_name(target_player_name)
        
        # On cherche d'abord la correspondance exacte normalisée
        self.df_raw['Name_Norm'] = self.df_raw['Nom'].apply(normalize_name)
        mask = self.df_raw['Name_Norm'] == target_norm
        
        if not mask.any():
            # Puis une recherche partielle normalisée
            mask = self.df_raw['Name_Norm'].str.contains(target_norm)
            
        if not mask.any(): 
            return {"error": "PLAYER_NOT_FOUND"}

        name_a = self.df_raw[mask].iloc[0]['Nom']
        row_a_norm = self.df_norm[self.df_norm['Nom'] == name_a].iloc[0]
        row_a_raw = self.df_raw[self.df_raw['Nom'] == name_a].iloc[0]
        
        # Identifier le poste NATIF de la cible (Yamal = ATT ou MOF)
        pos_map = {'F': 'ATT', 'M': 'MOF', 'D': 'DEF', 'G': 'GK'}
        target_native_pos = pos_map.get(row_a_raw.get('Poste_Cat', 'M'), 'MOF')
        
        # 1. Stratégie de sélection des candidats multiniveaux
        # Niveau 1 : Même poste, équipe différente
        target_team = row_a_raw.get('Team', 'Inconnu')
        candidates = self.df_norm[
            (self.df_norm['Poste_Cat'] == row_a_raw['Poste_Cat']) & 
            (self.df_norm['Team'] != target_team) &
            (self.df_norm['Nom'] != name_a)
        ]
        
        # Niveau 2 : Si trop peu de candidats, on accepte d'autres postes (équipe différente)
        if len(candidates) < 5:
            candidates = self.df_norm[
                (self.df_norm['Team'] != target_team) & 
                (self.df_norm['Nom'] != name_a)
            ]
            
        # Niveau 3 : Si toujours trop peu (ex: équipe isolée), on accepte même équipe (autres joueurs)
        if len(candidates) < 3:
            candidates = self.df_norm[self.df_norm['Nom'] != name_a]
            
        results = []

        for _, row_b_norm in candidates.iterrows():
            name_b = row_b_norm['Nom']
            row_b_raw = self.df_raw[self.df_raw['Nom'] == name_b].iloc[0]
            candidate_native_pos = pos_map.get(row_b_raw.get('Poste_Cat', 'M'), 'MOF')

            # 2. CALCUL DE SIMILARITÉ
            score_hybrid = self.compute_hybrid_score(name_a, name_b, alpha, target_native_pos)
            
            # 3. PÉNALITÉ DE COMPATIBILITÉ (Mutation Fit)
            mutation_fit = POSITION_FIT[candidate_native_pos].get(target_native_pos, 0.1)
            
            # Score Final
            final_score = round(score_hybrid * mutation_fit * 100, 1)

            # Seuil de visibilité abaissé pour garantir des résultats pour tous les profils
            if final_score >= 10: 
                # On récupère le résumé ET la liste complète des raisons
                explanation, full_reasons = self.generate_explanation(row_a_raw, row_b_raw, alpha)
                
                # Calcul de la polyvalence pour l'affichage (juste pour l'UI)
                versatility = {}
                for p_key in ["ATT", "MOF", "MDF", "DEF"]:
                    fit = POSITION_FIT[candidate_native_pos].get(p_key, 0.1)
                    versatility[p_key] = round(score_hybrid * fit * 100, 1)

                # Stats brutes pour comparaison (Tableau)
                raw_stats_b = {feat: round(float(row_b_raw.get(feat, 0)), 2) for feat in self.features_list}
                raw_stats_a = {feat: round(float(row_a_raw.get(feat, 0)), 2) for feat in self.features_list}

                # Stats normalisées par PERCENTILES (0-100)
                # Reflète le rang du joueur par rapport au reste de la base de données
                radar_stats_b = {}
                radar_stats_a = {}
                
                for feat in self.features_list:
                    # On calcule le rang (percentile)
                    series = self.df_raw[feat]
                    val_b = float(row_b_raw.get(feat, 0))
                    val_a = float(row_a_raw.get(feat, 0))
                    
                    # Percentile rank : quel % de la population est en dessous de cette valeur ?
                    # On ajoute une petite pondération pour éviter les 0 absolus si le joueur a une stat non-nulle
                    pct_b = (series < val_b).mean() * 100
                    pct_a = (series < val_a).mean() * 100
                    
                    # On lisse un peu pour le visuel
                    radar_stats_b[feat] = round(max(5, pct_b), 1)
                    radar_stats_a[feat] = round(max(5, pct_a), 1)

                results.append({
                    "name": name_b,
                    "team": row_b_raw.get('Team', 'Inconnu'), # Équipe actuelle du candidat
                    "target_name": name_a,
                    "target_team": row_a_raw.get('Team', 'Cible'), # Équipe de la cible
                    "native_pos": candidate_native_pos,
                    "final_score": int(final_score),
                    "versatility": versatility,
                    "medical_risk": "High" if row_b_raw.get('Medical_Risk_Score', 0) > 0.6 else "Low",
                    "confidence": int(row_b_raw.get('Match_Num', 10)),
                    "age": int(row_b_raw.get('Age', 25)),
                    "explanation": explanation,
                    "full_reasons": full_reasons,
                    "raw_stats": raw_stats_b,
                    "target_stats": raw_stats_a,
                    "radar_stats": radar_stats_b,
                    "target_radar_stats": radar_stats_a
                })

        return sorted(results, key=lambda x: x['final_score'], reverse=True)[:top_n]

    def compute_hybrid_score(self, name_a, name_b, alpha, target_pos):
        try:
            row_a = self.df_norm[self.df_norm['Nom'] == name_a].iloc[0]
            row_b = self.df_norm[self.df_norm['Nom'] == name_b].iloc[0]
        except: return 0

        vec_a = row_a[self.features_list].values.reshape(1, -1).astype(float)
        vec_b = row_b[self.features_list].values.reshape(1, -1).astype(float)
        
        # Utilisation des poids du poste de la cible
        w_dict = POSITION_WEIGHTS.get(target_pos, POSITION_WEIGHTS['MOF'])
        weights = np.array([w_dict.get(f, 1.0) for f in self.features_list])
        
        s_actuel = cosine_similarity(vec_a * weights, vec_b * weights)[0][0]
        
        # Potentiel bridé : le boost ne peut pas dépasser 20% du score actuel
        age_fact = min(1.2, row_b.get('Age_Factor', 1.0))
        slope = 1 + (0.15 * max(0, float(row_b.get('Rating_Slope5', 0) or 0)))
        
        s_potentiel = s_actuel * age_fact * slope
        # Clipping rigoureux à 1.0
        return min(1.0, max(0.0, (alpha * s_actuel) + ((1 - alpha) * s_potentiel)))

    def generate_explanation(self, player_a, player_b, alpha):
        """Génère une analyse de scouting professionnelle."""
        reasons = []
        
        # 1. Analyse du style de percussion
        drib_a = player_a.get('Successful_Dribbles', 0)
        drib_b = player_b.get('Successful_Dribbles', 0)
        if drib_b >= drib_a * 0.85:
            reasons.append("Capacité d'élimination en 1v1 identique, capable de briser les lignes par le dribble.")
        
        # 2. Analyse de la création
        xa_a = player_a.get('xA_P90', 0)
        xa_b = player_b.get('xA_P90', 0)
        if xa_b >= xa_a * 0.85:
            reasons.append("Vision de jeu créative similaire, excellente qualité de centre et de passe clé.")
        
        # 3. Analyse de la finition
        xg_a = player_a.get('xG_P90', 0)
        xg_b = player_b.get('xG_P90', 0)
        if xg_b >= xg_a * 0.85:
            reasons.append("Présence chirurgicale dans la surface et flair offensif pour la finition.")
            
        # 4. Analyse du volume de jeu (Workrate)
        dist_a = player_a.get('distanceRun', 0)
        dist_b = player_b.get('distanceRun', 0)
        if dist_b >= dist_a * 0.95:
            reasons.append("Gros volume de courses et intensité physique constante durant les 90 minutes.")

        # 5. Facteur d'âge / Projet
        age_b = player_b.get('Age', 25)
        if age_b <= 21:
            reasons.append(f"Profil très précoce ({age_b} ans) avec une courbe de progression similaire à l'élite mondiale.")

        if not reasons:
            reasons = ["Profil technique hybride offrant une polyvalence tactique rare sur les phases de transition."]
        
        # On crée le petit résumé pour la carte
        short_explanation = " " + " ".join(reasons[:2])
        
        # On renvoie le résumé ET la liste complète
        return short_explanation, reasons
