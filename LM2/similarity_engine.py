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

POSITION_FIT = {
    "ATT": {"ATT": 1.0, "WING": 0.8, "MOF": 0.5, "MC": 0.2, "MDF": 0.1, "CB": 0.0, "FB": 0.1, "GK": 0.0},
    "WING": {"WING": 1.0, "ATT": 0.8, "MOF": 0.7, "FB": 0.6, "MC": 0.3, "MDF": 0.2, "CB": 0.0, "GK": 0.0},
    "MOF": {"MOF": 1.0, "WING": 0.7, "ATT": 0.5, "MC": 0.8, "MDF": 0.4, "CB": 0.1, "FB": 0.3, "GK": 0.0},
    "MC": {"MC": 1.0, "MOF": 0.8, "MDF": 0.8, "WING": 0.4, "ATT": 0.2, "FB": 0.4, "CB": 0.3, "GK": 0.0},
    "MDF": {"MDF": 1.0, "MC": 0.8, "CB": 0.6, "FB": 0.4, "MOF": 0.4, "WING": 0.1, "ATT": 0.0, "GK": 0.0},
    "CB": {"CB": 1.0, "FB": 0.4, "MDF": 0.6, "MC": 0.3, "MOF": 0.0, "WING": 0.0, "ATT": 0.0, "GK": 0.0},
    "FB": {"FB": 1.0, "CB": 0.4, "WING": 0.6, "MDF": 0.4, "MC": 0.4, "MOF": 0.2, "ATT": 0.1, "GK": 0.0},
    "GK": {"GK": 1.0}
}

POSTE_MAP = {
    'CF': 'ATT', 'ST': 'ATT', 'F': 'ATT', 'ATT': 'ATT',
    'LW': 'WING', 'RW': 'WING', 'AG': 'WING', 'AD': 'WING',
    'AM': 'MOF', 'RM': 'MOF', 'LM': 'MOF', 'MOF': 'MOF',
    'CM': 'MC', 'M': 'MC', 'MC': 'MC',
    'DM': 'MDF', 'MDF': 'MDF',
    'CB': 'CB', 'D': 'CB',
    'LB': 'FB', 'RB': 'FB', 'WB': 'FB',
    'GK': 'GK', 'G': 'GK'
}

POSITION_WEIGHTS = {
    "ATT": {
        "Rating_MA10": 1.0, "xG_P90": 2.0, "xA_P90": 1.2, "Pass_Accuracy": 0.5,
        "Tackles_P90": 0.0, "distanceRun": 0.8, "Possession_Security": 0.8,
        "Ground_Duels_Won_P90": 0.7, "Dribbles_P90": 1.8,
        "Fatigue_IA": 1.5, "Medical_Risk_Score": 2.0
    },
    "WING": {
        "Rating_MA10": 1.0, "xG_P90": 1.5, "xA_P90": 1.8, "Pass_Accuracy": 1.0,
        "Tackles_P90": 0.5, "distanceRun": 1.2, "Possession_Security": 1.0,
        "Ground_Duels_Won_P90": 1.0, "Dribbles_P90": 2.0,
        "Fatigue_IA": 1.5, "Medical_Risk_Score": 2.0
    },
    "MOF": {
        "Rating_MA10": 1.0, "xG_P90": 0.9, "xA_P90": 2.0, "Pass_Accuracy": 1.5,
        "Tackles_P90": 0.4, "distanceRun": 1.2, "Possession_Security": 1.3,
        "Ground_Duels_Won_P90": 1.0, "Dribbles_P90": 1.5,
        "Fatigue_IA": 1.5, "Medical_Risk_Score": 1.8
    },
    "MC": {
        "Rating_MA10": 1.0, "xG_P90": 0.5, "xA_P90": 1.2, "Pass_Accuracy": 1.8,
        "Tackles_P90": 1.0, "distanceRun": 1.5, "Possession_Security": 1.5,
        "Ground_Duels_Won_P90": 1.2, "Dribbles_P90": 1.0,
        "Fatigue_IA": 1.8, "Medical_Risk_Score": 2.0
    },
    "MDF": {
        "Rating_MA10": 1.0, "xG_P90": 0.2, "xA_P90": 1.0, "Pass_Accuracy": 1.8,
        "Tackles_P90": 1.8, "distanceRun": 1.4, "Possession_Security": 1.5,
        "Ground_Duels_Won_P90": 1.5, "Dribbles_P90": 0.5,
        "Fatigue_IA": 2.2, "Medical_Risk_Score": 2.5
    },
    "CB": {
        "Rating_MA10": 1.2, "xG_P90": 0.1, "xA_P90": 0.2, "Pass_Accuracy": 1.4,
        "Tackles_P90": 2.0, "distanceRun": 1.0, "Possession_Security": 1.2,
        "Ground_Duels_Won_P90": 2.0, "Dribbles_P90": 0.3,
        "Fatigue_IA": 1.8, "Medical_Risk_Score": 2.2
    },
    "FB": {
        "Rating_MA10": 1.0, "xG_P90": 0.3, "xA_P90": 1.5, "Pass_Accuracy": 1.2,
        "Tackles_P90": 1.5, "distanceRun": 1.8, "Possession_Security": 1.0,
        "Ground_Duels_Won_P90": 1.5, "Dribbles_P90": 1.2,
        "Fatigue_IA": 1.8, "Medical_Risk_Score": 2.0
    }
}

class SimilarityEngine:
    def __init__(self, df_features):
        self.df = df_features.copy()
        self.features_list = [
            "Rating_MA10", "xG_P90", "xA_P90", "Pass_Accuracy", 
            "Tackles_P90", "distanceRun", "Possession_Security", 
            "Ground_Duels_Won_P90", "Dribbles_P90",
            "Fatigue_IA", "Medical_Risk_Score"
        ]
        
        for f in self.features_list:
            if f in self.df.columns:
                self.df[f] = pd.to_numeric(self.df[f], errors='coerce').fillna(0)
            else:
                self.df[f] = 0.0

        self.df_raw = self.df.sort_values('Match_Date').groupby('Nom').tail(1).copy()
        perf_feats = [f for f in self.features_list if f not in ["Fatigue_IA", "Medical_Risk_Score"]]
        df_means = self.df.groupby('Nom')[perf_feats].mean()
        for feat in perf_feats:
            self.df_raw[feat] = self.df_raw['Nom'].map(df_means[feat])
        
        self.df_norm = self.df_raw.copy()
        self.scaler = StandardScaler()
        if not self.df_norm.empty:
            self.df_norm[self.features_list] = self.scaler.fit_transform(self.df_norm[self.features_list])

    def get_similar_players(self, target_player_name, alpha=0.5, top_n=10):
        target_norm = normalize_name(target_player_name)
        self.df_raw['Name_Norm'] = self.df_raw['Nom'].apply(normalize_name)
        mask = self.df_raw['Name_Norm'] == target_norm
        if not mask.any(): mask = self.df_raw['Name_Norm'].str.contains(target_norm, regex=False)
        if not mask.any(): return {"error": "PLAYER_NOT_FOUND"}

        row_a_raw = self.df_raw[mask].iloc[0]
        row_a_norm = self.df_norm[self.df_norm['Nom'] == row_a_raw['Nom']].iloc[0]
        name_a = row_a_raw['Nom']
        target_team = row_a_raw.get('Team', 'Inconnu')
        target_pos = POSTE_MAP.get(row_a_raw.get('Poste_Cat', 'M'), 'MOF')

        # --- DNA Cible ---
        team_players = self.df_raw[self.df_raw['Team'] == target_team]
        if len(team_players) > 3:
            team_dna = team_players[self.features_list].mean()
        else:
            team_dna = self.df_raw[self.features_list].mean()

        candidates = self.df_norm[self.df_norm['Nom'] != name_a]
        results = []
        
        for _, row_b_norm in candidates.iterrows():
            name_b = row_b_norm['Nom']
            row_b_raw = self.df_raw[self.df_raw['Nom'] == name_b].iloc[0]
            cand_pos = POSTE_MAP.get(row_b_raw.get('Poste_Cat', 'M'), 'MOF')

            # 1. Similarité Intrinsèque
            sim_score = self.compute_hybrid_score(row_a_norm, row_b_norm, alpha, target_pos)
            
            # 2. Squad DNA Fit
            dna_fit = self.calculate_dna_fit(row_b_raw, team_dna, target_pos)
            
            # 3. Fit Factor
            fit_factor = POSITION_FIT.get(cand_pos, {}).get(target_pos, 0.1)
            
            final_score = int(((sim_score * 0.6) + (dna_fit * 0.4)) * fit_factor * 100)

            if final_score > 10:
                explanation, full_reasons = self.generate_explanation(row_a_raw, row_b_raw, dna_fit)
                
                # Radar stats (Percentiles sécurisés)
                radar_b = {}
                radar_a = {}
                for f in self.features_list:
                    # On évite les erreurs si la colonne est vide ou constante
                    if self.df_raw[f].std() == 0:
                        rb = 50.0
                        ra = 50.0
                    else:
                        rb = (self.df_raw[f] < float(row_b_raw[f])).mean() * 100
                        ra = (self.df_raw[f] < float(row_a_raw[f])).mean() * 100
                    
                    radar_b[f] = round(max(5, rb), 1)
                    radar_a[f] = round(max(5, ra), 1)

                results.append({
                    "name": name_b,
                    "team": row_b_raw.get('Team', 'Inconnu'),
                    "target_name": name_a,
                    "target_team": target_team,
                    "native_pos": cand_pos,
                    "final_score": final_score,
                    "dna_fit_score": int(dna_fit * 100),
                    "explanation": explanation,
                    "full_reasons": full_reasons,
                    "medical_risk": "High" if row_b_raw.get('Medical_Risk_Score', 0) > 0.6 else "Low",
                    "confidence": int(row_b_raw.get('Match_Num', 10)),
                    "versatility": {p: int(sim_score * POSITION_FIT[cand_pos].get(p, 0.1) * 100) for p in ["ATT", "WING", "MOF", "MDF", "CB", "FB"]},
                    "raw_stats": {f: round(float(row_b_raw[f]), 2) for f in self.features_list},
                    "target_stats": {f: round(float(row_a_raw[f]), 2) for f in self.features_list},
                    "radar_stats": radar_b,
                    "target_radar_stats": radar_a
                })

        return sorted(results, key=lambda x: x['final_score'], reverse=True)[:top_n]

    def calculate_dna_fit(self, player_b, team_dna, pos):
        """Calcule si le joueur peut physiquement et techniquement s'intégrer à l'équipe cible."""
        # On se concentre sur les critères d'intensité et de discipline
        critical_feats = ["distanceRun", "Pass_Accuracy", "Possession_Security", "Tackles_P90"]
        diffs = []
        for f in critical_feats:
            target_val = team_dna[f]
            player_val = player_b[f]
            # Si le joueur est au-dessus ou proche de la moyenne équipe, c'est bon
            ratio = min(1.2, player_val / (target_val + 1e-6))
            diffs.append(ratio)
        
        # Malus si le risque médical est trop haut par rapport à la moyenne équipe
        med_ratio = 1.0
        if player_b['Medical_Risk_Score'] > team_dna['Medical_Risk_Score'] * 1.5:
            med_ratio = 0.8 # Pénalité de 20% car trop fragile pour le rythme de l'équipe
            
        return (sum(diffs) / len(diffs)) * med_ratio

    def compute_hybrid_score(self, row_a, row_b, alpha, target_pos):
        vec_a = row_a[self.features_list].values.reshape(1, -1).astype(float)
        vec_b = row_b[self.features_list].values.reshape(1, -1).astype(float)
        w_dict = POSITION_WEIGHTS.get(target_pos, POSITION_WEIGHTS['MOF'])
        weights = np.array([w_dict.get(f, 1.0) for f in self.features_list])
        s_actuel = cosine_similarity(vec_a * weights, vec_b * weights)[0][0]
        return min(1.0, max(0.0, s_actuel))

    def generate_explanation(self, p_a, p_b, dna_fit):
        reasons = []
        if p_b['Dribbles_P90'] >= p_a['Dribbles_P90'] * 0.85:
            reasons.append("Profil de percussion identique.")
        if dna_fit > 0.95:
            reasons.append(f"ADÉQUATION TACTIQUE ÉLEVÉE : Son volume de jeu ({int(p_b['distanceRun'])}m) correspond exactement au DNA de l'équipe.")
        else:
            reasons.append("Besoin d'adaptation physique pour coller au rythme de l'équipe.")
            
        if p_b['Medical_Risk_Score'] < 0.3:
            reasons.append("Fiabilité athlétique supérieure, capable d'enchaîner les matchs.")
            
        return " " + " ".join(reasons[:2]), reasons
