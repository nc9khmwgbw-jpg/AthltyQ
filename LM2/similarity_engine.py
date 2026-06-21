"""
AthlytIQ — Similarity Engine v3.0 (Moteur de Scouting Industrialisé)
======================================================================
Compare les joueurs via un profil multi-match pondéré, enrichi par :
  - Poids appris par ML (Ridge Regression par poste)
  - Clustering K-Means (archétypes de joueurs)
  - Calcul 100% vectorisé (scipy.spatial.distance)
  - Multi-métrique : Cosinus + Euclidean + PCA
  - Score de momentum + confiance + DNA Fit

Rétrocompatibilité API :
  SimilarityEngine(df_features)
  .get_similar_players(target_player_name, alpha=0.5, top_n=10)
"""

import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import joblib


# ══════════════════════════════════════════════════════════════════════
# NORMALISATION DES NOMS (accents, majuscules)
# ══════════════════════════════════════════════════════════════════════

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    return "".join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
    ).lower()


# ══════════════════════════════════════════════════════════════════════
# SYSTÈME DE POSTES GRANULAIRES (9 postes précis)
# ══════════════════════════════════════════════════════════════════════

POSITION_FIT = {
    "ATT": {"ATT":1.00,"AG":0.50,"AD":0.50,"MOF":0.30,"MC":0.05,"MDF":0.00,"CB":0.00,"LB":0.00,"RB":0.00},
    "AG":  {"ATT":0.50,"AG":1.00,"AD":0.70,"MOF":0.50,"MC":0.10,"MDF":0.00,"CB":0.00,"LB":0.05,"RB":0.00},
    "AD":  {"ATT":0.50,"AG":0.70,"AD":1.00,"MOF":0.50,"MC":0.10,"MDF":0.00,"CB":0.00,"LB":0.00,"RB":0.05},
    "MOF": {"ATT":0.30,"AG":0.50,"AD":0.50,"MOF":1.00,"MC":0.60,"MDF":0.20,"CB":0.00,"LB":0.00,"RB":0.00},
    "MC":  {"ATT":0.05,"AG":0.10,"AD":0.10,"MOF":0.60,"MC":1.00,"MDF":0.60,"CB":0.10,"LB":0.10,"RB":0.10},
    "MDF": {"ATT":0.00,"AG":0.00,"AD":0.00,"MOF":0.20,"MC":0.60,"MDF":1.00,"CB":0.30,"LB":0.20,"RB":0.20},
    "CB":  {"ATT":0.00,"AG":0.00,"AD":0.00,"MOF":0.00,"MC":0.10,"MDF":0.30,"CB":1.00,"LB":0.20,"RB":0.20},
    "LB":  {"ATT":0.00,"AG":0.05,"AD":0.00,"MOF":0.00,"MC":0.10,"MDF":0.20,"CB":0.20,"LB":1.00,"RB":0.60},
    "RB":  {"ATT":0.00,"AG":0.00,"AD":0.05,"MOF":0.00,"MC":0.10,"MDF":0.20,"CB":0.20,"LB":0.60,"RB":1.00},
}

POSTE_MAP = {
    'CF': 'ATT', 'ST': 'ATT', 'LW': 'AG', 'LF': 'AG',
    'RW': 'AD', 'RF': 'AD', 'SS': 'ATT', 'F': 'ATT',
    'AM': 'MOF', 'CAM': 'MOF', 'CM': 'MC', 'M': 'MC',
    'LM': 'AG', 'RM': 'AD', 'DM': 'MDF', 'CDM': 'MDF',
    'CB': 'CB', 'D': 'CB', 'LB': 'LB', 'LWB': 'LB',
    'RB': 'RB', 'RWB': 'RB', 'WB': 'RB',
    'ATT': 'ATT', 'AG': 'AG', 'AD': 'AD',
    'MOF': 'MOF', 'MC': 'MC', 'MDF': 'MDF',
    'DEF': 'CB',
}


# ══════════════════════════════════════════════════════════════════════
# 15 FEATURES ORTHOGONALES
# ══════════════════════════════════════════════════════════════════════

FEATURES_ORTHOGONALES = [
    "Rating_MA10", "xG_P90", "xA_P90", "Pass_Accuracy",
    "Defensive_Actions_P90", "distanceRun", "Possession_Security",
    "Dribbles_P90", "Key_Passes_P90", "Fatigue_Index",
    "Medical_Risk_Score", "Age", "Rating_Trend", "Rating_STD5",
    "Minutes_Played",
]

# Poids par défaut (fallback si learned_weights.joblib n'existe pas)
_DEFAULT_WEIGHTS = {
    "ATT": {"Rating_MA10":1.0,"xG_P90":2.5,"xA_P90":0.8,"Pass_Accuracy":0.4,
            "Defensive_Actions_P90":0.0,"distanceRun":0.6,"Possession_Security":0.7,
            "Dribbles_P90":1.2,"Key_Passes_P90":0.5,"Fatigue_Index":1.2,
            "Medical_Risk_Score":1.5,"Age":0.5,"Rating_Trend":1.5,"Rating_STD5":0.8,"Minutes_Played":0.5},
    "MOF": {"Rating_MA10":1.0,"xG_P90":0.8,"xA_P90":2.0,"Pass_Accuracy":1.6,
            "Defensive_Actions_P90":0.3,"distanceRun":1.0,"Possession_Security":1.4,
            "Dribbles_P90":1.5,"Key_Passes_P90":2.2,"Fatigue_Index":1.2,
            "Medical_Risk_Score":1.5,"Age":0.5,"Rating_Trend":1.3,"Rating_STD5":0.8,"Minutes_Played":0.6},
    "MC":  {"Rating_MA10":1.0,"xG_P90":0.5,"xA_P90":1.2,"Pass_Accuracy":1.8,
            "Defensive_Actions_P90":1.2,"distanceRun":1.8,"Possession_Security":1.5,
            "Dribbles_P90":0.8,"Key_Passes_P90":1.5,"Fatigue_Index":1.4,
            "Medical_Risk_Score":1.8,"Age":0.6,"Rating_Trend":1.2,"Rating_STD5":1.0,"Minutes_Played":0.7},
    "MDF": {"Rating_MA10":1.0,"xG_P90":0.1,"xA_P90":0.6,"Pass_Accuracy":1.8,
            "Defensive_Actions_P90":2.2,"distanceRun":1.8,"Possession_Security":1.6,
            "Dribbles_P90":0.3,"Key_Passes_P90":0.8,"Fatigue_Index":1.5,
            "Medical_Risk_Score":2.0,"Age":0.6,"Rating_Trend":1.2,"Rating_STD5":1.0,"Minutes_Played":0.7},
    "CB":  {"Rating_MA10":1.2,"xG_P90":0.1,"xA_P90":0.1,"Pass_Accuracy":1.6,
            "Defensive_Actions_P90":2.2,"distanceRun":0.8,"Possession_Security":1.4,
            "Dribbles_P90":0.2,"Key_Passes_P90":0.2,"Fatigue_Index":1.3,
            "Medical_Risk_Score":1.8,"Age":0.7,"Rating_Trend":1.0,"Rating_STD5":1.0,"Minutes_Played":0.8},
}
# AG/AD share with ATT-like, LB/RB share with CB-like
for _p in ["AG", "AD"]:
    _DEFAULT_WEIGHTS[_p] = dict(_DEFAULT_WEIGHTS["ATT"])
    _DEFAULT_WEIGHTS[_p]["Dribbles_P90"] = 2.5
    _DEFAULT_WEIGHTS[_p]["xA_P90"] = 1.8
for _p in ["LB", "RB"]:
    _DEFAULT_WEIGHTS[_p] = dict(_DEFAULT_WEIGHTS["CB"])
    _DEFAULT_WEIGHTS[_p]["distanceRun"] = 2.0
    _DEFAULT_WEIGHTS[_p]["Dribbles_P90"] = 1.2
    _DEFAULT_WEIGHTS[_p]["xA_P90"] = 1.4


# ══════════════════════════════════════════════════════════════════════
# AUTO-CLASSIFICATION DU POSTE (inférence par stats)
# ══════════════════════════════════════════════════════════════════════

def _infer_positions_bulk(df_full):
    """Infère le poste précis de chaque joueur (9 postes) via percentiles."""
    agg = df_full.groupby('Nom').agg(
        xG_P90=('xG_P90', 'median'),
        xA_P90=('xA_P90', 'median'),
        Key_Passes_P90=('Key_Passes_P90', 'median'),
        Dribbles_P90=('Dribbles_P90', 'median'),
        distanceRun=('distanceRun', 'median'),
        Defensive=('Defensive_Actions_P90', 'median') if 'Defensive_Actions_P90' in df_full.columns
                  else ('Interceptions', 'median'),
    ).reset_index()

    def pct_rank(col):
        if col in agg.columns and agg[col].std() > 0:
            return agg[col].rank(pct=True)
        return pd.Series(0.5, index=agg.index)

    xg  = pct_rank('xG_P90')
    xa  = pct_rank('xA_P90')
    df_ = pct_rank('Defensive')
    kp  = pct_rank('Key_Passes_P90')
    drb = pct_rank('Dribbles_P90')
    dst = pct_rank('distanceRun')

    pos = pd.Series('MC', index=agg.index)
    pos[(kp >= 0.55) & (xg < 0.65) & (df_ < 0.65)] = 'MOF'
    pos[(df_ >= 0.55) & (kp < 0.55) & (xg < 0.55)] = 'MDF'
    pos[((xg < 0.40) & (drb < 0.45) & (xa < 0.40)) |
        ((xg < 0.30) & (xa < 0.30) & (kp < 0.35))] = 'CB'
    pos[(xg < 0.45) & (dst >= 0.55) & (drb >= 0.35)] = 'LB'
    pos[(drb >= 0.55) & ((xg >= 0.50) | (xa >= 0.55))] = 'AG'
    pos[(xg >= 0.65) & (kp < 0.80)] = 'ATT'
    pos[(xg >= 0.50) & (xg < 0.70) & (kp >= 0.70)] = 'MOF'

    agg['Poste_Cat_Infere'] = pos.values
    return dict(zip(agg['Nom'], agg['Poste_Cat_Infere']))


# ══════════════════════════════════════════════════════════════════════
# MOTEUR DE SIMILARITÉ v3.0 (INDUSTRIALISÉ)
# ══════════════════════════════════════════════════════════════════════

class SimilarityEngine:
    """
    Moteur de scouting haute précision AthlytIQ v3.0.

    Améliorations v3 :
    1. Poids appris par ML (learned_weights.joblib) au lieu de codés en dur
    2. Clustering K-Means (archétypes de joueurs)
    3. Calcul 100% vectorisé (cdist + matrix ops)
    4. Multi-métrique : Cosinus pondéré + PCA
    5. Archétype dans la réponse API
    """

    def __init__(self, df_features):
        self.df = df_features.copy()
        self.features_list = FEATURES_ORTHOGONALES
        benchmark_dir = Path(__file__).resolve().parent / "benchmark"

        # ── Sécurisation des colonnes ──
        for f in self.features_list:
            if f in self.df.columns:
                self.df[f] = pd.to_numeric(self.df[f], errors='coerce').fillna(0)
            else:
                self.df[f] = 0.0

        # ── Construction du profil multi-match (EWM 15 matchs) ──
        self.df = self.df.sort_values('Match_Date')
        self.nb_matchs_par_joueur = self.df.groupby('Nom').size().to_dict()

        profils = []
        for nom, group in self.df.groupby('Nom'):
            last_n = group.tail(15)
            if len(last_n) >= 2:
                profil = last_n[self.features_list].ewm(span=5, min_periods=1).mean().iloc[-1]
            else:
                profil = last_n[self.features_list].iloc[-1]
            meta = last_n.iloc[-1]
            profil['Nom'] = nom
            for col in ['Team', 'Equipe', 'Poste_Cat', 'League', 'Match_Num', 'Match_Date']:
                if col in meta.index:
                    profil[col] = meta[col]
            profils.append(profil)

        self.df_raw = pd.DataFrame(profils)

        # ── Résolution des postes (fichier > inférence) ──
        positions_file = Path(__file__).resolve().parent.parent / "data" / "player_positions.csv"
        exact_pos_map = {}
        if positions_file.exists():
            pos_df = pd.read_csv(positions_file)
            if 'Nom' in pos_df.columns and 'Poste_Cat' in pos_df.columns:
                exact_pos_map = dict(zip(pos_df['Nom'], pos_df['Poste_Cat']))
                n_exact = self.df_raw['Nom'].map(exact_pos_map).notna().sum()
                print(f"[SimilarityEngine] ✅ player_positions.csv : {n_exact}/{len(self.df_raw)} joueurs couverts")

        inferred_pos_map = _infer_positions_bulk(self.df)
        self.df_raw['Poste_Cat'] = self.df_raw['Nom'].map(
            lambda n: exact_pos_map.get(n) or inferred_pos_map.get(n, 'MOF'))

        # ── Noms normalisés ──
        self.df_raw['Name_Norm'] = self.df_raw['Nom'].apply(normalize_name)

        # ── Chargement des poids appris (fallback: poids par défaut) ──
        weights_path = benchmark_dir / "learned_weights.joblib"
        if weights_path.exists():
            self.position_weights = joblib.load(weights_path)
            print("[SimilarityEngine] ✅ Poids appris chargés (learned_weights.joblib)")
        else:
            self.position_weights = _DEFAULT_WEIGHTS
            print("[SimilarityEngine] ⚠️ Poids par défaut (learned_weights.joblib absent)")

        # ── Chargement des clusters (optionnel) ──
        self.cluster_labels = None
        self.archetype_map = {}
        clusters_path = benchmark_dir / "scouting_profiles.joblib"
        if clusters_path.exists():
            try:
                cluster_profiles = joblib.load(clusters_path)
                if 'Cluster_ID' in cluster_profiles.columns and 'Archetype' in cluster_profiles.columns:
                    self.archetype_map = dict(zip(cluster_profiles['Nom'], cluster_profiles['Archetype']))
                    self.cluster_labels = dict(zip(cluster_profiles['Nom'], cluster_profiles['Cluster_ID']))
                    print(f"[SimilarityEngine] ✅ Archétypes chargés ({len(self.archetype_map)} joueurs)")
            except Exception:
                pass

        # ── Normalisation (StandardScaler) ──
        self.df_norm = self.df_raw.copy()
        self.scaler = StandardScaler()
        if len(self.df_norm) > 1:
            self.df_norm[self.features_list] = self.scaler.fit_transform(
                self.df_norm[self.features_list])

        # ── PCA ──
        self.pca = None
        self.X_pca = None
        if len(self.df_norm) > 10:
            n_comp = min(10, len(self.features_list), len(self.df_norm) - 1)
            self.pca = PCA(n_components=n_comp, random_state=42)
            X_scaled = self.df_norm[self.features_list].values.astype(float)
            self.X_pca = self.pca.fit_transform(X_scaled)

        # ── Pré-calcul des matrices vectorisées ──
        self._X_norm = self.df_norm[self.features_list].values.astype(float)
        self._cos_sim_matrix = cosine_similarity(self._X_norm)
        if self.X_pca is not None:
            self._pca_sim_matrix = cosine_similarity(self.X_pca)
        else:
            self._pca_sim_matrix = np.full((len(self.df_norm), len(self.df_norm)), 0.5)

    def get_similar_players(self, target_player_name, alpha=0.5, top_n=10):
        """
        Trouve les joueurs les plus similaires au joueur cible.
        Interface 100% rétrocompatible avec v2.
        """
        # ── Recherche du joueur cible ──
        target_norm = normalize_name(target_player_name)
        mask = self.df_raw['Name_Norm'] == target_norm
        if not mask.any():
            mask = self.df_raw['Name_Norm'].str.contains(target_norm, regex=False)
        if not mask.any():
            return {"error": "PLAYER_NOT_FOUND"}

        idx_a = self.df_raw[mask].index[0]
        pos_a = self.df_raw.index.get_loc(idx_a)
        row_a_raw = self.df_raw.loc[idx_a]
        name_a = row_a_raw['Nom']
        target_team = row_a_raw.get('Team', row_a_raw.get('Equipe', 'Inconnu'))
        target_pos = POSTE_MAP.get(row_a_raw.get('Poste_Cat', 'M'), 'MOF')

        # ── Poids pour ce poste ──
        w_dict = self.position_weights.get(target_pos, self.position_weights.get('MC', {}))
        weights = np.array([w_dict.get(f, 1.0) for f in self.features_list])

        # ── DNA de l'équipe cible ──
        team_col = 'Team' if 'Team' in self.df_raw.columns else 'Equipe'
        if team_col in self.df_raw.columns:
            team_mask = self.df_raw[team_col] == target_team
            team_players = self.df_raw[team_mask]
        else:
            team_players = pd.DataFrame()
        team_dna = team_players[self.features_list].mean() if len(team_players) > 3 \
            else self.df_raw[self.features_list].mean()

        # ── Calcul vectorisé de la similarité pondérée ──
        vec_a = self._X_norm[pos_a].reshape(1, -1) * weights
        X_weighted = self._X_norm * weights
        cos_scores = cosine_similarity(vec_a, X_weighted)[0]
        cos_scores = np.clip(cos_scores, 0, 1)

        # PCA scores
        pca_scores = self._pca_sim_matrix[pos_a]
        pca_scores = np.clip(pca_scores, 0, 1)

        # DNA fit vectorisé
        critical_feats_idx = [self.features_list.index(f) for f in
                              ["distanceRun", "Pass_Accuracy", "Possession_Security", "Defensive_Actions_P90"]
                              if f in self.features_list]
        team_dna_vals = np.array([team_dna[self.features_list[i]] for i in critical_feats_idx])
        player_vals = self.df_raw[self.features_list].values[:, critical_feats_idx]
        ratios = np.minimum(1.2, player_vals / (team_dna_vals + 1e-6))
        dna_scores = ratios.mean(axis=1)

        # Medical penalty
        med_col_idx = self.features_list.index("Medical_Risk_Score") if "Medical_Risk_Score" in self.features_list else -1
        if med_col_idx >= 0:
            team_med = float(team_dna.get("Medical_Risk_Score", 0))
            player_med = self.df_raw[self.features_list].values[:, med_col_idx]
            med_penalty = np.where(player_med > team_med * 1.5, 0.8, 1.0)
            dna_scores *= med_penalty

        # ── Score combiné ──
        base_scores = cos_scores * 0.45 + pca_scores * 0.25 + dna_scores * 0.30

        # ── Filtres multiplicatifs ──
        postes = self.df_raw['Poste_Cat'].apply(lambda p: POSTE_MAP.get(p, p)).values
        fit_factors = np.array([POSITION_FIT.get(p, {}).get(target_pos, 0.1) for p in postes])

        trends = pd.to_numeric(self.df_raw.get('Rating_Trend', 0), errors='coerce').fillna(0).values
        momentum = np.where(trends < -1.0, 0.80, np.where(trends < -0.3, 0.90, np.where(trends > 1.0, 1.10, 1.0)))

        nb_matchs_arr = np.array([self.nb_matchs_par_joueur.get(n, 1) for n in self.df_raw['Nom']])
        confidence = np.minimum(1.0, nb_matchs_arr / 12.0)

        final_scores = (base_scores * fit_factors * momentum * confidence * 100).astype(int)
        final_scores[pos_a] = -1  # Exclude self
        final_scores = np.clip(final_scores, 0, 99)

        # ── Top N ──
        top_indices = np.argsort(final_scores)[::-1][:top_n * 2]
        top_indices = [i for i in top_indices if final_scores[i] > 10][:top_n]

        # ── Construire les résultats ──
        results = []
        for pos_b in top_indices:
            idx_b = self.df_raw.index[pos_b]
            row_b_raw = self.df_raw.loc[idx_b]
            name_b = row_b_raw['Nom']
            cand_pos = POSTE_MAP.get(row_b_raw.get('Poste_Cat', 'M'), 'MOF')

            explanation, full_reasons = self._generate_explanation(
                row_a_raw, row_b_raw, dna_scores[pos_b], momentum[pos_b],
                int(nb_matchs_arr[pos_b]))

            # Radar stats (Percentiles)
            radar_b, radar_a = {}, {}
            for f in self.features_list:
                col_std = self.df_raw[f].std()
                if col_std == 0:
                    rb, ra = 50.0, 50.0
                else:
                    rb = (self.df_raw[f] < float(row_b_raw[f])).mean() * 100
                    ra = (self.df_raw[f] < float(row_a_raw[f])).mean() * 100
                radar_b[f] = round(max(5, rb), 1)
                radar_a[f] = round(max(5, ra), 1)

            results.append({
                "name": name_b,
                "team": row_b_raw.get('Team', row_b_raw.get('Equipe', 'Inconnu')),
                "target_name": name_a,
                "target_team": target_team,
                "native_pos": cand_pos,
                "archetype": self.archetype_map.get(name_b, "Non classifié"),
                "final_score": int(final_scores[pos_b]),
                "dna_fit_score": int(dna_scores[pos_b] * 100),
                "explanation": explanation,
                "full_reasons": full_reasons,
                "medical_risk": "High" if float(row_b_raw.get('Medical_Risk_Score', 0)) > 0.6 else "Low",
                "confidence": int(confidence[pos_b] * 100),
                "momentum": round(float(momentum[pos_b]), 2),
                "versatility": {
                    p: int(cos_scores[pos_b] * POSITION_FIT.get(cand_pos, {}).get(p, 0.1) * 100)
                    for p in ["ATT", "MOF", "MDF", "DEF"]
                },
                "raw_stats": {f: round(float(row_b_raw[f]), 2) for f in self.features_list},
                "target_stats": {f: round(float(row_a_raw[f]), 2) for f in self.features_list},
                "radar_stats": radar_b,
                "target_radar_stats": radar_a,
            })

        return sorted(results, key=lambda x: x['final_score'], reverse=True)

    def _generate_explanation(self, p_a, p_b, dna_fit, momentum, nb_matchs):
        """Génère une explication textuelle du score de similarité."""
        reasons = []

        if float(p_b.get('Dribbles_P90', 0)) >= float(p_a.get('Dribbles_P90', 0)) * 0.85:
            reasons.append("Profil de percussion identique.")
        if dna_fit > 0.95:
            reasons.append(f"ADÉQUATION TACTIQUE ÉLEVÉE : Son volume de jeu "
                           f"({int(float(p_b.get('distanceRun', 0)))}m) correspond au DNA de l'équipe.")
        else:
            reasons.append("Besoin d'adaptation physique pour coller au rythme de l'équipe.")
        if float(p_b.get('Medical_Risk_Score', 1)) < 0.3:
            reasons.append("Fiabilité athlétique supérieure, capable d'enchaîner les matchs.")
        if momentum >= 1.10:
            reasons.append("Joueur en pleine progression récente (+10%).")
        elif momentum <= 0.80:
            reasons.append("⚠️ Joueur en nette baisse de forme récente (-20%).")
        if nb_matchs < 8:
            reasons.append(f"Profil basé sur {nb_matchs} matchs seulement — confiance limitée.")

        archetype = self.archetype_map.get(p_b['Nom'], '')
        if archetype:
            reasons.append(f"Archétype : {archetype}.")

        std_b = float(p_b.get('Rating_STD5', 0))
        std_a = float(p_a.get('Rating_STD5', 0))
        if std_b < std_a * 0.7:
            reasons.append("Plus régulier et prévisible dans ses performances.")

        return " " + " ".join(reasons[:4]), reasons
