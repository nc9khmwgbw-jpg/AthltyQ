import os
import json
import math
import pandas as pd
from pathlib import Path

def calculer_physique_depuis_tracking(dossier_matches, dossier_sortie):
    """
    Parcourt les dossiers de matchs SkillCorner, lit les coordonnées X,Y brutes,
    et calcule mathématiquement la distance parcourue et les sprints par joueur.
    """
    chemin_matches = Path(dossier_matches)
    dossiers_match = [d for d in chemin_matches.iterdir() if d.is_dir()]
    
    print(f"📂 {len(dossiers_match)} dossiers de matchs trouvés pour l'analyse cinématique.")
    if not dossiers_match:
        return

    os.makedirs(dossier_sortie, exist_ok=True)
    
    # Dictionnaire global : { "Nom Joueur": [{"match_id": "123", "distanceRun": 10500, "sprints": 15}] }
    donnees_physiques_joueurs = {}

    # Constantes physiques
    FRAME_RATE = 10 # 10 frames par seconde (standard SkillCorner)
    DT = 1.0 / FRAME_RATE # Delta de temps entre deux frames (0.1s)
    SEUIL_SPRINT_MS = 7.0 # 7 mètres par seconde = 25.2 km/h

    for dossier in dossiers_match:
        match_id = dossier.name
        print(f"   ⏳ Traitement du match {match_id} (calcul frame-par-frame)...")
        
        fichier_tracking = list(dossier.glob("*tracking.json"))
        fichier_meta = list(dossier.glob("*metadata.json"))
        
        if not fichier_tracking or not fichier_meta:
            print(f"      ⚠️ Fichiers tracking/meta manquants pour le match {match_id}")
            continue
            
        try:
            # 1. Mapping des IDs avec les noms des joueurs
            with open(fichier_meta[0], 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            joueurs_mapping = {}
            for team in ['home_team', 'away_team']:
                for player in meta_data.get(team, {}).get('players', []):
                    joueurs_mapping[player['trackable_object']] = player.get('first_name', '') + " " + player.get('last_name', '')

            # 2. Lecture du tracking et calcul
            with open(fichier_tracking[0], 'r', encoding='utf-8') as f:
                tracking_data = json.load(f)
                
            stats_match = {pid: {'distance': 0.0, 'sprints': 0, 'is_sprinting': False} for pid in joueurs_mapping.keys()}
            positions_precedentes = {}

            # Analyse cinématique frame par frame
            for frame_data in tracking_data:
                for objet in frame_data.get('data', []):
                    pid = objet.get('trackable_object')
                    if pid not in joueurs_mapping:
                        continue
                        
                    x, y = objet.get('x'), objet.get('y')
                    
                    if x is None or y is None:
                        continue

                    # Si on a déjà une position précédente pour ce joueur
                    if pid in positions_precedentes:
                        x_prev, y_prev = positions_precedentes[pid]
                        
                        # Calcul de la distance euclidienne
                        dx = x - x_prev
                        dy = y - y_prev
                        dist = math.hypot(dx, dy)
                        
                        stats_match[pid]['distance'] += dist
                        
                        # Calcul de la vitesse (V = d / t)
                        vitesse = dist / DT
                        
                        # Détection des sprints
                        if vitesse >= SEUIL_SPRINT_MS:
                            if not stats_match[pid]['is_sprinting']:
                                stats_match[pid]['sprints'] += 1
                                stats_match[pid]['is_sprinting'] = True
                        else:
                            stats_match[pid]['is_sprinting'] = False
                            
                    # Mise à jour de la position
                    positions_precedentes[pid] = (x, y)

            # 3. Sauvegarde des résultats du match dans le dictionnaire global
            for pid, stats in stats_match.items():
                if stats['distance'] > 0:
                    nom_joueur = joueurs_mapping[pid].strip()
                    if nom_joueur not in donnees_physiques_joueurs:
                        donnees_physiques_joueurs[nom_joueur] = []
                        
                    donnees_physiques_joueurs[nom_joueur].append({
                        'match_id': match_id,
                        'distanceRun': round(stats['distance'], 2), # En mètres
                        'sprints': stats['sprints']
                    })

        except Exception as e:
            print(f"      ❌ Erreur sur le match {match_id}: {e}")

    # Génération des CSV finaux par joueur
    print(f"\n🧬 Calculs terminés. Génération des CSV pour {len(donnees_physiques_joueurs)} joueurs...")
    fichiers_crees = 0
    for nom, stats in donnees_physiques_joueurs.items():
        if not nom: continue
        
        df_joueur = pd.DataFrame(stats)
        nom_propre = str(nom).replace(' ', '_').replace('/', '').replace('\\', '')
        nom_fichier = os.path.join(dossier_sortie, f"run_{nom_propre}.csv")
        
        df_joueur.to_csv(nom_fichier, index=False, encoding='utf-8-sig')
        fichiers_crees += 1
        
    print(f"✅ Opération réussie ! {fichiers_crees} fichiers 'run_[Nom].csv' générés dans '{dossier_sortie}'.")


def calculer_distance_et_sprints(df):
    """
    Estime la distance et les sprints à partir des stats techniques (sans GPS).
    Utilisé après une réparation FotMob quand les données SkillCorner ne sont pas disponibles.
    """
    import numpy as np

    if df.empty:
        return df

    minutes   = pd.to_numeric(df.get('Minutes_Played', 0), errors='coerce').fillna(0)
    touches   = pd.to_numeric(df.get('Touches', 0), errors='coerce').fillna(0)
    recov     = pd.to_numeric(df.get('Ball_Recovery', 0), errors='coerce').fillna(0)
    dribbles  = pd.to_numeric(df.get('Successful_Dribbles', 0), errors='coerce').fillna(0)
    intercept = pd.to_numeric(df.get('Interceptions', 0), errors='coerce').fillna(0)

    # Distance en mètres : base 105m/min + bonus activités
    df['distanceRun'] = ((minutes * 105) + (touches * 4.2) + (recov * 12.5)).round(0)

    # Sprints : base cadence + intensité
    df['sprints'] = ((minutes / 5.2) + (dribbles * 2.8) + (intercept * 1.4)).round(0)

    # KPI work rate
    safe_min = minutes.replace(0, np.nan)
    df['kpi_work_rate'] = (df['distanceRun'] / safe_min).round(2)

    # KPI explosivité
    df['kpi_explosivity'] = ((df['sprints'] / safe_min) * 10).round(2)

    return df


if __name__ == "__main__":

    # Chemin vers tes dossiers bruts de matchs SkillCorner (à adapter si besoin)
    DOSSIER_MATCHES = "opendata/data/matches" 
    
    # Destination des fichiers finaux
    DOSSIER_SORTIE = "DATA_PIPELINE/SCRAPPING/raw/skillcorner_physique_calcule"
    
    calculer_physique_depuis_tracking(DOSSIER_MATCHES, DOSSIER_SORTIE)