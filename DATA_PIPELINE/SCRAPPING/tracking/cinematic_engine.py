import os
import json
import math
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("CinematicEngine")

class AthlytIQCinematicEngine:
    """
    Moteur d'analyse cinématique haute performance.
    Calcule la distance et les sprints à partir de données de tracking X,Y (ex: SkillCorner).
    """

    FRAME_RATE = 10 # frames per second
    DT = 1.0 / FRAME_RATE
    SPRINT_THRESHOLD_MS = 7.0 # 7 m/s = 25.2 km/h

    def __init__(self):
        pass

    def process_match_folder(self, match_folder: Path) -> Dict[str, List[Dict[str, Any]]]:
        """
        Traite un dossier de match contenant tracking.json et metadata.json.
        Retourne un dictionnaire {nom_joueur: [stats_match]}.
        """
        match_id = match_folder.name
        tracking_files = list(match_folder.glob("*tracking.json"))
        meta_files = list(match_folder.glob("*metadata.json"))

        if not tracking_files or not meta_files:
            logger.warning(f"Fichiers manquants dans {match_id}")
            return {}

        try:
            # 1. Mapping Joueurs
            with open(meta_files[0], 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            player_mapping = {}
            for team in ['home_team', 'away_team']:
                for p in meta.get(team, {}).get('players', []):
                    pid = p.get('trackable_object')
                    if pid:
                        full_name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                        player_mapping[pid] = full_name

            # 2. Analyse Tracking
            with open(tracking_files[0], 'r', encoding='utf-8') as f:
                tracking = json.load(f)

            stats = {pid: {'distance': 0.0, 'sprints': 0, 'is_sprinting': False} for pid in player_mapping}
            prev_positions = {}

            for frame in tracking:
                for obj in frame.get('data', []):
                    pid = obj.get('trackable_object')
                    if pid not in player_mapping: continue

                    x, y = obj.get('x'), obj.get('y')
                    if x is None or y is None: continue

                    if pid in prev_positions:
                        x_prev, y_prev = prev_positions[pid]
                        dist = math.hypot(x - x_prev, y - y_prev)
                        stats[pid]['distance'] += dist

                        speed = dist / self.DT
                        if speed >= self.SPRINT_THRESHOLD_MS:
                            if not stats[pid]['is_sprinting']:
                                stats[pid]['sprints'] += 1
                                stats[pid]['is_sprinting'] = True
                        else:
                            stats[pid]['is_sprinting'] = False

                    prev_positions[pid] = (x, y)

            # 3. Formatage résultats
            results = {}
            for pid, s in stats.items():
                if s['distance'] > 0:
                    name = player_mapping[pid]
                    results[name] = [{
                        'match_id': match_id,
                        'distanceRun': round(s['distance'], 2),
                        'sprints': s['sprints']
                    }]
            
            return results

        except Exception as e:
            logger.error(f"Erreur sur le match {match_id}: {e}")
            return {}

    def run_full_analysis(self, input_dir: Path, output_dir: Path):
        """Parcourt tout un dossier de matchs et génère les CSV par joueur."""
        if not input_dir.exists():
            logger.error(f"Dossier source inexistant : {input_dir}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        global_data = {}

        folders = [d for d in input_dir.iterdir() if d.is_dir()]
        logger.info(f"🚀 Analyse cinématique lancée sur {len(folders)} matchs.")

        for folder in folders:
            match_results = self.process_match_folder(folder)
            for name, data in match_results.items():
                if name not in global_data:
                    global_data[name] = []
                global_data[name].extend(data)

        # Export CSV
        for name, stats in global_data.items():
            clean_name = name.replace(' ', '_').replace('/', '').replace('\\', '')
            df = pd.DataFrame(stats)
            df.to_csv(output_dir / f"run_{clean_name}.csv", index=False, encoding='utf-8-sig')

        logger.info(f"✅ Analyse terminée. {len(global_data)} fichiers générés dans {output_dir}")
