import os
import sys
import pandas as pd
import glob
import requests
import urllib.parse
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPTS_DIR))

try:
    from fotmob_match_scraper import repair_player_with_fotmob
except ImportError as e:
    print(f"❌ Erreur module fotmob_match_scraper : {e}")
    sys.exit(1)

RAW_DIR = Path(__file__).resolve().parents[2] / "SCRAPPING" / "raw" / "sofascore"

# Toutes les stats à vérifier une par une
STATS_TO_CHECK = {
    'Goals':               '⚽ Buts',
    'Assists':             '🎯 Passes Décisives',
    'Minutes_Played':      '⏱️  Minutes',
    'Rating':              '⭐ Note',
    'Touches':             '🔘 Touches',
    'Total_Passes':        '📬 Passes Totales',
    'Accurate_Passes':     '✅ Passes Précises',
    'Key_Passes':          '🔑 Passes Clés',
    'Tackles':             '💪 Tacles',
    'Interceptions':       '🛡️  Interceptions',
    'Clearances':          '🧹 Dégagements',
    'Ball_Recovery':       '🔄 Récupérations',
    'Successful_Dribbles': '🌀 Dribbles Réussis',
    'Expected_Goals':      '📊 xG',
    'Expected_Assists':    '📊 xA',
}

# Anomalies locales que FotMob ne peut pas forcément corriger
LOCAL_CHECKS = [
    ('Minutes_Played', 'Goals', "But marqué avec 0 minute jouée"),
    ('Minutes_Played', 'Assists', "Passe décisive avec 0 minute jouée"),
]

def check_local_anomalies(df_local: pd.DataFrame) -> tuple[bool, str]:
    """
    Scanne le dataframe local pour détecter des impossibilités physiques ou mathématiques.
    """
    for _, row in df_local.iterrows():
        try:
            mins = float(row.get('Minutes_Played', 0))
            goals = float(row.get('Goals', 0))
            assists = float(row.get('Assists', 0))
            
            # RÈGLE 1 : Actions décisives sans jouer
            if mins == 0 and (goals > 0 or assists > 0):
                return True, "Action décisive avec 0 minute jouée"
                
            # RÈGLE 2 : Passes réussies supérieures aux passes totales
            if 'Accurate_Passes' in row and 'Total_Passes' in row:
                acc_passes = float(row['Accurate_Passes']) if pd.notnull(row['Accurate_Passes']) else 0
                tot_passes = float(row['Total_Passes']) if pd.notnull(row['Total_Passes']) else 0
                if acc_passes > tot_passes:
                    return True, f"Passes réussies ({acc_passes}) > Total ({tot_passes})"
                    
            # RÈGLE 3 : Plus de buts que de tirs (Shots)
            if 'Shots' in row:
                shots = float(row['Shots']) if pd.notnull(row['Shots']) else 0
                if goals > shots and shots > 0: # S'il y a des tirs comptabilisés
                    return True, f"Plus de buts ({goals}) que de tirs ({shots})"

        except Exception:
            continue
            
    return False, ""

def repair_local_anomaly(file_path, df, player_name):
    """
    Corrige directement les impossibilités physiques sans FotMob.
    Règle : si Minutes_Played == 0 → Goals = 0, Assists = 0 (impossible physiquement).
    """
    df_fixed = df.copy()
    changes = []

    for col in ['Goals', 'Assists']:
        if col not in df_fixed.columns:
            continue
        mask = (df_fixed['Minutes_Played'] == 0) & (df_fixed[col] > 0)
        if mask.any():
            rows = df_fixed[mask]
            for _, row in rows.iterrows():
                val = row[col]
                label = '⚽ Buts' if col == 'Goals' else '🎯 Assists'
                changes.append(f"   {label} {row['Match_Date']} : {val} ➡️  0 (0 min jouée)")
            df_fixed.loc[mask, col] = 0

    if changes:
        df_fixed.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ Corrigé directement (impossibilité physique) :")
        for c in changes:
            print(c)
        return True
    return False


def check_mismatch_fotmob(player_name, df_local):
    """
    Vérification rapide via l'API publique FotMob (non-bloquant).
    Retourne (True, raison) si anomalie confirmée,
    (False, 'OK') si tout va bien OU si l'API est inaccessible.
    NOTE : L'API FotMob publique est souvent bloquée (403/429).
    La réparation réelle passe par Playwright (repair_player_with_fotmob).
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Referer": "https://www.fotmob.com/",
        }
        encoded = urllib.parse.quote(player_name)
        search_url = f"https://www.fotmob.com/api/search?term={encoded}"

        session = requests.Session()
        res = session.get(search_url, headers=headers, timeout=(4, 8))

        # API bloquée ou indisponible → pas une anomalie
        if res.status_code in (403, 429, 503, 502, 404):
            return False, "API Bloquée"
        if res.status_code != 200:
            return False, "OK"

        data = res.json()
        player_id = None
        fm_total_goals = 0

        # Chercher dans playerSuggestions ou hits.players
        candidates = (
            data.get('playerSuggestions', [])
            or data.get('hits', {}).get('players', [])
        )
        for p in candidates:
            p_name = p.get('name', p.get('playerName', ''))
            if player_name.lower() in p_name.lower():
                player_id = p.get('id', p.get('playerId'))
                stats_list = p.get('stats', [])
                fm_total_goals = int(stats_list[0]) if stats_list else 0
                break

        if not player_id:
            return False, "OK"  # Non trouvé → on ne signale pas comme anomalie

        # Vérification totale de buts
        local_total = int(df_local['Goals'].sum()) if 'Goals' in df_local.columns else 0
        if local_total != fm_total_goals:
            return True, f"Total buts incorrect ({local_total} vs {fm_total_goals})"

        # Vérification distribution match par match
        player_url = f"https://www.fotmob.com/api/playerData?id={player_id}"
        res2 = session.get(player_url, headers=headers, timeout=(4, 8))
        if res2.status_code == 200:
            for m in res2.json().get('lastMatches', [])[:15]:
                date_raw = m.get('matchDate', {}).get('utcTime', '')
                date = date_raw.split('T')[0] if 'T' in date_raw else date_raw
                fm_goals = int(m.get('goals', 0))
                local_row = df_local[df_local['Match_Date'].astype(str) == date]
                if not local_row.empty:
                    if int(local_row.iloc[0].get('Goals', 0)) != fm_goals:
                        return True, f"Erreur distribution le {date}"

        return False, "OK"
    except Exception:
        return False, "OK"  # Toute erreur réseau → ignorer silencieusement


def run_quality_control():
    print("\n" + "="*65)
    print(" 🔬  ATHLYTIQ — VÉRIFICATION ABSOLUE (LA VÉRITÉ FOTMOB)")
    print("="*65)

    all_files = glob.glob(str(RAW_DIR / "**/*.csv"), recursive=True)
    total_corrections = 0

    for i, file_path in enumerate(all_files, 1):
        p_path = Path(file_path)
        player_name = p_path.stem.replace("_", " ")
        team_name = p_path.parent.name.replace("_", " ")

        try:
            df_before = pd.read_csv(file_path)
            if 'Match_Date' not in df_before.columns or 'Minutes_Played' not in df_before.columns:
                continue

            print(f"[{i:03d}/{len(all_files)}] 🕵️‍♂️ Vérification de la réalité pour : {player_name}...")

            # 💥 ON FORCE LA VÉRIFICATION SUR FOTMOB POUR TOUT LE MONDE
            # On ne fait plus de supposition locale. On demande directement au God Mode.
            if repair_player_with_fotmob(player_name, team_name):
                print(f"\n{'='*65}")
                print(f"🚨 MENSONGE DÉTECTÉ ET CORRIGÉ : {player_name}")
                print(f"{'='*65}")

                df_after = pd.read_csv(file_path)
                corrections_joueur = 0

                for date in df_after['Match_Date'].unique():
                    r_a = df_after[df_after['Match_Date'] == date].iloc[0]
                    r_b_list = df_before[df_before['Match_Date'] == str(date)]

                    if r_b_list.empty:
                        print(f"   🆕 {date} | Nouveau match ajouté par FotMob")
                        corrections_joueur += 1
                        continue

                    r_b = r_b_list.iloc[0]
                    match_diffs = []

                    for col, label in STATS_TO_CHECK.items():
                        if col not in r_a.index: continue
                        try:
                            v_b = float(r_b.get(col, 0))
                            v_a = float(r_a.get(col, 0))
                            if abs(v_b - v_a) > 0.01:
                                match_diffs.append(f"      {label:<30} {v_b} ➡️  {v_a}")
                        except: pass

                    if match_diffs:
                        print(f"\n   📝 {date} | {r_a.get('Home_Team','')} vs {r_a.get('Away_Team','')}")
                        for d in match_diffs: print(d)
                        corrections_joueur += len(match_diffs)
                
                total_corrections += corrections_joueur
                print(f"   ✅ {corrections_joueur} stat(s) corrigée(s).\n")
            
            else:
                # FotMob a vérifié les données, et tes données locales étaient exactement les mêmes !
                print(f"      ✅ Les données de Sofascore disaient la vérité.")

            # Pause vitale de 2 secondes pour ne pas faire exploser le serveur FotMob
            # et éviter un bannissement d'IP
            time.sleep(2)

        except Exception as e:
            print(f"      ⚠️ Erreur lors du traitement de {player_name} : {e}")
            continue

    print(f"\n{'='*65}")
    print(f"✅ ANALYSE TERMINÉE — {total_corrections} correction(s) réelles au total.")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    run_quality_control()
