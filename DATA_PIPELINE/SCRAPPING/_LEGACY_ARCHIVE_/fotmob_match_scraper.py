"""
AthlytIQ — Moteur de Réparation via FotMob (Playwright)
=========================================================
Utilise Playwright avec interception réseau pour contourner la
protection anti-bot de FotMob, capturer les appels API internes,
extraire les stats match par match et corriger les données locales.
"""

import json
import time
import re
import sys
import urllib.parse
import unicodedata
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPTS_DIR))
from calcul_tracking_brut import calculer_distance_et_sprints

ROOT = Path(__file__).resolve().parents[2]
SOFASCORE_RAW = ROOT / "SCRAPPING" / "raw" / "sofascore"

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    return name.lower().replace("-", " ").strip()

def _build_browser_context(playwright):
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--window-size=1920,1080",
        ],
    )
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-GB",
    )
    context.add_init_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
    )
    return browser, context

# ─────────────────────────────────────────────────────────────
#  ÉTAPE 1 : Chercher le joueur et capturer l'ID via réseau
# ─────────────────────────────────────────────────────────────

def _search_player_fotmob(page, player_name: str) -> dict | None:
    page.add_init_script("""
        window.__STEAL_DATA__ = null;
        const originalParse = JSON.parse;
        JSON.parse = function(text, reviver) {
            const result = originalParse.call(this, text, reviver);
            try {
                if (result && typeof result === 'object') {
                    if (result.recentMatches || result.lastMatches) {
                        window.__STEAL_DATA__ = result;
                    } else if (result.props && result.props.pageProps && result.props.pageProps.fallback) {
                        const fallback = result.props.pageProps.fallback;
                        for (let key in fallback) {
                            if (fallback[key] && (fallback[key].recentMatches || fallback[key].lastMatches)) {
                                window.__STEAL_DATA__ = fallback[key];
                            }
                        }
                    }
                }
            } catch(e) {}
            return result;
        };
        const originalFetch = window.fetch;
        window.fetch = async function(...args) {
            const response = await originalFetch.apply(this, args);
            try {
                const clone = response.clone();
                clone.json().then(data => {
                    if (data && typeof data === 'object' && (data.recentMatches || data.lastMatches)) {
                        window.__STEAL_DATA__ = data;
                    }
                }).catch(() => {});
            } catch(e) {}
            return response;
        };
    """)

    captured = {}
    
    def intercept_api(response):
        try:
            if "application/json" in response.headers.get("content-type", ""):
                data = response.json()
                if "recentMatches" in data or "lastMatches" in data:
                    captured["playerData"] = data
                elif "pageProps" in data:
                    fallback = data.get("pageProps", {}).get("fallback", {})
                    for k, v in fallback.items():
                        if "playerData" in k:
                            captured["playerData"] = v
        except Exception:
            pass

    page.on("response", intercept_api)
    
    try:
        page.goto("https://www.fotmob.com/", wait_until="domcontentloaded", timeout=15000)
        time.sleep(2)
        
        cookie_btn = page.locator("button").filter(has_text=re.compile(r"(Accept|Agree|OK|Got it)", re.IGNORECASE)).first
        if cookie_btn.is_visible():
            cookie_btn.click(force=True)
            time.sleep(1)

        search_icon = page.locator("button[aria-label*='Search' i], a[href*='/search']").first
        if search_icon.is_visible():
            search_icon.click(force=True)
        else:
            page.keyboard.press("/") 
        time.sleep(1)
        
        search_input = page.locator("input[type='search'], input[placeholder*='Search' i]").first
        search_input.click(force=True)
        search_input.press_sequentially(player_name, delay=150)
        
        first_name = player_name.split()[0]
        
        # 🧹 NOUVEAU : Nettoyage des accents pour éviter le crash sur des noms comme "Théo"
        import unicodedata
        safe_first_name = unicodedata.normalize('NFKD', first_name).encode('ascii', 'ignore').decode('utf-8')
        
        player_option = page.locator("a[href*='/players/']").filter(has_text=re.compile(safe_first_name, re.IGNORECASE)).first
        
        try:
            player_option.wait_for(state="visible", timeout=6000)
            player_option.click(force=True)
        except Exception:
            page.keyboard.press("Enter")
            time.sleep(3)
            result_link = page.locator("a[href*='/players/']").first
            if result_link.is_visible():
                result_link.click(force=True)

        page.wait_for_url("**/players/**", timeout=10000)
        time.sleep(2) 
        
        page.remove_listener("response", intercept_api)
        current_url = page.url
        match = re.search(r'/players/(\d+)', current_url)
        
        if match:
            player_id = match.group(1)
            return {"id": player_id, "name": player_name, "teamName": "Unknown", "_data": captured.get("playerData")}
            
    except Exception:
        pass 

    page.remove_listener("response", intercept_api)
    return None

def _fetch_player_data(page, player_id) -> dict | None:
    for _ in range(10):
        data = page.evaluate("() => window.__STEAL_DATA__")
        if data:
            return data
        time.sleep(0.5)
        
    try:
        page.goto(f"https://www.fotmob.com/players/{player_id}", wait_until="domcontentloaded", timeout=15000)
        time.sleep(3)
        data = page.evaluate("() => window.__STEAL_DATA__")
        if data:
            return data
    except Exception:
        pass
        
    return None

# ─────────────────────────────────────────────────────────────
#  ÉTAPE 3 : Parser les matchs depuis les données FotMob
# ─────────────────────────────────────────────────────────────

def _parse_fotmob_matches(player_data: dict) -> list[dict]:
    matches = []

    raw_matches = (
        player_data.get("recentMatches")
        or player_data.get("lastMatches")
        or player_data.get("stats", {}).get("matchStats", [])
        or []
    )

    for m in raw_matches[:15]:
        try:
            date_raw = (
                m.get("matchDate", {}).get("utcTime", "")
                or m.get("date", "")
                or m.get("matchTms", "")
            )
            date = date_raw.split("T")[0] if "T" in str(date_raw) else str(date_raw)[:10]

            # ── STATS CLASSIQUES ──
            goals   = int(m.get("goals", m.get("g", 0)) or 0)
            assists = int(m.get("assists", m.get("a", 0)) or 0)
            minutes = int(m.get("minutesPlayed", m.get("minsPlayed", m.get("mp", 0))) or 0)
            rating  = float(m.get("ratingProps", {}).get("num", m.get("rating", 0.0)) or 0.0)
            shots   = int(m.get("shots", m.get("totalShots", 0)) or 0)
            touches = int(m.get("touches", 0) or 0)
            
            # ── PASSES ──
            key_passes = int(m.get("keyPasses", m.get("keypasses", 0)) or 0)
            total_passes = int(m.get("totalPasses", m.get("passes", {}).get("total", 0)) or 0)
            acc_passes = int(m.get("accuratePasses", m.get("passes", {}).get("accurate", 0)) or 0)

            # ── NOUVELLES STATS (Défense & xG) ──
            xg = float(m.get("expectedGoals", m.get("xg", 0.0)) or 0.0)
            xa = float(m.get("expectedAssists", m.get("xa", 0.0)) or 0.0)
            tackles = int(m.get("tackles", m.get("tackle", 0)) or 0)
            interceptions = int(m.get("interceptions", 0) or 0)
            clearances = int(m.get("clearances", 0) or 0)
            recoveries = int(m.get("ballRecovery", m.get("recoveries", 0)) or 0)

            home_team = m.get("homeTeam", {}).get("name", "") if isinstance(m.get("homeTeam"), dict) else m.get("homeTeamName", "")
            away_team = m.get("awayTeam", {}).get("name", "") if isinstance(m.get("awayTeam"), dict) else m.get("awayTeamName", "")

            matches.append({
                "Match_Date":       date,
                "Home_Team":        home_team,
                "Away_Team":        away_team,
                "Goals":            goals,
                "Assists":          assists,
                "Minutes_Played":   minutes,
                "Rating":           rating,
                "Shots":            shots,
                "Touches":          touches,
                "Key_Passes":       key_passes,
                "Total_Passes":     total_passes,
                "Accurate_Passes":  acc_passes,
                "Expected_Goals":   xg,           # Ajouté
                "Expected_Assists": xa,           # Ajouté
                "Tackles":          tackles,      # Ajouté
                "Interceptions":    interceptions,# Ajouté
                "Clearances":       clearances,   # Ajouté
                "Ball_Recovery":    recoveries    # Ajouté
            })
        except Exception:
            continue

    return matches

# ─────────────────────────────────────────────────────────────
#  ÉTAPE 4 : Appliquer les corrections sur le CSV local
# ─────────────────────────────────────────────────────────────

def _apply_corrections(file_path: Path, df_local: pd.DataFrame, fm_matches: list[dict]) -> bool:
    if not fm_matches:
        return False

    # Standardisation de la date locale
    df_local["Match_Date_DT"] = pd.to_datetime(df_local["Match_Date"], errors='coerce')
    
    changed = False
    for fm_match in fm_matches:
        try:
            fm_date_dt = pd.to_datetime(fm_match["Match_Date"])
        except: continue

        # 1. FILTRE DE DATE (Tolérance 1 jour)
        mask = (df_local["Match_Date_DT"] >= fm_date_dt - pd.Timedelta(days=1)) & \
               (df_local["Match_Date_DT"] <= fm_date_dt + pd.Timedelta(days=1))
        
        if not mask.any(): continue

        # 2. 🛡️ LE VERROU PAR ÉQUIPE (Anti-80.0)
        # On ne traite le match que si le nom de l'équipe locale match avec FotMob
        possible_indices = df_local.index[mask]
        local_idx = None
        
        for idx in possible_indices:
            l_home = str(df_local.at[idx, "Home_Team"]).lower()
            l_away = str(df_local.at[idx, "Away_Team"]).lower()
            fm_home = str(fm_match.get("home_name", "")).lower()
            fm_away = str(fm_match.get("away_name", "")).lower()

            # Vérification : Est-ce qu'Auxerre ou l'adversaire est bien là ?
            if (l_home in fm_home or fm_home in l_home) or (l_away in fm_away or fm_away in l_away):
                local_idx = idx
                break

        if local_idx is None:
            # On ignore ce match car l'équipe ne correspond pas (ex: vieux match U19 ou Amical)
            continue

        # 3. CORRECTIONS CHIRURGICALES
        local_row = df_local.loc[local_idx]

        # Buts / Assists
        for col in ["Goals", "Assists"]:
            l_val = float(local_row.get(col, 0) or 0)
            fm_val = float(fm_match.get(col, 0) or 0)
            if abs(l_val - fm_val) > 0.01:
                df_local.at[local_idx, col] = fm_val
                print(f"      ⚽ {fm_match['Match_Date']} | {col}: {l_val} ➡️  {fm_val}")
                changed = True

        # Minutes (On répare le cas des 27 min)
        l_min = float(local_row.get("Minutes_Played", 0) or 0)
        fm_min = float(fm_match.get("Minutes_Played", 0) or 0)

        # Si SofaScore dit 90 (erreur classique) mais FotMob a le vrai temps (ex: 27)
        if l_min > 89.0 and 0 < fm_min < 90:
            df_local.at[local_idx, "Minutes_Played"] = fm_min
            print(f"      ⏱️  {fm_match['Match_Date']} | Minutes: {l_min} ➡️  {fm_min}")
            changed = True
        # Si SofaScore dit 0 (oubli) mais que le joueur a joué
        elif l_min < 1.0 and fm_min > 0:
            df_local.at[local_idx, "Minutes_Played"] = fm_min
            print(f"      ⏱️  {fm_match['Match_Date']} | Minutes: {l_min} ➡️  {fm_min}")
            changed = True

    if "Match_Date_DT" in df_local.columns:
        df_local = df_local.drop(columns=["Match_Date_DT"])

    if changed:
        df_local.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"      ✅ Dataset mis à jour avec précision.")
    
    return changed

# ─────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE PUBLIC
# ─────────────────────────────────────────────────────────────

def repair_player_with_fotmob(player_name: str, team_name: str) -> bool:
    csv_name = player_name.replace(" ", "_") + ".csv"
    csv_candidates = list(SOFASCORE_RAW.rglob(csv_name))
    if not csv_candidates:
        return False

    file_path = csv_candidates[0]
    try:
        df_local = pd.read_csv(file_path)
    except Exception:
        return False

    with sync_playwright() as pw:
        browser, context = _build_browser_context(pw)
        page = context.new_page()

        try:
            print(f"      [DEBUG] 1. Recherche de {player_name} sur FotMob...")
            player_info = _search_player_fotmob(page, player_name)
            
            if not player_info:
                print(f"      ❌ [DEBUG] ÉCHEC : Impossible de trouver '{player_name}' (Timeout ou Cloudflare).")
                return False
                
            player_id = player_info["id"]
            print(f"      [DEBUG] 2. Joueur trouvé ! ID: {player_id}. Interception des données...")

            player_data = player_info.get("_data")
            if not player_data:
                player_data = _fetch_player_data(page, player_id)

            if not player_data:
                print(f"      ❌ [DEBUG] ÉCHEC : Aucun JSON intercepté pour ce joueur.")
                return False
                
            print(f"      [DEBUG] 3. JSON intercepté avec succès. Parsing des matchs...")

            fm_matches = _parse_fotmob_matches(player_data)
            if not fm_matches:
                print(f"      ❌ [DEBUG] ÉCHEC : FotMob n'a renvoyé AUCUN match récent dans son JSON.")
                return False
                
            print(f"      [DEBUG] 4. {len(fm_matches)} matchs trouvés. L'Arbitre Logique entre en piste...")

            corrected = _apply_corrections(file_path, df_local, fm_matches)
            
            if not corrected:
                print(f"      [DEBUG] 5. L'Arbitre n'a détecté aucune anomalie après comparaison.")
                
            return corrected

        except Exception as e:
            print(f"      ⚠️ [DEBUG] ERREUR CRITIQUE : {e}")
            return False
        finally:
            context.close()
            browser.close()