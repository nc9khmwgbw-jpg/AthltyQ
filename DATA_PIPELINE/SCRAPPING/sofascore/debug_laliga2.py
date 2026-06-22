"""
debug_laliga2.py
================
Script de diagnostic pour capturer EXACTEMENT quels appels API
SofaScore fait pour LaLiga2 (ID: 54).

Usage: python debug_laliga2.py
"""

import json
import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def capture_all_xhr(driver, wait_seconds=10):
    """Capture tous les appels XHR après navigation."""
    print(f"  ⏳ Attente {wait_seconds}s pour que les XHR se chargent...")
    time.sleep(wait_seconds)
    
    logs = driver.get_log("performance")
    print(f"  📋 {len(logs)} entrées de log")
    
    api_calls = []
    for entry in logs:
        try:
            msg = json.loads(entry["message"])["message"]
            if msg.get("method") == "Network.responseReceived":
                url = msg["params"]["response"]["url"]
                status = msg["params"]["response"]["status"]
                request_id = msg["params"]["requestId"]
                if "sofascore" in url:
                    api_calls.append({
                        "url": url,
                        "status": status,
                        "request_id": request_id
                    })
        except Exception:
            continue
    
    return api_calls


def read_response_body(driver, request_id):
    """Lit le body d'une réponse via CDP."""
    try:
        body = driver.execute_cdp_cmd(
            "Network.getResponseBody",
            {"requestId": request_id}
        )
        if body and body.get("body"):
            return json.loads(body["body"])
    except Exception as e:
        return {"error": str(e)}
    return None


def main():
    print("🔍 DIAGNOSTIC SOFASCORE — LaLiga2 (ID: 54)\n")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    driver.execute_cdp_cmd("Network.enable", {})
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )

    results = {}

    try:
        # ── TEST 1 : Page d'accueil SofaScore (établir session) ──
        print("=" * 60)
        print("TEST 1 : Accueil SofaScore")
        driver.get("https://www.sofascore.com")
        time.sleep(4)
        logs = driver.get_log("performance")  # Vider les logs
        print(f"  ✓ Session établie, {len(logs)} logs vidés\n")

        # ── TEST 2 : Page tournoi LaLiga2 ──
        print("=" * 60)
        print("TEST 2 : Page tournoi LaLiga2")
        print("  URL: https://www.sofascore.com/tournament/football/spain/laliga-2/54")
        driver.get("https://www.sofascore.com/tournament/football/spain/laliga-2/54")
        
        calls = capture_all_xhr(driver, wait_seconds=8)
        print(f"\n  🌐 {len(calls)} appels SofaScore capturés :")
        for c in calls:
            print(f"    [{c['status']}] {c['url']}")
        
        # Chercher les appels intéressants
        for c in calls:
            url = c["url"]
            if any(kw in url for kw in ["season", "tournament", "teams", "standings", "transfer"]):
                print(f"\n  🎯 Appel intéressant : {url}")
                body = read_response_body(driver, c["request_id"])
                if body and "error" not in body:
                    keys = list(body.keys()) if isinstance(body, dict) else f"[liste de {len(body)}]"
                    print(f"     Clés : {keys}")
                    if "seasons" in body:
                        seasons = body["seasons"]
                        print(f"     Saisons : {[{'id': s.get('id'), 'name': s.get('name', s.get('year'))} for s in seasons[:3]]}")
                        results["seasons"] = seasons
                    if "teams" in body:
                        teams = body["teams"]
                        print(f"     Équipes : {len(teams)} trouvées, ex: {teams[0].get('name') if teams else 'aucune'}")
                        results["teams_url"] = url
                        results["teams"] = teams
        
        # Vider les logs pour le prochain test
        driver.get_log("performance")

        # ── TEST 3 : Page standings directe ──
        print("\n" + "=" * 60)
        print("TEST 3 : Page standings LaLiga2")
        
        # Utiliser le season_id trouvé si disponible
        season_id = None
        if "seasons" in results and results["seasons"]:
            season_id = results["seasons"][0].get("id")
            print(f"  Season ID récupéré : {season_id}")
        
        standings_url = f"https://www.sofascore.com/tournament/football/spain/laliga-2/54/standings/home"
        print(f"  URL: {standings_url}")
        driver.get(standings_url)
        
        calls2 = capture_all_xhr(driver, wait_seconds=8)
        print(f"\n  🌐 {len(calls2)} appels SofaScore capturés :")
        for c in calls2:
            print(f"    [{c['status']}] {c['url']}")
        
        # Chercher les équipes
        for c in calls2:
            url = c["url"]
            if "team" in url or "standing" in url:
                print(f"\n  🎯 {url}")
                body = read_response_body(driver, c["request_id"])
                if body and isinstance(body, dict):
                    print(f"     Clés : {list(body.keys())}")
                    if "standings" in body:
                        rows = body["standings"]
                        if rows and isinstance(rows, list):
                            row = rows[0]
                            print(f"     Ex row keys: {list(row.keys()) if isinstance(row, dict) else 'list'}")
                            if "rows" in row:
                                teams_in_standings = row["rows"]
                                print(f"     ✅ {len(teams_in_standings)} équipes dans standings !")
                                for t in teams_in_standings[:3]:
                                    team = t.get("team", t)
                                    print(f"       - {team.get('name')} (id: {team.get('id')})")
                                results["standings_teams"] = teams_in_standings
                                results["standings_url"] = url

        # ── TEST 4 : URL alternative avec season_id dans le hash ──
        driver.get_log("performance")  # Vider
        print("\n" + "=" * 60)
        print("TEST 4 : Accès direct API via fetch() browser")
        
        # Tenter un fetch direct depuis le navigateur (utilise les cookies/session existants)
        fetch_urls = []
        if season_id:
            fetch_urls.append(f"https://api.sofascore.com/api/v1/unique-tournament/54/season/{season_id}/teams")
        fetch_urls.append("https://api.sofascore.com/api/v1/unique-tournament/54/seasons")
        
        for furl in fetch_urls:
            print(f"\n  🔗 Fetch: {furl}")
            result = driver.execute_script(f"""
                try {{
                    const r = await fetch('{furl}', {{
                        headers: {{
                            'User-Agent': 'Mozilla/5.0',
                            'Accept': 'application/json',
                            'Referer': 'https://www.sofascore.com/'
                        }}
                    }});
                    const data = await r.json();
                    return {{ status: r.status, data: data }};
                }} catch(e) {{
                    return {{ error: e.toString() }};
                }}
            """)
            if result:
                status = result.get("status")
                data = result.get("data", {})
                error = result.get("error")
                if error:
                    print(f"  ❌ Erreur: {error}")
                elif status == 200:
                    keys = list(data.keys()) if isinstance(data, dict) else "?"
                    print(f"  ✅ Status {status} — Clés: {keys}")
                    if "seasons" in data:
                        slist = data["seasons"]
                        print(f"     Saisons: {[{'id': s.get('id'), 'year': s.get('year')} for s in slist[:3]]}")
                        if not season_id and slist:
                            season_id = slist[0].get("id")
                    if "teams" in data:
                        teams = data["teams"]
                        print(f"     ✅ {len(teams)} équipes!")
                        for t in teams[:3]:
                            print(f"       - {t.get('name')} (id: {t.get('id')})")
                else:
                    print(f"  ⚠️ Status {status}: {str(data)[:200]}")

        # ── RÉSUMÉ ──
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ DIAGNOSTIC")
        print(f"  Season ID trouvé : {season_id}")
        print(f"  Teams via XHR    : {'✅ ' + str(len(results.get('teams', []))) if 'teams' in results else '❌ Non'}")
        print(f"  Teams via standings: {'✅ ' + str(len(results.get('standings_teams', []))) if 'standings_teams' in results else '❌ Non'}")
        if results.get("standings_url"):
            print(f"  URL standings    : {results['standings_url']}")
        
        print("\n🔑 CONCLUSION:")
        if "teams" in results or "standings_teams" in results:
            print("  ✅ Les données sont accessibles — problème dans le timing/pattern de l'engine")
        else:
            print("  ❌ SofaScore bloque les requêtes — besoin de résoudre Cloudflare manuellement")
            print("     → Relancer avec headless=False pour voir ce qui se passe")

    finally:
        driver.quit()
        print("\n✅ Diagnostic terminé.")


if __name__ == "__main__":
    main()
