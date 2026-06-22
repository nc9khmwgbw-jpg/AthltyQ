"""
debug_fetch.py — Test minimal du fetch() JS sur l'API SofaScore
Usage: python debug_fetch.py
"""

import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def main():
    print("🔍 TEST FETCH SOFASCORE API\n")

    options = Options()
    # Mettre False pour voir ce qui se passe visuellement si besoin
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    # Timeout généreux
    driver.set_page_load_timeout(90)
    driver.set_script_timeout(30)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )

    try:
        # ── Étape 1 : charger SofaScore avec gestion timeout ──
        print("1️⃣  Chargement sofascore.com...")
        try:
            driver.get("https://www.sofascore.com")
        except Exception as e:
            print(f"   ⚠️  Timeout page (normal) : {str(e)[:80]}")
            print("   → Continuation avec la session partielle...")
        time.sleep(5)
        print(f"   URL courante : {driver.current_url}")

        # ── Étape 2 : tester fetch() synchrone d'abord ──
        print("\n2️⃣  Test fetch synchrone /seasons (tournoi 54 = LaLiga2)...")
        result = driver.execute_script("""
            var xhr = new XMLHttpRequest();
            xhr.open('GET', 'https://api.sofascore.com/api/v1/unique-tournament/54/seasons', false);
            xhr.setRequestHeader('Accept', 'application/json');
            xhr.send();
            return {status: xhr.status, body: xhr.responseText.substring(0, 500)};
        """)
        print(f"   Status XHR : {result.get('status')}")
        print(f"   Body (500c) : {result.get('body', '')[:300]}")

        # ── Étape 3 : tester fetch() async ──
        print("\n3️⃣  Test fetch async /seasons...")
        result2 = driver.execute_async_script("""
            var callback = arguments[arguments.length - 1];
            fetch('https://api.sofascore.com/api/v1/unique-tournament/54/seasons', {
                headers: {
                    'Accept': 'application/json',
                    'Referer': 'https://www.sofascore.com/'
                },
                credentials: 'include'
            })
            .then(r => r.text().then(t => callback({status: r.status, body: t.substring(0, 500)})))
            .catch(e => callback({error: e.toString()}));
        """)
        print(f"   Status : {result2.get('status')}")
        if result2.get('error'):
            print(f"   Erreur : {result2['error']}")
        else:
            body = result2.get('body', '')
            print(f"   Body : {body[:300]}")
            if result2.get('status') == 200:
                try:
                    data = json.loads(result2['body'])
                    seasons = data.get('seasons', [])
                    print(f"\n   ✅ {len(seasons)} saisons trouvées !")
                    for s in seasons[:3]:
                        print(f"      - ID: {s.get('id')} | {s.get('name', s.get('year'))}")
                except Exception as e:
                    print(f"   Parse error: {e}")

        # ── Étape 4 : naviguer vers la page tournoi et retester ──
        print("\n4️⃣  Navigation vers la page tournoi LaLiga2...")
        try:
            driver.get("https://www.sofascore.com/tournament/football/spain/laliga-2/54")
        except Exception as e:
            print(f"   ⚠️  Timeout (normal) : {str(e)[:60]}")
        time.sleep(6)
        print(f"   URL : {driver.current_url}")

        print("\n5️⃣  Retest fetch après navigation tournoi...")
        result3 = driver.execute_async_script("""
            var callback = arguments[arguments.length - 1];
            fetch('https://api.sofascore.com/api/v1/unique-tournament/54/seasons', {
                headers: {'Accept': 'application/json'},
                credentials: 'include'
            })
            .then(r => r.text().then(t => callback({status: r.status, body: t.substring(0, 800)})))
            .catch(e => callback({error: e.toString()}));
        """)
        status3 = result3.get('status')
        print(f"   Status : {status3}")
        if status3 == 200:
            data3 = json.loads(result3['body'])
            seasons3 = data3.get('seasons', [])
            print(f"   ✅ {len(seasons3)} saisons ! season_id courant = {seasons3[0].get('id') if seasons3 else 'N/A'}")
            
            # Tester avec le season_id
            if seasons3:
                sid = seasons3[0]['id']
                print(f"\n6️⃣  Test récupération équipes (season {sid})...")
                result4 = driver.execute_async_script(f"""
                    var callback = arguments[arguments.length - 1];
                    fetch('https://api.sofascore.com/api/v1/unique-tournament/54/season/{sid}/teams', {{
                        headers: {{'Accept': 'application/json'}},
                        credentials: 'include'
                    }})
                    .then(r => r.text().then(t => callback({{status: r.status, body: t.substring(0, 1000)}})))
                    .catch(e => callback({{error: e.toString()}}));
                """)
                print(f"   Status : {result4.get('status')}")
                if result4.get('status') == 200:
                    data4 = json.loads(result4['body'])
                    teams = data4.get('teams', [])
                    print(f"   ✅ {len(teams)} équipes !")
                    for t in teams[:3]:
                        print(f"      - {t.get('name')} (id: {t.get('id')})")
                else:
                    print(f"   Body: {result4.get('body', '')[:200]}")
        else:
            print(f"   Body: {result3.get('body', '')[:300]}")

        print("\n" + "="*50)
        print("✅ Diagnostic terminé.")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
