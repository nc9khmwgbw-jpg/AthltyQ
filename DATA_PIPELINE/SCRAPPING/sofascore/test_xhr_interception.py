"""
test_xhr_interception.py
========================
Script de test rapide pour vérifier que l'interception XHR fonctionne.
Lance ce script avant de lancer le scraping complet.

Usage:
    python test_xhr_interception.py
"""

import json
import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def test_xhr_interception():
    print("🧪 Test interception XHR SofaScore...")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    # CRITIQUE : logs de performance
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

    try:
        # Test 1 : Page d'un joueur connu (Vinicius Jr, id=886667)
        print("\n📡 Navigation vers la page joueur Vinicius Jr...")
        driver.get("https://www.sofascore.com/player/vinicius-jr/886667")
        time.sleep(6)

        # Capturer les logs réseau
        logs = driver.get_log("performance")
        print(f"📋 {len(logs)} entrées de log capturées")

        xhr_captured = []
        for entry in logs:
            try:
                msg = json.loads(entry["message"])["message"]
                if msg.get("method") == "Network.responseReceived":
                    url = msg["params"]["response"]["url"]
                    if "sofascore.com/api" in url:
                        xhr_captured.append(url)
            except Exception:
                continue

        print(f"\n✅ {len(xhr_captured)} appels API SofaScore capturés :")
        for url in xhr_captured[:20]:
            print(f"  → {url}")

        # Test 2 : Lire le body d'une réponse
        events_pattern = re.compile(r"/api/v1/player/\d+/events/last/\d+")
        for entry in logs:
            try:
                msg = json.loads(entry["message"])["message"]
                if msg.get("method") == "Network.responseReceived":
                    url = msg["params"]["response"]["url"]
                    request_id = msg["params"]["requestId"]
                    if events_pattern.search(url):
                        print(f"\n🎯 Cible trouvée : {url}")
                        body = driver.execute_cdp_cmd(
                            "Network.getResponseBody",
                            {"requestId": request_id}
                        )
                        if body and body.get("body"):
                            data = json.loads(body["body"])
                            events = data.get("events", [])
                            print(f"✅ {len(events)} matchs récupérés !")
                            if events:
                                e = events[0]
                                print(f"   Dernier match : {e.get('homeTeam', {}).get('name')} vs {e.get('awayTeam', {}).get('name')}")
                        break
            except Exception:
                continue
        else:
            print("\n⚠️ Aucun appel /events trouvé — SofaScore a peut-être bloqué la session")
            print("   → Essaie avec headless=False pour vérifier manuellement")

    finally:
        driver.quit()
        print("\n✅ Test terminé.")


if __name__ == "__main__":
    test_xhr_interception()
