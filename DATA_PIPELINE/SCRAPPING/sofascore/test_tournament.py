import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def test_seasons_interception():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.execute_cdp_cmd("Network.enable", {})
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"})
    
    print("📡 Navigation vers la page La Liga...")
    driver.get("https://www.sofascore.com/tournament/football/spain/laliga/8")
    time.sleep(8)
    
    logs = driver.get_log("performance")
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
    for url in xhr_captured:
        print(f"  → {url}")
    driver.quit()

if __name__ == "__main__":
    test_seasons_interception()
