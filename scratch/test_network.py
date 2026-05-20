from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

options = Options()
options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://www.sofascore.com/tournament/football/england/championship/18")
time.sleep(5)

logs = driver.get_log("performance")
found = False
for entry in logs:
    log = json.loads(entry["message"])["message"]
    if log["method"] == "Network.responseReceived":
        url = log["params"]["response"]["url"]
        if "unique-tournament/18/seasons" in url:
            request_id = log["params"]["requestId"]
            try:
                body = driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": request_id})
                print("FOUND SEASONS VIA CDP:", body['body'][:200])
                found = True
            except Exception as e:
                pass
if not found:
    print("Not found in network logs.")

driver.quit()
