from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import time

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.execute_cdp_cmd(
    "Page.addScriptToEvaluateOnNewDocument",
    {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
)

print("Navigating...")
driver.get("https://www.sofascore.com/tournament/football/united-arab-emirates/uae-pro-league/1322")
time.sleep(10)

soup = BeautifulSoup(driver.page_source, 'html.parser')
script = soup.find('script', id='__NEXT_DATA__')
if script:
    data = json.loads(script.string)
    print("Found __NEXT_DATA__!")
    print(list(data.keys()))
    if 'props' in data and 'pageProps' in data['props']:
        print(list(data['props']['pageProps'].keys()))
else:
    print("No __NEXT_DATA__ script found.")

driver.quit()
