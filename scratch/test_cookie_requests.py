import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
import time
import requests

b = SofaScoreBrowser(headless=False)
b.start()
print("Navigating to www.sofascore.com to get cookies...")
b.driver.get("https://www.sofascore.com/tournament/football/england/championship/18")
time.sleep(5)
print("Title:", b.driver.title)

cookies = b.driver.get_cookies()
s = requests.Session()
for c in cookies:
    s.cookies.set(c['name'], c['value'], domain=c['domain'])

print("Cookies loaded:", len(cookies))

url = "https://api.sofascore.com/api/v1/unique-tournament/18/seasons"
headers = {
    'User-Agent': b.driver.execute_script("return navigator.userAgent"),
    'Accept': 'application/json',
    'Origin': 'https://www.sofascore.com',
    'Referer': 'https://www.sofascore.com/'
}
res = s.get(url, headers=headers)
print("API Result:", res.status_code)
try:
    data = res.json()
    print("Keys:", data.keys())
    if 'seasons' in data:
        print(f"Success! {len(data['seasons'])} seasons.")
except:
    pass

b.stop()
