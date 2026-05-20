import cloudscraper
import json

scraper = cloudscraper.create_scraper()
url = "https://api.sofascore.com/api/v1/unique-tournament/18/seasons"
headers = {
    'Accept': 'application/json',
    'Origin': 'https://www.sofascore.com',
    'Referer': 'https://www.sofascore.com/'
}
res = scraper.get(url, headers=headers)
print("Status:", res.status_code)
try:
    print("Keys:", res.json().keys())
    if 'seasons' in res.json():
        print(f"Found {len(res.json()['seasons'])} seasons")
except:
    print(res.text[:200])
