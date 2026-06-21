from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
import time

b = SofaScoreBrowser(headless=False)
b.start()
try:
    print("Loading homepage...")
    b.driver.get('https://www.sofascore.com')
    print("Waiting 10 seconds for Cloudflare...")
    time.sleep(10)
    print("Fetching API data...")
    data = b.fetch_json('https://api.sofascore.com/api/v1/unique-tournament/17/seasons')
    print("DATA:", data)
finally:
    b.stop()
