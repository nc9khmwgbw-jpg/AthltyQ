from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
import time

b = SofaScoreBrowser(headless=True)
b.start()
b.driver.get('https://www.sofascore.com')
time.sleep(5)
data = b.fetch_json('https://api.sofascore.com/api/v1/search/all?q=uae%20pro%20league')
print("DATA:", data)
b.stop()
