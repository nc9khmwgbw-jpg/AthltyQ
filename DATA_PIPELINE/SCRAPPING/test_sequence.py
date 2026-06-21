import time
from selenium.common.exceptions import TimeoutException
from DATA_PIPELINE.SCRAPPING.sofascore.scrapers.league_scraper import SofaScoreLeagueScraper

scraper = SofaScoreLeagueScraper()
scraper.browser.start()
try:
    scraper.browser.driver.set_page_load_timeout(15)
    scraper.browser.driver.get("https://www.sofascore.com/tournament/football/united-arab-emirates/uae-pro-league/1322")
except TimeoutException:
    print("Timeout, moving on.")

data = scraper.engine._api_get('https://api.sofascore.com/api/v1/unique-tournament/1322/seasons')
print("DATA:", data)
scraper.browser.stop()
