import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import undetected_chromedriver as uc
import time
from bs4 import BeautifulSoup

options = uc.ChromeOptions()
options.add_argument("--window-size=1920,1080")

driver = uc.Chrome(options=options)
driver.get("https://www.sofascore.com/tournament/football/england/championship/18")
print("Waiting 10 seconds for React to render...")
time.sleep(10)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
teams = set()
for a in soup.find_all('a', href=True):
    if '/team/' in a['href']:
        teams.add(a['href'])

print(f"Found {len(teams)} teams in DOM!")
driver.quit()
