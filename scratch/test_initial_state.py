import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
import time

b = SofaScoreBrowser(headless=False)
b.start()
print("Navigating to www.sofascore.com...")
b.driver.get("https://www.sofascore.com/tournament/football/england/championship/18")
time.sleep(5)
src = b.driver.page_source
if '__INITIAL_STATE__' in src:
    print("Found INITIAL_STATE!")
    import re, json
    match = re.search(r'window\.__INITIAL_STATE__=(.*?);\s*</script>', src)
    if match:
        data = json.loads(match.group(1))
        print("Keys:", data.keys())
        if 'tournament' in data:
            print("Tournament keys:", data['tournament'].keys())
            if 'standings' in data['tournament']:
                print("Standings found!")
else:
    print("Not found.")
b.stop()
