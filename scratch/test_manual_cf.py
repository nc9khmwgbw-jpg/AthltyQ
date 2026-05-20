import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
import time

b = SofaScoreBrowser(headless=False)
b.start()
print("Navigating to www.sofascore.com...")
b.driver.get("https://www.sofascore.com/tournament/football/england/championship/18")

print("\n🚨 CLIQUEZ SUR LE BOUTON CLOUDFLARE DANS LE NAVIGATEUR S'IL APPARAIT !")
print("Attente de 15 secondes...")
time.sleep(15)

script = """
return await fetch('https://api.sofascore.com/api/v1/unique-tournament/18/seasons').then(r => r.json()).catch(e => ({'err': e.message}));
"""
try:
    res = b.execute_script(script)
    print("\nAPI Result:", res.keys() if isinstance(res, dict) else res)
    if isinstance(res, dict) and 'seasons' in res:
        print("BINGO! Cloudflare bypassed manually!")
except Exception as e:
    print("Error:", e)
b.stop()
