import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
import time

b = SofaScoreBrowser(headless=True)
b.start()
print("Navigating to www.sofascore.com...")
b.driver.get("https://www.sofascore.com/tournament/football/england/championship/18")
time.sleep(5)

script = """
return await fetch('https://api.sofascore.com/api/v1/unique-tournament/18/seasons').then(r => r.json()).catch(e => ({'err': e.message}));
"""
try:
    res = b.execute_script(script)
    print("API Result:", res.keys() if isinstance(res, dict) else res)
except Exception as e:
    print("Error:", e)
b.stop()
