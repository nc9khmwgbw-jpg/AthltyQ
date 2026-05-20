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

script = """
const r = await fetch('https://www.sofascore.com/api/v1/unique-tournament/18/seasons', {
    headers: {
        'Accept': 'application/json',
        'X-Requested-With': 'XMLHttpRequest'
    }
});
return await r.json();
"""
try:
    res = b.execute_script(script)
    print("API Result (www):", res.keys() if isinstance(res, dict) else type(res))
except Exception as e:
    print("Error:", e)
b.stop()
