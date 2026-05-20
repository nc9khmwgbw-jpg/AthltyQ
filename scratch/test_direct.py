import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser

b = SofaScoreBrowser(headless=True)
b.start()
b.driver.get("https://www.sofascore.com/tournament/football/england/championship/18")
import time
time.sleep(5)
print(b.driver.current_url)
print(b.driver.title)

script = """
const r = await fetch('https://api.sofascore.com/api/v1/unique-tournament/18/seasons', {
    headers: {
        'Accept': 'application/json',
        'X-Requested-With': 'XMLHttpRequest'
    }
});
return await r.json();
"""
try:
    res = b.execute_script(script)
    print("API Result:", res)
except Exception as e:
    print("Error:", e)
b.stop()
