import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
import time

b = SofaScoreBrowser(headless=False)
b.start()
print("Navigating to API domain to clear Cloudflare...")
b.driver.get("https://api.sofascore.com/api/v1/unique-tournament/18/seasons")
time.sleep(5)
print("URL:", b.driver.current_url)
print("Page Source excerpt:", b.driver.page_source[:200])

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
