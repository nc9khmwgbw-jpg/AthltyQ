import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DATA_PIPELINE.SCRAPPING.sofascore.browser import SofaScoreBrowser
import time
import json
import re

b = SofaScoreBrowser(headless=False)
b.start()
print("Navigating directly to API...")
b.driver.get("https://api.sofascore.com/api/v1/unique-tournament/18/seasons")
time.sleep(5)
src = b.driver.page_source

# Extract JSON from page source
match = re.search(r'(\{.*\})', src)
if match:
    try:
        data = json.loads(match.group(1))
        print("Success JSON keys:", data.keys())
        if 'seasons' in data:
            print(f"Found {len(data['seasons'])} seasons!")
    except Exception as e:
        print("JSON parse error:", e)
else:
    print("No JSON found.")
    print(src[:300])

b.stop()
