import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import undetected_chromedriver as uc
import time

options = uc.ChromeOptions()
options.add_argument("--window-size=1920,1080")

driver = uc.Chrome(options=options)
driver.get("https://www.sofascore.com/tournament/football/england/championship/18")
print("Page loaded, waiting for 5 seconds...")
time.sleep(5)
print("Title:", driver.title)

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
    res = driver.execute_script(script)
    print("API Result:", res.keys() if isinstance(res, dict) else type(res))
    if 'seasons' in res:
        print(f"Bypassed Cloudflare! Found {len(res['seasons'])} seasons.")
except Exception as e:
    print("Error:", e)

driver.quit()
