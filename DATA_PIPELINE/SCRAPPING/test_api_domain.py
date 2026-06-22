import time
from sofascore.browser import SofaScoreBrowser

b = SofaScoreBrowser(headless=True)
b.start()
print("Fetching from www.sofascore.com/api/v1/...")
res = b.execute_script("return await fetch('https://www.sofascore.com/api/v1/unique-tournament/54/seasons').then(r => r.json()).catch(e => e.toString())")
print(res)

print("Fetching from api.sofascore.com/api/v1/...")
res2 = b.execute_script("return await fetch('https://api.sofascore.com/api/v1/unique-tournament/54/seasons').then(r => r.json()).catch(e => e.toString())")
print(res2)

b.stop()
