import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import undetected_chromedriver as uc
import time

options = uc.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = uc.Chrome(options=options)
driver.get("https://www.sofascore.com")
time.sleep(3)
print("Page title:", driver.title)
res = driver.execute_script("return await fetch('https://www.sofascore.com/api/v1/unique-tournament/54/seasons').then(r => r.json()).catch(e => e.toString())")
print(res)
driver.quit()
