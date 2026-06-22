from seleniumbase import Driver
import time

driver = Driver(uc=True, headless=True)
driver.get("https://www.sofascore.com")
print("Page title:", driver.title)

res = driver.execute_script("return await fetch('https://www.sofascore.com/api/v1/unique-tournament/54/seasons').then(r => r.json()).catch(e => e.toString())")
print(res)
driver.quit()
