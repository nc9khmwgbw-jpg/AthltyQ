from DrissionPage import ChromiumPage, ChromiumOptions
import time

co = ChromiumOptions()
co.headless(False)
page = ChromiumPage(co)

print("Opening homepage...")
page.get('https://www.sofascore.com')
time.sleep(5)
print("Title:", page.title)

print("Checking API...")
page.get('https://www.sofascore.com/api/v1/unique-tournament/54/seasons')
time.sleep(2)
print("API Response:", page.html[:200])

page.quit()
