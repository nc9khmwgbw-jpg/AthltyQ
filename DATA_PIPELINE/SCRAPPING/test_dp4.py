from DrissionPage import ChromiumPage, ChromiumOptions
import time

co = ChromiumOptions()
co.headless(False)
page = ChromiumPage(co)

print("Opening homepage...")
page.get('https://www.sofascore.com')
time.sleep(3)

print("Checking API via JS...")
res = page.run_js("""
    return fetch('https://www.sofascore.com/api/v1/unique-tournament/54/seasons', {
        headers: {
            'Accept': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        }
    }).then(r => r.json()).catch(e => e.toString());
""")
print("API Response:", res)

page.quit()
