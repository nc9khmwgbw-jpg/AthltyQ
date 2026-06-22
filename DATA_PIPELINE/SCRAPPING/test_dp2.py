from DrissionPage import ChromiumPage, ChromiumOptions

co = ChromiumOptions()
co.headless(False)
page = ChromiumPage(co)

print("Opening homepage to solve captcha...")
page.get('https://www.sofascore.com')

# Wait for captcha to be solved if present
while True:
    iframe = page.get_frame('@src^https://challenges.cloudflare.com')
    if iframe:
        print("Found Cloudflare iframe, attempting to click...")
        try:
            ele = iframe.ele('.mark', timeout=5)
            if ele:
                ele.click()
                print("Clicked!")
        except Exception as e:
            print("Wait for manual click...", e)
        page.wait(2)
    else:
        break

print("Homepage loaded. Checking API...")
page.get('https://www.sofascore.com/api/v1/unique-tournament/54/seasons')
page.wait(2)
print("API Response:", page.html[:200])

page.quit()
