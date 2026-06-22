from DrissionPage import ChromiumPage, ChromiumOptions

co = ChromiumOptions()
co.headless(False)
page = ChromiumPage(co)

page.get('https://www.sofascore.com/api/v1/unique-tournament/54/seasons')
page.wait.load_start()
print("URL:", page.url)
print("Title:", page.title)

# If Cloudflare challenge, it's an iframe
iframe = page.get_frame('@src^https://challenges.cloudflare.com')
if iframe:
    print("Found Cloudflare iframe, attempting to click...")
    page.wait(2)
    # Checkbox class is often mark or cb-c
    try:
        ele = iframe.ele('.mark', timeout=5)
        if ele:
            ele.click()
            print("Clicked!")
    except:
        print("Could not find checkbox")

page.wait(5)
print("Body:", page.html[:200])
page.quit()
