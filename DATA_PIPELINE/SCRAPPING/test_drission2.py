from DrissionPage import ChromiumPage, ChromiumOptions

def test_drission():
    co = ChromiumOptions()
    co.set_browser_path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    # Use normal mode to let Cloudflare pass, then we can hide it later
    # co.set_argument('--headless=new') 
    
    page = ChromiumPage(co)
    page.get('https://www.sofascore.com')
    page.wait.load_start()
    print("Page loaded. Waiting for 10 seconds for Cloudflare...")
    page.wait(10)
    
    res = page.run_js("""
    return await fetch('https://api.sofascore.com/api/v1/unique-tournament/17/seasons')
        .then(r => r.json())
        .catch(e => ({error: e.message}));
    """)
    print("DATA:", res)
    page.quit()

if __name__ == '__main__':
    test_drission()
