from DrissionPage import ChromiumPage, ChromiumOptions

def test_drission():
    co = ChromiumOptions()
    co.set_argument('--headless=new')  # Try headless first
    co.set_argument('--no-sandbox')
    
    page = ChromiumPage(co)
    page.get('https://www.sofascore.com')
    page.wait.load_start()
    print("Page loaded.")
    
    # Wait a bit
    page.wait(5)
    
    res = page.run_js("""
    return await fetch('https://api.sofascore.com/api/v1/unique-tournament/17/seasons')
        .then(r => r.json())
        .catch(e => ({error: e.message}));
    """)
    print("DATA:", res)
    page.quit()

if __name__ == '__main__':
    test_drission()
