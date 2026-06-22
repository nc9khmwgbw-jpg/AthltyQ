from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
import time

def test_pw():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        stealth_sync(page)
        
        page.goto("https://www.sofascore.com")
        print("Page loaded, waiting for 10s...")
        time.sleep(10)
        
        res = page.evaluate("""
        async () => {
            return await fetch('https://api.sofascore.com/api/v1/unique-tournament/17/seasons')
                .then(r => r.json())
                .catch(e => ({error: e.message}));
        }
        """)
        print("DATA:", res)
        browser.close()

if __name__ == '__main__':
    test_pw()
