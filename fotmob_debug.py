"""
Debug v2 : extrait __NEXT_DATA__ de la page joueur FotMob et affiche les clés.
.venv/bin/python fotmob_debug2.py
"""
import time, re, json
from playwright.sync_api import sync_playwright

PLAYER_NAME = "Dan Burn"

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )
    page = context.new_page()

    page.goto("https://www.fotmob.com/", wait_until="domcontentloaded", timeout=20000)
    time.sleep(3)
    page.evaluate("""() => {
        const b = Array.from(document.querySelectorAll('button'));
        const a = b.find(x => /Accept|Agree|OK|Got it/i.test(x.innerText));
        if (a) a.click();
    }""")

    sb = page.locator("input[placeholder*='Search']").first
    sb.click(force=True); time.sleep(1)
    sb.fill(PLAYER_NAME); time.sleep(2)

    link = page.locator("a[href*='/players/']").first
    if link.is_visible():
        link.click(force=True)
    else:
        sb.press("Enter"); time.sleep(3)
        page.locator("a[href*='/players/']").first.click(force=True)

    page.wait_for_url("**/players/**", timeout=12000)
    page.wait_for_load_state("networkidle", timeout=15000)

    print(f"\nURL : {page.url}")

    # Extraction __NEXT_DATA__
    raw = page.evaluate("() => document.getElementById('__NEXT_DATA__')?.innerText")
    if raw:
        data = json.loads(raw)
        page_props = data.get("props", {}).get("pageProps", {})
        print(f"\n✅ __NEXT_DATA__ trouvé. Clés de pageProps : {list(page_props.keys())}")

        # Afficher récursivement les sous-clés importantes
        def show_keys(d, prefix="", depth=2):
            if depth == 0 or not isinstance(d, dict):
                return
            for k, v in d.items():
                print(f"  {'  ' * (2 - depth)}{prefix}{k}: {type(v).__name__}" +
                      (f" [{len(v)} items]" if isinstance(v, (list, dict)) else ""))
                show_keys(v, "", depth - 1)

        show_keys(page_props)

        # Sauvegarder localement
        with open("fotmob_next_data.json", "w") as f:
            json.dump(page_props, f, indent=2)
        print("\n✅ Sauvegardé dans fotmob_next_data.json")
    else:
        print("❌ __NEXT_DATA__ introuvable dans la page.")
        print("Contenu HTML (500 chars) :", page.content()[:500])

    browser.close()