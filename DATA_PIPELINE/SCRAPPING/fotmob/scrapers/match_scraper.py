import time
import re
import json
from typing import Optional, Dict, Any, List
from playwright.sync_api import sync_playwright, Page, BrowserContext, Browser
from DATA_PIPELINE.SCRAPPING.fotmob.extractors.match_extractor import FotMobMatchExtractor
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("FotMobScraper")


class FotMobMatchScraper:
    """Scraper FotMob utilisant Playwright + extraction __NEXT_DATA__."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.extractor = FotMobMatchExtractor()

    def _setup_browser(self, pw) -> tuple[Browser, BrowserContext]:
        browser = pw.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        return browser, context

    def _extract_next_data(self, page: Page) -> Optional[Dict[str, Any]]:
        """Extrait les données joueur depuis __NEXT_DATA__ (Next.js SSR)."""
        # Attendre que le script __NEXT_DATA__ soit présent dans le DOM
        try:
            page.wait_for_selector("#__NEXT_DATA__", timeout=10000)
        except Exception:
            logger.warning("⚠️  #__NEXT_DATA__ non trouvé dans le DOM.")

        try:
            raw = page.evaluate("() => document.getElementById('__NEXT_DATA__')?.textContent")
            if raw:
                data = json.loads(raw)
                page_props = data.get("props", {}).get("pageProps", {})
                if page_props:
                    logger.info("✅ __NEXT_DATA__ extrait avec succès.")
                    return page_props
        except Exception as e:
            logger.warning(f"Échec extraction __NEXT_DATA__ : {e}")

        # Fallback : window.__NEXT_DATA__
        try:
            data = page.evaluate("() => window.__NEXT_DATA__?.props?.pageProps")
            if data:
                logger.info("✅ window.__NEXT_DATA__ extrait (fallback).")
                return data
        except Exception as e:
            logger.warning(f"Échec fallback __NEXT_DATA__ : {e}")

        return None

    def search_player(self, page: Page, player_name: str) -> Optional[Dict[str, Any]]:
        """Recherche un joueur, navigue sur sa page et extrait les données SSR."""
        try:
            page.goto("https://www.fotmob.com/", wait_until="domcontentloaded", timeout=20000)
            time.sleep(3)

            # Fermeture cookies
            page.evaluate("""() => {
                const b = Array.from(document.querySelectorAll('button'));
                const a = b.find(x => /Accept|Agree|Accepter|OK|Got it/i.test(x.innerText));
                if (a) a.click();
            }""")

            # Recherche joueur
            search_box = page.locator("input[placeholder*='Search']").first
            search_box.click(force=True)
            time.sleep(1)
            search_box.fill(player_name)
            time.sleep(2)

            # Clic sur le premier résultat joueur
            player_link = page.locator("a[href*='/players/']").first
            if player_link.is_visible():
                logger.info("✅ Joueur détecté dans les suggestions.")
                player_link.click(force=True)
            else:
                search_box.press("Enter")
                time.sleep(3)
                player_link = page.locator("a[href*='/players/']").first
                if player_link.is_visible():
                    player_link.click(force=True)
                else:
                    logger.warning(f"❌ Aucun résultat pour {player_name}.")
                    return None

            # ✅ Attendre l'URL puis le DOM — pas networkidle (timeout garanti sur FotMob)
            page.wait_for_url("**/players/**", timeout=12000)
            page.wait_for_load_state("domcontentloaded", timeout=10000)

            url_match = re.search(r'/players/(\d+)', page.url)
            player_id = url_match.group(1) if url_match else None
            logger.info(f"🆔 ID trouvé : {player_id}. Extraction __NEXT_DATA__...")

            page_props = self._extract_next_data(page)
            if page_props:
                return {"id": player_id, "name": player_name, "_data": page_props}

            logger.warning(f"⚠️  Aucune donnée dans __NEXT_DATA__ pour {player_name}.")
            return None

        except Exception as e:
            logger.error(f"Erreur recherche FotMob {player_name}: {e}")
            return None

    def scrape_player(self, player_name: str) -> List[Dict[str, Any]]:
        """Méthode principale pour récupérer les matchs d'un joueur."""
        with sync_playwright() as pw:
            browser, context = self._setup_browser(pw)
            page = context.new_page()
            try:
                info = self.search_player(page, player_name)
                if info and info.get("_data"):
                    return self.extractor.extract(info["_data"])
            finally:
                browser.close()
        return []