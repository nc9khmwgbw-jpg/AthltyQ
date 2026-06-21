"""
browser_fixed.py — SofaScoreBrowser corrigé
============================================
Changement critique : activer les logs de performance Chrome (nécessaire
pour intercepter les réponses XHR via CDP).
"""

import time
from typing import Optional, Any
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("Browser")


class SofaScoreBrowser:
    """Gestionnaire de navigateur Selenium pour SofaScore — avec logs de performance."""

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless
        self.driver: Optional[webdriver.Chrome] = None
        self.cloudflare_resolved = False

    def start(self) -> webdriver.Chrome:
        """Initialise Chrome avec les options Stealth + logs de performance."""
        logger.info("⚡ Initialisation du navigateur (Stealth Mode + CDP)...")

        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )

        # ─── CRITIQUE : activer les logs de performance pour capturer les XHR ───
        chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        # ────────────────────────────────────────────────────────────────────────

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options,
        )

        self.driver.set_page_load_timeout(60)
        self.driver.set_script_timeout(60)
        self.driver.implicitly_wait(5)

        # Supprimer la signature webdriver
        self.driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )

        # Activer les logs réseau CDP dès le démarrage
        self.driver.execute_cdp_cmd("Network.enable", {})

        return self.driver

    def navigate_and_wait(self, url: str, wait_time: int = 3) -> None:
        if self.driver is None:
            self.start()
        self.driver.get(url)
        time.sleep(wait_time)

    def fetch_json(self, url: str) -> Optional[dict]:
        """Exécute un fetch via le navigateur (utilise la session existante)."""
        js = f"return await fetch('{url}').then(r => r.json()).catch(e => ({{}}));"
        return self.execute_script(js)

    def execute_script(self, script: str) -> Any:
        if self.driver is None:
            self.start()
            self.driver.get("https://www.sofascore.com")
            time.sleep(3)
        try:
            return self.driver.execute_script(script)
        except Exception as e:
            logger.error(f"Erreur execute_script: {e}")
            return None

    def stop(self) -> None:
        if self.driver is not None:
            self.driver.quit()
            self.driver = None

    def get_cf_clearance_cookie(self) -> Optional[str]:
        if self.driver is None:
            return None
        try:
            for cookie in self.driver.get_cookies():
                if cookie.get("name") == "cf_clearance":
                    return cookie.get("value")
        except Exception:
            pass
        return None

    def restart_visible(self, blocked_url: Optional[str] = None) -> None:
        """Relance en mode visible pour résoudre manuellement les CAPTCHAs."""
        self.stop()
        self.headless = False
        self.start()
        self.driver.get("https://www.sofascore.com")
        time.sleep(3)
        if blocked_url:
            self.driver.get(blocked_url)
            time.sleep(2)

        # Tentative auto-clic CAPTCHA
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            wait = WebDriverWait(self.driver, 20)
            iframe = wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, "//iframe[contains(@src, 'challenges.cloudflare.com')]")
                )
            )
            self.driver.switch_to.frame(iframe)
            checkbox = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "input[type='checkbox'], .ctp-checkbox-label, .mark, label")
                )
            )
            time.sleep(2)
            checkbox.click()
            self.driver.switch_to.default_content()
            time.sleep(15)
        except Exception:
            logger.info("Pas d'iframe Cloudflare détecté ou déjà résolu")
            time.sleep(5)

        self.driver.get("https://www.sofascore.com")
        time.sleep(5)
        self.cloudflare_resolved = True
        logger.info("✅ Session Cloudflare prête")