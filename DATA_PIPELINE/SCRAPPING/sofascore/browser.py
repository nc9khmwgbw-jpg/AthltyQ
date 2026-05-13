import time
from typing import Optional, Any
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("Browser")

class SofaScoreBrowser:
    """Gestionnaire de navigateur Selenium pour SofaScore."""

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless
        self.driver: Optional[webdriver.Chrome] = None

    def start(self) -> webdriver.Chrome:
        """Initialise le driver Chrome avec des options Stealth."""
        logger.info("⚡ Initialisation du navigateur (Stealth Mode)...")
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--window-size=1920,1080")

        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )

        # Timeouts de sécurité
        self.driver.set_page_load_timeout(60)
        self.driver.set_script_timeout(60)
        self.driver.implicitly_wait(5)

        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })

        return self.driver

    def fetch_json(self, url: str) -> Optional[dict]:
        """Exécute un fetch via le navigateur."""
        js_script = f"return await fetch('{url}').then(r => r.json()).catch(e => ({{}}));"
        return self.execute_script(js_script)

    def execute_script(self, script: str) -> Any:
        """Exécute un script JS arbitraire et retourne le résultat."""
        if self.driver is None:
            self.start()
            assert self.driver is not None
            self.driver.get("https://www.sofascore.com")
            logger.info("✅ Page d'accueil SofaScore chargée.")
            time.sleep(3)
        
        assert self.driver is not None
        try:
            return self.driver.execute_script(script)
        except Exception as e:
            logger.error(f"Erreur execute_script: {e}")
            return None

    def stop(self) -> None:
        """Arrête le driver."""
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
