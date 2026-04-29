import json
import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def repair_player_with_fotmob(player_name, team_name):
    driver = get_driver()
    try:
        print(f"      🔍 Recherche FotMob : {player_name} ({team_name})...")
        driver.get("https://www.fotmob.com/")
        # ... (Logique de recherche, clic sur le joueur, et extraction des 15 derniers matchs)
        return True
    except Exception as e:
        print(f"      ❌ Erreur FotMob : {e}")
        return False
    finally:
        driver.quit()
