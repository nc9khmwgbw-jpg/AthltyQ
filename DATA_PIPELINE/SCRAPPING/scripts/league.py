import os
import sys
import time
import pandas as pd
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Import config
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

def get_driver():
    options = Options()
    options.add_argument("--no-sandbox")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def main():
    print("============================================================")
    print("   AthlytIQ - SCRAPER DE LIGUE INTERACTIF")
    print("============================================================")
    print("🏆 CHOISISSEZ UNE LIGUE : [1] LaLiga, [2] Premier League...")
    # ... (Le code complet du menu et du scraping par équipe/joueur)
    
if __name__ == "__main__":
    main()
