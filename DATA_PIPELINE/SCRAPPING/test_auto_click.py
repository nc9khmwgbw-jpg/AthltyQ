from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
driver.get("https://www.sofascore.com/api/v1/unique-tournament/54/seasons")
print("Page loaded. Looking for CF iframe...")

try:
    wait = WebDriverWait(driver, 10)
    iframe = wait.until(EC.presence_of_element_located((By.XPATH, "//iframe[contains(@src, 'challenges.cloudflare.com')]")))
    print("Found iframe")
    driver.switch_to.frame(iframe)
    
    # Try different selectors
    selectors = [
        "input[type='checkbox']", 
        ".ctp-checkbox-label", 
        ".mark", 
        "label"
    ]
    
    clicked = False
    for sel in selectors:
        try:
            print(f"Trying selector {sel}...")
            element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
            element.click()
            print(f"Clicked {sel}!")
            clicked = True
            break
        except Exception as e:
            pass
            
    if not clicked:
        print("Could not click anything.")
        
    driver.switch_to.default_content()
    
    # Wait to see if it passed
    time.sleep(10)
    print("Final body:", driver.page_source[:200])

except Exception as e:
    print("Error:", e)

driver.quit()
