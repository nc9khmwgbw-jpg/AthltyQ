from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless=new")
driver = webdriver.Chrome(options=options)

try:
    script = """
        const r = await fetch('https://httpbin.org/json');
        return await r.json();
    """
    res = driver.execute_script(script)
    print("Result:", res)
except Exception as e:
    print("Exception:", e)

driver.quit()
