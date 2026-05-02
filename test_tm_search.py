import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}
resp = requests.get("https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche", params={"query": "Bukayo Saka", "Spieler_page": "0"}, headers=HEADERS)
soup = BeautifulSoup(resp.text, "html.parser")
table = soup.select_one("table.items")
if table:
    rows = table.select("tbody tr")
    for row in rows[:1]:
        cells = row.select("td")
        for i, cell in enumerate(cells):
            print(f"Cell {i}: {cell.get_text(strip=True)}")
