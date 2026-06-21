from curl_cffi import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}

try:
    # Let's try fetching the seasons for Premier League (id 17)
    res = requests.get("https://api.sofascore.com/api/v1/unique-tournament/17/seasons", headers=headers, impersonate="chrome120")
    print(res.status_code)
    print(res.json())
except Exception as e:
    print(e)
