from curl_cffi import requests
import json

url = "https://api.sofascore.com/api/v1/unique-tournament/18/seasons"
headers = {
    'Accept': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Origin': 'https://www.sofascore.com',
    'Referer': 'https://www.sofascore.com/'
}

# Use impersonate='chrome120' to mimic real Chrome TLS fingerprint
res = requests.get(url, headers=headers, impersonate="chrome120")
print("Status:", res.status_code)
try:
    data = res.json()
    if 'seasons' in data:
        print(f"SUCCESS! Found {len(data['seasons'])} seasons via curl_cffi.")
    else:
        print("Data keys:", data.keys())
except Exception as e:
    print("Error:", e)
    print(res.text[:200])
