from curl_cffi import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
    "X-Requested-With": "XMLHttpRequest"
}

print("Fetching API with curl_cffi...")
res = requests.get("https://www.sofascore.com/api/v1/unique-tournament/54/seasons", headers=headers, impersonate="chrome120")
print("Status:", res.status_code)
print("Response:", res.text[:200])
