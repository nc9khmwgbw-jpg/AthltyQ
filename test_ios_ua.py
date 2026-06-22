from curl_cffi import requests

headers_ios = {
    "User-Agent": "SofaScore/12.3.4 CFNetwork/1406.0.4 Darwin/22.4.0",
    "Accept-Encoding": "gzip",
    "Host": "api.sofascore.com",
    "Connection": "Keep-Alive"
}

r = requests.get("https://api.sofascore.com/api/v1/unique-tournament/54/seasons", impersonate="safari15_5", headers=headers_ios)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print(r.json().keys())
else:
    print(r.text[:200])
