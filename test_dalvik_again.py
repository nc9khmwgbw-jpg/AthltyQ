from curl_cffi import requests
import json

headers = {
    "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 12; Pixel 6 Build/SQ3A.220705.004)",
    "Accept-Encoding": "gzip",
    "Host": "api.sofascore.com",
    "Connection": "Keep-Alive"
}

def test_endpoint(name, url):
    print(f"\n--- {name} ---")
    try:
        r = requests.get(url, impersonate="chrome110", headers=headers)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print(f"Success! Keys: {list(r.json().keys())}")
        else:
            print(r.text[:200])
    except Exception as e:
        print(f"Error: {e}")

test_endpoint("Teams (LaLiga2)", "https://api.sofascore.com/api/v1/unique-tournament/54/seasons")
test_endpoint("Teams (UAE)", "https://api.sofascore.com/api/v1/unique-tournament/1322/seasons")
