from curl_cffi import requests

def test_impersonate(impersonate_val, ua_val=None):
    print(f"\n--- Testing impersonate={impersonate_val} ---")
    headers = {
        "Accept-Encoding": "gzip",
        "Host": "api.sofascore.com",
        "Connection": "Keep-Alive"
    }
    if ua_val:
        headers["User-Agent"] = ua_val

    try:
        r = requests.get("https://api.sofascore.com/api/v1/unique-tournament/54/seasons", impersonate=impersonate_val, headers=headers)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("SUCCESS!")
        else:
            print(f"Error: {r.text[:50]}")
    except Exception as e:
        print(f"Exception: {e}")

# Test Dalvik with different TLS fingerprints
test_impersonate("chrome120", "Dalvik/2.1.0 (Linux; U; Android 13; Pixel 7 Build/TQ3A.230705.001)")
test_impersonate("safari15_5", "Dalvik/2.1.0 (Linux; U; Android 14; SM-S918B Build/UP1A.231005.007)")

# Test actual Mobile Chrome User-Agents instead of Dalvik
test_impersonate("chrome120", "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.43 Mobile Safari/537.36")
test_impersonate("safari15_5", "Mozilla/5.0 (iPhone; CPU iPhone OS 15_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 Mobile/15E148 Safari/604.1")

# Test standard Desktop Chrome again
test_impersonate("chrome124")
