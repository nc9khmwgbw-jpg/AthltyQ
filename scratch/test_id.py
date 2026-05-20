from curl_cffi import requests
res = requests.get("https://api.sofascore.com/api/v1/unique-tournament/670/seasons", impersonate="chrome120")
print(res.json().get('seasons', [])[0] if res.status_code == 200 else "Error")
