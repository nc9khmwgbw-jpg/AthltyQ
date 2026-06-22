import tls_client

session = tls_client.Session(
    client_identifier="chrome_120",
    random_tls_extension_order=True
)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/"
}

res = session.get("https://www.sofascore.com/api/v1/unique-tournament/54/seasons", headers=headers)
print("Status:", res.status_code)
print("Text:", res.text[:200])
