from curl_cffi import requests
from bs4 import BeautifulSoup
import json

r = requests.get("https://www.sofascore.com/tournament/football/united-arab-emirates/uae-pro-league/1322", impersonate="chrome124")
print(f"Status: {r.status_code}")

if r.status_code == 200:
    soup = BeautifulSoup(r.text, 'html.parser')
    script = soup.find('script', id='__NEXT_DATA__')
    if script:
        data = json.loads(script.string)
        print("Found __NEXT_DATA__ keys:")
        print(list(data.keys()))
        props = data.get('props', {}).get('pageProps', {})
        print("PageProps keys:")
        print(list(props.keys()))
        
        # Check if we can find seasons
        if 'tournament' in props:
            print("Tournament found in props.")
        
        if 'initialSeasons' in props:
            seasons = props['initialSeasons']
            print(f"Found {len(seasons)} seasons!")
            print(seasons[0])
    else:
        print("No __NEXT_DATA__ script found.")
