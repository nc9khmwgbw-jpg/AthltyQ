import requests

url1 = "https://api.sofascore.com/api/v1/unique-tournament/18/seasons"
url2 = "https://api.sofascore.com/api/v1/team/42/players"

res1 = requests.get(url1, headers={'User-Agent': 'Mozilla/5.0'})
print("Seasons endpoint:", res1.status_code)

res2 = requests.get(url2, headers={'User-Agent': 'Mozilla/5.0'})
print("Players endpoint:", res2.status_code)

