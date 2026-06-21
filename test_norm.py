import unicodedata
import re

def normalize_team_name(name: str) -> str:
    n = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    n = re.sub(r'[^a-zA-Z0-9]', '', n)
    return n.lower()

print(normalize_team_name("1 Fc Koln") in normalize_team_name("1. FC Köln"))
print(normalize_team_name("Vfl Wolfsburg") in normalize_team_name("Wolfsburg"))
print(normalize_team_name("Wolfsburg") in normalize_team_name("Vfl Wolfsburg"))
