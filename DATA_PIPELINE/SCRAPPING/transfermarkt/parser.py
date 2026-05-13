from bs4 import BeautifulSoup
class TMParser:
    @staticmethod
    def parse_injuries(html):
        soup = BeautifulSoup(html, 'html.parser')
        return [] # Logique BeautifulSoup
