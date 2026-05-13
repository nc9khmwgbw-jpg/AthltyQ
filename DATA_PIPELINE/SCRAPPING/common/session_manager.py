import requests
from DATA_PIPELINE.SCRAPPING.common.headers import get_random_headers
class SessionManager:
    def __init__(self):
        self.session = requests.Session()
    def get_session(self):
        self.session.headers.update(get_random_headers())
        return self.session
