import json
from pathlib import Path
class CacheManager:
    def __init__(self, cache_dir="DATA_PIPELINE/SCRAPPING/.cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    def get(self, key):
        path = self.cache_dir / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None
    def set(self, key, data):
        (self.cache_dir / f"{key}.json").write_text(json.dumps(data))
