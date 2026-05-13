import pandas as pd
from pathlib import Path
class SofaScoreSaver:
    @staticmethod
    def save_to_csv(df, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False, encoding='utf-8-sig')
