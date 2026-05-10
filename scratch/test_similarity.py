import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LM2.similarity_engine import SimilarityEngine

# Load data
df = pd.read_csv(ROOT / "data" / "processed" / "features_dataset.csv")

# Init engine
engine = SimilarityEngine(df)

# Test search
player_name = df['Nom'].iloc[0]
print(f"Testing similarity for: {player_name}")
results = engine.get_similar_players(player_name, alpha=0.5)

print(f"Found {len(results)} similar players.")
for r in results[:3]:
    print(f"- {r['name']} ({r['final_score']}%)")
