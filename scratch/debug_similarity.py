import sys
from pathlib import Path
import pandas as pd

# Ajout du root au path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LM2.similarity_engine import SimilarityEngine
from DASHBOARD.backend import get_data

def test_engine():
    print("🚀 Test du moteur de similarité...")
    results, features = get_data()
    
    if features is None:
        print("❌ Erreur: Features non trouvées.")
        return
        
    try:
        engine = SimilarityEngine(features)
        print("✅ Moteur initialisé.")
        
        # Test sur Raphinha
        print("🔍 Recherche pour Raphinha...")
        res = engine.get_similar_players("Raphinha")
        
        if "error" in res:
            print(f"❌ Erreur moteur: {res['error']}")
        else:
            print(f"✅ Succès ! {len(res)} candidats trouvés.")
            for r in res[:3]:
                print(f" - {r['name']} ({r['final_score']}%)")
                
    except Exception as e:
        print(f"🔥 CRASH détecté: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_engine()
