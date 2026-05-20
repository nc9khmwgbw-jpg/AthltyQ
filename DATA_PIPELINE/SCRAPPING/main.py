import sys
import argparse
from pathlib import Path

# Ajout de la RACINE du projet (AthlytIQ) au PATH pour permettre les imports DATA_PIPELINE
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from DATA_PIPELINE.SCRAPPING.sofascore.scrapers.league_scraper import SofaScoreLeagueScraper
from DATA_PIPELINE.SCRAPPING.common.config import LEAGUES
from DATA_PIPELINE.SCRAPPING.common.logger import setup_logger

logger = setup_logger("AthlytIQ-Main")

def display_source_menu():
    print("\n" + "="*60)
    print("      🚀 ATHLYTIQ ENTERPRISE DATA PIPELINE (v2.0) 🚀")
    print("="*60)
    print("Sélectionnez la source de données :")
    print("  1. SOFASCORE (Statistiques de Matchs)")
    print("  2. TRANSFERMARKT (Historique des Blessures)")
    print("  Q. Quitter")
    print("-"*60, flush=True)
    choice = input("\n👉 Votre choix : ").strip().upper()
    return choice

def display_league_menu(source_name):
    leagues_list = list(LEAGUES.keys())
    print(f"\n--- SCRAPPING {source_name} ---")
    print("Sélectionnez la ligue :")
    for i, league_name in enumerate(leagues_list, 1):
        print(f"  {i:2}. {league_name}")
    print("  ALL. Toutes les ligues")
    print("  B.  Retour au menu principal")
    print("-"*60)
    choice = input("\n👉 Votre choix : ").strip().upper()
    return choice, leagues_list

def get_missing_leagues():
    """Identifie les ligues qui n'ont pas encore de dossier dans data/raw/sofascore."""
    base_dir = Path(__file__).resolve().parents[0] # Dossier SCRAPPING
    raw_path = base_dir / "data" / "raw" / "sofascore"
    
    existing = [d.name.lower() for d in raw_path.iterdir() if d.is_dir()] if raw_path.exists() else []
    
    missing = []
    for i, name in enumerate(LEAGUES.keys()):
        if name.lower() not in existing:
            missing.append((i + 1, name))
    return missing

def main():
    parser = argparse.ArgumentParser(description="AthlytIQ Orchestrator")
    parser.add_argument("--source", help="1 or 2")
    parser.add_argument("--league", help="League Number")
    parser.add_argument("--force-update", help="O or N")
    parser.add_argument("--mode", choices=['auto'], help="Mode auto: scrapper les ligues manquantes")
    args = parser.parse_args()

    # --- MODE AUTO (DÉTECTION DES MANQUES) ---
    if args.mode == 'auto':
        missing = get_missing_leagues()
        if not missing:
            print("✅ Toutes les ligues sont déjà présentes dans raw/sofascore.")
        else:
            print(f"🔍 Mode Auto : {len(missing)} ligues manquantes détectées.")
            scraper = SofaScoreLeagueScraper()
            for idx, name in missing:
                print(f"\n🚀 Scraping automatique de : {name}")
                scraper.scrape(name, force_update=False)
            print("\n✅ Mode Auto terminé. Toutes les ligues sont à jour.")
        return

    while True:
        source_choice = args.source if args.source else display_source_menu()
        
        if source_choice == 'Q':
            print("Au revoir ! 👋")
            break
            
        if source_choice in ['1', '2']:
            source_name = "SOFASCORE" if source_choice == '1' else "TRANSFERMARKT"
            
            while True:
                leagues_list = list(LEAGUES.keys())
                if args.source and args.league:
                    league_choice = args.league
                else:
                    league_choice, _ = display_league_menu(source_name)
                
                if league_choice == 'B':
                    break

                if league_choice in ['ALL', 'all']:
                    selected_leagues = leagues_list
                else:
                    try:
                        index = int(league_choice) - 1
                        if 0 <= index < len(leagues_list):
                            selected_leagues = [leagues_list[index]]
                        else:
                            print(f"❌ Numéro invalide : {league_choice}")
                            break
                    except ValueError:
                        print(f"❌ Entrée invalide : {league_choice}")
                        break
                
                # --- GESTION NON-INTERACTIVE DU FORCE UPDATE ---
                if args.force_update:
                    rep = args.force_update.lower()
                elif args.source:
                    # En mode CLI, on ne demande pas, on prend le défaut (non)
                    rep = 'n'
                else:
                    rep = input("🔄 Mettre à jour les joueurs existants ? (o/N) : ").strip().lower()
                
                force_update = (rep == 'o')
                
                for selected_league in selected_leagues:
                    print(f"\n⚡ Lancement de {source_name} pour : {selected_league}")
                    
                    if source_choice == '1':
                        scraper = SofaScoreLeagueScraper()
                        scraper.scrape(selected_league, force_update=force_update)
                    else:
                        from DATA_PIPELINE.SCRAPPING.pipelines.injury_pipeline import InjuryPipeline
                        pipeline = InjuryPipeline()
                        pipeline.run(force_update=force_update, league_filter=selected_league)
                    
                    print(f"✅ TERMINÉ : {selected_league} est à jour.")
                
                if not args.source:
                    input("\nAppuyez sur Entrée pour continuer...")
                
                if args.source: break # Sortie si mode CLI
        else:
            print("❌ Choix inconnu.")
        
        if args.source: break # Sortie si mode CLI

if __name__ == "__main__":
    main()
