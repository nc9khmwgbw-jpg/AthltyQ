"""
AthlytIQ — Transfermarkt Injury History Scraper
=================================================
Scrape l'historique complet de blessures pour chaque joueur du dataset.

Pipeline :
  1. Charge les joueurs depuis merged_dataset_clean.csv
  2. Recherche chaque joueur sur Transfermarkt
  3. Extrait son historique de blessures (type, durée, saison)
  4. Classifie la cause (musculaire, ligament, os, etc.)
  5. Sauvegarde incrémentale dans injury_history.csv

Output : DATA_PIPELINE/SCRAPPING/raw/transfermarkt/injury_history.csv
Colonnes : Nom, Team, Season, Injury_Type, Date_From, Date_To,
           Duration_Days, Cause_Category, Transfermarkt_ID

Usage :
  python transfermarkt_injury_scraper.py               # Tous les joueurs
  python transfermarkt_injury_scraper.py --limit 50    # 50 premiers
  python transfermarkt_injury_scraper.py --resume      # Reprend depuis la dernière fois
"""

import sys
import re
import time
import random
import logging
import argparse
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# ══════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

ROOT       = Path(__file__).resolve().parents[3]
INPUT_CSV  = ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"
OUTPUT_DIR = ROOT / "DATA_PIPELINE" / "SCRAPPING" / "raw" / "transfermarkt"
OUTPUT_CSV = OUTPUT_DIR / "injury_history.csv"
CACHE_CSV  = OUTPUT_DIR / "players_processed.txt"   # Joueurs déjà scrapés
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL   = "https://www.transfermarkt.com"
SEARCH_URL = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.transfermarkt.com/",
    "DNT": "1",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "scraper.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("TM_Scraper")


# ══════════════════════════════════════════════════════════════════════
# 1. CLASSIFICATION DES BLESSURES
# ══════════════════════════════════════════════════════════════════════

INJURY_CATEGORIES = {
    "MUSCULAIRE": [
        "muscle", "muscular", "hamstring", "quadricep", "calf", "strain",
        "adductor", "groin", "thigh", "psoas", "ischios", "mollet",
    ],
    "LIGAMENT": [
        "ligament", "acl", "mcl", "pcl", "ankle sprain", "cruciate",
        "entorse", "cheville", "genou", "knee sprain",
    ],
    "TENDON": [
        "tendon", "achilles", "patellar", "tendinitis", "tendinopathy",
        "achille", "rotulien",
    ],
    "OS": [
        "fracture", "bone", "stress fracture", "broken", "metatarsal",
        "tibia", "fibula", "rib", "fractura",
    ],
    "GENOU": [
        "knee", "meniscus", "cartilage", "genou", "ménisque",
        "patella", "kneecap",
    ],
    "DOS_HANCHE": [
        "back", "hip", "spine", "lumbar", "dos", "hanche", "pubis",
        "hernia", "pubalgia",
    ],
    "TETE_COU": [
        "head", "neck", "concussion", "jaw", "shoulder", "collarbone",
        "tête", "cou", "commotion",
    ],
    "MALADIE": [
        "illness", "virus", "covid", "flu", "sick", "maladie",
        "appendix", "gastro",
    ],
    "AUTRE": [],  # fallback
}


def classifier_blessure(injury_text: str) -> str:
    """Classifie une blessure en catégorie standardisée."""
    if not injury_text:
        return "AUTRE"
    text = injury_text.lower()
    for category, keywords in INJURY_CATEGORIES.items():
        if any(kw in text for kw in keywords):
            return category
    return "AUTRE"


# ══════════════════════════════════════════════════════════════════════
# 2. SESSION HTTP AVEC RETRY
# ══════════════════════════════════════════════════════════════════════

def creer_session() -> requests.Session:
    """Crée une session HTTP avec retry automatique."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


def pause(min_s=1.5, max_s=4.0):
    """Pause aléatoire pour éviter le blocage."""
    time.sleep(random.uniform(min_s, max_s))


# ══════════════════════════════════════════════════════════════════════
# 3. RECHERCHE DU JOUEUR SUR TRANSFERMARKT
# ══════════════════════════════════════════════════════════════════════

def rechercher_joueur(session: requests.Session, nom: str) -> dict | None:
    """
    Recherche un joueur sur Transfermarkt.
    Retourne {'id': str, 'name': str, 'url': str, 'team': str} ou None.
    """
    try:
        resp = session.get(
            SEARCH_URL,
            params={"query": nom, "Spieler_page": "0"},
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"Erreur recherche '{nom}': {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Table des joueurs dans les résultats de recherche
    tables = soup.select("div#yw1 table.items")
    if not tables:
        # Essai avec une sélection plus large
        tables = soup.select("table.items")

    for table in tables:
        rows = table.select("tbody tr")
        for row in rows:
            # Lien du joueur
            link = row.select_one("td.hauptlink a[href*='/profil/spieler/']")
            if not link:
                continue

            href = link.get("href", "")
            match = re.search(r"/spieler/(\d+)", href)
            if not match:
                continue

            player_id = match.group(1)
            player_name = link.get_text(strip=True)

            # Équipe actuelle
            team_cell = row.select_one("td.zentriert a[href*='/startseite/verein/']")
            team_name = team_cell.get_text(strip=True) if team_cell else ""

            return {
                "id": player_id,
                "name": player_name,
                "url": f"{BASE_URL}{href}",
                "team": team_name,
            }

    return None


# ══════════════════════════════════════════════════════════════════════
# 4. EXTRACTION DE L'HISTORIQUE DE BLESSURES
# ══════════════════════════════════════════════════════════════════════

def extraire_blessures(session: requests.Session, player_id: str, player_name: str, team: str) -> list[dict]:
    """
    Extrait l'historique complet de blessures depuis la page Transfermarkt.
    URL : /player-name/verletzungen/spieler/{id}
    """
    url = f"{BASE_URL}/player/verletzungen/spieler/{player_id}"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"Erreur blessures '{player_name}' (ID {player_id}): {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    injuries = []

    table = soup.select_one("table.items")
    if not table:
        log.info(f"   ℹ️  {player_name} — Aucun historique de blessures")
        return []

    rows = table.select("tbody tr")
    for row in rows:
        cells = row.select("td")
        if len(cells) < 5:
            continue

        try:
            season      = cells[0].get_text(strip=True)
            injury_type = cells[1].get_text(strip=True)
            date_from   = cells[2].get_text(strip=True)
            date_to     = cells[3].get_text(strip=True)
            duration    = cells[4].get_text(strip=True)

            # Nettoyage de la durée
            duration_days = 0
            dur_match = re.search(r"(\d+)", duration.replace(",", ""))
            if dur_match:
                duration_days = int(dur_match.group(1))

            # Parsing des dates
            def parse_date(d):
                for fmt in ("%b %d, %Y", "%d/%m/%Y", "%d.%m.%Y", "%B %d, %Y"):
                    try:
                        return datetime.strptime(d, fmt).strftime("%Y-%m-%d")
                    except ValueError:
                        continue
                return d  # Retourne la string brute si parsing échoue

            injuries.append({
                "Nom":             player_name,
                "Team":            team,
                "Transfermarkt_ID": player_id,
                "Season":          season,
                "Injury_Type":     injury_type,
                "Date_From":       parse_date(date_from),
                "Date_To":         parse_date(date_to),
                "Duration_Days":   duration_days,
                "Cause_Category":  classifier_blessure(injury_type),
            })
        except Exception:
            continue

    return injuries


# ══════════════════════════════════════════════════════════════════════
# 5. CACHE — JOUEURS DÉJÀ TRAITÉS
# ══════════════════════════════════════════════════════════════════════

def charger_cache() -> set:
    if CACHE_CSV.exists():
        return set(CACHE_CSV.read_text(encoding="utf-8").strip().splitlines())
    return set()


def sauvegarder_cache(nom: str):
    with open(CACHE_CSV, "a", encoding="utf-8") as f:
        f.write(nom + "\n")


# ══════════════════════════════════════════════════════════════════════
# 6. SAUVEGARDE INCRÉMENTALE
# ══════════════════════════════════════════════════════════════════════

def sauvegarder_blessures(rows: list[dict]):
    """Ajoute les nouvelles lignes au CSV de sortie (append)."""
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if OUTPUT_CSV.exists():
        df_new.to_csv(OUTPUT_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════════════
# 7. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def run(limit: int = None, resume: bool = True):
    print("\n" + "═" * 65)
    print("   AthlytIQ — SCRAPER HISTORIQUE BLESSURES (Transfermarkt)")
    print("═" * 65)

    # Chargement des joueurs
    if not INPUT_CSV.exists():
        log.error(f"❌ Dataset introuvable : {INPUT_CSV}")
        log.error("   Exécutez d'abord : python data_cleaner.py")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    joueurs = sorted(df["Nom"].dropna().unique().tolist())

    if limit:
        joueurs = joueurs[:limit]

    # Cache (joueurs déjà traités)
    cache = charger_cache() if resume else set()
    joueurs_restants = [j for j in joueurs if j not in cache]

    print(f"\n   👥 Joueurs total      : {len(joueurs)}")
    print(f"   ✅ Déjà traités      : {len(cache)}")
    print(f"   🔄 À traiter         : {len(joueurs_restants)}")
    print(f"   💾 Output            : {OUTPUT_CSV}\n")

    if not joueurs_restants:
        print("✅ Tous les joueurs ont déjà été traités !")
        return

    session = creer_session()
    stats = {"trouvés": 0, "non_trouvés": 0, "blessures": 0, "erreurs": 0}

    for i, nom in enumerate(joueurs_restants, 1):
        print(f"[{i:>4}/{len(joueurs_restants)}] 🔍 {nom}...", end=" ", flush=True)

        try:
            # 1. Recherche du joueur
            pause(1.5, 3.5)
            joueur = rechercher_joueur(session, nom)

            if not joueur:
                print("❌ Non trouvé sur Transfermarkt")
                stats["non_trouvés"] += 1
                sauvegarder_cache(nom)
                continue

            stats["trouvés"] += 1

            # 2. Extraction des blessures
            pause(1.0, 2.5)
            blessures = extraire_blessures(session, joueur["id"], nom, joueur["team"])

            if blessures:
                sauvegarder_blessures(blessures)
                stats["blessures"] += len(blessures)
                print(f"✅ {len(blessures)} blessure(s) | {joueur['team']}")
            else:
                print(f"🟢 Aucune blessure trouvée | {joueur['team']}")

            sauvegarder_cache(nom)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interruption manuelle. Progression sauvegardée.")
            break
        except Exception as e:
            log.error(f"Erreur inattendue pour '{nom}': {e}")
            stats["erreurs"] += 1
            sauvegarder_cache(nom)
            continue

    # Rapport final
    print("\n" + "═" * 65)
    print("   📊 RAPPORT FINAL")
    print("═" * 65)
    print(f"   ✅ Joueurs trouvés    : {stats['trouvés']}")
    print(f"   ❌ Non trouvés       : {stats['non_trouvés']}")
    print(f"   🏥 Blessures scrapées: {stats['blessures']}")
    print(f"   ⚠️  Erreurs           : {stats['erreurs']}")
    print(f"\n   💾 Fichier sauvegardé : {OUTPUT_CSV}")

    if OUTPUT_CSV.exists():
        df_out = pd.read_csv(OUTPUT_CSV)
        print(f"\n   📈 Statistiques du dataset :")
        print(f"      Lignes totales      : {len(df_out)}")
        print(f"      Joueurs avec bless. : {df_out['Nom'].nunique()}")
        print(f"      Durée moyenne       : {df_out['Duration_Days'].mean():.0f} jours")
        print(f"\n   🔬 Distribution par type :")
        for cat, count in df_out["Cause_Category"].value_counts().items():
            pct = count / len(df_out) * 100
            bar = "█" * int(pct / 3)
            print(f"      {cat:<15} {bar} {count} ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# 8. INTÉGRATION AVEC LE PIPELINE ATHLYTIQ
# ══════════════════════════════════════════════════════════════════════

def charger_historique_pour_pipeline() -> pd.DataFrame:
    """
    Charge l'historique de blessures pour le feature engineering.
    Compatible avec integrer_historique_medical() dans feature_engineering.py.

    Returns:
        DataFrame avec colonnes : Nom, Duration_Days, Cause_Category,
                                  Injury_Count, Injury_Prone_Index
    """
    if not OUTPUT_CSV.exists():
        log.warning("⚠️  Historique de blessures non disponible.")
        log.warning("   Exécutez : python transfermarkt_injury_scraper.py")
        return pd.DataFrame()

    df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")

    # Agrégation par joueur
    agg = df.groupby("Nom").agg(
        Total_Injury_Days=("Duration_Days", "sum"),
        Injury_Count=("Duration_Days", "count"),
        Avg_Injury_Duration=("Duration_Days", "mean"),
        Dominant_Injury_Cause=("Cause_Category", lambda x: x.value_counts().index[0]),
        Last_Injury_Date=("Date_From", "max"),
        Had_ACL=("Cause_Category", lambda x: int("LIGAMENT" in x.values)),
        Had_Muscle=("Cause_Category", lambda x: int("MUSCULAIRE" in x.values)),
    ).reset_index()

    # Score de fragilité normalisé 0-1
    max_days = agg["Total_Injury_Days"].max() if not agg.empty else 1
    agg["Injury_Prone_Index"] = (agg["Total_Injury_Days"] / max(max_days, 1)).clip(0, 1)

    return agg


# ══════════════════════════════════════════════════════════════════════
# 9. ENTRÉE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AthlytIQ — Scraper historique blessures Transfermarkt"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Nombre maximum de joueurs à traiter (défaut: tous)"
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Reprendre depuis la dernière fois (défaut: True)"
    )
    parser.add_argument(
        "--reset", action="store_true", default=False,
        help="Repart de zéro (efface le cache)"
    )
    args = parser.parse_args()

    if args.reset:
        if CACHE_CSV.exists():
            CACHE_CSV.unlink()
        if OUTPUT_CSV.exists():
            OUTPUT_CSV.unlink()
        print("🗑️  Cache et données effacés. Nouveau départ.")

    run(limit=args.limit, resume=not args.reset)
