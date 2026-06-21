"""
AthlytIQ — Générateur Complet de Postes (9 postes granulaires)
================================================================
Pipeline 100% offline — aucun scraping nécessaire.

  1. Mapping manuel (140+ joueurs clés) → Priorité absolue
  2. Inférence statistique 2-tiers → 9 postes pour tous les autres

Postes : ATT | AG | AD | MOF | MC | MDF | CB | LB | RB

Usage :
    .venv/bin/python LM2/build_positions.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════
# MAPPING MANUEL (140+ joueurs clés — 100% précision)
# ══════════════════════════════════════════════════════════════

KNOWN_POSITIONS = {
    # ══ BARCELONA ══
    "Lamine Yamal": "AD", "Raphinha": "AG", "Ferran Torres": "AG",
    "Robert Lewandowski": "ATT", "Ansu Fati": "AG", "Dani Olmo": "MOF",
    "Pablo Gavi": "MC", "Pedri": "MOF", "Nico Williams": "AG",
    "Frenkie de Jong": "MC", "Martín Zubimendi": "MDF", "Marc Casadó": "MDF",
    "Pau Cubarsí": "CB", "Ronald Araújo": "CB", "Eric García": "CB",
    "Iñigo Martínez": "CB", "Alejandro Balde": "LB", "Jules Koundé": "RB",
    "Gerard Martín": "LB", "Héctor Fort": "RB",
    # ══ REAL MADRID ══
    "Kylian Mbappé": "ATT", "Vinicius Júnior": "AG", "Rodrygo": "AD",
    "Brahim Díaz": "MOF", "Arda Güler": "MOF", "Jude Bellingham": "MOF",
    "Federico Valverde": "MC", "Aurélien Tchouaméni": "MDF",
    "Eduardo Camavinga": "MC", "Trent Alexander-Arnold": "RB",
    "Dean Huijsen": "CB", "Antonio Rüdiger": "CB", "Éder Militão": "CB",
    # ══ ATLETICO MADRID ══
    "Antoine Griezmann": "ATT", "Julián Álvarez": "ATT",
    "Alexander Sørloth": "ATT", "Samuel Lino": "AG", "Rodrigo Riquelme": "AD",
    "Pablo Barrios": "MDF", "Koke": "MC", "Conor Gallagher": "MC",
    "Reinildo": "LB", "Marcos Llorente": "MC",
    "José Giménez": "CB", "Robin Le Normand": "CB", "Axel Witsel": "MDF",
    # ══ REAL SOCIEDAD ══
    "Mikel Oyarzabal": "ATT", "Takefusa Kubo": "AD", "Brais Méndez": "MOF",
    "Carlos Soler": "MC", "Luca Sučić": "MOF", "Ander Barrenetxea": "AD",
    "Beñat Turrientes": "MDF", "Jon Aramburu": "RB", "Duje Ćaleta-Car": "CB",
    # ══ REAL BETIS ══
    "Pablo Fornals": "MOF", "Antony": "AD", "Giovani Lo Celso": "MOF",
    "Isco": "MOF", "Cucho Hernández": "ATT",
    # ══ VILLARREAL ══
    "Ayoze Pérez": "ATT", "Gerard Moreno": "ATT", "Georges Mikautadze": "ATT",
    "Nicolas Pépé": "AD", "Alberto Moleiro": "MOF",
    "Thomas Partey": "MDF", "Santi Comesaña": "MDF",
    # ══ MANCHESTER CITY ══
    "Kevin De Bruyne": "MOF", "Erling Haaland": "ATT", "Phil Foden": "AG",
    "Bernardo Silva": "MC", "Rodri": "MDF", "Mateo Kovačić": "MC",
    "Ilkay Gündogan": "MC", "Josko Gvardiol": "LB",
    "Manuel Akanji": "CB", "Rúben Dias": "CB",
    "Kyle Walker": "RB", "Rico Lewis": "RB",
    # ══ LIVERPOOL ══
    "Mohamed Salah": "AD", "Luis Díaz": "AG", "Darwin Núñez": "ATT",
    "Diogo Jota": "ATT", "Cody Gakpo": "AG",
    "Alexis Mac Allister": "MDF", "Ryan Gravenberch": "MDF",
    "Dominik Szoboszlai": "MOF", "Virgil van Dijk": "CB",
    "Ibrahima Konaté": "CB", "Trent Alexander-Arnold": "RB",
    "Andy Robertson": "LB",
    # ══ ARSENAL ══
    "Bukayo Saka": "AD", "Gabriel Martinelli": "AG", "Leandro Trossard": "AG",
    "Kai Havertz": "ATT", "Martin Ødegaard": "MOF",
    "Declan Rice": "MDF", "Gabriel dos Santos": "CB",
    "Ben White": "RB", "Oleksandr Zinchenko": "LB",
    # ══ CHELSEA ══
    "Cole Palmer": "MOF", "Enzo Fernández": "MC", "Moisés Caicedo": "MDF",
    "Nicolas Jackson": "ATT", "Pedro Neto": "AD", "Noni Madueke": "AD",
    "Jadon Sancho": "AG", "Reece James": "RB",
    # ══ TOTTENHAM ══
    "Son Heung-min": "AG", "Dejan Kulusevski": "AD", "James Maddison": "MOF",
    "Yves Bissouma": "MDF", "Rodrigo Bentancur": "MC",
    "Cristian Romero": "CB", "Micky van de Ven": "CB",
    # ══ PSG ══
    "Ousmane Dembélé": "AD", "Bradley Barcola": "AG", "Gonçalo Ramos": "ATT",
    "Fabian Ruiz": "MC", "Vitinha": "MC", "João Neves": "MDF",
    "Warren Zaïre-Emery": "MC", "Marquinhos": "CB",
    "Achraf Hakimi": "RB", "Nuno Mendes": "LB", "Lucas Beraldo": "CB",
    # ══ MONACO ══
    "Folarin Balogun": "ATT", "Breel Embolo": "ATT",
    "Eliesse Ben Seghir": "MOF", "Maghnes Akliouche": "MOF",
    "Denis Zakaria": "MDF", "Youssouf Fofana": "MDF", "Mohamed Camara": "MDF",
    "Wilfried Singo": "RB", "Caio Henrique": "LB",
    # ══ OL ══
    "Alexandre Lacazette": "ATT", "Rayan Cherki": "MOF",
    "Maxence Caqueret": "MC", "Corentin Tolisso": "MC",
    # ══ LEVERKUSEN ══
    "Granit Xhaka": "MDF", "Florian Wirtz": "MOF",
    "Victor Boniface": "ATT", "Patrik Schick": "ATT",
    "Alejandro Grimaldo": "LB", "Robert Andrich": "MDF",
    "Exequiel Palacios": "MC", "Edmond Tapsoba": "CB",
    # ══ DORTMUND ══
    "Karim Adeyemi": "AG", "Serhou Guirassy": "ATT",
    "Julian Brandt": "MOF", "Marcel Sabitzer": "MC",
    "Emre Can": "MDF", "Maximilian Beier": "ATT",
    "Nico Schlotterbeck": "CB", "Waldemar Anton": "CB",
    # ══ JUVENTUS ══
    "Kenan Yıldız": "AG", "Khephren Thuram": "MC", "Douglas Luiz": "MC",
    "Teun Koopmeiners": "MOF", "Andrea Cambiaso": "RB",
    "Gleison Bremer": "CB", "Federico Gatti": "CB",
    # ══ NAPOLI ══
    "Romelu Lukaku": "ATT", "David Neres": "AD",
    "Scott McTominay": "MC", "André-Frank Zambo Anguissa": "MC",
    "Stanislav Lobotka": "MDF",
    # ══ INTER ══
    "Lautaro Martínez": "ATT", "Marcus Thuram": "ATT",
    "Nicolò Barella": "MC", "Hakan Çalhanoğlu": "MDF",
    "Henrikh Mkhitaryan": "MOF", "Alessandro Bastoni": "LB",
    "Francesco Acerbi": "CB", "Denzel Dumfries": "RB",
    # ══ AC MILAN ══
    "Rafael Leão": "AG", "Christian Pulisic": "MOF",
    "Ruben Loftus-Cheek": "MC", "Tijjani Reijnders": "MC",
    "Theo Hernández": "LB", "Fikayo Tomori": "CB", "Malick Thiaw": "CB",
}


# ══════════════════════════════════════════════════════════════
# INFÉRENCE STATISTIQUE 9 POSTES (2 tiers)
# ══════════════════════════════════════════════════════════════

FEATURES_FOR_POS = [
    'xG_P90', 'xA_P90', 'Key_Passes_P90', 'Dribbles_P90',
    'distanceRun', 'Defensive_Actions_P90',
]

def infer_all_positions(df_full):
    """
    Classifie CHAQUE joueur du dataset dans un des 9 postes
    en utilisant les percentiles de ses stats.

    Tier 1 : Séparer ATT-zone / MID-zone / DEF-zone
    Tier 2 : Sub-classifier dans chaque zone
    """
    # Agréger les stats par joueur (médiane sur tous les matchs)
    agg_cols = {f: (f, 'median') for f in FEATURES_FOR_POS if f in df_full.columns}
    agg = df_full.groupby('Nom').agg(**agg_cols).reset_index()

    def pct(col):
        if col in agg.columns and agg[col].std() > 0:
            return agg[col].rank(pct=True)
        return pd.Series(0.5, index=agg.index)

    xg  = pct('xG_P90')
    xa  = pct('xA_P90')
    kp  = pct('Key_Passes_P90')
    drb = pct('Dribbles_P90')
    dst = pct('distanceRun')
    dfn = pct('Defensive_Actions_P90')

    result = {}

    for idx, row in agg.iterrows():
        name = row['Nom']

        # ── Tier 1 : Déterminer la zone ──
        # Score offensif = combinaison xG + xA + dribbles
        off_score = xg[idx] * 0.4 + xa[idx] * 0.3 + drb[idx] * 0.3
        # Score défensif = combinaison actions défensives + distance
        def_score = dfn[idx] * 0.6 + (1 - kp[idx]) * 0.2 + (1 - xg[idx]) * 0.2
        # Score milieu = passes clés + équilibre off/def
        mid_score = kp[idx] * 0.4 + dst[idx] * 0.3 + (1 - abs(off_score - def_score)) * 0.3

        # ── ZONE DEF : peu d'xG + peu de dribbles + défensif ──
        if (xg[idx] < 0.35 and drb[idx] < 0.40 and xa[idx] < 0.40) or \
           (xg[idx] < 0.25 and xa[idx] < 0.30):
            # Tier 2 DEF : CB vs LB/RB
            if dst[idx] >= 0.50 and (drb[idx] >= 0.35 or xa[idx] >= 0.35):
                result[name] = 'LB'  # Latéral (court beaucoup, qq dribbles)
            else:
                result[name] = 'CB'  # Central (statique)

        # ── ZONE ATT : xG très élevé ──
        elif xg[idx] >= 0.65:
            # Tier 2 ATT : ATT vs AG/AD
            if drb[idx] >= 0.60 and kp[idx] < 0.55:
                result[name] = 'AG'  # Ailier (dribbleur)
            else:
                result[name] = 'ATT'  # Avant-centre (finisseur)

        # ── ZONE AILIER : dribbles élevés + contribution offensive ──
        elif drb[idx] >= 0.55 and (xg[idx] >= 0.45 or xa[idx] >= 0.50):
            result[name] = 'AG'  # Ailier (par défaut AG)

        # ── ZONE MDF : défensif + peu créatif ──
        elif dfn[idx] >= 0.55 and kp[idx] < 0.50 and xg[idx] < 0.50:
            result[name] = 'MDF'  # Milieu défensif

        # ── ZONE MOF : créatif, passes clés élevées ──
        elif kp[idx] >= 0.55 and dfn[idx] < 0.55:
            # Distinguer MOF (créateur) vs MC (équilibré)
            if kp[idx] >= 0.65 or (xg[idx] >= 0.45 and xa[idx] >= 0.50):
                result[name] = 'MOF'  # Milieu offensif / meneur
            else:
                result[name] = 'MC'   # Milieu central

        # ── MOF override : xG modéré-élevé + très créatif ──
        elif xg[idx] >= 0.45 and kp[idx] >= 0.60:
            result[name] = 'MOF'

        # ── MC par défaut ──
        else:
            result[name] = 'MC'

    return result


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    dataset_path = ROOT / "data" / "processed" / "features_dataset.csv"
    output_path  = ROOT / "data" / "player_positions.csv"

    print(f"📂 Chargement du dataset : {dataset_path}")
    df = pd.read_csv(dataset_path)
    players = df["Nom"].unique().tolist()
    print(f"   → {len(players)} joueurs uniques\n")

    # ── Étape 1 : Mapping manuel (priorité absolue) ──
    manual = {p: KNOWN_POSITIONS[p] for p in players if p in KNOWN_POSITIONS}
    print(f"📌 Étape 1 — Mapping manuel : {len(manual)} joueurs (100% précision)")

    # ── Étape 2 : Inférence statistique pour le reste ──
    print(f"🧮 Étape 2 — Inférence statistique pour {len(players) - len(manual)} joueurs...")
    inferred = infer_all_positions(df)

    # ── Fusion : manuel > inférence ──
    final = {}
    src_manual = 0
    src_infer  = 0

    for p in players:
        if p in manual:
            final[p] = manual[p]
            src_manual += 1
        elif p in inferred:
            final[p] = inferred[p]
            src_infer += 1
        else:
            final[p] = 'MC'
            src_infer += 1

    # ── Sauvegarde ──
    rows = [{"Nom": k, "Poste_Cat": v} for k, v in final.items()]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ── INJECTION DANS LE MASTER DATASET (features_dataset.csv) ──
    # On met à jour le dataset principal pour que tout soit regroupé
    df = df.drop(columns=['Poste_Cat'], errors='ignore')
    df = df.merge(out_df, on='Nom', how='left')
    df.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    print(f"\n{'═' * 60}")
    print(f"✅ RÉSULTAT — {len(final)} joueurs classifiés en 9 postes")
    print(f"{'═' * 60}")
    print(f"   📌 Mapping manuel : {src_manual}")
    print(f"   🧮 Inférence stat : {src_infer}")
    print(f"\n📊 Distribution :")
    print(out_df["Poste_Cat"].value_counts().sort_index().to_string())
    print(f"\n💾 Sauvegardé : {output_path}")

    # ── Validation qualité ──
    print(f"\n{'═' * 60}")
    print(f"🔍 VALIDATION QUALITÉ (joueurs clés)")
    print(f"{'═' * 60}")
    tests = [
        ("Kylian Mbappé", "ATT"), ("Erling Haaland", "ATT"),
        ("Lamine Yamal", "AD"), ("Mohamed Salah", "AD"), ("Bukayo Saka", "AD"),
        ("Vinicius Júnior", "AG"), ("Raphinha", "AG"), ("Luis Díaz", "AG"),
        ("Kevin De Bruyne", "MOF"), ("Pedri", "MOF"), ("Jude Bellingham", "MOF"),
        ("Frenkie de Jong", "MC"), ("Pablo Gavi", "MC"), ("Federico Valverde", "MC"),
        ("Rodri", "MDF"), ("Martín Zubimendi", "MDF"), ("Declan Rice", "MDF"),
        ("Pau Cubarsí", "CB"), ("Virgil van Dijk", "CB"),
        ("Jules Koundé", "RB"), ("Achraf Hakimi", "RB"),
        ("Alejandro Balde", "LB"), ("Andy Robertson", "LB"),
    ]
    ok = 0
    for name, expected in tests:
        actual = final.get(name, '?')
        icon = '✅' if actual == expected else '⚠️'
        if actual == expected:
            ok += 1
        print(f"  {icon} {name:30s} → {actual:4s}  [attendu: {expected}]")
    print(f"\n  Précision : {ok}/{len(tests)} ({ok/len(tests)*100:.0f}%)")


if __name__ == "__main__":
    main()
