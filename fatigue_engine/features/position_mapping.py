"""Player position mapping for Dataset_Fatigue_V3."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

UNKNOWN_POSITION = "Unknown"
UNKNOWN_VALUES = {"", "nan", "none", "unknown", "na", "n/a"}


def normalize_name(value: object) -> str:
    """Normalize player names for cross-source joins."""

    if not isinstance(value, str):
        return ""
    normalized = "".join(
        char
        for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()


def _is_known_position(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() not in UNKNOWN_VALUES


def _mode_or_first(values: pd.Series) -> str:
    mode = values.dropna().astype(str).mode()
    if not mode.empty:
        return str(mode.iloc[0])
    return str(values.dropna().astype(str).iloc[0])


def build_injury_position_maps(injury_history_path: Path) -> tuple[pd.DataFrame, dict[str, str], set[str]]:
    """Build a normalized-name -> position map from Transfermarkt injury history."""

    if not injury_history_path.exists():
        empty = pd.DataFrame(columns=["position_key", "positions", "position_count"])
        return empty, {}, set()

    injury = pd.read_csv(injury_history_path, low_memory=False)
    required = {"Nom", "Position"}
    if not required.issubset(injury.columns):
        empty = pd.DataFrame(columns=["position_key", "positions", "position_count"])
        return empty, {}, set()

    injury = injury.copy()
    injury["position_key"] = injury["Nom"].map(normalize_name)
    injury["Position"] = injury["Position"].astype(str).str.strip()
    injury = injury[injury["position_key"].ne("") & injury["Position"].map(_is_known_position)]

    ambiguity = (
        injury.groupby("position_key")["Position"]
        .agg(
            positions=lambda s: ", ".join(sorted(set(s.astype(str)))),
            position_count=lambda s: int(s.astype(str).nunique()),
        )
        .reset_index()
    )
    ambiguous_keys = set(ambiguity.loc[ambiguity["position_count"] > 1, "position_key"])

    reliable = injury[~injury["position_key"].isin(ambiguous_keys)]
    mapping = reliable.groupby("position_key")["Position"].agg(_mode_or_first).to_dict()
    return ambiguity, mapping, ambiguous_keys


def apply_position_mapping(
    df: pd.DataFrame,
    injury_history_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add reliable player positions from the best available local source."""

    out = df.copy()
    before_known = (
        out["Position"].map(_is_known_position)
        if "Position" in out.columns
        else pd.Series(False, index=out.index)
    )

    if "Position" not in out.columns:
        out["Position"] = UNKNOWN_POSITION
    out["Position"] = out["Position"].where(before_known, UNKNOWN_POSITION)
    out["Position_Source"] = "unknown"
    out.loc[before_known, "Position_Source"] = "input_dataset"

    out["position_key"] = out["Nom"].map(normalize_name)
    ambiguity, injury_map, ambiguous_keys = build_injury_position_maps(injury_history_path)
    out["Position_Ambiguous"] = out["position_key"].isin(ambiguous_keys)

    needs_mapping = ~before_known
    mapped = out.loc[needs_mapping, "position_key"].map(injury_map)
    has_mapped_position = mapped.map(_is_known_position).fillna(False)
    mapped_index = mapped[has_mapped_position].index

    out.loc[mapped_index, "Position"] = mapped.loc[mapped_index]
    out.loc[mapped_index, "Position_Source"] = "injury_history"
    out.loc[out["Position_Ambiguous"] & out["Position"].eq(UNKNOWN_POSITION), "Position_Source"] = (
        "ambiguous_injury_history"
    )

    out = out.drop(columns=["position_key"])
    return out, ambiguity


def write_position_mapping_report(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    ambiguity: pd.DataFrame,
    report_path: Path,
) -> Path:
    """Write a markdown report describing position mapping coverage and risks."""

    report_path.parent.mkdir(parents=True, exist_ok=True)

    player_before = before_df.drop_duplicates("Nom").copy()
    player_after = after_df.drop_duplicates("Nom").copy()
    before_known = (
        player_before["Position"].map(_is_known_position)
        if "Position" in player_before.columns
        else pd.Series(False, index=player_before.index)
    )
    after_known = player_after["Position"].map(_is_known_position)

    total_players = int(player_after["Nom"].nunique())
    found_players = int(after_known.sum())
    unknown_players = int(total_players - found_players)
    coverage = found_players / total_players * 100 if total_players else 0.0

    position_dist = (
        player_after.loc[after_known, "Position"].value_counts().rename_axis("Position").reset_index(name="players")
    )
    source_dist = (
        player_after["Position_Source"].value_counts().rename_axis("Position_Source").reset_index(name="players")
    )
    examples = player_after.loc[
        after_known,
        ["Nom", "Team", "League", "Position", "Position_Source", "Position_Ambiguous"],
    ].head(25)
    unknown_examples = player_after.loc[
        ~after_known,
        ["Nom", "Team", "League", "Position", "Position_Source", "Position_Ambiguous"],
    ].head(25)

    ambiguous = ambiguity[ambiguity["position_count"] > 1].copy()

    lines = [
        "# Position Mapping Report",
        "",
        "## Sources analysées",
        "",
        "- `DATA_PIPELINE/NETTOYAGE/data/dataset_v2_injury.csv`: aucune colonne position fiable.",
        "- `DATA_PIPELINE/NETTOYAGE/data/merged_dataset_clean.csv`: aucune colonne position.",
        "- `DATA_PIPELINE/SCRAPPING/data/raw/transfermarkt/injury_history.csv`: colonne `Position`, source utilisée.",
        "- `DATA_PIPELINE/SCRAPPING/data/raw/sofascore/**/*.csv`: fichiers match/stats sans colonne position exploitable.",
        "",
        "## Stratégie de mapping",
        "",
        "1. Conserver `Position` existante si elle est déjà connue dans le dataset d'entrée.",
        "2. Fallback Transfermarkt `injury_history.csv` via nom joueur normalisé.",
        "3. Exclure les noms homonymes avec plusieurs postes Transfermarkt contradictoires.",
        "4. Fallback `Unknown` si aucune source fiable n'est disponible.",
        "",
        "## Couverture",
        "",
        f"- Joueurs totaux: `{total_players:,}`",
        f"- Joueurs avec position avant mapping: `{int(before_known.sum()):,}`",
        f"- Joueurs avec position après mapping: `{found_players:,}`",
        f"- Joueurs encore Unknown: `{unknown_players:,}`",
        f"- Taux de couverture après mapping: `{coverage:.2f}%`",
        "",
        "## Distribution des sources",
        "",
        source_dist.to_markdown(index=False),
        "",
        "## Distribution des postes",
        "",
        position_dist.to_markdown(index=False),
        "",
        "## Exemples de mappings",
        "",
        examples.to_markdown(index=False),
        "",
        "## Exemples encore Unknown",
        "",
        unknown_examples.to_markdown(index=False),
        "",
        "## Homonymes / positions contradictoires",
        "",
    ]

    if ambiguous.empty:
        lines.append("_Aucun homonyme à positions contradictoires détecté._")
    else:
        lines.append(ambiguous.head(50).to_markdown(index=False))

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
