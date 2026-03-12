from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


# -----------------------------
# Manual cleanup rules
# -----------------------------

# Entities we want to drop entirely (too noisy / generic / bad extraction)
DROP_ENTITY_LABELS = {
    "Unknown",
    "Beautiful",
    "Fork",
    "Grandmaster",
}

# Manual type corrections for obvious real persons
FORCE_ENTITY_CLASS = {
    "Walter Tevis": "REAL_PERSON",
    "Scott Frank": "REAL_PERSON",
    "Anya Taylor-Joy": "REAL_PERSON",
    "Bill Camp": "REAL_PERSON",
    "Thomas Brodie-Sangster": "REAL_PERSON",
    "Jacob Fortune-Lloyd": "REAL_PERSON",
    "Moses Ingram": "REAL_PERSON",
    "Marielle Heller": "REAL_PERSON",
    "Harry Melling": "REAL_PERSON",
    "Chloe Pirrie": "REAL_PERSON",
    "Millie Brady": "REAL_PERSON",
    "Isla Johnston": "REAL_PERSON",
    "Bruce Pandolfini": "REAL_PERSON",
    "Paul Morphy": "REAL_PERSON",
    "Judit Polgár": "REAL_PERSON",
    "Hou Yifan": "REAL_PERSON",
    "Nona Gaprindashvili": "REAL_PERSON",
}

# Manual alias canonicalization for remaining obvious cases
FORCE_CANONICAL_TEXT = {
    "Shaibel": "William Shaibel",
    "Cloe": "Cleo",
    "Benjamin Watts": "Benny Watts",
    "Elizabeth Harmon": "Beth Harmon",
    'Elizabeth "Beth" Harmon': "Beth Harmon",
    'Elizabeth "Beth" Olivia Harmon': "Beth Harmon",
}

# Explicit bad relation triples to remove
DROP_RELATION_TRIPLES = {
    ("Paul Morphy", "DIVORCED", "Alice Harmon"),
    ("Allston Wheatley", "LEARNED_FROM", "Beth Harmon"),
    ("Beth Harmon", "LOST_TO", "Najdorf"),
}

# Very noisy object labels we never want in person-vs-person relations
BAD_RELATION_OBJECTS = {
    "Unknown",
    "Beautiful",
    "Fork",
    "Grandmaster",
}

DROP_ENTITY_LABELS_NORM = {x.lower().strip() for x in DROP_ENTITY_LABELS}
BAD_RELATION_OBJECTS_NORM = {x.lower().strip() for x in BAD_RELATION_OBJECTS}

# Human-only relations
HUMAN_ONLY_RELATIONS = {
    "MET",
    "HELPED",
    "SUPPORTED",
    "BEFRIENDED",
    "LOVED",
    "MARRIED",
    "DIVORCED",
    "TRAINED_BY",
    "TAUGHT_BY",
    "COACHED_BY",
    "LEARNED_FROM",
    "PLAYED",
    "DEFEATED",
    "LOST_TO",
    "DREW_WITH",
}

HUMAN_CLASSES = {"CHARACTER", "REAL_PERSON"}

EVENT_ONLY_RELATIONS = {"WON", "LOST", "ENTERED", "COMPETED_IN"}
EVENT_CLASSES = {"EVENT", "WORK", "TOURNAMENT", "GAME", "EPISODE", "BOOK"}


# -----------------------------
# Helpers
# -----------------------------

def normalize_text(s: str) -> str:
    return str(s).strip()


def apply_entity_fixes(ent_df: pd.DataFrame) -> pd.DataFrame:
    ent_df = ent_df.copy()

    if "canonical_text" not in ent_df.columns:
        raise ValueError("entities CSV must contain column 'canonical_text'")

    if "entity_class" not in ent_df.columns:
        raise ValueError("entities CSV must contain column 'entity_class'")

    # Drop obvious junk
    ent_df = ent_df[~ent_df["canonical_text"].astype(str).str.strip().str.lower().isin(DROP_ENTITY_LABELS_NORM)].copy()

    # Force canonical text
    ent_df["canonical_text"] = ent_df["canonical_text"].apply(
        lambda x: FORCE_CANONICAL_TEXT.get(normalize_text(x), normalize_text(x))
    )

    # Force class where obvious
    ent_df["entity_class"] = ent_df.apply(
        lambda r: FORCE_ENTITY_CLASS.get(normalize_text(r["canonical_text"]), r["entity_class"]),
        axis=1,
    )

    # Recompute entity_id if present, based on canonical_text + entity_class
    if "entity_id" in ent_df.columns:
        ent_df["entity_id"] = ent_df.apply(
            lambda r: make_canonical_entity_id(r["entity_class"], r["canonical_text"]),
            axis=1,
        )

    # Re-aggregate after text corrections
    agg_dict = {}
    for col in ent_df.columns:
        if col in {"entity_class", "canonical_text"}:
            continue
        if col == "entity_id":
            agg_dict[col] = "first"
        elif col in {"total_count", "mention_count"}:
            agg_dict[col] = "sum"
        elif col == "urls":
            agg_dict[col] = merge_pipe_values
        elif col == "aliases":
            agg_dict[col] = merge_pipe_values
        elif col == "url":
            agg_dict[col] = "first"
        else:
            agg_dict[col] = "first"

    ent_df = (
        ent_df.groupby(["entity_class", "canonical_text"], as_index=False)
        .agg(agg_dict)
        .copy()
    )

    # Ensure entity_id exists
    if "entity_id" not in ent_df.columns:
        ent_df["entity_id"] = ent_df.apply(
            lambda r: make_canonical_entity_id(r["entity_class"], r["canonical_text"]),
            axis=1,
        )

    return ent_df


def merge_pipe_values(series: pd.Series) -> str:
    vals = []
    for v in series.dropna().astype(str):
        vals.extend([p.strip() for p in v.split("|") if p.strip()])
    return "|".join(sorted(set(vals)))


def make_canonical_entity_id(entity_class: str, canonical_text: str) -> str:
    import hashlib
    import re

    def slug(s: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
        return s or "unknown"

    h = hashlib.sha1(f"{entity_class}:{canonical_text}".encode("utf-8")).hexdigest()[:6]
    return f"{entity_class}:{slug(canonical_text)}-{h}"


def build_entity_lookup(ent_df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """
    canonical_text -> (entity_id, entity_class)
    """
    lookup = {}
    for _, row in ent_df.iterrows():
        lookup[normalize_text(row["canonical_text"])] = (
            row["entity_id"],
            row["entity_class"],
        )
    return lookup


def apply_relation_fixes(rel_df: pd.DataFrame, entity_lookup: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rel_df = rel_df.copy()

    required = {"subject", "relation", "object"}
    missing = required - set(rel_df.columns)
    if missing:
        raise ValueError(f"relations CSV missing columns: {sorted(missing)}")

    # Force canonical labels on subject/object if known
    def resolve_text(x: str) -> str:
        x = FORCE_CANONICAL_TEXT.get(normalize_text(x), normalize_text(x))
        return x

    rel_df["subject"] = rel_df["subject"].apply(resolve_text)
    rel_df["object"] = rel_df["object"].apply(resolve_text)

    # Update IDs and classes from cleaned entity table where possible
    def update_side(text_col: str, id_col: str, class_col: str):
        if id_col not in rel_df.columns:
            rel_df[id_col] = ""
        if class_col not in rel_df.columns:
            rel_df[class_col] = ""

        new_ids = []
        new_classes = []
        for txt, old_id, old_class in zip(rel_df[text_col], rel_df[id_col], rel_df[class_col]):
            hit = entity_lookup.get(normalize_text(txt))
            if hit:
                new_ids.append(hit[0])
                new_classes.append(hit[1])
            else:
                new_ids.append(old_id)
                new_classes.append(old_class)
        rel_df[id_col] = new_ids
        rel_df[class_col] = new_classes

    update_side("subject", "subject_id", "subject_class")
    update_side("object", "object_id", "object_class")

    # Drop relations involving dropped/noisy labels
    rel_df = rel_df[
    ~rel_df["subject"].astype(str).str.strip().str.lower().isin(DROP_ENTITY_LABELS_NORM)
    & ~rel_df["object"].astype(str).str.strip().str.lower().isin(DROP_ENTITY_LABELS_NORM)
    & ~rel_df["object"].astype(str).str.strip().str.lower().isin(BAD_RELATION_OBJECTS_NORM)].copy()

    # Drop explicit bad triples
    rel_df = rel_df[
        ~rel_df.apply(
            lambda r: (normalize_text(r["subject"]), normalize_text(r["relation"]), normalize_text(r["object"])) in DROP_RELATION_TRIPLES,
            axis=1,
        )
    ].copy()

    # Remove self-loops
    if "subject_id" in rel_df.columns and "object_id" in rel_df.columns:
        rel_df = rel_df[rel_df["subject_id"] != rel_df["object_id"]].copy()
    else:
        rel_df = rel_df[rel_df["subject"] != rel_df["object"]].copy()

    # Enforce class constraints if class columns exist
    if "object_class" in rel_df.columns:
        rel_df["object_class"] = rel_df["object_class"].astype(str)
    if "subject_class" in rel_df.columns:
        rel_df["subject_class"] = rel_df["subject_class"].astype(str)

    if {"subject_class", "object_class"}.issubset(rel_df.columns):
        rel_df = rel_df[
            ~(
                rel_df["relation"].isin(HUMAN_ONLY_RELATIONS)
                & ~rel_df["subject_class"].isin(HUMAN_CLASSES)
            )
        ].copy()

        rel_df = rel_df[
            ~(
                rel_df["relation"].isin(HUMAN_ONLY_RELATIONS)
                & ~rel_df["object_class"].isin(HUMAN_CLASSES)
            )
        ].copy()

        rel_df = rel_df[
            ~(
                rel_df["relation"].isin(EVENT_ONLY_RELATIONS)
                & ~rel_df["object_class"].isin(EVENT_CLASSES)
            )
        ].copy()

    # Deduplicate relation rows
    subset = [c for c in ["subject_id", "relation", "object_id"] if c in rel_df.columns]
    if len(subset) < 3:
        subset = ["subject", "relation", "object"]

    rel_df = rel_df.drop_duplicates(subset=subset).copy()

    return rel_df


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean entities/relations before RDF export.")
    parser.add_argument("--entities-in", default="data/entities_dedup.csv")
    parser.add_argument("--relations-in", default="data/relations.csv")
    parser.add_argument("--entities-out", default="data/entities_clean.csv")
    parser.add_argument("--relations-out", default="data/relations_clean.csv")
    args = parser.parse_args()

    ent_df = pd.read_csv(args.entities_in)
    rel_df = pd.read_csv(args.relations_in)

    ent_clean = apply_entity_fixes(ent_df)
    entity_lookup = build_entity_lookup(ent_clean)
    rel_clean = apply_relation_fixes(rel_df, entity_lookup)

    ent_clean = ent_clean.sort_values(["entity_class", "canonical_text"]).reset_index(drop=True)
    rel_clean = rel_clean.sort_values(["relation", "subject", "object"]).reset_index(drop=True)

    Path(args.entities_out).parent.mkdir(parents=True, exist_ok=True)
    ent_clean.to_csv(args.entities_out, index=False)
    rel_clean.to_csv(args.relations_out, index=False)

    print(f"Wrote cleaned entities: {args.entities_out}")
    print(f"Wrote cleaned relations: {args.relations_out}")
    print(f"Entities: {len(ent_df)} -> {len(ent_clean)}")
    print(f"Relations: {len(rel_df)} -> {len(rel_clean)}")


if __name__ == "__main__":
    main()