#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

QG = Namespace("http://example.org/qg#")
WD = Namespace("http://www.wikidata.org/entity/")

# -------------------------------------------------------------------
# Global reject patterns
# -------------------------------------------------------------------
ALWAYS_REJECT_DESC_CONTAINS = {
    "given name",
    "male given name",
    "female given name",
    "family name",
    "disambiguation page",
    "wikimedia disambiguation page",
}

# -------------------------------------------------------------------
# Manual accept list
# -------------------------------------------------------------------
MANUAL_ACCEPT_LABELS = {
    "Beth Harmon",
    "The Queen's Gambit",
    "The Queen's Gambit (novel)",
    "Openings",
    "Exchanges",
    "Middle Game",
    "Doubled Pawns",
    "Fork",
    "Adjournment",
    "End Game",
    "Anya Taylor-Joy",
    "Scott Frank",
    "Walter Tevis",
    "Netflix",
}

# -------------------------------------------------------------------
# Manual reject controls
# Add bad links here as you discover them.
# -------------------------------------------------------------------
MANUAL_REJECT_QIDS = {
    "Q29540143",   # Allan Scott professor
    "Q112843087",  # Charles Levy radio broadcaster
    "Q76067418",   # Crawford Walker peerage person
    "Q109420063",  # Diana Lanni chess player
    "Q5272895",    # Dick Evans footballer
    "Q241732",     # Rosa Bonheur painter
    "Q28736316",   # Rudolph from Bon Voyage
    "Q115918608",  # Wolff from Fire Emblem
    "Q119143457",  # W. B. Yeats painting
    "Q21856367",   # The White House painting by Soutine
    "Q830149",     # Paris, Texas
    "Q961598",     # Lucerne village in Missouri
    "Q959584",     # Nevada city in Missouri
    "Q928044",     # Wakefield village in Quebec
}

MANUAL_REJECT_PRIVATE_LABELS = {
    # use only if you really never want these auto-links accepted
}

MANUAL_REJECT_WD_LABELS = {
    # optional exact bad wd labels
}

MANUAL_REJECT_DESC_CONTAINS = {
    "footballer",
    "radio broadcaster",
    "peerage person",
    "painting by",
    "painter",
    "artist (",
    "researcher",
    "professor",
    "fire emblem",
    "ninja turtles",
    "bon voyage",
}

# Pair-level rejects: (private_label, wikidata_id)
MANUAL_REJECT_PAIRS: Set[Tuple[str, str]] = {
    ("Allan Scott", "Q29540143"),
    ("Charles Levy", "Q112843087"),
    ("Crawford Walker", "Q76067418"),
    ("Diana Lanni", "Q109420063"),
    ("Dick Evans", "Q5272895"),
    ("Rosa Bonheur", "Q241732"),
    ("Rudolph", "Q28736316"),
    ("Wolff", "Q115918608"),
    ("W. B. Yeats", "Q119143457"),
    ("the White House", "Q21856367"),
    ("Paris", "Q830149"),
    ("Lucerne", "Q961598"),
    ("Nevada", "Q959584"),
    ("Wakefield", "Q928044"),
}

# -------------------------------------------------------------------
# Type-specific rules
# -------------------------------------------------------------------
TYPE_RULES = {
    "Character": {
        "min_conf": 0.78,
        "good": {
            "fictional character",
            "character from",
            "character in",
            "queen's gambit",
        },
        "bad": {
            "actor",
            "actress",
            "city",
            "country",
            "television episode",
            "film",
            "given name",
            "footballer",
            "painter",
            "artist",
            "radio broadcaster",
            "peerage",
            "professor",
            "researcher",
            "scientist",
            "chess player",
        },
    },
    "RealPerson": {
        "min_conf": 0.76,
        "good": {
            "human",
            "actor",
            "actress",
            "writer",
            "director",
            "screenwriter",
            "producer",
            "chess player",
            "author",
        },
        "bad": {
            "fictional character",
            "television episode",
            "television series",
            "novel",
            "painting by",
            "city",
            "country",
        },
    },
    "Organization": {
        "min_conf": 0.80,
        "good": {
            "organization",
            "company",
            "publisher",
            "club",
            "federation",
            "streaming",
            "newspaper",
            "network",
            "magazine",
            "university",
            "high school",
            "school",
            "periodical",
        },
        "bad": {
            "album",
            "song",
            "person",
            "actor",
            "city",
            "country",
            "episode",
            "painting",
            "film",
            "aircraft",
            "fictional character",
        },
    },
    "Location": {
        "min_conf": 0.82,
        "good": {
            "city",
            "country",
            "state",
            "region",
            "town",
            "capital",
            "village",
            "municipality",
            "county seat",
        },
        "bad": {
            "person",
            "actor",
            "organization",
            "company",
            "fictional character",
            "episode",
            "painting",
            "film",
        },
    },
    "Event": {
        "min_conf": 0.78,
        "good": {
            "event",
            "competition",
            "championship",
            "tournament",
            "match",
            "olympiad",
        },
        "bad": {
            "person",
            "actor",
            "city",
            "country",
            "organization",
            "episode",
            "fictional character",
        },
    },
    "Episode": {
        "min_conf": 0.70,
        "good": {
            "episode",
            "television episode",
            "episode of the queen's gambit",
            "netflix miniseries",
            "2020 episode of the queen's gambit",
        },
        "bad": {
            "person",
            "actor",
            "city",
            "country",
            "chess opening",
            "book",
            "novel",
            "painting",
        },
    },
    "Book": {
        "min_conf": 0.72,
        "good": {
            "novel",
            "book",
            "written by walter tevis",
            "novel by walter tevis",
        },
        "bad": {
            "television episode",
            "television series",
            "actor",
            "city",
            "country",
        },
    },
    "Work": {
        "min_conf": 0.72,
        "good": {
            "television series",
            "miniseries",
            "novel",
            "fictional work",
            "netflix",
            "queen's gambit",
        },
        "bad": {
            "given name",
            "city",
            "country",
            "person",
            "painting",
        },
    },
}


def short_local(uri: str) -> str:
    uri = str(uri or "").strip()
    if "#" in uri:
        return uri.split("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def contains_any(text: str, patterns: set[str]) -> bool:
    low = str(text or "").lower()
    return any(p.lower() in low for p in patterns)


def normalize_cell(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def accept_row(row: pd.Series) -> bool:
    label = normalize_cell(row.get("private_label"))
    ptype = short_local(normalize_cell(row.get("private_type")))
    desc = normalize_cell(row.get("wd_description")).lower()
    wd_label = normalize_cell(row.get("wd_label"))
    qid = normalize_cell(row.get("wikidata_id"))
    conf_raw = row.get("confidence", 0.0)

    try:
        conf = float(conf_raw)
    except Exception:
        conf = 0.0

    # -----------------------------
    # Manual hard rejections first
    # -----------------------------
    if qid in MANUAL_REJECT_QIDS:
        return False

    if label in MANUAL_REJECT_PRIVATE_LABELS:
        return False

    if wd_label in MANUAL_REJECT_WD_LABELS:
        return False

    if (label, qid) in MANUAL_REJECT_PAIRS:
        return False

    if contains_any(desc, MANUAL_REJECT_DESC_CONTAINS):
        return False

    # -----------------------------
    # Manual accepts
    # -----------------------------
    if label in MANUAL_ACCEPT_LABELS:
        return True

    # -----------------------------
    # Global rejects
    # -----------------------------
    if contains_any(desc, ALWAYS_REJECT_DESC_CONTAINS):
        return False

    # empty description: be more conservative
    if not desc and conf < 0.96:
        return False

    rules = TYPE_RULES.get(ptype)
    if not rules:
        return False

    if conf < rules["min_conf"]:
        return False

    # type-specific bad patterns
    if contains_any(desc, rules["bad"]):
        return False

    # special safeguards
    if ptype == "Episode":
        # must really look like an episode from the right universe
        if "episode" not in desc and conf < 0.98:
            return False

    if ptype == "Character":
        # if not explicitly fictional / queen's gambit and confidence not extreme -> reject
        if not contains_any(desc, {"fictional", "queen's gambit"}) and conf < 0.95:
            return False

    if ptype == "Location":
        # for location, descriptions with village/town may still be okay,
        # but obscure places at low confidence are rejected
        if contains_any(desc, {"village", "municipality"}) and conf < 0.95:
            return False

    # type-specific good patterns
    if contains_any(desc, rules["good"]):
        return True

    # fallback: only extremely strong confidence survives sparse descriptions
    return conf >= max(0.96, rules["min_conf"] + 0.18)


def build_sameas_graph(rows: List[dict], out_ns: Namespace) -> Graph:
    g = Graph()
    g.bind("qg", out_ns)
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("wd", WD)

    CONF = out_ns.confidence
    WD_LABEL = out_ns.wikidataLabel
    WD_DESC = out_ns.wikidataDescription

    for r in rows:
        s = URIRef(r["private_entity"])
        wd_uri = URIRef(r["wikidata_uri"])
        g.add((s, OWL.sameAs, wd_uri))

        st = URIRef(f"{r['private_entity']}/sameAs/{r['wikidata_id']}")
        g.add((st, RDF.type, RDF.Statement))
        g.add((st, RDF.subject, s))
        g.add((st, RDF.predicate, OWL.sameAs))
        g.add((st, RDF.object, wd_uri))
        g.add((st, CONF, Literal(str(r["confidence"]), datatype=XSD.decimal)))

        if r.get("wd_label"):
            g.add((st, WD_LABEL, Literal(str(r["wd_label"]), lang="en")))
        if r.get("wd_description"):
            g.add((st, WD_DESC, Literal(str(r["wd_description"]), lang="en")))

    return g


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter Wikidata entity linking output for balanced precision/recall."
    )
    parser.add_argument("--in_csv", default="data/entity_wikidata_mapping.csv")
    parser.add_argument("--out_csv", default="data/entity_wikidata_mapping_clean.csv")
    parser.add_argument("--out_ttl", default="data/entity_sameas_clean.ttl")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    required = {
        "private_entity",
        "private_label",
        "private_type",
        "wikidata_id",
        "wikidata_uri",
        "confidence",
        "wd_label",
        "wd_description",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

    # Normalize columns to avoid NaN / pandas masking issues
    for col in [
        "private_entity",
        "private_label",
        "private_type",
        "wikidata_id",
        "wikidata_uri",
        "wd_label",
        "wd_description",
    ]:
        df[col] = df[col].fillna("").astype(str)

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)

    df["accepted"] = df.apply(accept_row, axis=1)
    df["accepted"] = df["accepted"].fillna(False).astype(bool)

    clean = df[df["accepted"]].copy()
    clean = clean.sort_values(
        by=["private_entity", "confidence"],
        ascending=[True, False],
    ).drop_duplicates(subset=["private_entity"], keep="first")

    clean = clean.drop_duplicates(
        subset=["private_entity", "wikidata_id"]
    ).drop(columns=["accepted"]).reset_index(drop=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    sameas_g = build_sameas_graph(clean.to_dict(orient="records"), out_ns=QG)
    Path(args.out_ttl).parent.mkdir(parents=True, exist_ok=True)
    sameas_g.serialize(destination=args.out_ttl, format="turtle")

    print(f"Input rows: {len(df)}")
    print(f"Accepted rows: {len(clean)}")
    print(f"Clean CSV: {args.out_csv}")
    print(f"Clean TTL: {args.out_ttl}")


if __name__ == "__main__":
    main()