#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
predicate_alignment.py — Align private predicates with Wikidata properties.

Still conservative, but extended a bit so the KG has more useful bridges.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDFS

QG = Namespace("http://example.org/qg#")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
WD_SPARQL = "https://query.wikidata.org/sparql"

PREDICATE_ALIGNMENT: Dict[str, Tuple[str, str]] = {
    # Family
    "parentOf": ("equivalent", "P40"),
    "siblingOf": ("equivalent", "P3373"),

    # Participation / competition
    "competedIn": ("subproperty", "P1344"),
    "playedIn": ("subproperty", "P1344"),
    "entered": ("subproperty", "P1344"),
    "won": ("subproperty", "P1346"),
    "tookPlaceIn": ("subproperty", "P276"),

    # Social
    "married": ("equivalent", "P26"),

    # Training / performance (loose but useful as subproperties)
    "trainedBy": ("subproperty", "P1066"),
    "taughtBy": ("subproperty", "P1066"),
    "coachedBy": ("subproperty", "P1066"),
    "learnedFrom": ("subproperty", "P1066"),
}

ALIGNMENT_NOTES: Dict[str, str] = {
    "parentOf": "Aligned to Wikidata P40 (child). Direction matches: subject has child object.",
    "siblingOf": "Direct equivalence with Wikidata sibling property.",
    "competedIn": "Private predicate is narrower than generic participation in an event.",
    "playedIn": "Modeled as a narrower form of participation.",
    "entered": "Entering a tournament is treated as a narrower form of participation.",
    "won": "Winning is aligned as a narrower relation to Wikidata winner.",
    "tookPlaceIn": "Event location aligned as a narrower relation to location.",
    "married": "Private married relation aligned to Wikidata spouse property.",
    "trainedBy": "Training/teaching relations mapped conservatively to influenced by.",
    "taughtBy": "Training/teaching relations mapped conservatively to influenced by.",
    "coachedBy": "Training/teaching relations mapped conservatively to influenced by.",
    "learnedFrom": "Training/teaching relations mapped conservatively to influenced by.",
}

UNALIGNED_EXPLANATIONS: Dict[str, str] = {
    "childOf": "Not aligned automatically because Wikidata P40 has opposite direction.",
    "divorced": "No clean direct Wikidata property equivalent in this modeling.",
    "lost": "Winner is not the same as losing; do not force a direct alignment.",
    "met": "Too weak / context-dependent for automatic alignment.",
    "befriended": "Too vague for a strong Wikidata property mapping.",
    "loved": "Too vague semantically; do not force to spouse/partner properties.",
    "helped": "Too vague semantically.",
    "supported": "Too vague semantically.",
    "played": "Opposition in a game is not cleanly captured by one direct Wikidata property.",
    "defeated": "Could map to winner/participant patterns, but not as a clean direct property.",
    "lostTo": "No direct clean mapping chosen here.",
    "drewWith": "No direct clean mapping chosen here.",
    "usedOpening": "No safe direct Wikidata property chosen in this private modeling.",
}


def emit_alignment_ttl(out_path: Path, include_notes: bool = True) -> None:
    g = Graph()
    g.bind("qg", QG)
    g.bind("wdt", WDT)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    for private_local, (kind, pid) in PREDICATE_ALIGNMENT.items():
        p_private = QG[private_local]
        p_wd = WDT[pid]
        if kind == "equivalent":
            g.add((p_private, OWL.equivalentProperty, p_wd))
        elif kind == "subproperty":
            g.add((p_private, RDFS.subPropertyOf, p_wd))
        else:
            raise ValueError(f"Unknown alignment kind: {kind} for {private_local}")
        if include_notes and private_local in ALIGNMENT_NOTES:
            g.add((p_private, RDFS.comment, Literal(ALIGNMENT_NOTES[private_local])))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="turtle")
    print(f"Wrote predicate alignment TTL: {out_path}")
    if UNALIGNED_EXPLANATIONS:
        print("\nPredicates intentionally left unaligned:")
        for pred, reason in sorted(UNALIGNED_EXPLANATIONS.items()):
            print(f" - {pred}: {reason}")


def escape_sparql_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")


def suggest_properties(keyword: str, limit: int = 50) -> List[Tuple[str, str]]:
    kw = keyword.strip().lower()
    if not kw:
        return []
    kw_escaped = escape_sparql_string(kw)
    query = f"""
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?property ?propertyLabel WHERE {{
  ?property a wikibase:Property .
  ?property rdfs:label ?propertyLabel .
  FILTER(LANG(?propertyLabel) = "en")
  FILTER(CONTAINS(LCASE(STR(?propertyLabel)), "{kw_escaped}"))
}}
LIMIT {int(limit)}
"""
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "QueensGambitKG/2.0 (predicate alignment lookup)",
    }
    r = requests.get(WD_SPARQL, params={"query": query}, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    out: List[Tuple[str, str]] = []
    for b in data["results"]["bindings"]:
        out.append((b["property"]["value"], b["propertyLabel"]["value"]))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predicate alignment for the private KG.")
    p.add_argument("--mode", choices=["emit", "suggest"], default="emit")
    p.add_argument("--out", type=Path, default=Path("data/predicate_alignment.ttl"))
    p.add_argument("--keyword", type=str, default="")
    p.add_argument("--limit", type=int, default=75)
    p.add_argument("--no-notes", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "emit":
        emit_alignment_ttl(args.out, include_notes=not args.no_notes)
    else:
        props = suggest_properties(args.keyword, limit=args.limit)
        for uri, label in props:
            print(uri, "->", label)


if __name__ == "__main__":
    main()
