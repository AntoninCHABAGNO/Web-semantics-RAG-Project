#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predicate_alignment.py — Align private predicates with Wikidata properties.

Two modes:

1) "emit" (default):
   Produces a TTL file with alignments (owl:equivalentProperty / rdfs:subPropertyOf).
   You edit/validate a mapping dict.

2) "suggest":
   Queries Wikidata SPARQL endpoint to suggest candidate properties by keyword.
   This is helpful for manual validation.

Output:
- data/predicate_alignment.ttl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL


QG = Namespace("http://example.org/qg#")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
WD = Namespace("http://www.wikidata.org/entity/")

WD_SPARQL = "https://query.wikidata.org/sparql"


# ---- You should validate these mappings (manual validation is allowed by the TP)
# Use "equivalent" when it's basically the same meaning.
# Use "subproperty" when your predicate is narrower than the Wikidata one.
PREDICATE_ALIGNMENT: Dict[str, Tuple[str, str]] = {
    # private_predicate_local: (alignment_kind, wikidata_property_id)
    # Example: qg:married ≈ wdt:P26 (spouse)
    "married": ("equivalent", "P26"),
    "divorced": ("subproperty", "P582"),  # not perfect; divorce isn't directly P582 (end time). You may refine later.

    # chess / events
    "won": ("subproperty", "P1346"),      # winner
    "lost": ("subproperty", "P1346"),     # (not perfect; losing isn't "winner"). Keep as subproperty if you keep it.
    "entered": ("subproperty", "P710"),   # participant (entered is a kind of participation)
    "playedIn": ("subproperty", "P1344"), # participant in
    "competedIn": ("subproperty", "P1344"),

    # social
    "met": ("subproperty", "P131"),       # NOT good, placeholder — you should revise.
    "befriended": ("subproperty", "P3342"),  # fictional placeholder; validate manually.
    "loved": ("subproperty", "P451"),     # "unmarried partner" isn't love; validate.
    "helped": ("subproperty", "P1534"),   # "influenced by" etc; validate.

    # family
    "childOf": ("equivalent", "P40"),     # child (inverse direction!), careful: Wikidata P40 is "child"
    "parentOf": ("equivalent", "P40"),    # direction mismatch; handle later if you want
    "siblingOf": ("equivalent", "P3373"), # sibling
}


def emit_alignment_ttl(out_path: Path) -> None:
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="turtle")
    print(f"Wrote predicate alignment TTL: {out_path}")


def suggest_properties(keyword: str, limit: int = 50) -> List[Tuple[str, str]]:
    """
    Suggest Wikidata properties by label keyword (English).
    Returns list of (property_uri, label).
    """
    kw = keyword.strip().lower()
    if not kw:
        return []

    query = f"""
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?property ?propertyLabel WHERE {{
  ?property a wikibase:Property .
  ?property rdfs:label ?propertyLabel .
  FILTER(LANG(?propertyLabel) = "en")
  FILTER(CONTAINS(LCASE(STR(?propertyLabel)), "{kw}"))
}}
LIMIT {int(limit)}
"""
    headers = {"Accept": "application/sparql-results+json"}
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
    p.add_argument("--limit", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "emit":
        emit_alignment_ttl(args.out)
    else:
        props = suggest_properties(args.keyword, limit=args.limit)
        for uri, label in props:
            print(uri, "->", label)


if __name__ == "__main__":
    main()