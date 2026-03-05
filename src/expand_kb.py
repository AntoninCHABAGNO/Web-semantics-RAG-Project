#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
expand_kb.py — Expand your private KB using Wikidata SPARQL (anchored expansion)

Inputs:
- ABox TTL (private KG)
- entity mapping CSV from entity_linking.py (private_entity -> wikidata_uri)
- (optional) predicate alignment TTL to map / filter predicates

Output:
- expanded TTL graph (keeps your private triples + adds Wikidata triples)

Strategy:
- Only expand from confidently aligned entities (confidence >= min_conf)
- 1-hop expansion: wd:Qxxx ?p ?o
- Keep predicates to wdt: (direct properties) by default
- Control volume with per-entity limit and global max triples
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL


WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
QG = Namespace("http://example.org/qg#")

WD_SPARQL = "https://query.wikidata.org/sparql"


@dataclass
class LinkRow:
    private_entity: str
    wikidata_uri: str
    confidence: float


def load_mapping_csv(path: Path, min_conf: float) -> List[LinkRow]:
    rows: List[LinkRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                conf = float(row.get("confidence", "0") or 0)
            except Exception:
                conf = 0.0
            if conf < min_conf:
                continue
            pe = (row.get("private_entity") or "").strip()
            wu = (row.get("wikidata_uri") or "").strip()
            if not pe or not wu:
                continue
            rows.append(LinkRow(private_entity=pe, wikidata_uri=wu, confidence=conf))
    return rows


def sparql_json(query: str, sleep: float = 0.0) -> dict:
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "ProjetRAG-KGExpansion/1.0 (educational; contact: none)",
    }
    r = requests.get(WD_SPARQL, params={"query": query}, headers=headers, timeout=90)
    r.raise_for_status()
    if sleep:
        time.sleep(sleep)
    return r.json()


def expand_one_entity(wd_uri: str, per_entity_limit: int, include_labels: bool) -> List[Tuple[str, str, str, bool, Optional[str]]]:
    """
    Return list of triples (s, p, o, o_is_literal, o_lang)
    Only retrieves wdt: direct properties by default.
    """
    wd = f"<{wd_uri}>"
    label_block = ""
    if include_labels:
        # try to fetch labels for object URIs (optional)
        label_block = """
  OPTIONAL {
    ?o rdfs:label ?oLabel .
    FILTER(LANG(?oLabel) = "en")
  }
"""

    query = f"""
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?p ?o ?oLabel WHERE {{
  {wd} ?p ?o .
  FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
  {label_block}
}}
LIMIT {int(per_entity_limit)}
"""
    data = sparql_json(query)
    out: List[Tuple[str, str, str, bool, Optional[str]]] = []
    for b in data["results"]["bindings"]:
        p = b["p"]["value"]
        o_bind = b["o"]
        if o_bind["type"] == "uri":
            out.append((wd_uri, p, o_bind["value"], False, None))
        else:
            # literal
            lang = o_bind.get("xml:lang")
            out.append((wd_uri, p, o_bind["value"], True, lang))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expand a private KG using aligned Wikidata entities.")
    p.add_argument("--abox", type=Path, default=Path("data/queens_gambit_graph_abox.ttl"), help="Private ABox TTL.")
    p.add_argument("--sameas_ttl", type=Path, default=Path("data/entity_sameas.ttl"), help="owl:sameAs TTL produced by entity_linking.py")
    p.add_argument("--mapping_csv", type=Path, default=Path("data/entity_wikidata_mapping.csv"), help="Mapping CSV from entity_linking.py")
    p.add_argument("--out", type=Path, default=Path("data/expanded_kb.ttl"), help="Output expanded TTL.")
    p.add_argument("--min_conf", type=float, default=0.85, help="Min confidence to expand from.")
    p.add_argument("--per_entity_limit", type=int, default=300, help="Max triples fetched per aligned entity.")
    p.add_argument("--max_new_triples", type=int, default=15000, help="Global cap on added Wikidata triples.")
    p.add_argument("--include_labels", action="store_true", help="Try to fetch English labels for object URIs (slower).")
    p.add_argument("--sleep", type=float, default=0.15, help="Sleep between SPARQL calls.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load base graphs
    g = Graph()
    g.parse(args.abox)
    if args.sameas_ttl.exists():
        g.parse(args.sameas_ttl)

    g.bind("qg", QG)
    g.bind("wd", WD)
    g.bind("wdt", WDT)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    links = load_mapping_csv(args.mapping_csv, min_conf=args.min_conf)
    print(f"Anchors (confidence >= {args.min_conf}): {len(links)}")

    added = 0
    seen: Set[Tuple[str, str, str]] = set()

    # Ensure sameAs links exist in the graph (safety)
    for lr in links:
        s = URIRef(lr.private_entity)
        o = URIRef(lr.wikidata_uri)
        g.add((s, OWL.sameAs, o))

    for i, lr in enumerate(links, start=1):
        if added >= args.max_new_triples:
            break

        wd_uri = lr.wikidata_uri
        try:
            triples = expand_one_entity(wd_uri, per_entity_limit=args.per_entity_limit, include_labels=args.include_labels)
        except Exception as e:
            print(f"[WARN] expand failed for {wd_uri}: {e}")
            continue

        for (s_uri, p_uri, o_val, o_is_lit, o_lang) in triples:
            if added >= args.max_new_triples:
                break

            key = (s_uri, p_uri, o_val)
            if key in seen:
                continue
            seen.add(key)

            s = URIRef(s_uri)
            p = URIRef(p_uri)
            if o_is_lit:
                # Keep literals short-ish to avoid huge KB
                lit_text = o_val
                if isinstance(lit_text, str) and len(lit_text) > 300:
                    continue
                if o_lang:
                    o = Literal(lit_text, lang=o_lang)
                else:
                    o = Literal(lit_text)
            else:
                o = URIRef(o_val)

            g.add((s, p, o))
            added += 1

        time.sleep(args.sleep)
        if i % 10 == 0:
            print(f"Processed {i}/{len(links)} anchors — added {added} Wikidata triples so far...")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(args.out), format="turtle")
    print(f"Done. Added Wikidata triples: {added}")
    print(f"Expanded KB written to: {args.out}")


if __name__ == "__main__":
    main()