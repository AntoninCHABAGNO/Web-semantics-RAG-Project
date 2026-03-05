#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
entity_linking.py — Link private KG entities to Wikidata (owl:sameAs)

Inputs:
- ABox TTL (your private data graph): data/queens_gambit_graph_abox.ttl
  We read entities from rdf:type + rdfs:label.
- Optional: restrict by qg classes (Character, RealPerson, Location, Organization, Event, ...)

Outputs:
- mapping CSV: private_entity, private_label, private_type, wikidata_id, wikidata_uri, confidence, wd_label, wd_description
- TTL file containing owl:sameAs links + confidence as an annotation on the statement node.

Notes:
- Wikidata Search API does not return a numeric score; we compute a conservative heuristic confidence.
- We use caching and rate limiting to be kind to Wikidata.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from requests.exceptions import HTTPError
import requests
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WD_ENTITY = "http://www.wikidata.org/entity/"
WD = Namespace(WD_ENTITY)

QG = Namespace("http://example.org/qg#")

DEFAULT_HEADERS = {
    "User-Agent": "QueensGambitKG/1.0 (mailto: anton.tu@exemple.com) requests/2.x",
    "Accept": "application/json",
    "Accept-Language": "en",
    "Connection": "keep-alive",
}

@dataclass
class Candidate:
    qid: str
    uri: str
    label: str
    description: str


def normalize_label(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)

    # ignore labels that are too short (avoid linking "Beth", "Alma", etc.)
    if len(s) < 5:
        return ""

    return s

def make_session(no_proxy: bool = True) -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)

    if no_proxy:
        # ignore les variables proxy du système (souvent cause des 403)
        s.trust_env = False
        s.proxies = {"http": None, "https": None}

    return s

def wikidata_search(
    label: str,
    language: str = "en",
    limit: int = 5,
    session: Optional[requests.Session] = None,
    sleep: float = 0.2,
    retries: int = 3,
    debug: bool = False,
) -> List[Candidate]:
    label = normalize_label(label)
    if not label:
        return []

    sess = session or make_session(no_proxy=True)

    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": language,
        "format": "json",
        "limit": str(limit),
        # parfois utile (pas obligatoire, mais inoffensif)
        "uselang": language,
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = sess.get(WIKIDATA_API, params=params, timeout=30)

            if debug:
                print("URL:", r.url)
                print("Status:", r.status_code)
                print("Resp headers:", dict(r.headers))

            # Handle rate limiting
            if r.status_code in (429, 503):
                wait = sleep * (2 ** (attempt - 1))
                if debug:
                    print(f"Rate limited ({r.status_code}), sleeping {wait:.2f}s...")
                time.sleep(wait)
                continue

            # Handle forbidden
            if r.status_code == 403:
                if debug:
                    print("403 body (first 300 chars):", r.text[:300])
                # No point retrying too much, but we do a small backoff once/twice
                wait = sleep * (2 ** (attempt - 1))
                time.sleep(wait)
                last_err = HTTPError(f"403 Forbidden for {r.url}")
                continue

            r.raise_for_status()
            data = r.json()

            out: List[Candidate] = []
            for item in data.get("search", []):
                qid = item.get("id") or ""
                concepturi = item.get("concepturi") or (WD_ENTITY + qid if qid else "")
                out.append(
                    Candidate(
                        qid=qid,
                        uri=concepturi,
                        label=item.get("label") or "",
                        description=item.get("description") or "",
                    )
                )
            time.sleep(sleep)
            return out

        except Exception as e:
            last_err = e
            wait = sleep * (2 ** (attempt - 1))
            if debug:
                print(f"Error attempt {attempt}/{retries}: {e} — sleeping {wait:.2f}s")
            time.sleep(wait)

    # If all attempts fail:
    if debug and last_err:
        print("Final error:", repr(last_err))
    return []


def heuristic_confidence(query_label: str, cand: Candidate, private_type: str) -> float:
    """
    Conservative heuristic confidence:
    - exact label match => high
    - near match => medium
    - type mismatch hints reduce confidence
    """
    q = normalize_label(query_label).lower()
    c = normalize_label(cand.label).lower()

    if not cand.qid:
        return 0.0

    # base on label match
    if q == c:
        conf = 0.95
    elif q in c or c in q:
        conf = 0.85
    else:
        conf = 0.70

    desc = (cand.description or "").lower()

    # some gentle type hints
    # (you can adjust these based on your domain needs)
    if private_type.endswith("Character"):
        # fictional character should mention "fictional" often
        if "fictional" in desc or "character" in desc:
            conf += 0.03
        elif any(k in desc for k in ["actor", "actress", "film", "singer", "chess player", "grandmaster", "person"]):
            conf -= 0.10

    if private_type.endswith("Location"):
        if any(k in desc for k in ["city", "country", "town", "village", "region", "capital"]):
            conf += 0.03

    if private_type.endswith("Organization"):
        if any(k in desc for k in ["company", "organization", "publisher", "federation", "club", "magazine"]):
            conf += 0.03

    if private_type.endswith("Event"):
        if any(k in desc for k in ["tournament", "championship", "event", "competition"]):
            conf += 0.03

    # clamp
    conf = max(0.0, min(0.99, conf))
    return conf


def extract_private_entities(g: Graph) -> List[Tuple[URIRef, str, str]]:
    """
    Return (entity_uri, label, type_uri) for all individuals in the ABox.
    We require rdfs:label, and rdf:type.
    """
    entities: List[Tuple[URIRef, str, str]] = []
    for s in set(g.subjects(RDF.type, None)):
        if not isinstance(s, URIRef):
            continue
        # skip reified statement blank nodes
        if (s, RDF.type, RDF.Statement) in g:
            continue

        label = None
        for lit in g.objects(s, RDFS.label):
            if isinstance(lit, Literal):
                label = str(lit)
                break
        if not label:
            continue

        type_uri = None
        for t in g.objects(s, RDF.type):
            if isinstance(t, URIRef):
                type_uri = str(t)
                break
        if not type_uri:
            continue

        entities.append((s, normalize_label(label), type_uri))
    return entities


def short_local(uri: str) -> str:
    """Best-effort localname for filtering (qg#Character -> Character)."""
    if "#" in uri:
        return uri.split("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def build_sameas_graph(rows: List[dict], out_ns: Namespace) -> Graph:
    """
    Create a TTL graph containing owl:sameAs links + confidence on a reified statement.
    """
    out = Graph()
    out.bind("qg", out_ns)
    out.bind("owl", OWL)
    out.bind("rdf", RDF)
    out.bind("rdfs", RDFS)
    out.bind("xsd", XSD)
    out.bind("wd", WD)

    CONF = out_ns.confidence  # reuse qg:confidence if it exists in your ontology

    for r in rows:
        s = URIRef(r["private_entity"])
        wd_uri = URIRef(r["wikidata_uri"])
        out.add((s, OWL.sameAs, wd_uri))

        # reify for confidence (optional but nice)
        st = URIRef(f"{r['private_entity']}/sameAs/{r['wikidata_id']}")
        out.add((st, RDF.type, RDF.Statement))
        out.add((st, RDF.subject, s))
        out.add((st, RDF.predicate, OWL.sameAs))
        out.add((st, RDF.object, wd_uri))
        out.add((st, CONF, Literal(r["confidence"], datatype=XSD.decimal)))

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entity linking from private ABox TTL to Wikidata.")
    p.add_argument("--abox", type=Path, default=Path("data/queens_gambit_graph_abox.ttl"), help="Path to ABox TTL.")
    p.add_argument("--out_csv", type=Path, default=Path("data/entity_wikidata_mapping.csv"), help="Output mapping CSV.")
    p.add_argument("--out_ttl", type=Path, default=Path("data/entity_sameas.ttl"), help="Output TTL with owl:sameAs.")
    p.add_argument("--min_conf", type=float, default=0.80, help="Minimum confidence to accept a match.")
    p.add_argument("--limit", type=int, default=3, help="How many candidates to consider per entity (top-k).")
    p.add_argument("--sleep", type=float, default=0.10, help="Sleep between API calls (seconds).")
    p.add_argument(
        "--types",
        type=str,
        default="Character,RealPerson,Location,Organization,Event,Group,Work,Episode,Book",
        help="Comma-separated local class names to link (from qg:Class localname).",
    )
    p.add_argument("--lang", type=str, default="en", help="Language for Wikidata search.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wanted = {t.strip() for t in args.types.split(",") if t.strip()}

    g = Graph()
    g.parse(args.abox)

    entities = extract_private_entities(g)

    sess = make_session(no_proxy=True)
    cache_path = Path("data/.wikidata_search_cache.json")
    cache: Dict[str, List[dict]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    rows_out: List[dict] = []

    for (uri, label, type_uri) in entities:
        local_type = short_local(type_uri)
        if wanted and local_type not in wanted:
            continue

        # small skip list for obviously useless labels
        if label.lower() in {"unknown", "episode", "season"}:
            continue
        if len(label) < 2:
            continue

        key = f"{label}||{args.lang}"
        if key in cache:
            candidates = [Candidate(**c) for c in cache[key]]
        else:
            try:
                cands = wikidata_search(label, language=args.lang, limit=max(1, args.limit), session=sess)
            except Exception as e:
                print(f"[WIKIDATA ERROR] label='{label}' type='{short_local(type_uri)}' -> {e}")
                cands = []
            cache[key] = [c.__dict__ for c in cands]
            candidates = cands
            time.sleep(args.sleep)

        if not candidates:
            continue

        best = candidates[0]
        conf = heuristic_confidence(label, best, type_uri)

        if conf < args.min_conf:
            continue

        rows_out.append(
            {
                "private_entity": str(uri),
                "private_label": label,
                "private_type": type_uri,
                "wikidata_id": best.qid,
                "wikidata_uri": best.uri,
                "confidence": f"{conf:.2f}",
                "wd_label": best.label,
                "wd_description": best.description,
            }
        )

    # persist cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # write csv
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "private_entity",
                "private_label",
                "private_type",
                "wikidata_id",
                "wikidata_uri",
                "confidence",
                "wd_label",
                "wd_description",
            ],
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # write ttl (sameAs)
    sameas_g = build_sameas_graph(rows_out, out_ns=QG)
    args.out_ttl.parent.mkdir(parents=True, exist_ok=True)
    sameas_g.serialize(destination=str(args.out_ttl), format="turtle")

    print(f"Linked entities: {len(rows_out)}")
    print(f"Mapping CSV: {args.out_csv}")
    print(f"sameAs TTL: {args.out_ttl}")


if __name__ == "__main__":
    main()