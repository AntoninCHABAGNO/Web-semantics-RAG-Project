#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
expand_kb.py — Multi-hop controlled KG expansion from Wikidata.

This version is designed to reach large triple counts without blowing up into
irrelevant noise. It uses the Wikidata EntityData JSON endpoint instead of
SPARQL for the main fetch path, which is usually more stable for repeated
entity expansion.
"""

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import requests
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

QG = Namespace("http://example.org/qg#")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
SCHEMA = Namespace("http://schema.org/")

ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
HEADERS = {
    "User-Agent": "QueensGambitKG/2.0 (controlled Wikidata expansion)",
    "Accept": "application/json",
}

DEFAULT_PROPERTY_WHITELIST = {
    # typing / taxonomy
    "P31", "P279", "P361", "P527", "P179", "P921",
    # people
    "P106", "P27", "P19", "P20", "P69", "P1412", "P166", "P800", "P463",
    # works / media
    "P50", "P57", "P58", "P161", "P162", "P170", "P175", "P86", "P136",
    "P364", "P495", "P577", "P449", "P123", "P144", "P1441", "P1080",
    "P155", "P156", "P674", "P453",
    # places / orgs / events
    "P17", "P131", "P159", "P452", "P571", "P112", "P749", "P1344", "P1346", "P276",
}

EXPAND_THROUGH_PROPERTIES = {
    "P31", "P279", "P361", "P527", "P179", "P50", "P57", "P58", "P161", "P162",
    "P170", "P175", "P136", "P495", "P449", "P17", "P131", "P159", "P112", "P749",
    "P155", "P156", "P144", "P1441", "P1080", "P1344", "P1346", "P276", "P800",
}

LABEL_LANG_PRIORITY = ("en", "fr")


@dataclass
class LinkRow:
    private_entity: str
    private_label: str
    private_type: str
    wikidata_uri: str
    confidence: float

    @property
    def qid(self) -> str:
        return self.wikidata_uri.rsplit("/", 1)[-1]


def short_local(uri: str) -> str:
    if "#" in uri:
        return uri.split("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def load_mapping_csv(path: Path, min_conf: float, allowed_types: Set[str]) -> List[LinkRow]:
    rows: List[LinkRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                conf = float(row.get("confidence", "0") or 0.0)
            except Exception:
                conf = 0.0
            if conf < min_conf:
                continue
            pe = (row.get("private_entity") or "").strip()
            pl = (row.get("private_label") or "").strip()
            pt = (row.get("private_type") or "").strip()
            wu = (row.get("wikidata_uri") or "").strip()
            if not pe or not pt or not wu:
                continue
            local_type = short_local(pt)
            if allowed_types and local_type not in allowed_types:
                continue
            rows.append(LinkRow(pe, pl, pt, wu, conf))
    return rows


def load_private_graph(abox: Path, sameas_ttl: Optional[Path]) -> Graph:
    g = Graph()
    g.parse(abox)
    if sameas_ttl and sameas_ttl.exists():
        g.parse(sameas_ttl)
    g.bind("qg", QG)
    g.bind("wd", WD)
    g.bind("wdt", WDT)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)
    g.bind("schema", SCHEMA)
    return g


def load_cache(path: Path) -> Dict[str, dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


import random

def fetch_entity_json(qid: str, session: requests.Session, cache: Dict[str, dict], sleep: float = 0.0) -> dict:

    if qid in cache:
        return cache[qid]

    url = ENTITY_DATA_URL.format(qid=qid)

    for attempt in range(6):

        try:

            r = session.get(url, headers=HEADERS, timeout=60)

            if r.status_code == 429:
                wait = min(60, 2 ** attempt) + random.uniform(0.2, 1.0)
                print(f"429 for {qid}, waiting {wait:.1f}s")
                time.sleep(wait)
                continue

            r.raise_for_status()

            data = r.json()
            cache[qid] = data

            if sleep:
                time.sleep(sleep)

            return data

        except requests.RequestException as e:

            if attempt == 5:
                raise e

            wait = min(60, 2 ** attempt) + random.uniform(0.2, 1.0)
            print(f"Retry {qid} in {wait:.1f}s ({e})")
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch {qid}")


def parse_qid_from_datavalue(obj: dict) -> Optional[str]:
    value = obj.get("value")
    if not isinstance(value, dict):
        return None
    entity_type = value.get("entity-type")
    numeric_id = value.get("numeric-id")
    if entity_type != "item" or numeric_id is None:
        return None
    return f"Q{numeric_id}"

import re

WD_TIME_RE = re.compile(
    r"^[+-](\d{1,16})-(\d{2})-(\d{2})T\d{2}:\d{2}:\d{2}Z$"
)

def literal_from_snak_datavalue(dv: dict) -> Optional[Tuple[Literal, str]]:
    vtype = dv.get("type")
    value = dv.get("value")
    if vtype == "string" and isinstance(value, str):
        return Literal(value), "string"
    if vtype == "time" and isinstance(value, dict):
        time_val = value.get("time")
        if isinstance(time_val, str):

            m = WD_TIME_RE.match(time_val)
            if not m:
                return Literal(time_val), "time"

            year, month, day = m.groups()

            # mois ou jour inconnus (00)
            if month == "00" or day == "00":
                return Literal(f"{year}-{month}-{day}"), "time"

            try:
                return Literal(f"{int(year):04d}-{int(month):02d}-{int(day):02d}", datatype=XSD.date), "time"
            except Exception:
                return Literal(time_val), "time"
    if vtype == "quantity" and isinstance(value, dict):
        amount = value.get("amount")
        if isinstance(amount, str):
            return Literal(amount), "quantity"
    if vtype == "monolingualtext" and isinstance(value, dict):
        txt = value.get("text")
        lang = value.get("language") or "en"
        if isinstance(txt, str):
            return Literal(txt, lang=lang), "monolingualtext"
    return None


def best_lang_value(data: dict, key: str) -> Optional[Tuple[str, str]]:
    block = data.get(key, {})
    for lang in LABEL_LANG_PRIORITY:
        if lang in block and isinstance(block[lang], dict):
            val = block[lang].get("value")
            if isinstance(val, str) and val.strip():
                return val.strip(), lang
    for lang, item in block.items():
        if isinstance(item, dict):
            val = item.get("value")
            if isinstance(val, str) and val.strip():
                return val.strip(), lang
    return None


def add_entity_metadata(g: Graph, qid: str, entity_data: dict) -> int:
    entity_obj = entity_data.get("entities", {}).get(qid, {})
    uri = WD[qid]
    added = 0
    label = best_lang_value(entity_obj, "labels")
    if label:
        triple = (uri, RDFS.label, Literal(label[0], lang=label[1]))
        if triple not in g:
            g.add(triple)
            added += 1
    desc = best_lang_value(entity_obj, "descriptions")
    if desc:
        triple = (uri, SCHEMA.description, Literal(desc[0], lang=desc[1]))
        if triple not in g:
            g.add(triple)
            added += 1

    aliases = entity_obj.get("aliases", {})
    for lang in LABEL_LANG_PRIORITY:
        for item in aliases.get(lang, [])[:5]:
            val = item.get("value")
            if val:
                triple = (uri, SKOS.altLabel, Literal(val, lang=lang))
                if triple not in g:
                    g.add(triple)
                    added += 1
    return added


def extract_claim_triples(
    qid: str,
    entity_data: dict,
    allowed_properties: Set[str],
    include_literals: bool,
    literal_max_len: int,
    max_claims_per_property: int,
) -> Tuple[List[Tuple[URIRef, URIRef, object]], Set[str]]:
    entity_obj = entity_data.get("entities", {}).get(qid, {})
    claims = entity_obj.get("claims", {})
    subject = WD[qid]
    triples: List[Tuple[URIRef, URIRef, object]] = []
    neighbor_qids: Set[str] = set()

    for pid, statements in claims.items():
        if pid not in allowed_properties:
            continue
        predicate = WDT[pid]
        kept = 0
        for st in statements:
            if kept >= max_claims_per_property:
                break
            mainsnak = st.get("mainsnak", {})
            if mainsnak.get("snaktype") != "value":
                continue
            dv = mainsnak.get("datavalue")
            if not isinstance(dv, dict):
                continue
            qid_obj = parse_qid_from_datavalue(dv)
            if qid_obj:
                triples.append((subject, predicate, WD[qid_obj]))
                neighbor_qids.add(qid_obj)
                kept += 1
                continue
            if not include_literals:
                continue
            lit_res = literal_from_snak_datavalue(dv)
            if lit_res is None:
                continue
            lit, _ = lit_res
            if isinstance(lit.value, str) and len(str(lit.value)) > literal_max_len:
                continue
            triples.append((subject, predicate, lit))
            kept += 1
    return triples, neighbor_qids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-hop controlled expansion of a private KG using Wikidata.")
    p.add_argument("--abox", type=Path, default=Path("data/queens_gambit_graph_abox.ttl"))
    p.add_argument("--sameas_ttl", type=Path, default=Path("data/entity_sameas_clean.ttl"))
    p.add_argument("--mapping_csv", type=Path, default=Path("data/entity_wikidata_mapping_clean.csv"))
    p.add_argument("--alignment_ttl", type=Path, default=Path("data/predicate_alignment.ttl"))
    p.add_argument("--out", type=Path, default=Path("data/expanded_kb.ttl"))
    p.add_argument("--cache", type=Path, default=Path("data/.wikidata_entity_cache.json"))
    p.add_argument(
        "--anchor_types",
        type=str,
        default="Character,RealPerson,Organization,Location,Event,Work,Episode,Book",
    )
    p.add_argument("--min_conf", type=float, default=0.72)
    p.add_argument("--max_hops", type=int, default=3)
    p.add_argument("--max_entities", type=int, default=5000)
    p.add_argument("--max_new_triples", type=int, default=100000)
    p.add_argument("--max_claims_per_property", type=int, default=20)
    p.add_argument("--include_literals", action="store_true")
    p.add_argument("--literal_max_len", type=int, default=200)
    p.add_argument("--sleep", type=float, default=0.05)
    return p.parse_args()

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def main() -> None:
    args = parse_args()
    allowed_anchor_types = {x.strip() for x in args.anchor_types.split(",") if x.strip()}
    g = load_private_graph(args.abox, args.sameas_ttl)
    links = load_mapping_csv(args.mapping_csv, min_conf=args.min_conf, allowed_types=allowed_anchor_types)

    # sameAs seed links are always preserved
    for lr in links:
        g.add((URIRef(lr.private_entity), OWL.sameAs, URIRef(lr.wikidata_uri)))

    retry = Retry(
        total=6,
        connect=4,
        read=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    cache = load_cache(args.cache)

    queue: Deque[Tuple[str, int]] = deque()
    seen_qids: Set[str] = set()
    visited_qids: Set[str] = set()
    triple_count_before = len(g)

    for lr in links:
        qid = lr.qid
        if qid not in seen_qids:
            seen_qids.add(qid)
            queue.append((qid, 0))

    while queue and len(visited_qids) < args.max_entities and (len(g) - triple_count_before) < args.max_new_triples:
        qid, depth = queue.popleft()
        if qid in visited_qids:
            continue
        visited_qids.add(qid)

        try:
            entity_data = fetch_entity_json(qid, session=session, cache=cache, sleep=args.sleep)
        except Exception as e:  # pragma: no cover - network dependent
            print(f"Skip {qid}: {e}")
            continue

        add_entity_metadata(g, qid, entity_data)
        triples, neighbors = extract_claim_triples(
            qid=qid,
            entity_data=entity_data,
            allowed_properties=DEFAULT_PROPERTY_WHITELIST,
            include_literals=args.include_literals,
            literal_max_len=args.literal_max_len,
            max_claims_per_property=args.max_claims_per_property,
        )

        for s, p, o in triples:
            if (len(g) - triple_count_before) >= args.max_new_triples:
                break
            if (s, p, o) not in g:
                g.add((s, p, o))

        if depth < args.max_hops:
            for neigh in neighbors:
                # only expand through a subset of semantically rich properties
                # infer this by checking current graph
                expand_ok = False
                for p in DEFAULT_PROPERTY_WHITELIST & EXPAND_THROUGH_PROPERTIES:
                    if (WD[qid], WDT[p], WD[neigh]) in g:
                        expand_ok = True
                        break
                if expand_ok and neigh not in seen_qids and len(seen_qids) < args.max_entities:
                    seen_qids.add(neigh)
                    queue.append((neigh, depth + 1))

    save_cache(args.cache, cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(args.out), format="turtle")

    print(f"Seed links used: {len(links)}")
    print(f"Visited Wikidata entities: {len(visited_qids)}")
    print(f"New triples added: {len(g) - triple_count_before}")
    print(f"Total triples in graph: {len(g)}")
    print(f"Expanded TTL: {args.out}")


if __name__ == "__main__":
    main()
