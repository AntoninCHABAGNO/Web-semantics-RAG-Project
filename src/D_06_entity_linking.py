#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
entity_linking.py — Type-aware entity linking from private ABox TTL to Wikidata.

Inputs:
- ABox TTL: data/queens_gambit_graph_abox.ttl

Outputs:
- CSV mapping:
    private_entity, private_label, private_type,
    wikidata_id, wikidata_uri, confidence,
    wd_label, wd_description, linking_method, matched_query
- TTL file with owl:sameAs links + confidence annotation

Main ideas:
- manual overrides for critical The Queen's Gambit entities
- multiple search variants per entity
- type-aware scoring
- strong domain boosts for The Queen's Gambit universe
- conservative rejection for ambiguous labels
"""

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WD_ENTITY = "http://www.wikidata.org/entity/"
WD = Namespace(WD_ENTITY)
QG = Namespace("http://example.org/qg#")

DEFAULT_HEADERS = {
    "User-Agent": "QueensGambitKG/3.0 (academic project; entity linking)",
    "Accept": "application/json",
    "Accept-Language": "en",
    "Connection": "keep-alive",
}

# ---------------------------------------------------------------------
# Hard manual mappings for high-value entities
# ---------------------------------------------------------------------
MANUAL_LINKS: Dict[Tuple[str, str], str] = {
    # Work / series
    ("The Queen's Gambit", "Work"): "Q85808226",
    ("The Queen's Gambit", "Book"): "Q85808226",
    ("The Queen's Gambit, season 1", "Work"): "Q125208437",
    ("The Queen's Gambit season 1", "Work"): "Q125208437",

    # Episodes
    ("Openings", "Episode"): "Q104650181",
    ("Exchanges", "Episode"): "Q104650188",
    ("Doubled Pawns", "Episode"): "Q104650207",
    ("Middle Game", "Episode"): "Q104650246",
    ("Fork", "Episode"): "Q104650266",
    ("Adjournment", "Episode"): "Q104650276",
    ("End Game", "Episode"): "Q104650295",

    # Safe extras by label only when type is compatible
    ("Beth Harmon", "Character"): "Q102378740",
    ("Anya Taylor-Joy", "RealPerson"): "Q20882479",
    ("Scott Frank", "RealPerson"): "Q557208",
    ("Walter Tevis", "RealPerson"): "Q740928",
    ("Netflix", "Organization"): "Q907311",
}

MANUAL_LABEL_ONLY: Dict[str, str] = {
    "The Queen's Gambit": "Q85808226",
    "Openings": "Q104650181",
    "Exchanges": "Q104650188",
    "Doubled Pawns": "Q104650207",
    "Middle Game": "Q104650246",
    "Fork": "Q104650266",
    "Adjournment": "Q104650276",
    "End Game": "Q104650295",
}

# ---------------------------------------------------------------------
# Labels too generic / too risky for automatic linking
# ---------------------------------------------------------------------
SKIP_LABELS = {
    "unknown",
    "episode",
    "season",
    "white",
    "black",
    "american",
    "british",
    "english",
    "french",
    "russian",
    "soviet",
    "grandmaster",
    "champion",
    "player",
    "teacher",
    "mother",
    "father",
    "son",
    "daughter",
}

AMBIGUOUS_SINGLE_TOKEN_SKIP = {
    "italy",
    "norway",
    "sweden",
    "france",
    "russia",
    "nevada",
    "wakefield",
    "lucerne",
    "rudolph",
    "wolff",
    "hellstrom",
    "southwest",
}

QUEENS_GAMBIT_HINTS = {
    "queen's gambit",
    "the queen's gambit",
    "netflix miniseries",
    "netflix series",
    "novel by walter tevis",
    "episode of the queen's gambit",
}

# ---------------------------------------------------------------------
# Type hints for scoring
# ---------------------------------------------------------------------
TYPE_HINTS: Dict[str, Dict[str, Set[str]]] = {
    "Character": {
        "good": {
            "fictional character",
            "character in",
            "character from",
            "fictional human",
            "queen's gambit",
            "netflix miniseries",
            "novel by walter tevis",
        },
        "bad": {
            "city",
            "country",
            "state of the united states",
            "television episode",
            "film",
            "novel",
            "actor",
            "actress",
            "writer",
            "researcher",
            "professor",
            "painter",
            "newspaper",
            "magazine",
            "university",
        },
    },
    "RealPerson": {
        "good": {
            "actor",
            "actress",
            "writer",
            "director",
            "screenwriter",
            "producer",
            "human",
            "chess player",
            "grandmaster",
            "author",
        },
        "bad": {
            "fictional character",
            "television episode",
            "television series",
            "novel",
            "city",
            "country",
            "magazine",
            "newspaper",
            "organization",
        },
    },
    "Organization": {
        "good": {
            "organization",
            "company",
            "publisher",
            "federation",
            "club",
            "newspaper",
            "magazine",
            "streaming service",
            "streaming",
            "university",
            "high school",
            "school",
        },
        "bad": {
            "person",
            "actor",
            "actress",
            "fictional character",
            "city",
            "country",
            "episode",
            "film",
            "aircraft",
            "building",
        },
    },
    "Location": {
        "good": {
            "city",
            "country",
            "state",
            "capital",
            "region",
            "village",
            "municipality",
            "county seat",
            "country in",
            "state of the united states",
        },
        "bad": {
            "fictional character",
            "actor",
            "actress",
            "company",
            "television series",
            "television episode",
            "magazine",
            "painting",
            "film",
            "university",
        },
    },
    "Event": {
        "good": {
            "event",
            "competition",
            "championship",
            "tournament",
            "match",
            "olympiad",
            "annual chess tournament",
            "biennial chess tournament",
        },
        "bad": {
            "person",
            "actor",
            "city",
            "country",
            "company",
            "television episode",
            "series",
            "novel",
        },
    },
    "Episode": {
        "good": {
            "episode",
            "television episode",
            "episode of the queen's gambit",
            "netflix miniseries",
            "2020 episode of the queen's gambit",
        },
        "bad": {
            "city",
            "country",
            "person",
            "actor",
            "actress",
            "chess opening",
            "book",
            "novel",
            "painting",
            "magazine",
        },
    },
    "Book": {
        "good": {
            "novel",
            "book",
            "written by walter tevis",
            "novel by walter tevis",
            "literary work",
        },
        "bad": {
            "television episode",
            "television series",
            "actor",
            "city",
            "country",
            "organization",
        },
    },
    "Work": {
        "good": {
            "television series",
            "miniseries",
            "novel",
            "film",
            "fictional work",
            "netflix miniseries",
            "series",
        },
        "bad": {
            "given name",
            "city",
            "country",
            "person",
            "actor",
            "actress",
            "organization",
        },
    },
}

# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------
@dataclass
class Candidate:
    qid: str
    uri: str
    label: str
    description: str
    match: Optional[str] = None
    method: str = "search"


@dataclass
class PrivateEntity:
    uri: URIRef
    label: str
    type_uri: str

    @property
    def local_type(self) -> str:
        if "#" in self.type_uri:
            return self.type_uri.split("#", 1)[1]
        return self.type_uri.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------
def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_label(s: str) -> str:
    s = normalize_spaces(s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return s


def canonical_search_label(s: str) -> str:
    s = normalize_label(s)
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    s = s.replace('"', "")
    s = normalize_spaces(s)
    return s


def asciiish(s: str) -> str:
    s = canonical_search_label(s).lower()
    s = re.sub(r"[^a-z0-9\s'-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def short_local(uri: str) -> str:
    if "#" in uri:
        return uri.split("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def token_set(s: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", asciiish(s)))


def token_overlap_ratio(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, asciiish(a), asciiish(b)).ratio()


def looks_like_country_label(label: str) -> bool:
    low = asciiish(label)
    return low in {
        "austria", "france", "hungary", "italy", "norway", "russia",
        "sweden", "yugoslavia", "united states", "the united states",
        "mexico", "canada", "united kingdom",
    }


# ---------------------------------------------------------------------
# Search variant generation
# ---------------------------------------------------------------------
def search_variants(label: str, local_type: str) -> List[str]:
    label = canonical_search_label(label)
    variants: List[str] = [label]

    # remove leading article
    no_article = re.sub(r"^(the|a|an)\s+", "", label, flags=re.I).strip()
    if no_article and no_article != label:
        variants.append(no_article)

    # remove commas
    no_comma = label.replace(",", " ").strip()
    no_comma = normalize_spaces(no_comma)
    if no_comma and no_comma != label:
        variants.append(no_comma)

    # quoted nicknames
    no_quotes = label.replace('"', "")
    if no_quotes and no_quotes != label:
        variants.append(no_quotes)

    # people: try first+last if there are middle pieces
    if local_type in {"Character", "RealPerson"}:
        toks = label.split()
        if len(toks) >= 3:
            first_last = f"{toks[0]} {toks[-1]}"
            variants.append(first_last)

    # work/episode variants
    if local_type == "Episode":
        variants.append(f"{label} The Queen's Gambit")
        variants.append(f"{label} episode")
    if local_type in {"Work", "Book"} and "queen's gambit" in asciiish(label):
        variants.append("The Queen's Gambit")

    # unique keep order
    out: List[str] = []
    seen: Set[str] = set()
    for v in variants:
        vv = canonical_search_label(v)
        if vv and vv.lower() not in seen:
            seen.add(vv.lower())
            out.append(vv)
    return out


# ---------------------------------------------------------------------
# HTTP / Wikidata search
# ---------------------------------------------------------------------
def make_session(no_proxy: bool = True) -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    if no_proxy:
        s.trust_env = False
        s.proxies = {"http": None, "https": None}
    return s


def wikidata_search(
    label: str,
    language: str = "en",
    limit: int = 20,
    session: Optional[requests.Session] = None,
    sleep: float = 0.15,
    retries: int = 4,
    debug: bool = False,
) -> List[Candidate]:
    query = canonical_search_label(label)
    if not query:
        return []

    sess = session or make_session(no_proxy=True)
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": language,
        "format": "json",
        "limit": str(limit),
        "uselang": language,
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = sess.get(WIKIDATA_API, params=params, timeout=30)
            if debug:
                print("URL:", r.url)
                print("Status:", r.status_code)

            if r.status_code in {429, 503}:
                wait = sleep * (2 ** (attempt - 1))
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()

            out: List[Candidate] = []
            for item in data.get("search", []):
                qid = item.get("id") or ""
                if not qid:
                    continue
                out.append(
                    Candidate(
                        qid=qid,
                        uri=item.get("concepturi") or (WD_ENTITY + qid),
                        label=item.get("label") or "",
                        description=item.get("description") or "",
                        match=query,
                        method="search",
                    )
                )
            time.sleep(sleep)
            return out
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(sleep * (2 ** (attempt - 1)))

    if debug and last_err:
        print("Search failed:", repr(last_err))
    return []


def fetch_wikidata_entity_summary(
    qid: str,
    session: Optional[requests.Session] = None,
    sleep: float = 0.05,
    retries: int = 3,
) -> Optional[Candidate]:
    sess = session or make_session(no_proxy=True)
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "languages": "en",
        "format": "json",
        "props": "labels|descriptions",
    }
    for attempt in range(1, retries + 1):
        try:
            r = sess.get(WIKIDATA_API, params=params, timeout=30)
            if r.status_code in {429, 503}:
                time.sleep(sleep * (2 ** (attempt - 1)))
                continue
            r.raise_for_status()
            data = r.json()
            ent = data.get("entities", {}).get(qid, {})
            label = ent.get("labels", {}).get("en", {}).get("value", qid)
            desc = ent.get("descriptions", {}).get("en", {}).get("value", "")
            time.sleep(sleep)
            return Candidate(
                qid=qid,
                uri=WD_ENTITY + qid,
                label=label,
                description=desc,
                method="manual",
            )
        except Exception:
            time.sleep(sleep * (2 ** (attempt - 1)))
    return Candidate(qid=qid, uri=WD_ENTITY + qid, label=qid, description="", method="manual")


# ---------------------------------------------------------------------
# Manual link handling
# ---------------------------------------------------------------------
def manual_link_for(entity: PrivateEntity, session: requests.Session) -> Optional[Candidate]:
    key = (entity.label, entity.local_type)
    if key in MANUAL_LINKS:
        return fetch_wikidata_entity_summary(MANUAL_LINKS[key], session=session)

    # softer fallback for label-only critical entities
    if entity.label in MANUAL_LABEL_ONLY and entity.local_type in {"Episode", "Work", "Book"}:
        return fetch_wikidata_entity_summary(MANUAL_LABEL_ONLY[entity.label], session=session)

    return None


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------
def score_candidate(entity: PrivateEntity, cand: Candidate) -> float:
    q = canonical_search_label(entity.label)
    c = canonical_search_label(cand.label)
    desc = normalize_label(cand.description).lower()
    local_type = entity.local_type

    sim = string_similarity(q, c)
    tok = token_overlap_ratio(q, c)

    # lexical base
    if sim >= 0.985:
        score = 0.82
    elif sim >= 0.94:
        score = 0.76
    elif sim >= 0.88:
        score = 0.68
    elif sim >= 0.75:
        score = 0.58
    elif tok >= 0.80:
        score = 0.54
    elif tok >= 0.50:
        score = 0.45
    else:
        score = 0.28

    if q.lower() == c.lower():
        score += 0.08

    # if the matched query variant itself is close, reward it
    if cand.match:
        score += max(0.0, string_similarity(cand.match, cand.label) - 0.80) * 0.12

    # type hints
    hints = TYPE_HINTS.get(local_type, {})
    for good in hints.get("good", set()):
        if good in desc:
            score += 0.06
    for bad in hints.get("bad", set()):
        if bad in desc:
            score -= 0.10

    # domain-aware boosts
    if any(h in desc for h in QUEENS_GAMBIT_HINTS):
        score += 0.14

    # type-specific boosts/penalties
    if local_type == "Episode":
        if "episode" in desc:
            score += 0.08
        if "queen's gambit" in desc:
            score += 0.20
        else:
            score -= 0.22

    if local_type == "Character":
        if "fictional" in desc:
            score += 0.10
        if "queen's gambit" in desc:
            score += 0.15
        if "actor" in desc or "actress" in desc:
            score -= 0.15

    if local_type == "RealPerson":
        if "fictional" in desc:
            score -= 0.20
        if any(x in desc for x in {"actor", "actress", "director", "writer", "screenwriter", "producer", "chess player"}):
            score += 0.07

    if local_type in {"Work", "Book"}:
        if any(x in desc for x in {"novel", "series", "miniseries", "television series"}):
            score += 0.08
        if "queen's gambit" in desc:
            score += 0.16

    if local_type == "Location":
        if looks_like_country_label(q):
            # prefer countries/states/capitals over obscure towns for country-like labels
            if "country" in desc or "state of the united states" in desc or "capital" in desc:
                score += 0.10
            if "town in" in desc or "village in" in desc or "municipality" in desc:
                score -= 0.14

    if local_type == "Organization":
        if "building" in desc or "film" in desc or "aircraft" in desc:
            score -= 0.18

    # risky single token labels
    if len(q.split()) == 1 and local_type in {"Character", "Location", "Organization", "Event"}:
        score -= 0.07
        if entity.label.lower() in AMBIGUOUS_SINGLE_TOKEN_SKIP:
            score -= 0.08

    return max(0.0, min(0.99, score))


# ---------------------------------------------------------------------
# Entity extraction from ABox
# ---------------------------------------------------------------------
def extract_private_entities(g: Graph) -> List[PrivateEntity]:
    entities: List[PrivateEntity] = []
    seen: Set[Tuple[str, str, str]] = set()

    for s in set(g.subjects(RDF.type, None)):
        if not isinstance(s, URIRef):
            continue
        if (s, RDF.type, RDF.Statement) in g:
            continue

        label: Optional[str] = None
        type_uri: Optional[str] = None

        for lit in g.objects(s, RDFS.label):
            if isinstance(lit, Literal):
                label = normalize_label(str(lit))
                if label:
                    break

        for t in g.objects(s, RDF.type):
            if isinstance(t, URIRef):
                type_uri = str(t)
                break

        if not label or not type_uri:
            continue

        key = (str(s), label, type_uri)
        if key in seen:
            continue
        seen.add(key)
        entities.append(PrivateEntity(uri=s, label=label, type_uri=type_uri))

    return entities


# ---------------------------------------------------------------------
# TTL sameAs graph
# ---------------------------------------------------------------------
def build_sameas_graph(rows: List[dict], out_ns: Namespace) -> Graph:
    out = Graph()
    out.bind("qg", out_ns)
    out.bind("owl", OWL)
    out.bind("rdf", RDF)
    out.bind("rdfs", RDFS)
    out.bind("xsd", XSD)
    out.bind("wd", WD)

    CONF = out_ns.confidence
    WD_LABEL = out_ns.wikidataLabel
    WD_DESC = out_ns.wikidataDescription
    LINK_METHOD = out_ns.linkingMethod
    MATCH_QUERY = out_ns.matchedQuery

    for r in rows:
        s = URIRef(r["private_entity"])
        wd_uri = URIRef(r["wikidata_uri"])
        out.add((s, OWL.sameAs, wd_uri))

        st = URIRef(f"{r['private_entity']}/sameAs/{r['wikidata_id']}")
        out.add((st, RDF.type, RDF.Statement))
        out.add((st, RDF.subject, s))
        out.add((st, RDF.predicate, OWL.sameAs))
        out.add((st, RDF.object, wd_uri))
        out.add((st, CONF, Literal(str(r["confidence"]), datatype=XSD.decimal)))

        if r.get("wd_label"):
            out.add((st, WD_LABEL, Literal(str(r["wd_label"]), lang="en")))
        if r.get("wd_description"):
            out.add((st, WD_DESC, Literal(str(r["wd_description"]), lang="en")))
        if r.get("linking_method"):
            out.add((st, LINK_METHOD, Literal(str(r["linking_method"]))))
        if r.get("matched_query"):
            out.add((st, MATCH_QUERY, Literal(str(r["matched_query"]))))

    return out


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------
def load_cache(path: Path) -> Dict[str, List[dict]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: Dict[str, List[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Type-aware entity linking from private ABox TTL to Wikidata.")
    p.add_argument("--abox", type=Path, default=Path("data/queens_gambit_graph_abox.ttl"))
    p.add_argument("--out_csv", type=Path, default=Path("data/entity_wikidata_mapping.csv"))
    p.add_argument("--out_ttl", type=Path, default=Path("data/entity_sameas.ttl"))
    p.add_argument("--cache", type=Path, default=Path("data/.wikidata_search_cache.json"))
    p.add_argument("--min_conf", type=float, default=0.70)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--sleep", type=float, default=0.10)
    p.add_argument(
        "--types",
        type=str,
        default="Character,RealPerson,Organization,Location,Event,Work,Episode,Book",
        help="Comma-separated local class names to link.",
    )
    p.add_argument("--lang", type=str, default="en")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    wanted = {t.strip() for t in args.types.split(",") if t.strip()}

    g = Graph()
    g.parse(args.abox)

    entities = extract_private_entities(g)
    sess = make_session(no_proxy=True)
    cache = load_cache(args.cache)

    rows_out: List[dict] = []

    for ent in entities:
        local_type = ent.local_type

        if wanted and local_type not in wanted:
            continue

        low_label = ent.label.strip().lower()
        if low_label in SKIP_LABELS:
            continue
        if len(ent.label) < 2:
            continue

        # risky short labels
        if local_type in {"Character", "Organization", "Location", "Event"}:
            if len(ent.label.split()) == 1 and len(ent.label) < 5:
                continue

        # manual high-confidence override
        manual = manual_link_for(ent, sess)
        if manual is not None:
            rows_out.append(
                {
                    "private_entity": str(ent.uri),
                    "private_label": ent.label,
                    "private_type": ent.type_uri,
                    "wikidata_id": manual.qid,
                    "wikidata_uri": manual.uri,
                    "confidence": "1.00",
                    "wd_label": manual.label,
                    "wd_description": manual.description,
                    "linking_method": "manual",
                    "matched_query": ent.label,
                }
            )
            continue

        merged_candidates: Dict[str, Candidate] = {}

        for variant in search_variants(ent.label, local_type):
            cache_key = f"{variant}||{local_type}||{args.lang}||{args.limit}"
            if cache_key in cache:
                cands = [Candidate(**c) for c in cache[cache_key]]
            else:
                cands = wikidata_search(
                    variant,
                    language=args.lang,
                    limit=max(1, args.limit),
                    session=sess,
                    sleep=args.sleep,
                    debug=args.debug,
                )
                cache[cache_key] = [asdict(c) for c in cands]

            for c in cands:
                old = merged_candidates.get(c.qid)
                if old is None:
                    c.match = variant
                    merged_candidates[c.qid] = c
                else:
                    new_sim = string_similarity(variant, c.label)
                    old_sim = string_similarity(old.match or ent.label, old.label)
                    if new_sim > old_sim:
                        c.match = variant
                        merged_candidates[c.qid] = c

        if not merged_candidates:
            continue

        scored: List[Tuple[float, Candidate]] = []
        for cand in merged_candidates.values():
            conf = score_candidate(ent, cand)
            scored.append((conf, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_conf, best = scored[0]
        second_conf = scored[1][0] if len(scored) > 1 else 0.0

        # ambiguity control
        margin = best_conf - second_conf
        if best_conf < args.min_conf:
            continue
        if best_conf < 0.86 and margin < 0.04:
            continue

        # extra strictness for episodes
        if local_type == "Episode" and "queen's gambit" not in best.description.lower():
            continue

        rows_out.append(
            {
                "private_entity": str(ent.uri),
                "private_label": ent.label,
                "private_type": ent.type_uri,
                "wikidata_id": best.qid,
                "wikidata_uri": best.uri,
                "confidence": f"{best_conf:.2f}",
                "wd_label": best.label,
                "wd_description": best.description,
                "linking_method": best.method,
                "matched_query": best.match or ent.label,
            }
        )

    # deduplicate one private entity -> one Wikidata entity
    rows_out = sorted(
        rows_out,
        key=lambda r: (r["private_entity"], float(r["confidence"])),
        reverse=False,
    )
    dedup_best: Dict[str, dict] = {}
    for row in rows_out:
        pe = row["private_entity"]
        old = dedup_best.get(pe)
        if old is None or float(row["confidence"]) > float(old["confidence"]):
            dedup_best[pe] = row
    rows_final = list(dedup_best.values())

    save_cache(args.cache, cache)

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
                "linking_method",
                "matched_query",
            ],
        )
        w.writeheader()
        for r in rows_final:
            w.writerow(r)

    sameas_g = build_sameas_graph(rows_final, out_ns=QG)
    args.out_ttl.parent.mkdir(parents=True, exist_ok=True)
    sameas_g.serialize(destination=str(args.out_ttl), format="turtle")

    print(f"Input entities: {len(entities)}")
    print(f"Linked entities: {len(rows_final)}")
    print(f"Mapping CSV: {args.out_csv}")
    print(f"sameAs TTL: {args.out_ttl}")


if __name__ == "__main__":
    main()