#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL


QG_PREFIX = "http://example.org/qg#"

WD_PREDICATE_LABELS = {
    "P31": "instanceOf",
    "P106": "occupation",
    "P1412": "languageSpoken",
    "P1441": "presentInWork",
    "P170": "creator",
    "P175": "performer",
    "P50": "author",
    "P57": "director",
    "P161": "castMember",
    "P179": "partOfSeries",
    "P361": "partOf",
    "P527": "hasPart",
    "P17": "country",
    "P131": "locatedIn",
    "P276": "location",
    "P495": "countryOfOrigin",
    "P449": "originalBroadcaster",
}

BLOCKED_PREDICATES = {
    str(RDFS.label).lower(),
    str(OWL.sameAs).lower(),
    "http://schema.org/description",
    f"{QG_PREFIX}wikidatalabel".lower(),
    f"{QG_PREFIX}wikidatadescription".lower(),
    f"{QG_PREFIX}confidence".lower(),
    f"{QG_PREFIX}linkingmethod".lower(),
    f"{QG_PREFIX}matchedquery".lower(),
}

LOW_VALUE_PREDICATES = {
    "altlabel",
    "matchedquery",
    "confidence",
    "linkingmethod",
    "subject",
    "predicate",
    "object",
}


# =========================
# Helpers
# =========================

def load_graph(path: Path) -> Graph:
    g = Graph()
    g.parse(str(path))
    return g


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9\s'_:-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def short_uri(uri: str) -> str:
    if "#" in uri:
        return uri.split("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def predicate_label(uri: str) -> str:
    local = short_uri(uri)
    return WD_PREDICATE_LABELS.get(local, local)


def extract_entity_name(question: str) -> str:
    q = question.strip().rstrip("?")

    m = re.match(r"^(who|what)\s+(is|are)\s+(.+)$", q, flags=re.I)
    if m:
        tail = m.group(3).strip()

        m2 = re.match(r"^happening in (?:the )?episode (.+)$", tail, flags=re.I)
        if m2:
            return m2.group(1).strip()

        return tail

    m3 = re.match(r"^(what)\s+happens in (?:the )?episode (.+)$", q, flags=re.I)
    if m3:
        return m3.group(2).strip()

    m4 = re.match(r"^who\s+portrays\s+(.+)$", q, flags=re.I)
    if m4:
        return m4.group(1).strip()

    return q


def build_label_index(g: Graph) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}

    for s, _, o in g.triples((None, RDFS.label, None)):
        if isinstance(s, URIRef) and isinstance(o, Literal):
            label = str(o).strip()
            if label:
                key = normalize_text(label)
                index.setdefault(key, [])
                if str(s) not in index[key]:
                    index[key].append(str(s))

    for s in set(g.subjects()):
        if isinstance(s, URIRef):
            local = short_uri(str(s)).replace("_", " ")
            key = normalize_text(local)
            if key:
                index.setdefault(key, [])
                if str(s) not in index[key]:
                    index[key].append(str(s))

    return index


def get_best_label(g: Graph, uri: str) -> str:
    u = URIRef(uri)
    labels = [str(o) for o in g.objects(u, RDFS.label) if isinstance(o, Literal)]
    if labels:
        return labels[0]
    return short_uri(uri)


def entity_class(g: Graph, uri: str) -> str:
    u = URIRef(uri)
    for o in g.objects(u, RDF.type):
        if isinstance(o, URIRef):
            cls = short_uri(str(o))
            if cls != "Statement":
                return cls
    return "Unknown"


def match_entities(question: str, label_index: Dict[str, List[str]], top_k: int = 5) -> List[str]:
    q = normalize_text(question)
    matches = []

    for label_norm, uris in label_index.items():
        if len(label_norm) < 3:
            continue
        if label_norm in q:
            for uri in uris:
                matches.append((len(label_norm), uri))

    matches.sort(reverse=True)
    seen = set()
    out = []
    for _, uri in matches:
        if uri not in seen:
            seen.add(uri)
            out.append(uri)

    return out[:top_k]


def is_blocked_predicate(p: str) -> bool:
    return p.lower() in BLOCKED_PREDICATES


def classify_question(question: str) -> str:
    q = question.lower().strip()

    if "what is happening in" in q or "what happens in" in q:
        return "episode_summary"

    if q.startswith("who is ") or q.startswith("what is ") or q.startswith("who are ") or q.startswith("what are "):
        return "identity"

    if q.startswith("who portrays ") or "portrays" in q or "played by" in q:
        return "portrayal"

    if "which episode" in q or "what episode" in q or "appears in" in q:
        return "episode_membership"

    return "generic"


# =========================
# NL -> pseudo-SPARQL routing
# =========================

def question_to_sparql(question: str, entity_uri: str, entity_label: str) -> str:
    q = normalize_text(question)

    if q.startswith("who is") or q.startswith("what is"):
        return f"""
SELECT ?type WHERE {{
    <{entity_uri}> rdf:type ?type .
}}
""".strip()

    if q.startswith("who portrays ") or "played by" in q or "portrays" in q:
        return f"""
SELECT ?p ?o WHERE {{
    <{entity_uri}> ?p ?o .
}}
LIMIT 10
""".strip()

    return f"""
SELECT ?p ?o WHERE {{
    <{entity_uri}> ?p ?o .
}}
LIMIT 10
""".strip()


def run_sparql_query(g: Graph, query: str):
    try:
        return list(g.query(query))
    except Exception as e:
        print("[SPARQL ERROR]", e)
        return []


def fact_dict(
    *,
    score: float,
    subject_uri: str,
    subject_label: str,
    subject_class: str,
    predicate_uri: str,
    predicate_label_: str,
    object_uri: str | None,
    object_label: str,
    object_class: str,
) -> Dict:
    return {
        "score": score,
        "subject_uri": subject_uri,
        "subject_label": subject_label,
        "subject_class": subject_class,
        "predicate_uri": predicate_uri,
        "predicate_label": predicate_label_,
        "object_uri": object_uri,
        "object_label": object_label,
        "object_class": object_class,
        "fact_text": f"{subject_label} --{predicate_label_}--> {object_label}",
    }


def dedupe_facts(facts: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for f in facts:
        key = (
            f["subject_uri"],
            f["predicate_uri"],
            f["object_uri"],
            f["object_label"],
        )
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


# =========================
# Main retrieval
# =========================

def sparql_retrieve(
    question: str,
    kg_path: Path,
    top_k_entities: int = 5,
    top_k_facts: int = 10,
) -> Dict:
    g = load_graph(kg_path)
    label_index = build_label_index(g)
    matched = match_entities(question, label_index, top_k=top_k_entities)
    qtype = classify_question(question)

    all_facts: List[Dict] = []
    sparql_queries: List[str] = []

    for uri in matched:
        if "/sameAs/" in uri:
            continue

        subj_label = get_best_label(g, uri)
        subj_class = entity_class(g, uri)
        s = URIRef(uri)

        query = question_to_sparql(question, uri, subj_label)
        sparql_queries.append(query)
        _ = run_sparql_query(g, query)

        if qtype == "identity":
            for _, _, o in g.triples((s, RDF.type, None)):
                if isinstance(o, URIRef):
                    cls_label = short_uri(str(o))
                    all_facts.append(
                        fact_dict(
                            score=6.0,
                            subject_uri=uri,
                            subject_label=subj_label,
                            subject_class=subj_class,
                            predicate_uri=str(RDF.type),
                            predicate_label_="type",
                            object_uri=str(o),
                            object_label=cls_label,
                            object_class="Class",
                        )
                    )

        for _, p, o in g.triples((s, None, None)):
            p_str = str(p)
            if is_blocked_predicate(p_str):
                continue

            pred_label_ = predicate_label(p_str)
            if normalize_text(pred_label_) in LOW_VALUE_PREDICATES:
                continue

            obj_label = get_best_label(g, str(o)) if isinstance(o, URIRef) else str(o)
            obj_class = entity_class(g, str(o)) if isinstance(o, URIRef) else "Literal"

            score = 5.0
            if qtype == "portrayal" and normalize_text(pred_label_) in {"portrays", "performer", "castmember"}:
                score = 7.0
            elif qtype == "episode_membership" and normalize_text(pred_label_) in {"appearsinepisode", "partofseries"}:
                score = 7.0
            elif qtype == "episode_summary" and normalize_text(pred_label_) in {"partofseries", "director", "originalbroadcaster"}:
                score = 6.5

            all_facts.append(
                fact_dict(
                    score=score,
                    subject_uri=uri,
                    subject_label=subj_label,
                    subject_class=subj_class,
                    predicate_uri=p_str,
                    predicate_label_=pred_label_,
                    object_uri=str(o) if isinstance(o, URIRef) else None,
                    object_label=obj_label,
                    object_class=obj_class,
                )
            )

        for s2, p, _ in g.triples((None, None, s)):
            if isinstance(s2, URIRef) and "/sameAs/" in str(s2):
                continue

            p_str = str(p)
            if is_blocked_predicate(p_str):
                continue

            pred_label_ = predicate_label(p_str)
            if normalize_text(pred_label_) in LOW_VALUE_PREDICATES:
                continue

            subj2_label = get_best_label(g, str(s2))
            subj2_class = entity_class(g, str(s2))

            score = 4.5
            if qtype == "portrayal" and normalize_text(pred_label_) in {"portrays", "performer", "castmember"}:
                score = 7.0

            all_facts.append(
                fact_dict(
                    score=score,
                    subject_uri=str(s2),
                    subject_label=subj2_label,
                    subject_class=subj2_class,
                    predicate_uri=p_str,
                    predicate_label_=pred_label_,
                    object_uri=uri,
                    object_label=subj_label,
                    object_class=subj_class,
                )
            )

    all_facts = dedupe_facts(all_facts)
    all_facts.sort(key=lambda x: x["score"], reverse=True)
    all_facts = all_facts[:top_k_facts]

    return {
        "matched_entities": [
            {
                "uri": uri,
                "label": get_best_label(g, uri),
                "class": entity_class(g, uri),
            }
            for uri in matched
            if "/sameAs/" not in uri
        ],
        "facts": all_facts,
        "sparql_queries": sparql_queries,
    }