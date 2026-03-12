#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL


QG_PREFIX = "http://example.org/qg#"
WD_PREFIX = "http://www.wikidata.org/entity/"


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


def is_entity_uri(x) -> bool:
    return isinstance(x, URIRef)


def is_literal_heavy_predicate(p: str) -> bool:
    low = p.lower()
    bad = {
        str(RDFS.label).lower(),
        str(RDF.type).lower(),
        str(OWL.sameAs).lower(),
        "http://schema.org/description",
        f"{QG_PREFIX}wikidatadescription".lower(),
        f"{QG_PREFIX}wikidatalabel".lower(),
        f"{QG_PREFIX}confidence".lower(),
        f"{QG_PREFIX}linkingmethod".lower(),
        f"{QG_PREFIX}matchedquery".lower(),
    }
    return low in bad


def load_graph(path: Path) -> Graph:
    g = Graph()
    g.parse(str(path))
    return g


def build_label_index(g: Graph) -> Dict[str, Set[str]]:
    """
    Index simple label -> set(uri)
    """
    index: Dict[str, Set[str]] = defaultdict(set)

    for s, _, o in g.triples((None, RDFS.label, None)):
        if isinstance(s, URIRef) and isinstance(o, Literal):
            label = str(o).strip()
            if label:
                index[normalize_text(label)].add(str(s))

    # fallback: also index local name
    for s in set(g.subjects()):
        if isinstance(s, URIRef):
            local = short_uri(str(s)).replace("_", " ")
            norm = normalize_text(local)
            if norm:
                index[norm].add(str(s))

    return index


def match_entities_in_question(question: str, label_index: Dict[str, Set[str]]) -> List[str]:
    q = normalize_text(question)
    matches: List[Tuple[int, str]] = []

    for label_norm, uris in label_index.items():
        if not label_norm or len(label_norm) < 3:
            continue
        if label_norm in q:
            for uri in uris:
                matches.append((len(label_norm), uri))

    # longest matches first, dedup
    matches.sort(reverse=True)
    seen = set()
    out = []
    for _, uri in matches:
        if uri not in seen:
            seen.add(uri)
            out.append(uri)

    return out[:10]


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
            if cls not in {"Statement"}:
                return cls
    return "Unknown"


def get_one_hop_facts(
    g: Graph,
    seed_uris: List[str],
    max_facts_per_entity: int = 20,
) -> List[Dict]:
    facts: List[Dict] = []

    for uri in seed_uris:
        s = URIRef(uri)
        count = 0

        for _, p, o in g.triples((s, None, None)):
            p_str = str(p)
            if is_literal_heavy_predicate(p_str):
                continue

            subj_label = get_best_label(g, uri)
            pred_label = short_uri(p_str)
            obj_label = get_best_label(g, str(o)) if isinstance(o, URIRef) else str(o)

            facts.append(
                {
                    "subject_uri": uri,
                    "subject_label": subj_label,
                    "subject_class": entity_class(g, uri),
                    "predicate_uri": p_str,
                    "predicate_label": pred_label,
                    "object_uri": str(o) if isinstance(o, URIRef) else None,
                    "object_label": obj_label,
                    "object_class": entity_class(g, str(o)) if isinstance(o, URIRef) else "Literal",
                    "direction": "out",
                    "fact_text": f"{subj_label} --{pred_label}--> {obj_label}",
                }
            )
            count += 1
            if count >= max_facts_per_entity:
                break

        count = 0
        for s2, p, _ in g.triples((None, None, s)):
            p_str = str(p)
            if is_literal_heavy_predicate(p_str):
                continue

            subj_label = get_best_label(g, str(s2))
            pred_label = short_uri(p_str)
            obj_label = get_best_label(g, uri)

            facts.append(
                {
                    "subject_uri": str(s2),
                    "subject_label": subj_label,
                    "subject_class": entity_class(g, str(s2)),
                    "predicate_uri": p_str,
                    "predicate_label": pred_label,
                    "object_uri": uri,
                    "object_label": obj_label,
                    "object_class": entity_class(g, uri),
                    "direction": "in",
                    "fact_text": f"{subj_label} --{pred_label}--> {obj_label}",
                }
            )
            count += 1
            if count >= max_facts_per_entity:
                break

    return facts


def rank_facts(question: str, facts: List[Dict]) -> List[Dict]:
    q = normalize_text(question)

    def score_fact(f: Dict) -> float:
        score = 0.0
        s = normalize_text(f["subject_label"])
        p = normalize_text(f["predicate_label"])
        o = normalize_text(f["object_label"])

        if s in q:
            score += 2.0
        if o in q:
            score += 2.0
        if p in q:
            score += 1.0

        # heuristique : on favorise les faits lisibles entité -> entité
        if f["object_class"] != "Literal":
            score += 0.5

        # bonus léger pour les classes utiles
        if f["subject_class"] in {"Character", "Episode", "RealPerson", "Work"}:
            score += 0.2
        if f["object_class"] in {"Character", "Episode", "RealPerson", "Work", "Location", "Organization"}:
            score += 0.2

        return score

    facts = sorted(facts, key=score_fact, reverse=True)
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve one-hop KG facts relevant to a question")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="User question",
    )
    parser.add_argument(
        "--kg",
        type=Path,
        default=Path("data/expanded_kb.ttl"),
        help="Path to the expanded KG TTL",
    )
    parser.add_argument(
        "--top_k_entities",
        type=int,
        default=5,
        help="Max matched seed entities",
    )
    parser.add_argument(
        "--top_k_facts",
        type=int,
        default=10,
        help="Max facts to print",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to save results",
    )
    args = parser.parse_args()

    print(f"Loading KG: {args.kg}")
    g = load_graph(args.kg)

    print("Building label index...")
    label_index = build_label_index(g)

    print(f"Question: {args.question}")
    matched_entities = match_entities_in_question(args.question, label_index)[: args.top_k_entities]

    print("\n=== MATCHED ENTITIES ===")
    for i, uri in enumerate(matched_entities, start=1):
        print(f"[{i}] {get_best_label(g, uri)} ({entity_class(g, uri)}) -> {uri}")

    facts = get_one_hop_facts(g, matched_entities, max_facts_per_entity=20)
    facts = rank_facts(args.question, facts)
    facts = facts[: args.top_k_facts]

    print("\n=== TOP KG FACTS ===\n")
    for i, f in enumerate(facts, start=1):
        print(f"[{i}] {f['fact_text']}")
        print(f"    subject_class={f['subject_class']} | object_class={f['object_class']}")
        print("-" * 80)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as out:
            json.dump(
                {
                    "question": args.question,
                    "matched_entities": [
                        {
                            "uri": uri,
                            "label": get_best_label(g, uri),
                            "class": entity_class(g, uri),
                        }
                        for uri in matched_entities
                    ],
                    "facts": facts,
                },
                out,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved results to: {args.output_json}")


if __name__ == "__main__":
    main()