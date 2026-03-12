#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from sentence_transformers import SentenceTransformer


QG_PREFIX = "http://example.org/qg#"

# ---------------------------------------------------------
# Wikidata predicate labels
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Predicates to ignore in final evidence
# ---------------------------------------------------------
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

IDENTITY_PRIORITY_PREDICATES = {
    "type",
    "hastype",
    "instanceof",
    "presentinwork",
    "partofseries",
    "appearsinepisode",
    "portrays",
    "creator",
    "author",
    "director",
    "castmember",
    "performer",
}

IDENTITY_PENALTY_PREDICATES = {
    "met",
    "defeated",
    "lostto",
    "wonagainst",
    "interactswith",
}


# =========================================================
# Generic helpers
# =========================================================

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9\s'_:-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def short_uri(uri: str) -> str:
    if "#" in uri:
        return uri.split("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def predicate_label(uri: str) -> str:
    local = short_uri(uri)
    return WD_PREDICATE_LABELS.get(local, local)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# =========================================================
# Question typing
# =========================================================

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


def extract_focus_phrase(question: str) -> str:
    q = question.strip()
    m = re.match(r"^(who|what)\s+(is|are)\s+(.+?)\??$", q, flags=re.I)
    if m:
        return m.group(3).strip()
    return q.rstrip("?").strip()


# =========================================================
# Text retrieval
# =========================================================

def text_retrieve(
    question: str,
    index_dir: Path,
    chunks_path: Path,
    top_k: int = 10,
) -> List[Dict]:
    config = load_json(index_dir / "index_config.json")
    metadata: List[Dict] = load_json(index_dir / "chunk_metadata.json")
    embeddings = np.load(index_dir / "chunk_embeddings.npy")

    chunk_rows = load_jsonl(chunks_path)
    chunk_text_by_id: Dict[str, str] = {str(r["chunk_id"]): str(r.get("text", "")) for r in chunk_rows}

    model = SentenceTransformer(config["model_name"])
    query_vec = model.encode([question], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    query_vec = l2_normalize(query_vec)

    faiss_path = index_dir / "chunk_faiss.index"
    if HAS_FAISS and faiss_path.exists():
        index = faiss.read_index(str(faiss_path))
        scores, indices = index.search(query_vec.astype(np.float32), top_k)
        scores = scores[0]
        indices = indices[0]
    else:
        embeddings = l2_normalize(embeddings.astype(np.float32))
        sims = embeddings @ query_vec[0]
        indices = np.argsort(-sims)[:top_k]
        scores = sims[indices]

    results = []
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[int(idx)]
        chunk_id = str(meta.get("chunk_id"))
        text = chunk_text_by_id.get(chunk_id, "")
        results.append(
            {
                "rank": rank,
                "vector_score": float(score),
                "chunk_id": chunk_id,
                "title": meta.get("title"),
                "url": meta.get("url"),
                "page_id": meta.get("page_id"),
                "chunk_index": meta.get("chunk_index"),
                "entities": meta.get("entities", []),
                "text": text,
            }
        )
    return results


def rerank_text_results(question: str, results: List[Dict]) -> List[Dict]:
    qtype = classify_question(question)
    focus = normalize_text(extract_focus_phrase(question))

    reranked = []
    for r in results:
        score = r["vector_score"]
        title = normalize_text(r.get("title", ""))
        text = normalize_text(r.get("text", ""))

        if qtype == "identity":
            if title == focus:
                score += 0.50
            elif focus in title:
                score += 0.30

            if text.startswith(focus):
                score += 0.40

            intro_zone = text[:300]
            if f"{focus} is " in intro_zone:
                score += 0.35
            if f"{focus} was " in intro_zone:
                score += 0.25
            if f"{focus} (" in intro_zone:
                score += 0.20

        if focus and focus in text:
            score += 0.10

        rr = dict(r)
        rr["rerank_score"] = score
        reranked.append(rr)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    deduped = []
    seen = set()
    for r in reranked:
        key = (r["title"], r["chunk_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    for i, r in enumerate(deduped, start=1):
        r["rank"] = i

    return deduped


# =========================================================
# KG retrieval
# =========================================================

def load_graph(path: Path) -> Graph:
    g = Graph()
    g.parse(str(path))
    return g


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


def score_fact(question_type: str, pred_label: str, subj_class: str, obj_class: str) -> float:
    pred_norm = normalize_text(pred_label)
    score = 1.0

    if question_type == "identity":
        if pred_norm in IDENTITY_PRIORITY_PREDICATES:
            score += 4.0
        if pred_norm in IDENTITY_PENALTY_PREDICATES:
            score -= 1.5

    if question_type == "portrayal" and pred_norm == "portrays":
        score += 5.0

    if question_type == "episode_membership" and pred_norm in {"appearsinepisode", "partofseries"}:
        score += 5.0

    if subj_class in {"Character", "Episode", "RealPerson", "Work"}:
        score += 0.2
    if obj_class in {"Character", "Episode", "RealPerson", "Work", "Location", "Organization", "Class"}:
        score += 0.2

    if pred_norm in LOW_VALUE_PREDICATES:
        score -= 2.0

    return score


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


def kg_retrieve(question: str, kg_path: Path, top_k_entities: int = 5, top_k_facts: int = 12) -> Dict:
    g = load_graph(kg_path)
    label_index = build_label_index(g)
    matched = match_entities(question, label_index, top_k=top_k_entities)
    qtype = classify_question(question)

    facts = []
    for uri in matched:
        s = URIRef(uri)
        subj_label = get_best_label(g, uri)
        subj_class = entity_class(g, uri)

        # explicit rdf:type for identity questions
        if qtype == "identity":
            for _, _, o in g.triples((s, RDF.type, None)):
                if isinstance(o, URIRef):
                    cls_label = short_uri(str(o))
                    facts.append(
                        {
                            "score": 6.0,
                            "subject_uri": uri,
                            "subject_label": subj_label,
                            "subject_class": subj_class,
                            "predicate_uri": str(RDF.type),
                            "predicate_label": "type",
                            "object_uri": str(o),
                            "object_label": cls_label,
                            "object_class": "Class",
                            "fact_text": f"{subj_label} --type--> {cls_label}",
                        }
                    )

        # outgoing
        for _, p, o in g.triples((s, None, None)):
            p_str = str(p)
            if is_blocked_predicate(p_str):
                continue

            pred_label = predicate_label(p_str)
            pred_norm = normalize_text(pred_label)
            obj_label = get_best_label(g, str(o)) if isinstance(o, URIRef) else str(o)
            obj_class = entity_class(g, str(o)) if isinstance(o, URIRef) else "Literal"

            if pred_norm in LOW_VALUE_PREDICATES:
                continue

            score = score_fact(qtype, pred_label, subj_class, obj_class)

            facts.append(
                {
                    "score": score,
                    "subject_uri": uri,
                    "subject_label": subj_label,
                    "subject_class": subj_class,
                    "predicate_uri": p_str,
                    "predicate_label": pred_label,
                    "object_uri": str(o) if isinstance(o, URIRef) else None,
                    "object_label": obj_label,
                    "object_class": obj_class,
                    "fact_text": f"{subj_label} --{pred_label}--> {obj_label}",
                }
            )

        # incoming
        for s2, p, _ in g.triples((None, None, s)):
            p_str = str(p)
            if is_blocked_predicate(p_str):
                continue

            subj2_label = get_best_label(g, str(s2))
            subj2_class = entity_class(g, str(s2))
            pred_label = predicate_label(p_str)
            pred_norm = normalize_text(pred_label)

            if pred_norm in LOW_VALUE_PREDICATES:
                continue

            score = score_fact(qtype, pred_label, subj2_class, subj_class)

            facts.append(
                {
                    "score": score,
                    "subject_uri": str(s2),
                    "subject_label": subj2_label,
                    "subject_class": subj2_class,
                    "predicate_uri": p_str,
                    "predicate_label": pred_label,
                    "object_uri": uri,
                    "object_label": subj_label,
                    "object_class": subj_class,
                    "fact_text": f"{subj2_label} --{pred_label}--> {subj_label}",
                }
            )

    facts = dedupe_facts(facts)
    facts.sort(key=lambda x: x["score"], reverse=True)
    facts = facts[:top_k_facts]

    return {
        "matched_entities": [
            {
                "uri": uri,
                "label": get_best_label(g, uri),
                "class": entity_class(g, uri),
            }
            for uri in matched
        ],
        "facts": facts,
    }


# =========================================================
# Evidence pack
# =========================================================

def build_evidence_pack(question: str, text_results: List[Dict], kg_results: Dict) -> Dict:
    return {
        "question": question,
        "question_type": classify_question(question),
        "matched_entities": kg_results["matched_entities"],
        "text_evidence": [
            {
                "rank": r["rank"],
                "score": r["rerank_score"],
                "title": r["title"],
                "url": r["url"],
                "chunk_id": r["chunk_id"],
                "text": r["text"],
            }
            for r in text_results
        ],
        "kg_evidence": [
            {
                "rank": i + 1,
                "score": f["score"],
                "fact_text": f["fact_text"],
                "subject_class": f["subject_class"],
                "object_class": f["object_class"],
                "subject_uri": f["subject_uri"],
                "predicate_uri": f["predicate_uri"],
                "predicate_label": f["predicate_label"],
                "object_uri": f["object_uri"],
                "object_label": f["object_label"],
            }
            for i, f in enumerate(kg_results["facts"])
        ],
    }


def print_pack(pack: Dict) -> None:
    print("\n==============================")
    print("QUESTION")
    print("==============================")
    print(pack["question"])
    print(f"type: {pack['question_type']}")

    print("\n==============================")
    print("MATCHED ENTITIES")
    print("==============================")
    for i, e in enumerate(pack["matched_entities"], start=1):
        print(f"[{i}] {e['label']} ({e['class']}) -> {e['uri']}")

    print("\n==============================")
    print("TEXT EVIDENCE")
    print("==============================")
    for e in pack["text_evidence"]:
        print(f"[{e['rank']}] score={e['score']:.4f} | {e['title']}")
        print(f"url: {e['url']}")
        print(f"text: {clean_text(e['text'])[:450]}")
        print("-" * 80)

    print("\n==============================")
    print("KG EVIDENCE")
    print("==============================")
    for e in pack["kg_evidence"]:
        print(f"[{e['rank']}] score={e['score']:.4f}")
        print(e["fact_text"])
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retriever with reranking")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--chunks_path", type=Path, default=Path("data/rag/chunks.jsonl"))
    parser.add_argument("--index_dir", type=Path, default=Path("data/rag/index"))
    parser.add_argument("--kg", type=Path, default=Path("data/expanded_kb.ttl"))
    parser.add_argument("--top_k_text", type=int, default=6)
    parser.add_argument("--top_k_entities", type=int, default=5)
    parser.add_argument("--top_k_facts", type=int, default=10)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    text_results = text_retrieve(
        question=args.question,
        index_dir=args.index_dir,
        chunks_path=args.chunks_path,
        top_k=args.top_k_text,
    )
    text_results = rerank_text_results(args.question, text_results)

    kg_results = kg_retrieve(
        question=args.question,
        kg_path=args.kg,
        top_k_entities=args.top_k_entities,
        top_k_facts=args.top_k_facts,
    )

    pack = build_evidence_pack(args.question, text_results, kg_results)
    print_pack(pack)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(pack, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {args.output_json}")


if __name__ == "__main__":
    main()