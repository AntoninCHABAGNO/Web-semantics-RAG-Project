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


def extract_focus(question: str) -> str:
    q = question.strip().rstrip("?")

    m = re.match(r"^(Who|What)\s+(is|are)\s+(.+)$", q, flags=re.I)
    if m:
        tail = m.group(3).strip()
        m2 = re.match(r"^happening in (?:the )?episode (.+)$", tail, flags=re.I)
        if m2:
            return m2.group(1).strip()
        return tail

    m3 = re.match(r"^(What)\s+happens in (?:the )?episode (.+)$", q, flags=re.I)
    if m3:
        return m3.group(2).strip()

    return q


def dedupe_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        key = x.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(x)
    return out


def parse_fact_text(fact_text: str) -> Tuple[str, str, str] | None:
    m = re.match(r"^(.*?)\s+--(.*?)-->\s+(.*?)$", fact_text)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


# =======================
# TEXT RETRIEVAL
# =======================

def text_retrieve(question: str, index_dir: Path, chunks_path: Path, top_k: int = 10) -> List[Dict]:
    config = load_json(index_dir / "index_config.json")
    metadata: List[Dict] = load_json(index_dir / "chunk_metadata.json")
    embeddings = np.load(index_dir / "chunk_embeddings.npy")

    chunk_rows = load_jsonl(chunks_path)
    chunk_text_by_id = {str(r["chunk_id"]): str(r.get("text", "")) for r in chunk_rows}

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
        results.append(
            {
                "rank": rank,
                "vector_score": float(score),
                "chunk_id": chunk_id,
                "title": meta.get("title"),
                "url": meta.get("url"),
                "text": chunk_text_by_id.get(chunk_id, ""),
            }
        )
    return results


def rerank_text_results(question: str, results: List[Dict]) -> List[Dict]:
    qtype = classify_question(question)
    focus = normalize_text(extract_focus(question))

    reranked = []
    for r in results:
        score = r["vector_score"]
        title = normalize_text(r.get("title", ""))
        text = normalize_text(r.get("text", ""))

        if qtype in {"identity", "episode_summary"}:
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
    for i, r in enumerate(reranked, start=1):
        r["rank"] = i
    return reranked


# =======================
# KG RETRIEVAL
# =======================

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

    if question_type == "portrayal" and pred_norm in {"portrays", "performer"}:
        score += 5.0

    if question_type == "episode_membership" and pred_norm in {"appearsinepisode", "partofseries"}:
        score += 5.0

    if question_type == "episode_summary" and pred_norm in {"partofseries", "instanceof", "director", "originalbroadcaster"}:
        score += 4.0

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
        key = (f["subject_uri"], f["predicate_uri"], f["object_uri"], f["object_label"])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def kg_retrieve(question: str, kg_path: Path, top_k_entities: int = 5, top_k_facts: int = 10) -> Dict:
    g = load_graph(kg_path)
    label_index = build_label_index(g)
    matched = match_entities(question, label_index, top_k=top_k_entities)
    qtype = classify_question(question)

    facts = []
    for uri in matched:
        if "/sameAs/" in uri:
            continue

        s = URIRef(uri)
        subj_label = get_best_label(g, uri)
        subj_class = entity_class(g, uri)

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

        for _, p, o in g.triples((s, None, None)):
            p_str = str(p)
            if is_blocked_predicate(p_str):
                continue

            pred_label = predicate_label(p_str)
            pred_norm = normalize_text(pred_label)
            if pred_norm in LOW_VALUE_PREDICATES:
                continue

            obj_label = get_best_label(g, str(o)) if isinstance(o, URIRef) else str(o)
            obj_class = entity_class(g, str(o)) if isinstance(o, URIRef) else "Literal"

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

        for s2, p, _ in g.triples((None, None, s)):
            if isinstance(s2, URIRef) and "/sameAs/" in str(s2):
                continue

            p_str = str(p)
            if is_blocked_predicate(p_str):
                continue

            pred_label = predicate_label(p_str)
            pred_norm = normalize_text(pred_label)
            if pred_norm in LOW_VALUE_PREDICATES:
                continue

            subj2_label = get_best_label(g, str(s2))
            subj2_class = entity_class(g, str(s2))
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
            {"uri": uri, "label": get_best_label(g, uri), "class": entity_class(g, uri)}
            for uri in matched
            if "/sameAs/" not in uri
        ],
        "facts": facts,
    }


# =======================
# ANSWER GENERATION
# =======================

def select_best_text(pack: Dict, max_snippets: int = 3) -> List[Dict]:
    return pack.get("text_evidence", [])[:max_snippets]


def select_best_kg(pack: Dict, max_facts: int = 6) -> List[Dict]:
    facts = pack.get("kg_evidence", [])
    useful = []

    for f in facts:
        pred = str(f.get("predicate_label", "")).strip().lower()
        obj = str(f.get("object_label", "")).strip().lower()
        if pred in {"altlabel", "languagespoken", "occupation"}:
            continue
        if obj in {"q1860", "q10873124", "q15360275"}:
            continue
        useful.append(f)

    out = []
    seen = set()
    for f in useful:
        key = str(f["fact_text"]).strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(f)

    return out[:max_facts]


def generate_identity_answer(question: str, pack: Dict) -> str:
    focus = extract_focus(question)
    kg = select_best_kg(pack, max_facts=8)
    texts = select_best_text(pack, max_snippets=3)

    classes, works, creators, performers, instances = [], [], [], [], []

    for fact in kg:
        parsed = parse_fact_text(fact["fact_text"])
        if not parsed:
            continue
        _, pred, obj = parsed
        pred_low = pred.lower()

        if pred_low == "type":
            classes.append(obj)
        elif pred_low == "presentinwork":
            works.append(obj)
        elif pred_low == "creator":
            creators.append(obj)
        elif pred_low in {"performer", "portrays"}:
            performers.append(obj)
        elif pred_low == "instanceof":
            instances.append(obj)

    classes = dedupe_keep_order(classes)
    works = dedupe_keep_order(works)
    creators = dedupe_keep_order(creators)
    performers = dedupe_keep_order(performers)
    instances = dedupe_keep_order(instances)

    sentences = []

    main_class = None
    for c in classes + instances:
        c_low = c.lower()
        if "character" in c_low or "fictional" in c_low:
            main_class = c
            break

    if works:
        if main_class:
            sentences.append(f"{focus} is a {main_class.lower()} in *{works[0]}*.")
        else:
            sentences.append(f"{focus} is associated with *{works[0]}*.")
    else:
        if main_class:
            sentences.append(f"{focus} is a {main_class.lower()}.")
        else:
            sentences.append(f"{focus} is a character.")

    if creators:
        sentences.append(f"The character was created by {creators[0]}.")

    performers = [p for p in performers if p.lower() != focus.lower()]
    performers = dedupe_keep_order(performers)
    if performers:
        if len(performers) == 1:
            sentences.append(f"In screen adaptations, {focus} is performed or portrayed by {performers[0]}.")
        else:
            joined = ", ".join(performers[:-1]) + f" and {performers[-1]}"
            sentences.append(f"In screen adaptations, {focus} is performed or portrayed by {joined}.")

    if texts:
        top_title = texts[0].get("title", "")
        if normalize_text(top_title) == normalize_text(focus):
            sentences.append("The retrieved source passages also show that this character is central to the narrative and closely connected to major events and relationships in the series.")

    return " ".join(dedupe_keep_order(sentences))


def generate_portrayal_answer(question: str, pack: Dict) -> str:
    focus = extract_focus(question)
    kg = select_best_kg(pack, max_facts=8)

    for fact in kg:
        parsed = parse_fact_text(fact["fact_text"])
        if not parsed:
            continue
        subj, pred, obj = parsed
        if pred.lower() == "portrays":
            return f"{obj} is portrayed by {subj}."
        if pred.lower() == "performer":
            return f"{focus} is performed or portrayed by {obj}."

    return "I could not find a high-confidence portrayal relation in the retrieved evidence."


def generate_episode_membership_answer(pack: Dict) -> str:
    kg = select_best_kg(pack, max_facts=8)
    lines = []

    for fact in kg:
        parsed = parse_fact_text(fact["fact_text"])
        if not parsed:
            continue
        subj, pred, obj = parsed
        if pred.lower() == "appearsinepisode":
            lines.append(f"{subj} appears in the episode {obj}.")
        elif pred.lower() == "partofseries":
            lines.append(f"{subj} is part of the series {obj}.")

    lines = dedupe_keep_order(lines)
    if lines:
        return " ".join(lines[:3])

    return "I could not find a direct episode-related fact in the retrieved evidence."


def generate_episode_summary_answer(question: str, pack: Dict) -> str:
    focus = extract_focus(question)
    texts = select_best_text(pack, max_snippets=2)
    kg = select_best_kg(pack, max_facts=6)

    sentences = [f"*{focus}* is an episode of *The Queen's Gambit*."]

    for fact in kg:
        parsed = parse_fact_text(fact["fact_text"])
        if not parsed:
            continue
        _, pred, obj = parsed

        if pred.lower() == "partofseries":
            sentences.append(f"It is part of the series *{obj}*.")
        elif pred.lower() == "director":
            sentences.append(f"It was directed by {obj}.")
        elif pred.lower() == "originalbroadcaster":
            sentences.append(f"It was released on {obj}.")

    if texts:
        top_text = clean_text(texts[0]["text"])
        summary_piece = top_text[:500].strip()
        if summary_piece:
            sentences.append(f"According to the retrieved episode page, {summary_piece}")

    return " ".join(dedupe_keep_order(sentences))


def generate_generic_answer(pack: Dict) -> str:
    kg = select_best_kg(pack, max_facts=5)
    lines = []
    pred_map = {
        "type": "is a",
        "presentinwork": "appears in",
        "creator": "was created by",
        "performer": "is performed or portrayed by",
        "instanceof": "is an instance of",
        "partofseries": "is part of the series",
        "director": "was directed by",
    }

    for fact in kg:
        parsed = parse_fact_text(fact["fact_text"])
        if not parsed:
            continue
        subj, pred, obj = parsed
        lines.append(f"{subj} {pred_map.get(pred.lower(), pred)} {obj}.")

    lines = dedupe_keep_order(lines)
    if lines:
        return " ".join(lines[:4])

    return "I could not generate a grounded answer from the retrieved evidence."


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


def generate_answer(question: str, pack: Dict) -> Dict:
    qtype = classify_question(question)

    if qtype == "identity":
        answer = generate_identity_answer(question, pack)
    elif qtype == "portrayal":
        answer = generate_portrayal_answer(question, pack)
    elif qtype == "episode_membership":
        answer = generate_episode_membership_answer(pack)
    elif qtype == "episode_summary":
        answer = generate_episode_summary_answer(question, pack)
    else:
        answer = generate_generic_answer(pack)

    text_snippets = select_best_text(pack, max_snippets=3)
    kg_facts = select_best_kg(pack, max_facts=5)

    evidence_lines = []
    for i, t in enumerate(text_snippets, start=1):
        evidence_lines.append(f"[Text {i}] {t['title']} — {t['url']}")
    for i, f in enumerate(kg_facts, start=1):
        evidence_lines.append(f"[KG {i}] {f['fact_text']}")

    return {
        "question": question,
        "question_type": qtype,
        "answer": answer,
        "evidence": evidence_lines,
        "used_text_evidence": text_snippets,
        "used_kg_evidence": kg_facts,
    }


def print_debug(result: Dict, pack: Dict) -> None:
    print("\n==============================")
    print("QUESTION")
    print("==============================")
    print(result["question"])
    print(f"type: {result['question_type']}")

    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print(result["answer"])

    print("\n==============================")
    print("TEXT EVIDENCE")
    print("==============================")
    for e in pack["text_evidence"][:5]:
        print(f"[{e['rank']}] score={e['score']:.4f} | {e['title']}")
        print(f"url: {e['url']}")
        print(f"text: {clean_text(e['text'])[:350]}")
        print("-" * 80)

    print("\n==============================")
    print("KG EVIDENCE")
    print("==============================")
    for e in pack["kg_evidence"][:8]:
        print(f"[{e['rank']}] score={e['score']:.4f}")
        print(e["fact_text"])
        print("-" * 80)


def run_one_question(question: str, chunks_path: Path, index_dir: Path, kg_path: Path, mode: str) -> None:
    text_results = text_retrieve(question, index_dir=index_dir, chunks_path=chunks_path, top_k=8)
    text_results = rerank_text_results(question, text_results)
    kg_results = kg_retrieve(question, kg_path=kg_path, top_k_entities=5, top_k_facts=10)

    pack = build_evidence_pack(question, text_results, kg_results)
    result = generate_answer(question, pack)

    if mode == "simple":
        print("\n" + result["answer"] + "\n")
    else:
        print_debug(result, pack)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple CLI for hybrid KG + text RAG assistant")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["simple", "debug"], default="simple")
    parser.add_argument("--chunks_path", type=Path, default=Path("data/rag/chunks.jsonl"))
    parser.add_argument("--index_dir", type=Path, default=Path("data/rag/index"))
    parser.add_argument("--kg", type=Path, default=Path("data/expanded_kb.ttl"))
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        print("Hybrid RAG assistant")
        print("Tape ta question, ou 'quit' pour sortir.\n")
        while True:
            q = input("Question> ").strip()
            if not q:
                continue
            if q.lower() in {"quit", "exit", "q"}:
                break
            run_one_question(q, args.chunks_path, args.index_dir, args.kg, args.mode)
    else:
        if not args.question:
            raise ValueError("Use --question or --interactive")
        run_one_question(args.question, args.chunks_path, args.index_dir, args.kg, args.mode)


if __name__ == "__main__":
    main()