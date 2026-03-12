#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


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

def generate_episode_summary_answer(question: str, pack: Dict) -> str:
    focus = extract_focus(question)
    texts = select_best_text(pack, max_snippets=2)
    kg = select_best_kg(pack, max_facts=6)

    sentences = []

    sentences.append(f"*{focus}* is an episode of *The Queen's Gambit*.")

    for fact in kg:
        parsed = parse_fact_text(fact["fact_text"])
        if not parsed:
            continue
        subj, pred, obj = parsed

        pred_low = pred.lower()
        if pred_low == "partofseries":
            sentences.append(f"It is part of the series *{obj}*.")
        elif pred_low == "director":
            sentences.append(f"It was directed by {obj}.")
        elif pred_low == "originalbroadcaster":
            sentences.append(f"It was released on {obj}.")

    if texts:
        top_text = normalize_text(texts[0]["text"])
        if top_text:
            summary_piece = top_text[:500].strip()
            sentences.append(f"According to the retrieved episode page, {summary_piece}")

    return " ".join(dedupe_keep_order(sentences))

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


def select_best_text(pack: Dict, max_snippets: int = 3) -> List[Dict]:
    texts = pack.get("text_evidence", [])
    return texts[:max_snippets]


def select_best_kg(pack: Dict, max_facts: int = 8) -> List[Dict]:
    facts = pack.get("kg_evidence", [])

    useful = []
    for f in facts:
        pred = str(f.get("predicate_label", "")).strip().lower()
        obj = str(f.get("object_label", "")).strip().lower()

        if pred in {"altlabel", "languageSpoken".lower(), "occupation".lower()}:
            continue

        if obj in {"q1860", "q10873124", "q15360275"}:
            continue

        useful.append(f)

    # dédupe par texte de fait
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

    classes = []
    works = []
    creators = []
    performers = []
    instances = []

    for fact in kg:
        parsed = parse_fact_text(fact["fact_text"])
        if not parsed:
            continue
        subj, pred, obj = parsed
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

    # phrase 1 : définition
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
            sentences.append(f"{focus} is an entity found in the retrieved knowledge base evidence.")

    # phrase 2 : origine / création
    if creators:
        if len(creators) == 1:
            sentences.append(f"The character was created by {creators[0]}.")
        else:
            joined = ", ".join(creators[:-1]) + f" and {creators[-1]}"
            sentences.append(f"The character is linked to creators including {joined}.")

    # phrase 3 : portrayal
    if performers:
        performers_clean = [p for p in performers if p.lower() != focus.lower()]
        performers_clean = dedupe_keep_order(performers_clean)
        if performers_clean:
            if len(performers_clean) == 1:
                sentences.append(f"In screen adaptations, {focus} is performed or portrayed by {performers_clean[0]}.")
            else:
                joined = ", ".join(performers_clean[:-1]) + f" and {performers_clean[-1]}"
                sentences.append(f"In screen adaptations, {focus} is performed or portrayed by {joined}.")

    # phrase 4 : léger enrichissement textuel
    if texts:
        top_title = texts[0].get("title", "")
        if normalize_text(top_title).lower() == focus.lower():
            sentences.append(
                "The retrieved source passages also show that this character is central to the narrative and closely connected to major events and relationships in the series."
            )

    return " ".join(dedupe_keep_order(sentences))


def generate_portrayal_answer(question: str, pack: Dict) -> str:
    kg = select_best_kg(pack, max_facts=8)
    focus = extract_focus(question)

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


def generate_episode_answer(question: str, pack: Dict) -> str:
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


def generate_generic_answer(question: str, pack: Dict) -> str:
    kg = select_best_kg(pack, max_facts=5)
    lines = []

    for fact in kg:
        parsed = parse_fact_text(fact["fact_text"])
        if not parsed:
            continue
        subj, pred, obj = parsed

        pred_map = {
            "type": "is a",
            "presentinwork": "appears in",
            "creator": "was created by",
            "performer": "is performed or portrayed by",
            "instanceof": "is an instance of",
            "partofseries": "is part of the series",
        }

        pred_pretty = pred_map.get(pred.lower(), pred)
        lines.append(f"{subj} {pred_pretty} {obj}.")

    lines = dedupe_keep_order(lines)
    if lines:
        return " ".join(lines[:4])

    return "I could not generate a grounded answer from the retrieved evidence."


def build_grounded_answer(question: str, pack: Dict) -> Dict:
    qtype = classify_question(question)

    if qtype == "identity":
        answer = generate_identity_answer(question, pack)
    elif qtype == "portrayal":
        answer = generate_portrayal_answer(question, pack)
    elif qtype == "episode_membership":
        answer = generate_episode_answer(question, pack)
    elif qtype == "episode_summary":
        answer = generate_episode_summary_answer(question, pack)
    else:
        answer = generate_generic_answer(question, pack)

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


def pretty_print_result(result: Dict) -> None:
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
    print("EVIDENCE")
    print("==============================")
    for e in result["evidence"]:
        print("-", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grounded answer from hybrid evidence pack")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--evidence_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    pack = load_json(args.evidence_json)
    result = build_grounded_answer(args.question, pack)
    pretty_print_result(result)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved final answer to: {args.output_json}")


if __name__ == "__main__":
    main()