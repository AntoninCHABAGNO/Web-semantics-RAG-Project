# src/nlp_extract.py
"""
NLP extraction pipeline for a KG-first RAG assistant.

What this script does:
- Load cleaned pages from JSONL (url + text [+ optional title/page_id]).
- Extract entities (SpaCy NER + light heuristics).
- Extract relations (dependency / pattern rules) with evidence + audit metadata.
- Output:
    - data/entities.csv
    - data/relations.csv
- Then runs a dedup step (dedupe_entities_csv) as in the original pipeline.

Key design decisions (aligned with your project):
- Fandom domain is IN_UNIVERSE by default (do NOT flip whole pages to REAL_WORLD on bio words).
- Actor detection is local: "portrayed/played by X" makes X a REAL_PERSON (not the whole page).
- Entity classes and relation labels are normalized to SCREAMING_SNAKE_CASE to avoid schema mismatches.
- win/lose relations are routed based on object type/context (WON/LOST vs DEFEATED/LOST_TO).
- Relations carry extra audit fields: relation_rule, context_window, and component confidence scores.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import spacy

from dedupe_entities import dedupe_entities_csv

# -----------------------------
# Config
# -----------------------------
NLP_MODEL = "en_core_web_trf"

# Entities we keep (RAG-friendly + graph-friendly)
TARGET_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT"}

# The Queen's Gambit (Fandom) is IN_UNIVERSE by default
IN_UNIVERSE_DOMAINS = {
    "the-queens-gambit.fandom.com",
}

# External strong signals only (do NOT include "fandom.com" or generic "wiki")
REALWORLD_URL_HINTS = [
    "imdb.com",
    "wikipedia.org",
    "/actor",
    "/actress",
    "/cast",
    "/production",
    "/filming",
]

# Priority when multiple NER labels tie
LABEL_PRIORITY = {
    "GPE": 5,
    "LOC": 4,
    "FAC": 4,
    "ORG": 3,
    "NORP": 2,
    "PERSON": 1,
    "EVENT": 1,
}

# Canonical entity classes (stable schema-friendly)
# Everything emitted in entities.csv uses these values.
NER_TO_CLASS_BASE = {
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "NORP": "GROUP",
    "EVENT": "EVENT",
}

# Work-title detection (keep minimal)
WORK_TITLE_KEYWORDS = {"the queen's gambit", "queen's gambit"}
WORK_TITLE_STARTERS = {"a", "an", "the"}
WORK_TITLE_RE = re.compile(r"^(a|an|the)\s+.+", re.IGNORECASE)

# Actor pattern detection: "portrayed by X", "played by X", "voiced by X"
ACTOR_TRIGGER_LEMMAS = {"play", "portray", "voice"}

# Family / relationship nominal patterns
FAMILY_TRIGGERS = {
    "son": "CHILD_OF",
    "daughter": "CHILD_OF",
    "child": "CHILD_OF",
    "father": "CHILD_OF",
    "mother": "CHILD_OF",
}

KINSHIP_NOUNS = {
    "father": "CHILD_OF",
    "mother": "CHILD_OF",
    "parent": "CHILD_OF",
    "son": "CHILD_OF",
    "daughter": "CHILD_OF",
    "child": "CHILD_OF",
    "brother": "SIBLING_OF",
    "sister": "SIBLING_OF",
}

SPOUSE_NOUNS = {
    "wife": "MARRIED",
    "husband": "MARRIED",
    "spouse": "MARRIED",
}

COPULA_LEMMAS = {"be"}  # "is/was/are"

MAX_EVIDENCE_CHARS = 250
CONTEXT_WINDOW_CHARS = 120

# Cleanup helpers
POSSESSIVE_RE = re.compile(r"(\b\w+)'s\b", flags=re.IGNORECASE)
TRAILING_WIKI_REFS_RE = re.compile(r"(?:\s*\[[^\]]*\]?)+\s*$")
TRAILING_ORPHAN_RE = re.compile(r"[\[\]\(\)]+(?:\s*)$")

# Very common junk entities that are almost always noise for KG facts
STOP_ENTITY_CANONICALS = {
    "episode",
    "episodes",
    "chapter",
    "chapters",
    "season",
    "seasons",
    "wikipedia",
    "fandom",
    "external links",
    "references",
    "see also",
}

# Context keywords to detect "win a tournament/title/event" rather than "defeat a person"
EVENTISH_KEYWORDS = {
    "tournament",
    "championship",
    "title",
    "final",
    "match",
    "event",
    "open",
    "cup",
    "masters",
}


# -----------------------------
# Text normalization / IDs
# -----------------------------
def normalize_entity_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = " ".join(s.split()).strip()
    return s


def canonicalize_entity_text(s: str) -> str:
    s = normalize_entity_text(s)
    s = TRAILING_WIKI_REFS_RE.sub("", s)
    s = POSSESSIVE_RE.sub(r"\1", s)
    s = s.strip(" \t\n\r.,;:!?'\"{}")
    s = TRAILING_ORPHAN_RE.sub("", s)
    s = " ".join(s.split())
    return s


def slugify(s: str) -> str:
    s = canonicalize_entity_text(s).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


def normalize_entity_class(c: str) -> str:
    """
    Ensure all entity classes are stable SCREAMING_SNAKE_CASE.
    Accepts common variants (e.g., CamelCase) and normalizes them.
    """
    c = (c or "").strip()
    if not c:
        return "THING"

    aliases = {
        "RealPerson": "REAL_PERSON",
        "REALPERSON": "REAL_PERSON",
        "Character": "CHARACTER",
        "Organization": "ORGANIZATION",
        "Location": "LOCATION",
        "Work": "WORK",
        "Event": "EVENT",
        "Group": "GROUP",
        "Thing": "THING",
        "PERSON": "REAL_PERSON",  # should not be used directly, but safe fallback
    }

    if c in aliases:
        return aliases[c]

    # Convert CamelCase -> CAMEL_CASE -> CAMEL_CASE (then map if known)
    if re.match(r"^[A-Za-z]+(?:[A-Z][a-z]+)+$", c):
        c2 = re.sub(r"(?<!^)(?=[A-Z])", "_", c).upper()
        return aliases.get(c2, c2)

    c = c.upper()
    c = re.sub(r"[^A-Z0-9_]+", "_", c).strip("_")
    return aliases.get(c, c)


def make_entity_id(ent_text: str, ent_class: str) -> str:
    ent_class = normalize_entity_class(ent_class)
    base = f"{ent_class}:{canonicalize_entity_text(ent_text)}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:6]
    return f"{ent_class}:{slugify(ent_text)}-{h}"


def shorten_evidence(text: str, max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    text = normalize_entity_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


# -----------------------------
# Page context: IN_UNIVERSE vs REAL_WORLD
# -----------------------------
def looks_realworld_page(url: str, text: str) -> bool:
    """
    Conservative policy:
    - Fandom domain is IN_UNIVERSE by default -> return False always.
    - Mark page REAL_WORLD only if URL strongly indicates external real-world sources.
    - Unknown domains -> default False (safer for this project).
    """
    u = (url or "").strip()
    try:
        parsed = urlparse(u)
        netloc = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        full = (netloc + path).lower()
    except Exception:
        netloc = ""
        full = u.lower()

    # IN_UNIVERSE domain always stays in-universe
    if netloc in IN_UNIVERSE_DOMAINS:
        return False

    # Strong URL hints for real-world sources
    if any(h in full for h in REALWORLD_URL_HINTS):
        return True
    if any(k in netloc for k in ["wikipedia.org", "imdb.com"]):
        return True

    # Default safe
    return False


# -----------------------------
# Work-title detection
# -----------------------------
def looks_like_work_title(canonical: str) -> bool:
    c = (canonical or "").strip()
    if not c:
        return False
    low = c.lower()

    if any(kw in low for kw in WORK_TITLE_KEYWORDS):
        return True

    words = c.split()
    if len(words) >= 3 and words[0].lower() in WORK_TITLE_STARTERS and WORK_TITLE_RE.match(c):
        return True

    return False


def get_entity_class(final_label: str, canonical: str, page_is_realworld: bool, is_actor: bool) -> str:
    """
    Resolve entity class.
    - PERSON -> CHARACTER by default on in-universe pages
    - actor pattern forces REAL_PERSON for the actor entity only
    - work title forces WORK
    """
    final_label = (final_label or "").strip()
    canonical = canonical or ""

    if final_label == "PERSON":
        if looks_like_work_title(canonical):
            return "WORK"
        if is_actor:
            return "REAL_PERSON"
        # Keep conservative for this project: in-universe pages -> CHARACTER
        return "REAL_PERSON" if page_is_realworld else "CHARACTER"

    return NER_TO_CLASS_BASE.get(final_label, "THING")


# -----------------------------
# Actor detection ("portrayed by Y")
# -----------------------------
def find_actor_entities_in_sentence(sent, doc) -> Set[str]:
    """
    Return set of canonical actor names found via patterns:
    - portrayed by X
    - played by X
    - voiced by X
    """
    actors: Set[str] = set()

    for tok in sent:
        if tok.lemma_.lower() not in ACTOR_TRIGGER_LEMMAS:
            continue

        by_prep = None
        for child in tok.children:
            if child.dep_ == "prep" and child.lemma_.lower() == "by":
                by_prep = child
                break
        if by_prep is None:
            continue

        pobj = None
        for gc in by_prep.children:
            if gc.dep_ == "pobj":
                pobj = gc
                break
        if pobj is None:
            continue

        for ent in doc.ents:
            if ent.label_ == "PERSON" and ent.start <= pobj.i < ent.end:
                actors.add(canonicalize_entity_text(ent.text))

    return actors


# -----------------------------
# Label resolution
# -----------------------------
def choose_majority_label(labels: List[str]) -> str:
    counts = Counter(labels)
    max_count = max(counts.values())
    tied = [lb for lb, c in counts.items() if c == max_count]
    if len(tied) == 1:
        return tied[0]
    tied.sort(key=lambda lb: LABEL_PRIORITY.get(lb, 0), reverse=True)
    return tied[0]


# -----------------------------
# IO
# -----------------------------
def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


# -----------------------------
# Small utilities
# -----------------------------
def is_stop_entity(canonical: str) -> bool:
    low = (canonical or "").strip().lower()
    if not low:
        return True
    if low in STOP_ENTITY_CANONICALS:
        return True
    # Too numeric / too short handled elsewhere; keep extra guards
    if low.startswith("http") or "www." in low:
        return True
    return False


def compute_context_window(sent_text: str, token_text: str, window: int = CONTEXT_WINDOW_CHARS) -> str:
    """
    Without full original page text offsets, use sentence text as context.
    Also try to keep it short and centered around the token if possible.
    """
    s = normalize_entity_text(sent_text)
    if len(s) <= (2 * window + 40):
        return s

    # Center around first occurrence of token_text if possible
    t = (token_text or "").strip()
    if t:
        idx = s.lower().find(t.lower())
    else:
        idx = -1

    if idx == -1:
        return s[: (2 * window + 40)].rstrip() + "…"

    start = max(0, idx - window)
    end = min(len(s), idx + len(t) + window)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(s) else ""
    return prefix + s[start:end].strip() + suffix


# -----------------------------
# Entity extraction
# -----------------------------
def extract_entities(
    doc,
    url: str,
    page_is_realworld: bool,
    page_title: str = "",
    page_id: str = "",
):
    occ_labels: Dict[str, List[str]] = defaultdict(list)      # canonical -> [labels...]
    occ_forms: Dict[str, Counter] = defaultdict(Counter)      # canonical -> Counter(surface_forms)
    occ_spans: Dict[str, List[str]] = defaultdict(list)       # canonical -> few snippets

    actor_canonicals: Set[str] = set()
    for sent in doc.sents:
        actor_canonicals |= find_actor_entities_in_sentence(sent, doc)

    for ent in doc.ents:
        if ent.label_ not in TARGET_ENTITY_LABELS:
            continue

        surface = normalize_entity_text(ent.text)
        canonical = canonicalize_entity_text(surface)
        if not canonical:
            continue

        # Noise filters
        if len(canonical) < 3:
            continue
        if is_stop_entity(canonical):
            continue
        if sum(ch.isdigit() for ch in canonical) >= max(2, len(canonical) // 2):
            continue

        occ_labels[canonical].append(ent.label_)
        occ_forms[canonical][surface] += 1

        # Keep a couple of short span examples for debug / later scoring
        if len(occ_spans[canonical]) < 3:
            snippet = compute_context_window(ent.sent.text, ent.text)
            occ_spans[canonical].append(snippet)

    rows = []
    for canonical, labels in occ_labels.items():
        final_label = choose_majority_label(labels)
        is_actor = canonical in actor_canonicals
        ent_class = normalize_entity_class(get_entity_class(final_label, canonical, page_is_realworld, is_actor))
        ent_id = make_entity_id(canonical, ent_class)

        forms_counter = occ_forms[canonical]
        preferred, _ = forms_counter.most_common(1)[0]
        aliases = [f for f, _ in forms_counter.most_common(10)]
        mention_count = sum(forms_counter.values())

        rows.append({
            "entity_id": ent_id,
            "canonical_text": canonical,
            "preferred_text": preferred,
            "aliases": "|".join(aliases),
            "mention_count": int(mention_count),

            "entity_label": final_label,
            "entity_class": ent_class,

            "page_context": "REAL_WORLD" if page_is_realworld else "IN_UNIVERSE",
            "url": url,
            "source_page_title": page_title or "",
            "page_id": page_id or "",

            "span_examples": " || ".join(occ_spans.get(canonical, [])),
        })

    return rows


# -----------------------------
# Relation extraction helpers
# -----------------------------
def find_subject_object(verb_token):
    subj = None
    obj = None

    for child in verb_token.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj = child
            break

    for dep in ("dobj", "attr", "oprd", "dative"):
        for child in verb_token.children:
            if child.dep_ == dep:
                obj = child
                break
        if obj is not None:
            break

    if obj is None:
        for child in verb_token.children:
            if child.dep_ == "prep":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        obj = gc
                        break
            if obj is not None:
                break

    return subj, obj


def token_to_entity_span(token, doc):
    for ent in doc.ents:
        if ent.start <= token.i < ent.end and ent.label_ in TARGET_ENTITY_LABELS:
            return ent
    return None


def sentence_offsets(sent):
    return sent.start_char, sent.end_char


def entity_span_to_row(
    ent,
    url: str,
    page_is_realworld: bool,
    actor_canonicals: Set[str],
    page_title: str = "",
    page_id: str = "",
):
    surface = normalize_entity_text(ent.text)
    canonical = canonicalize_entity_text(surface)
    if not canonical:
        return None
    if is_stop_entity(canonical):
        return None

    label = ent.label_
    if label not in TARGET_ENTITY_LABELS:
        return None

    is_actor = canonical in actor_canonicals
    ent_class = normalize_entity_class(get_entity_class(label, canonical, page_is_realworld, is_actor))
    ent_id = make_entity_id(canonical, ent_class)

    return {
        "entity_id": ent_id,
        "canonical_text": canonical,
        "surface_text": surface,
        "entity_label": label,
        "entity_class": ent_class,
        "page_context": "REAL_WORLD" if page_is_realworld else "IN_UNIVERSE",
        "url": url,
        "source_page_title": page_title or "",
        "page_id": page_id or "",
    }


def resolve_relation_label(
    verb_lemma: str,
    obj_class: str,
    sentence_text: str,
) -> Tuple[str, Dict[str, float]]:
    """
    Route relation labels based on lemma + object type + context keywords.

    Returns (relation_label, component_scores).
    """
    lemma = (verb_lemma or "").lower()
    obj_class = normalize_entity_class(obj_class)
    sent_low = (sentence_text or "").lower()

    # Default component scores (rough, but audit-friendly)
    scores = {
        "pattern_score": 0.75,
        "ner_score": 0.75,
        "distance_score": 0.75,
        "page_bias": 0.75,
    }

    # Verb baseline mappings (non win/lose)
    base = {
        "play": "PLAYED",
        "compete": "COMPETED_IN",
        "enter": "ENTERED",

        "beat": "DEFEATED",
        "defeat": "DEFEATED",
        "draw": "DREW_WITH",

        "learn": "LEARNED_FROM",
        "study": "STUDIED",
        "train": "TRAINED_BY",
        "teach": "TAUGHT_BY",
        "coach": "COACHED_BY",

        "meet": "MET",
        "help": "HELPED",
        "support": "SUPPORTED",
        "befriend": "BEFRIENDED",
        "love": "LOVED",
        "marry": "MARRIED",
        "divorce": "DIVORCED",
    }

    if lemma not in {"win", "lose"}:
        rel = base.get(lemma)
        if rel:
            return rel, scores
        return "", scores

    # win/lose routing
    # If the object is clearly a person/character => DEFEATED / LOST_TO
    if obj_class in {"CHARACTER", "REAL_PERSON"}:
        if lemma == "win":
            scores["pattern_score"] = 0.85
            return "DEFEATED", scores
        scores["pattern_score"] = 0.85
        return "LOST_TO", scores

    # If object looks like an event/title, or context has eventish words => WON / LOST
    if obj_class in {"EVENT", "WORK"} or any(k in sent_low for k in EVENTISH_KEYWORDS):
        if lemma == "win":
            scores["pattern_score"] = 0.85
            return "WON", scores
        scores["pattern_score"] = 0.85
        return "LOST", scores

    # Otherwise default conservatively
    if lemma == "win":
        return "WON", scores
    return "LOST", scores


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def combine_confidence(scores: Dict[str, float]) -> float:
    # Simple weighted average; keep explainable
    w = {
        "pattern_score": 0.40,
        "ner_score": 0.25,
        "distance_score": 0.20,
        "page_bias": 0.15,
    }
    total = 0.0
    for k, wk in w.items():
        total += wk * float(scores.get(k, 0.0))
    return clamp01(total)


# -----------------------------
# Relation extraction
# -----------------------------
def extract_relations(
    doc,
    url: str,
    page_is_realworld: bool,
    page_title: str = "",
    page_id: str = "",
):
    rels: List[Dict[str, Any]] = []

    actor_canonicals: Set[str] = set()
    for sent in doc.sents:
        actor_canonicals |= find_actor_entities_in_sentence(sent, doc)

    def add_rel(
        subj_ent,
        obj_ent,
        rel_label: str,
        evidence_sent,
        relation_rule: str,
        verb_lemma: str = "",
        scores: Optional[Dict[str, float]] = None,
    ):
        subj = entity_span_to_row(subj_ent, url, page_is_realworld, actor_canonicals, page_title, page_id)
        obj = entity_span_to_row(obj_ent, url, page_is_realworld, actor_canonicals, page_title, page_id)
        if not subj or not obj:
            return
        if not rel_label:
            return

        ev_start, ev_end = sentence_offsets(evidence_sent)
        evidence = shorten_evidence(evidence_sent.text)
        context_window = compute_context_window(evidence_sent.text, verb_lemma or rel_label)

        scores = scores or {
            "pattern_score": 0.75,
            "ner_score": 0.75,
            "distance_score": 0.75,
            "page_bias": 0.75,
        }
        confidence = combine_confidence(scores)

        rels.append({
            "subject_id": subj["entity_id"],
            "subject": subj["canonical_text"],
            "subject_label": subj["entity_label"],
            "subject_class": subj["entity_class"],

            "relation": rel_label,

            "object_id": obj["entity_id"],
            "object": obj["canonical_text"],
            "object_label": obj["entity_label"],
            "object_class": obj["entity_class"],

            "verb_lemma": verb_lemma or "",
            "relation_rule": relation_rule,

            "confidence": float(confidence),
            "pattern_score": float(scores.get("pattern_score", 0.0)),
            "ner_score": float(scores.get("ner_score", 0.0)),
            "distance_score": float(scores.get("distance_score", 0.0)),
            "page_bias": float(scores.get("page_bias", 0.0)),

            "evidence": evidence,
            "context_window": context_window,
            "evidence_start_char": ev_start,
            "evidence_end_char": ev_end,

            "page_context": "REAL_WORLD" if page_is_realworld else "IN_UNIVERSE",
            "url": url,
            "source_page_title": page_title or "",
            "page_id": page_id or "",
        })

    # (1) Verb-based relations
    for tok in doc:
        if tok.pos_ != "VERB":
            continue

        lemma = tok.lemma_.lower()
        if lemma not in {
            "play", "compete", "enter",
            "win", "beat", "defeat", "lose", "draw",
            "learn", "study", "train", "teach", "coach",
            "meet", "help", "support", "befriend", "love", "marry", "divorce",
        }:
            continue

        subj_tok, obj_tok = find_subject_object(tok)
        if not subj_tok or not obj_tok:
            continue

        subj_ent = token_to_entity_span(subj_tok, doc)
        obj_ent = token_to_entity_span(obj_tok, doc)
        if not subj_ent or not obj_ent:
            continue

        # Determine object class for win/lose routing (and for future rules)
        obj_row = entity_span_to_row(obj_ent, url, page_is_realworld, actor_canonicals, page_title, page_id)
        if not obj_row:
            continue

        # Pattern vs pobj affects confidence a bit
        relation_rule = "dep:verb_nsubj_obj"
        scores = {
            "pattern_score": 0.85,
            "ner_score": 0.80,
            "distance_score": 0.75,
            "page_bias": 0.75,
        }
        if obj_tok.dep_ == "pobj":
            relation_rule = "dep:verb_nsubj_pobj"
            scores["pattern_score"] = 0.70
            scores["distance_score"] = 0.65

        # Resolve label (win/lose routing included)
        rel_label, scores2 = resolve_relation_label(lemma, obj_row["entity_class"], tok.sent.text)
        # Merge scores (keep the lower where relevant)
        for k in scores2:
            scores[k] = min(scores.get(k, 1.0), scores2.get(k, 1.0))

        add_rel(
            subj_ent=subj_ent,
            obj_ent=obj_ent,
            rel_label=rel_label,
            evidence_sent=tok.sent,
            relation_rule=relation_rule,
            verb_lemma=lemma,
            scores=scores,
        )

    # (2) Family nominal patterns: "X, daughter of Y"
    for tok in doc:
        lemma = tok.lemma_.lower()
        if lemma not in FAMILY_TRIGGERS:
            continue

        parent_tok = None
        for child in tok.children:
            if child.dep_ == "prep" and child.lemma_.lower() == "of":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        parent_tok = gc
                        break
        if not parent_tok:
            continue

        child_tok = tok.head if tok.dep_ == "appos" else None
        if not child_tok:
            continue

        child_ent = token_to_entity_span(child_tok, doc)
        parent_ent = token_to_entity_span(parent_tok, doc)
        if not child_ent or not parent_ent:
            continue

        add_rel(
            subj_ent=child_ent,
            obj_ent=parent_ent,
            rel_label=FAMILY_TRIGGERS[lemma],
            evidence_sent=tok.sent,
            relation_rule="pattern:family_appos_of",
            verb_lemma=lemma,
            scores={
                "pattern_score": 0.80,
                "ner_score": 0.75,
                "distance_score": 0.70,
                "page_bias": 0.75,
            },
        )

    # (3) Possessive patterns: "X's father/mother/wife/husband ..."
    for tok in doc:
        noun = tok.lemma_.lower()
        if noun not in KINSHIP_NOUNS and noun not in SPOUSE_NOUNS:
            continue
        if tok.pos_ not in ("NOUN", "PROPN"):
            continue

        poss = None
        for child in tok.children:
            if child.dep_ == "poss":
                poss = child
                break
        if poss is None:
            continue

        poss_ent = token_to_entity_span(poss, doc)
        if not poss_ent:
            continue

        # Prefer appos after the kinship noun, else first following entity in sentence
        y_ent = None
        for child in tok.children:
            if child.dep_ == "appos":
                y_ent = token_to_entity_span(child, doc)
                if y_ent:
                    break

        if y_ent is None:
            candidates = [e for e in tok.sent.ents if e.start_char >= tok.idx]
            if candidates and candidates[0].label_ in TARGET_ENTITY_LABELS:
                y_ent = candidates[0]

        if not y_ent:
            continue

        if noun in KINSHIP_NOUNS:
            rel_label = KINSHIP_NOUNS[noun]
            rule = "pattern:possessive_kinship"
            scores = {
                "pattern_score": 0.80,
                "ner_score": 0.75,
                "distance_score": 0.70,
                "page_bias": 0.75,
            }
        else:
            rel_label = SPOUSE_NOUNS[noun]
            rule = "pattern:possessive_spouse"
            scores = {
                "pattern_score": 0.75,
                "ner_score": 0.75,
                "distance_score": 0.65,
                "page_bias": 0.75,
            }

        add_rel(
            subj_ent=poss_ent,
            obj_ent=y_ent,
            rel_label=rel_label,
            evidence_sent=tok.sent,
            relation_rule=rule,
            verb_lemma=noun,
            scores=scores,
        )

    return rels


# -----------------------------
# Reporting
# -----------------------------
def print_report(ent_df: pd.DataFrame, rel_df: pd.DataFrame) -> None:
    print("\n=== SCHEMA REPORT ===")

    if ent_df.empty:
        print("No entities extracted.")
    else:
        print("\nTop entity_class:")
        print(ent_df["entity_class"].value_counts().head(20).to_string())

        print("\nTop entity_label:")
        print(ent_df["entity_label"].value_counts().head(20).to_string())

        print("\nTop entities (by mention_count):")
        if "mention_count" in ent_df.columns:
            top = ent_df.sort_values("mention_count", ascending=False).head(20)[
                ["canonical_text", "entity_class", "mention_count", "url"]
            ]
            print(top.to_string(index=False))

    if rel_df.empty:
        print("\nNo relations extracted.")
    else:
        print("\nTop relations:")
        print(rel_df["relation"].value_counts().head(20).to_string())

        print("\nConfidence buckets:")
        buckets = pd.cut(rel_df["confidence"], bins=[0, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
        print(buckets.value_counts().sort_index().to_string())

        print("\nExamples (5 per relation):")
        for rel in rel_df["relation"].value_counts().head(10).index.tolist():
            ex = rel_df[rel_df["relation"] == rel].head(5)[
                ["subject", "relation", "object", "confidence", "relation_rule", "url"]
            ]
            print(f"\n--- {rel} ---")
            print(ex.to_string(index=False))

    print("\n=== END REPORT ===\n")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_jsonl", default="data/raw_jsonl/pages.jsonl")
    parser.add_argument("--out_entities", default="data/entities.csv")
    parser.add_argument("--out_relations", default="data/relations.csv")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_process", type=int, default=-1)
    args = parser.parse_args()

    nlp = spacy.load(NLP_MODEL)

    records = list(load_jsonl(args.in_jsonl))

    urls = [r.get("url", "") for r in records]
    texts = [r.get("text", "") for r in records]
    titles = [r.get("title", "") or r.get("page_title", "") for r in records]
    page_ids = [str(r.get("page_id", "") or r.get("id", "") or "") for r in records]

    entity_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []

    # Process docs
    for doc, url, text, title, pid in zip(
        nlp.pipe(texts, batch_size=args.batch_size, n_process=args.n_process),
        urls,
        texts,
        titles,
        page_ids,
    ):
        page_is_realworld = looks_realworld_page(url, text)

        entity_rows.extend(extract_entities(doc, url, page_is_realworld, page_title=title, page_id=pid))
        relation_rows.extend(extract_relations(doc, url, page_is_realworld, page_title=title, page_id=pid))

    ent_df = pd.DataFrame(entity_rows).drop_duplicates()
    rel_df = pd.DataFrame(relation_rows).drop_duplicates()

    if not ent_df.empty:
        ent_df = ent_df.sort_values(["entity_class", "canonical_text", "url"])
    if not rel_df.empty:
        rel_df = rel_df.sort_values(["relation", "confidence"], ascending=[True, False])

    ent_df.to_csv(args.out_entities, index=False)
    dedupe_entities_csv(path_in=args.out_entities)
    rel_df.to_csv(args.out_relations, index=False)

    if args.report:
        print_report(ent_df, rel_df)


if __name__ == "__main__":
    main()
    