# src/nlp_extract.py
import json
import pandas as pd
import spacy
import re
import hashlib
import unicodedata
from collections import defaultdict, Counter
from urllib.parse import urlparse
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

# Relation verbs -> relation label (simple baseline for this universe)
RELATION_VERBS = {
    # chess / competition
    "play": "PLAYED",
    "compete": "COMPETED_IN",
    "enter": "ENTERED",
    "win": "DEFEATED",
    "beat": "DEFEATED",
    "defeat": "DEFEATED",
    "lose": "LOST_TO",
    "draw": "DREW_WITH",

    # training / learning
    "learn": "LEARNED_FROM",
    "study": "STUDIED",
    "train": "TRAINED_BY",
    "teach": "TAUGHT_BY",
    "coach": "COACHED_BY",

    # social / narrative
    "meet": "MET",
    "help": "HELPED",
    "support": "SUPPORTED",
    "befriend": "BEFRIENDED",
    "love": "LOVED",
    "marry": "MARRIED",
    "divorce": "DIVORCED",
}

LABEL_PRIORITY = {
    "GPE": 5,
    "LOC": 4,
    "FAC": 4,
    "ORG": 3,
    "NORP": 2,
    "PERSON": 1,
    "EVENT": 1,
}

# "Ontology-ish" mapping
NER_TO_CLASS_BASE = {
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
    "NORP": "Group",
    "EVENT": "Event",
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

# Cleanup helpers
POSSESSIVE_RE = re.compile(r"(\b\w+)'s\b", flags=re.IGNORECASE)
TRAILING_WIKI_REFS_RE = re.compile(r"(?:\s*\[[^\]]*\]?)+\s*$")
TRAILING_ORPHAN_RE = re.compile(r"[\[\]\(\)]+(?:\s*)$")


# -----------------------------
# Text normalization / IDs
# -----------------------------
def normalize_entity_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
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


def make_entity_id(ent_text: str, ent_class: str) -> str:
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
    The Queen's Gambit fandom:
    - Domain is IN_UNIVERSE by default
    - Some pages may be about real actors (bio-like pages)
    """
    u = (url or "").strip()
    low_text = (text or "").lower()

    try:
        parsed = urlparse(u)
        netloc = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        full = (netloc + path).lower()
    except Exception:
        netloc = ""
        full = u.lower()

    # External websites
    if any(h in full for h in REALWORLD_URL_HINTS):
        return True
    if any(k in netloc for k in ["wikipedia.org", "imdb.com"]):
        return True

    # In-universe domain: default False, except strong bio signals
    if netloc in IN_UNIVERSE_DOMAINS:
        bio_signals = [
            "born", "nationality", "filmography", "career",
            "actress", "actor", "television", "film", "role"
        ]
        score = sum(1 for s in bio_signals if s in low_text)
        return score >= 3

    # Unknown domains: conservative
    return True


# -----------------------------
# Work-title detection
# -----------------------------
def looks_like_work_title(canonical: str) -> bool:
    c = canonical.strip()
    if not c:
        return False
    low = c.lower()

    if any(kw in low for kw in WORK_TITLE_KEYWORDS):
        return True

    words = c.split()
    if len(words) >= 3 and words[0].lower() in WORK_TITLE_STARTERS and WORK_TITLE_RE.match(c):
        return True

    return False


def person_class_for_context(page_is_realworld: bool) -> str:
    return "RealPerson" if page_is_realworld else "Character"


def get_entity_class(final_label: str, canonical: str, page_is_realworld: bool, is_actor: bool) -> str:
    """
    Resolve classes:
    - PERSON -> Character by default on in-universe pages
    - actor pattern forces RealPerson
    - work title forces Work
    """
    if final_label == "PERSON":
        if looks_like_work_title(canonical):
            return "Work"
        if is_actor:
            return "RealPerson"
        return person_class_for_context(page_is_realworld)

    return NER_TO_CLASS_BASE.get(final_label, "Thing")


# -----------------------------
# Actor detection ("portrayed by Y")
# -----------------------------
def find_actor_entities_in_sentence(sent, doc) -> set[str]:
    actors = set()

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
def choose_majority_label(labels: list[str], ent_text: str) -> str:
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
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# -----------------------------
# Entity extraction
# -----------------------------
def extract_entities(doc, url: str, page_is_realworld: bool):
    occ_labels = defaultdict(list)    # canonical -> [labels...]
    occ_forms = defaultdict(Counter)  # canonical -> Counter(surface_forms)

    actor_canonicals = set()
    for sent in doc.sents:
        actor_canonicals |= find_actor_entities_in_sentence(sent, doc)

    for ent in doc.ents:
        if ent.label_ not in TARGET_ENTITY_LABELS:
            continue

        surface = normalize_entity_text(ent.text)
        canonical = canonicalize_entity_text(surface)
        if not canonical:
            continue

        # Light noise filters
        if len(canonical) < 3:
            continue
        if sum(ch.isdigit() for ch in canonical) >= max(2, len(canonical) // 2):
            continue

        occ_labels[canonical].append(ent.label_)
        occ_forms[canonical][surface] += 1

    rows = []
    for canonical, labels in occ_labels.items():
        final_label = choose_majority_label(labels, canonical)

        is_actor = canonical in actor_canonicals
        ent_class = get_entity_class(final_label, canonical, page_is_realworld, is_actor)
        ent_id = make_entity_id(canonical, ent_class)

        forms_counter = occ_forms[canonical]
        preferred, _ = forms_counter.most_common(1)[0]
        aliases = [f for f, _ in forms_counter.most_common(10)]
        count_in_page = sum(forms_counter.values())

        rows.append({
            "entity_id": ent_id,
            "canonical_text": canonical,
            "preferred_text": preferred,
            "aliases": "|".join(aliases),
            "count_in_page": count_in_page,
            "entity_label": final_label,
            "entity_class": ent_class,
            "page_context": "REAL_WORLD" if page_is_realworld else "IN_UNIVERSE",
            "url": url,
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


def entity_span_to_row(ent, url, page_is_realworld: bool, actor_canonicals: set[str]):
    surface = normalize_entity_text(ent.text)
    canonical = canonicalize_entity_text(surface)
    if not canonical:
        return None

    label = ent.label_
    if label not in TARGET_ENTITY_LABELS:
        return None

    is_actor = canonical in actor_canonicals
    ent_class = get_entity_class(label, canonical, page_is_realworld, is_actor)
    ent_id = make_entity_id(canonical, ent_class)

    return {
        "entity_id": ent_id,
        "canonical_text": canonical,
        "surface_text": surface,
        "entity_label": label,
        "entity_class": ent_class,
        "page_context": "REAL_WORLD" if page_is_realworld else "IN_UNIVERSE",
        "url": url,
    }


# -----------------------------
# Relation extraction
# -----------------------------
def extract_relations(doc, url: str, page_is_realworld: bool):
    rels = []

    actor_canonicals = set()
    for sent in doc.sents:
        actor_canonicals |= find_actor_entities_in_sentence(sent, doc)

    def add_rel(subj_ent, obj_ent, rel_label, evidence_sent, pattern, verb_lemma=None, confidence=1.0):
        subj = entity_span_to_row(subj_ent, url, page_is_realworld, actor_canonicals)
        obj = entity_span_to_row(obj_ent, url, page_is_realworld, actor_canonicals)
        if not subj or not obj:
            return

        ev_start, ev_end = sentence_offsets(evidence_sent)
        evidence = shorten_evidence(evidence_sent.text)

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
            "pattern": pattern,
            "confidence": float(confidence),

            "evidence": evidence,
            "evidence_start_char": ev_start,
            "evidence_end_char": ev_end,
            "page_context": "REAL_WORLD" if page_is_realworld else "IN_UNIVERSE",
            "url": url,
        })

    # (1) Verb-based relations
    for tok in doc:
        if tok.pos_ != "VERB":
            continue

        lemma = tok.lemma_.lower()
        if lemma not in RELATION_VERBS:
            continue

        subj_tok, obj_tok = find_subject_object(tok)
        if not subj_tok or not obj_tok:
            continue

        subj_ent = token_to_entity_span(subj_tok, doc)
        obj_ent = token_to_entity_span(obj_tok, doc)
        if not subj_ent or not obj_ent:
            continue

        pattern = "verb_nsubj_obj"
        confidence = 0.9
        if obj_tok.dep_ == "pobj":
            pattern = "verb_nsubj_pobj"
            confidence = 0.7

        add_rel(
            subj_ent=subj_ent,
            obj_ent=obj_ent,
            rel_label=RELATION_VERBS[lemma],
            evidence_sent=tok.sent,
            pattern=pattern,
            verb_lemma=lemma,
            confidence=confidence
        )

    # (2) Family nominal patterns: "son/daughter of ..."
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

        # Child via apposition: "X, daughter of Y"
        child_tok = tok.head if tok.dep_ == "appos" else None

        if child_tok:
            child_ent = token_to_entity_span(child_tok, doc)
            parent_ent = token_to_entity_span(parent_tok, doc)
            if child_ent and parent_ent:
                add_rel(
                    subj_ent=child_ent,
                    obj_ent=parent_ent,
                    rel_label=FAMILY_TRIGGERS[lemma],
                    evidence_sent=tok.sent,
                    pattern="family_appos_of",
                    verb_lemma=lemma,
                    confidence=0.85
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
            conf = 0.85
            patt = "possessive_kinship"
        else:
            rel_label = SPOUSE_NOUNS[noun]
            conf = 0.8
            patt = "possessive_spouse"

        add_rel(
            subj_ent=poss_ent,
            obj_ent=y_ent,
            rel_label=rel_label,
            evidence_sent=tok.sent,
            pattern=patt,
            verb_lemma=noun,
            confidence=conf
        )

    return rels


# -----------------------------
# Main
# -----------------------------
def main(in_jsonl="data/raw_jsonl/pages.jsonl"):
    nlp = spacy.load(NLP_MODEL)

    records = list(load_jsonl(in_jsonl))
    urls = [r.get("url", "") for r in records]
    texts = [r.get("text", "") for r in records]

    entity_rows = []
    relation_rows = []

    for doc, url, text in zip(nlp.pipe(texts, batch_size=8, n_process=-1), urls, texts):
        page_is_realworld = looks_realworld_page(url, text)

        entity_rows.extend(extract_entities(doc, url, page_is_realworld))
        relation_rows.extend(extract_relations(doc, url, page_is_realworld))

    ent_df = pd.DataFrame(entity_rows).drop_duplicates()
    rel_df = pd.DataFrame(relation_rows).drop_duplicates()

    if not ent_df.empty:
        ent_df = ent_df.sort_values(["entity_class", "canonical_text", "url"])
    if not rel_df.empty:
        rel_df = rel_df.sort_values(["relation", "confidence"], ascending=[True, False])

    ent_df.to_csv("data/entities.csv", index=False)
    rel_df.to_csv("data/relations.csv", index=False)


if __name__ == "__main__":
    main()
    dedupe_entities_csv()