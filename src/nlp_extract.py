# src/nlp_extract.py
import json
import pandas as pd
import spacy
import re
import hashlib
import unicodedata
from collections import defaultdict, Counter
from urllib.parse import urlparse

NLP_MODEL = "en_core_web_trf"

TARGET_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT"}

# verb lemmas -> relation label (simple baseline)
RELATION_VERBS = {
    # rule/lead
    "rule": "RULES",
    "reign": "RULES",
    "govern": "RULES",
    "lead": "LEADS",
    "command": "COMMANDS",
    "serve": "SERVED",
    "appoint": "APPOINTED",
    "name": "NAMED",
    "crown": "CROWNED",

    # alliances/conflict
    "ally": "ALLIED_WITH",
    "join": "JOINED",
    "swear": "SWORE_FEALTY_TO",
    "pledge": "SWORE_FEALTY_TO",
    "betray": "BETRAYED",
    "attack": "ATTACKED",
    "invade": "ATTACKED",
    "defeat": "DEFEATED",
    "capture": "CAPTURED",
    "siege": "BESIEGED",

    # violence
    "kill": "KILLED",
    "murder": "KILLED",
    "slay": "KILLED",
    "execute": "KILLED",
    "assassinate": "KILLED",

    # family/relationships
    "marry": "MARRIED",
    "wed": "MARRIED",
    "love": "LOVED",
    "hate": "HATED",

    # misc
    "protect": "PROTECTED",
    "follow": "FOLLOWED",
    "bear": "BORN_IN",   # often used in "was born in" -> lemma can be "bear"
    "born": "BORN_IN",   # sometimes appears weirdly depending on parse
    "die": "DIED_IN",
    "live": "LIVED_IN",
    "reside": "LIVED_IN",
    "locate": "LOCATED_IN",
}

KNOWN_REGIONS = {
    "dorne", "westeros", "essos", "valyria",
    "the reach", "the vale", "the riverlands", "the stormlands",
    "the westerlands", "the north", "the crownlands", "iron islands",
    "beyond the wall",
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

ROMAN_SUFFIX_RE = re.compile(r"\b[IVX]+\b\.?$")  # "I.", "VIII", etc.

# --- Lightweight "ontology-ish" mapping for later Structure step
# NOTE: PERSON is resolved dynamically (Character vs RealPerson) via heuristics.
NER_TO_CLASS_BASE = {
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
    "NORP": "Group",
    "EVENT": "Event",
}

# Relations / patterns helpers
FAMILY_TRIGGERS = {
    "son": "CHILD_OF",
    "daughter": "CHILD_OF",
    "child": "CHILD_OF",
    # Keep simple (you can later split into FATHER_OF/MOTHER_OF)
    "father": "CHILD_OF",
    "mother": "CHILD_OF",
}

# Possessive / kinship / simple attributes
KINSHIP_NOUNS = {
    "father": "CHILD_OF",  # (X's father Y => X CHILD_OF Y)
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

HOUSE_NOUNS = {"house"}  # for patterns like "House Stark"

HOUSE_MEMBERSHIP_PREPS = {"of"}  # "of House Stark"
ORIGIN_PREPS = {"from", "in", "at"}  # "from Winterfell", "in King's Landing"

COPULA_LEMMAS = {"be"}  # "is/was/are" -> lemma "be"
TITLE_HEADS = {
    "lord", "lady", "king", "queen", "prince", "princess",
    "ruler", "warden", "hand", "commander", "protector"
}

MAX_EVIDENCE_CHARS = 250

# --- Real-person vs character heuristics
# Make this conservative: only classify REAL_WORLD if strong URL hints OR non-AWOIAF domains.
REALWORLD_URL_HINTS = [
    "imdb", "wikipedia", "fandom.com", "wiki", "encyclopedia",
    "/actor", "/actress", "/cast", "/production", "/filming",
]

# AWOIAF is in-universe by default (key fix)
IN_UNIVERSE_DOMAINS = {
    "awoiaf.westeros.org",
    "www.awoiaf.westeros.org",
}

# --- Work-title filter (key fix)
# Titles like "A Game of Thrones", "A Clash of Kings", etc.
WORK_TITLE_STARTERS = {"a", "an", "the"}
WORK_TITLE_KEYWORDS = {
    "game of thrones", "clash of kings", "storm of swords",
    "feast for crows", "dance with dragons", "fire & blood",
}
WORK_TITLE_RE = re.compile(r"^(a|an|the)\s+.+", re.IGNORECASE)


def normalize_entity_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = " ".join(s.split()).strip()
    return s


POSSESSIVE_RE = re.compile(r"(\b\w+)'s\b", flags=re.IGNORECASE)
# Match trailing wiki refs even if the closing ']' is missing (truncated text)
TRAILING_WIKI_REFS_RE = re.compile(r"(?:\s*\[[^\]]*\]?)+\s*$")

# Remove leftover orphan punctuation/brackets at the end
TRAILING_ORPHAN_RE = re.compile(r"[\[\]\(\)]+(?:\s*)$")


def canonicalize_entity_text(s: str) -> str:
    s = normalize_entity_text(s)

    # 1) Remove trailing wiki-style references, even if truncated (missing ']')
    # Examples handled:
    #   "Addam Velaryon[10" -> "Addam Velaryon"
    #   "Aegon III Targaryen[21][22" -> "Aegon III Targaryen"
    #   "Aegon Targaryen.[3][37" -> "Aegon Targaryen."
    s = TRAILING_WIKI_REFS_RE.sub("", s)

    # 2) Remove possessive 's
    s = POSSESSIVE_RE.sub(r"\1", s)

    # 3) Strip common punctuation (avoid stripping '[' here; we already handled refs)
    s = s.strip(" \t\n\r.,;:!?'\"{}")

    # 4) Remove orphan closing parentheses/brackets left behind (e.g., "Aemon)")
    s = TRAILING_ORPHAN_RE.sub("", s)

    # 5) Final whitespace cleanup
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


def is_chapter_like(ent_text: str) -> bool:
    toks = ent_text.split()
    return len(toks) >= 2 and ROMAN_SUFFIX_RE.search(toks[-1]) is not None


def force_label_if_region(ent_text: str, current_label: str) -> str:
    key = ent_text.lower().strip(" .,'\"")
    if key in KNOWN_REGIONS:
        return "GPE"
    return current_label


def choose_majority_label(labels: list[str], ent_text: str) -> str:
    labels = [force_label_if_region(ent_text, lb) for lb in labels]
    counts = Counter(labels)
    max_count = max(counts.values())
    tied = [lb for lb, c in counts.items() if c == max_count]

    if len(tied) == 1:
        return tied[0]

    tied.sort(key=lambda lb: LABEL_PRIORITY.get(lb, 0), reverse=True)
    return tied[0]


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def head_noun(token) -> str:
    return token.lemma_.lower() if token is not None else ""

def find_nearest_person_entity_before(token, doc):
    # last PERSON entity ending before token
    candidates = [e for e in doc.ents if e.label_ == "PERSON" and e.end_char <= token.idx]
    return candidates[-1] if candidates else None

def span_is_entity(span, doc):
    # return the entity that covers span (if any)
    for ent in doc.ents:
        if ent.start <= span.start and ent.end >= span.end and ent.label_ in TARGET_ENTITY_LABELS:
            return ent
    return None

# -----------------------------
# Fix #1: Conservative REAL_WORLD detection + domain override
# -----------------------------
def looks_realworld_page(url: str, text: str) -> bool:
    """
    Conservative classifier:
    - If domain is AWOIAF => IN_UNIVERSE (False)
    - If domain looks like Wikipedia/IMDB/etc => REAL_WORLD (True)
    - Otherwise rely on URL hints only (not text words like "born", too noisy)
    """
    u = (url or "").strip()
    try:
        parsed = urlparse(u)
        netloc = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        full = (netloc + path).lower()
    except Exception:
        netloc = ""
        full = (u or "").lower()

    if netloc in IN_UNIVERSE_DOMAINS:
        return False  # key fix: AWOIAF pages are in-universe by default

    # Strong external signals
    if any(h in full for h in REALWORLD_URL_HINTS):
        return True

    # If it's clearly a known external domain, treat as real-world
    if any(k in netloc for k in ["wikipedia.org", "imdb.com"]):
        return True

    # Otherwise: default to in-universe
    return False


# -----------------------------
# Fix #3: Detect titles of works that spaCy may tag as PERSON
# -----------------------------
def looks_like_work_title(canonical: str) -> bool:
    """
    Heuristic:
    - Starts with A/An/The and has >=3 words, OR
    - Matches known GoT book-series patterns, OR
    - Contains typical title casing patterns and length (conservative)
    """
    c = canonical.strip()
    if not c:
        return False

    low = c.lower()

    # Exact-ish known keywords
    for kw in WORK_TITLE_KEYWORDS:
        if kw in low:
            return True

    words = c.split()
    if len(words) >= 3 and words[0].lower() in WORK_TITLE_STARTERS and WORK_TITLE_RE.match(c):
        return True

    # e.g. "A Game of Thrones" pattern: has "of" and first word article
    if len(words) >= 4 and words[0].lower() in WORK_TITLE_STARTERS and "of" in [w.lower() for w in words]:
        return True

    return False


def person_class_for_context(page_is_realworld: bool) -> str:
    return "RealPerson" if page_is_realworld else "Character"


def get_entity_class(final_label: str, canonical: str, page_is_realworld: bool, is_actor: bool) -> str:
    """
    Resolve classes.
    Fix #3: If a PERSON looks like a work title => Work
    Fix #2: Actor detection can force RealPerson even on in-universe pages.
    """
    if final_label == "PERSON":
        if looks_like_work_title(canonical):
            return "Work"
        if is_actor:
            return "RealPerson"
        return person_class_for_context(page_is_realworld)

    return NER_TO_CLASS_BASE.get(final_label, "Thing")


# -----------------------------
# Fix #2: Actor detection: ONLY mark the pobj after "by" as actor
# -----------------------------
ACTOR_TRIGGER_LEMMAS = {"play", "portray", "voice"}  # lemma forms
def find_actor_entities_in_sentence(sent, doc) -> set[str]:
    """
    Return canonical texts of actors in patterns like:
      "X is portrayed by Y" / "X is played by Y"
    We only tag Y (pobj after 'by') as RealPerson.
    """
    actors = set()

    for tok in sent:
        # Look for played/portrayed/voiced verbs (or participles)
        if tok.lemma_.lower() not in ACTOR_TRIGGER_LEMMAS:
            continue

        # Find a "by" preposition attached somewhere near this verb
        by_prep = None
        for child in tok.children:
            if child.dep_ == "prep" and child.lemma_.lower() == "by":
                by_prep = child
                break

        # Sometimes "portrayed" is an acl on a noun; the "by" can be under tok anyway.
        if by_prep is None:
            continue

        pobj = None
        for gc in by_prep.children:
            if gc.dep_ == "pobj":
                pobj = gc
                break

        if pobj is None:
            continue

        # If pobj is inside a PERSON entity => actor
        for ent in doc.ents:
            if ent.label_ == "PERSON" and ent.start <= pobj.i < ent.end:
                actors.add(canonicalize_entity_text(ent.text))

    return actors


def extract_entities(doc, url, page_is_realworld: bool):
    """
    One row per canonical entity and per url, with:
      - majority label (+region rule)
      - entity_class (Character vs RealPerson resolved by page + actor pattern)
      - Work-title fix
      - entity_id stable
      - aliases
      - count_in_page
    """
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

        if is_chapter_like(surface) or is_chapter_like(canonical):
            continue

        occ_labels[canonical].append(ent.label_)
        occ_forms[canonical][surface] += 1

    rows = []
    for canonical, labels in occ_labels.items():
        final_label = choose_majority_label(labels, canonical)
        final_label = force_label_if_region(canonical, final_label)

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

    label = force_label_if_region(canonical, ent.label_)
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


def extract_relations(doc, url, page_is_realworld: bool):
    rels = []

    actor_canonicals = set()
    for sent in doc.sents:
        actor_canonicals |= find_actor_entities_in_sentence(sent, doc)

    def add_rel(subj_ent, obj_ent, rel_label, evidence_sent, pattern, verb_lemma=None, confidence=1.0):
        subj = entity_span_to_row(subj_ent, url, page_is_realworld, actor_canonicals)
        obj = entity_span_to_row(obj_ent, url, page_is_realworld, actor_canonicals)
        if not subj or not obj:
            return

        # Skip chapter-like junk
        if is_chapter_like(subj["canonical_text"]) or is_chapter_like(obj["canonical_text"]):
            return

        # Optional: avoid relations involving Work unless you want them
        # if subj["entity_class"] == "Work" or obj["entity_class"] == "Work":
        #     return

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

    # (1) Verb-based relations (enriched)
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

    # (2) Copula patterns: "X is Y" (be + attr) and "X is <title> of Y"
    for tok in doc:
        if tok.pos_ not in ("AUX", "VERB"):
            continue
        if tok.lemma_.lower() not in COPULA_LEMMAS:
            continue

        subj_tok = None
        attr_tok = None
        for child in tok.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subj_tok = child
            if child.dep_ in ("attr", "acomp", "oprd"):
                attr_tok = child

        if not subj_tok or not attr_tok:
            continue

        subj_ent = token_to_entity_span(subj_tok, doc)
        if not subj_ent:
            continue

        attr_ent = token_to_entity_span(attr_tok, doc)

        # Case A: "X is Y" where Y is an entity
        if attr_ent:
            add_rel(
                subj_ent=subj_ent,
                obj_ent=attr_ent,
                rel_label="IS_ASSOCIATED_WITH",
                evidence_sent=tok.sent,
                pattern="copula_attr_entity",
                verb_lemma="be",
                confidence=0.6
            )
            continue

        # Case B: "X is Lord of Winterfell" -> RULES (heuristic)
        pobj = None
        for child in attr_tok.children:
            if child.dep_ == "prep" and child.lemma_.lower() == "of":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        pobj = gc
                        break

        if pobj:
            pobj_ent = token_to_entity_span(pobj, doc)
            if pobj_ent:
                head = attr_tok.lemma_.lower()
                if head in TITLE_HEADS:
                    rel_label = "RULES"
                    conf = 0.75
                    patt = "copula_title_of"
                else:
                    rel_label = "RELATED_TO"
                    conf = 0.55
                    patt = "copula_attr_of"

                add_rel(
                    subj_ent=subj_ent,
                    obj_ent=pobj_ent,
                    rel_label=rel_label,
                    evidence_sent=tok.sent,
                    pattern=patt,
                    verb_lemma="be",
                    confidence=conf
                )

    # (3) Family nominal patterns: "son/daughter of"
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

        # Child via apposition: "Arya, daughter of Eddard"
        child_tok = None
        if tok.dep_ == "appos" and tok.head:
            child_tok = tok.head

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
            continue

        # Fallback: last PERSON entity before token in the same sentence
        sent_ents = [e for e in tok.sent.ents if e.end_char <= tok.idx and e.label_ == "PERSON"]
        if sent_ents:
            child_ent = sent_ents[-1]
            parent_ent = token_to_entity_span(parent_tok, doc)
            if parent_ent:
                add_rel(
                    subj_ent=child_ent,
                    obj_ent=parent_ent,
                    rel_label=FAMILY_TRIGGERS[lemma],
                    evidence_sent=tok.sent,
                    pattern="family_<role>_of",
                    verb_lemma=lemma,
                    confidence=0.8
                )
        # --- (4) Possessive patterns: "X's father/mother/wife/husband ..."
    # Dependency pattern: possessor (X) -> possessed noun (father) -> appos/compound entity (Y)
    for tok in doc:
        # tok is the possessed noun, e.g., "father", "wife"
        noun = tok.lemma_.lower()
        if noun not in KINSHIP_NOUNS and noun not in SPOUSE_NOUNS:
            continue
        if tok.pos_ not in ("NOUN", "PROPN"):
            continue

        # Find possessor X: child with dep_ == 'poss'
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

        # Find Y: often appos or compound/proper noun attached to tok
        # Try appos first: "father, Eddard Stark" or "father Eddard Stark"
        y_ent = None

        # appos child
        for child in tok.children:
            if child.dep_ == "appos":
                y_ent = token_to_entity_span(child, doc)
                if y_ent:
                    break

        # if no appos, try a proper noun to the right within the same sentence
        if y_ent is None:
            # take the first entity after tok in same sentence
            candidates = [e for e in tok.sent.ents if e.start_char >= tok.idx]
            if candidates:
                # pick the closest PERSON/ORG/GPE etc
                y_ent = candidates[0] if candidates[0].label_ in TARGET_ENTITY_LABELS else None

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
    
        # --- (5) Apposition title: "X, Lord/King/... of Y"
    # Look for appositions where a title noun is appos to a PERSON entity.
    for tok in doc:
        # tok is title noun, often appos: "Lord" appos to "Eddard Stark"
        if tok.dep_ != "appos":
            continue
        if tok.lemma_.lower() not in TITLE_HEADS:
            continue

        # head of appos is the entity X
        head = tok.head
        x_ent = token_to_entity_span(head, doc)
        if not x_ent:
            continue

        # find "of" -> pobj => Y
        pobj = None
        for child in tok.children:
            if child.dep_ == "prep" and child.lemma_.lower() == "of":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        pobj = gc
                        break
        if pobj is None:
            continue

        y_ent = token_to_entity_span(pobj, doc)
        if not y_ent:
            continue

        add_rel(
            subj_ent=x_ent,
            obj_ent=y_ent,
            rel_label="RULES",
            evidence_sent=tok.sent,
            pattern="appos_title_of",
            verb_lemma=tok.lemma_.lower(),
            confidence=0.85
        )
    
        # --- (6) "of House X" membership: "Arya of House Stark"
    for tok in doc:
        # Look for token "house" and a proper noun right after / attached
        if tok.lemma_.lower() != "house":
            continue

        # Find the House name entity around "House Stark"
        house_ent = token_to_entity_span(tok, doc)
        if not house_ent:
            # fallback: build a span "House" + following proper nouns in sentence
            # but keep it simple: skip if NER didn't catch it
            continue

        # Find the member X: usually a PERSON entity earlier in the sentence, linked via prep "of"
        # Look for a preposition "of" whose pobj is this "house"
        of_prep = None
        for child in tok.children:
            # rare: house has head "of" but often "of" is attached to X not to house
            pass

        # So: find "of" token in sentence that governs "house" as pobj
        x_ent = None
        for t in tok.sent:
            if t.dep_ == "prep" and t.lemma_.lower() == "of":
                # does it have pobj that is within the house entity span?
                for gc in t.children:
                    if gc.dep_ == "pobj" and (house_ent.start <= gc.i < house_ent.end):
                        # X is often the head of this prep, or nearest PERSON before "of"
                        x_ent = token_to_entity_span(t.head, doc) or find_nearest_person_entity_before(t, doc)
                        break
            if x_ent:
                break

        if not x_ent:
            continue

        add_rel(
            subj_ent=x_ent,
            obj_ent=house_ent,
            rel_label="MEMBER_OF",
            evidence_sent=tok.sent,
            pattern="of_house_membership",
            verb_lemma="of",
            confidence=0.8
        )

    return rels


def main(in_jsonl="data/raw_jsonl/pages.jsonl"):
    nlp = spacy.load(NLP_MODEL)

    records = list(load_jsonl(in_jsonl))
    urls = [r.get("url", "") for r in records]
    texts = [r.get("text", "") for r in records]

    entity_rows = []
    relation_rows = []

    # Faster processing
    for doc, url, text in zip(nlp.pipe(texts, batch_size=8,n_process=-1), urls, texts):
        page_is_realworld = looks_realworld_page(url, text)

        entity_rows.extend(extract_entities(doc, url, page_is_realworld))
        relation_rows.extend(extract_relations(doc, url, page_is_realworld))

    ent_df = pd.DataFrame(entity_rows).drop_duplicates()
    rel_df = pd.DataFrame(relation_rows).drop_duplicates()

    # Optional: sort for readability
    if not ent_df.empty:
        ent_df = ent_df.sort_values(["entity_class", "canonical_text", "url"])
    if not rel_df.empty:
        rel_df = rel_df.sort_values(["relation", "confidence"], ascending=[True, False])

    ent_df.to_csv("data/entities.csv", index=False)
    rel_df.to_csv("data/relations.csv", index=False)


if __name__ == "__main__":
    main()