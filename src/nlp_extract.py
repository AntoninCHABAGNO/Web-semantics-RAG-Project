# src/nlp_extract.py
import json
import pandas as pd
import spacy
import re
from collections import defaultdict, Counter

NLP_MODEL = "en_core_web_trf"

TARGET_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT"}


# verb lemmas -> relation label (simple baseline)
RELATION_VERBS = {
    "rule": "RULES",
    "reign": "RULES",
    "lead": "LEADS",
    "command": "COMMANDS",
    "serve": "SERVED",
    "ally": "ALLIED_WITH",
    "betray": "BETRAYED",
    "attack": "ATTACKED",
    "defeat": "DEFEATED",
    "capture": "CAPTURED",
    "kill": "KILLED",
    "murder": "KILLED",
    "marry": "MARRIED",
    "wed": "MARRIED",
    "love": "LOVED",
    "hate": "HATED",
    "protect": "PROTECTED",
    "follow": "FOLLOWED",
    "born": "BORN_IN",
    "die": "DIED_IN",
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

def normalize_entity_text(s: str) -> str:
    return " ".join(s.split()).strip()

def is_chapter_like(ent_text: str) -> bool:
    # Ex: "Bran I.", "Catelyn VIII" -> souvent des chapitres, pas des entités utiles
    toks = ent_text.split()
    return len(toks) >= 2 and ROMAN_SUFFIX_RE.search(toks[-1]) is not None

def force_label_if_region(ent_text: str, current_label: str) -> str:
    # Règle demandée : si entité correspond à une région connue -> GPE
    # On compare en minuscules, et on gère les apostrophes/points
    key = ent_text.lower().strip(" .,'\"")
    if key in KNOWN_REGIONS:
        return "GPE"
    return current_label

def choose_majority_label(labels: list[str], ent_text: str) -> str:
    """
    - Prend le label le plus fréquent sur la page (majorité).
    - En cas d'égalité, choisit selon une priorité (GPE/LOC > ORG > PERSON).
    - Applique la règle région connue -> GPE.
    """
    labels = [force_label_if_region(ent_text, lb) for lb in labels]
    counts = Counter(labels)
    max_count = max(counts.values())
    tied = [lb for lb, c in counts.items() if c == max_count]

    if len(tied) == 1:
        return tied[0]

    # tie-break par priorité
    tied.sort(key=lambda lb: LABEL_PRIORITY.get(lb, 0), reverse=True)
    return tied[0]



def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def extract_entities(doc, url):
    """
    Retourne 1 seule ligne par entité_text et par url,
    avec un label choisi par majorité + règles de priorité.
    """
    occurrences = defaultdict(list)  # ent_text -> [labels...]

    for ent in doc.ents:
        if ent.label_ not in TARGET_ENTITY_LABELS:
            continue

        ent_text = normalize_entity_text(ent.text)

        # optionnel mais conseillé : ignorer les entités "chapitre"
        if is_chapter_like(ent_text):
            continue

        occurrences[ent_text].append(ent.label_)

    rows = []
    for ent_text, labels in occurrences.items():
        final_label = choose_majority_label(labels, ent_text)
        rows.append({
            "entity_text": ent_text,
            "entity_label": final_label,
            "url": url,
        })

    return rows

def find_subject_object(verb_token):
    # Heuristique: subject = nsubj, object = dobj/attr/pobj
    subj = None
    obj = None
    for child in verb_token.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj = child
        if child.dep_ in ("dobj", "attr", "oprd", "dative"):

            obj = child
    # cas prépositionnel : "hosted in Paris" => pobj de "in"
    if obj is None:
        for child in verb_token.children:
            if child.dep_ == "prep":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        obj = gc
                        break
    return subj, obj

def token_to_entity_span(token, doc):
    # Remonter à une entité NER si le token est dedans
    for ent in doc.ents:
        if ent.start <= token.i < ent.end and ent.label_ in TARGET_ENTITY_LABELS:
            return ent
    return None

def extract_relations(doc, url):
    rels = []
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

        subj_text = normalize_entity_text(subj_ent.text)
        obj_text = normalize_entity_text(obj_ent.text)

        # éviter les relations avec chapitres
        if is_chapter_like(subj_text) or is_chapter_like(obj_text):
            continue

        # evidence courte
        evidence = normalize_entity_text(tok.sent.text)
        MAX_EVIDENCE_CHARS = 250
        if len(evidence) > MAX_EVIDENCE_CHARS:
            evidence = evidence[:MAX_EVIDENCE_CHARS].rsplit(" ", 1)[0] + "…"

        # types cohérents pour régions connues
        subj_type = force_label_if_region(subj_text, subj_ent.label_)
        obj_type = force_label_if_region(obj_text, obj_ent.label_)

        rels.append({
            "subject": subj_text,
            "subject_type": subj_type,
            "relation": RELATION_VERBS[lemma],
            "object": obj_text,
            "object_type": obj_type,
            "evidence": evidence,
            "url": url
        })
    return rels


def main(in_jsonl="data/raw_jsonl/pages.jsonl"):
    nlp = spacy.load(NLP_MODEL)

    entity_rows = []
    relation_rows = []

    for rec in load_jsonl(in_jsonl):
        url = rec["url"]
        text = rec["text"]

        doc = nlp(text)

        entity_rows.extend(extract_entities(doc, url))
        relation_rows.extend(extract_relations(doc, url))

    ent_df = pd.DataFrame(entity_rows).drop_duplicates()
    rel_df = pd.DataFrame(relation_rows).drop_duplicates()

    ent_df.to_csv("data/entities.csv", index=False)
    rel_df.to_csv("data/relations.csv", index=False)

if __name__ == "__main__":
    main()
