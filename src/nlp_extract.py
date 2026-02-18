# src/nlp_extract.py
import json
import pandas as pd
import spacy
from collections import defaultdict

NLP_MODEL = "en_core_web_trf"

TARGET_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT"}

# verb lemmas -> relation label (simple baseline)
RELATION_VERBS = {
    "host": "HOSTED",
    "hold": "HOSTED",
    "organize": "ORGANIZED",
    "win": "WON",
    "earn": "WON",
    "participate": "PARTICIPATED_IN",
    "compete": "PARTICIPATED_IN",
    "be": "IS",   # attention: trop général, à filtrer avec patterns
    "locate": "LOCATED_IN",
    "born": "BORN_IN",
}

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def extract_entities(doc, url):
    rows = []
    for ent in doc.ents:
        if ent.label_ in TARGET_ENTITY_LABELS:
            rows.append({
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "url": url,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })
    return rows

def find_subject_object(verb_token):
    # Heuristique: subject = nsubj, object = dobj/attr/pobj
    subj = None
    obj = None
    for child in verb_token.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj = child
        if child.dep_ in ("dobj", "attr", "oprd"):
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
        if tok.pos_ == "VERB":
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

            rels.append({
                "subject": subj_ent.text,
                "subject_type": subj_ent.label_,
                "relation": RELATION_VERBS[lemma],
                "object": obj_ent.text,
                "object_type": obj_ent.label_,
                "evidence": doc[max(0, tok.sent.start):min(len(doc), tok.sent.end)].text,
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
