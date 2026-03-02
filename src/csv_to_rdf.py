# src/csv_to_rdf.py
from __future__ import annotations

import os
import re
import pandas as pd

BASE = "http://example.org/got#"
ONTO_IRI = "http://example.org/got"
GRAPH_IRI = "http://example.org/got-graph"  # IRI of the generated ABox file

# Map CSV relation labels -> ontology property local names (the ones you already defined in got.ttl)
REL_MAP = {
    "MEMBER_OF": "memberOf",
    "CHILD_OF": "childOf",
    "PARENT_OF": "parentOf",
    "KILLED": "killed",
    "MARRIED": "marriedTo",
    "RULES": "rules",
    "LOCATED_IN": "locatedIn",
    "ALLIED_WITH": "alliedWith",
}

# Map entity_class -> ontology class local names (the ones you defined in got.ttl)
CLASS_MAP = {
    "Entity": "Entity",
    "Person": "Person",
    "Character": "Character",
    "RealPerson": "RealPerson",
    "Group": "Group",
    "House": "House",
    "Organization": "Organization",
    "Place": "Place",
    "Location": "Location",
    "Region": "Region",
    "Event": "Event",
    "Battle": "Battle",
    "Work": "Work",
}

SAFE_LOCAL_RE = re.compile(r"[^A-Za-z0-9_]+")


def safe_local_name(raw: str) -> str:
    """
    Turn your entity_id (e.g. 'Character:addam-90295f') into a Turtle-safe local name.
    """
    raw = (raw or "").strip()
    raw = raw.replace(":", "_")
    raw = SAFE_LOCAL_RE.sub("_", raw)
    raw = raw.strip("_")
    if not raw:
        raw = "unknown"
    if raw[0].isdigit():
        raw = "i_" + raw
    return "i_" + raw


def prop_local_name(rel_label: str) -> str:
    """
    Convert relation label into a property local name.
    If not in REL_MAP, create a conservative lowerCamel-ish name and declare it in the output
    as owl:ObjectProperty with domain/range got:Entity.
    """
    rel = (rel_label or "").strip().upper()
    if rel in REL_MAP:
        return REL_MAP[rel]

    # fallback: ATTACKED -> attacked, DEFEATED -> defeated, SWORE_FEALTY_TO -> sworeFealtyTo, etc.
    parts = rel.lower().split("_")
    if not parts:
        return "relatedTo"
    return parts[0] + "".join(p.title() for p in parts[1:])


def turtle_escape(s: str) -> str:
    s = (s or "")
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\n", "\\n").replace("\r", "\\r")
    return s


def write_graph(
    entities_csv: str,
    relations_csv: str,
    out_ttl: str,
) -> None:
    ent = pd.read_csv(entities_csv)
    rel = pd.read_csv(relations_csv)

    # Collect which properties we will use (for declaring unknown ones)
    used_props: set[str] = set()

    lines = []
    lines.append('@prefix : <http://example.org/got#> .')
    lines.append('@prefix got: <http://example.org/got#> .')
    lines.append('@prefix owl: <http://www.w3.org/2002/07/owl#> .')
    lines.append('@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .')
    lines.append('@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .')
    lines.append('@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .')
    lines.append("")

    # ---- Individuals (entities)
    # We'll generate:
    #   :i_Character_addam_90295f a got:Character ; got:canonicalName "Addam" ; got:sourceUrl "..."^^xsd:anyURI .
    for _, row in ent.iterrows():
        entity_id = str(row.get("entity_id", "")).strip()
        canonical = str(row.get("canonical_text", "")).strip()
        url = str(row.get("url", "")).strip()

        if not entity_id:
            continue

        local = safe_local_name(entity_id)
        ent_class = str(row.get("entity_class", "Entity")).strip()
        cls_local = CLASS_MAP.get(ent_class, "Entity")

        lines.append(f':{local} a got:{cls_local} ;')
        if canonical:
            lines.append(f'  got:canonicalName "{turtle_escape(canonical)}"^^xsd:string ;')
        if url:
            lines.append(f'  got:sourceUrl "{turtle_escape(url)}"^^xsd:anyURI ;')
        # End individual statement (replace last ';' with '.')
        if lines[-1].endswith(" ;"):
            lines[-1] = lines[-1][:-2] + " ."
        else:
            # if no canonical/url were added (rare), close the triple
            lines.append(" .")
        lines.append("")

    # ---- Property declarations for any relations not in ontology yet
    # We will declare them as ObjectProperty with domain/range got:Entity, so the file is self-contained.
    # (Your TBox can later refine these domain/range.)
    for _, row in rel.iterrows():
        used_props.add(prop_local_name(str(row.get("relation", ""))))

    # Properties already in your TBox (no need to redeclare, but harmless).
    # We'll declare only those NOT in REL_MAP values.
    known_props = set(REL_MAP.values())
    extra_props = sorted(p for p in used_props if p not in known_props)

    if extra_props:
        lines.append("##########")
        lines.append("# Extra object properties (auto-declared from relations.csv)")
        lines.append("##########")
        lines.append("")
        for p in extra_props:
            lines.append(f'got:{p} a owl:ObjectProperty ;')
            lines.append(f'  rdfs:domain got:Entity ;')
            lines.append(f'  rdfs:range got:Entity .')
            lines.append("")
        lines.append("")

    # ---- Facts (relations) + axiom annotations (evidence/confidence)
    #
    # We generate:
    #   :S got:attacked :O .
    #   [ a owl:Axiom ;
    #     owl:annotatedSource :S ;
    #     owl:annotatedProperty got:attacked ;
    #     owl:annotatedTarget :O ;
    #     got:confidence "0.9"^^xsd:float ;
    #     got:evidenceText "..."^^xsd:string ;
    #     got:evidenceStart "123"^^xsd:integer ;
    #     got:evidenceEnd "456"^^xsd:integer ;
    #   ] .
    for _, row in rel.iterrows():
        sid = str(row.get("subject_id", "")).strip()
        oid = str(row.get("object_id", "")).strip()
        rel_label = str(row.get("relation", "")).strip()

        if not sid or not oid or not rel_label:
            continue

        s_local = safe_local_name(sid)
        o_local = safe_local_name(oid)
        p_local = prop_local_name(rel_label)

        # the asserted triple
        lines.append(f':{s_local} got:{p_local} :{o_local} .')

        # axiom annotation (evidence)
        conf = row.get("confidence", None)
        ev = str(row.get("evidence", "") or "").strip()
        ev_start = row.get("evidence_start_char", None)
        ev_end = row.get("evidence_end_char", None)

        lines.append("[ a owl:Axiom ;")
        lines.append(f'  owl:annotatedSource :{s_local} ;')
        lines.append(f'  owl:annotatedProperty got:{p_local} ;')
        lines.append(f'  owl:annotatedTarget :{o_local} ;')

        if conf is not None and str(conf) != "nan":
            try:
                c = float(conf)
                lines.append(f'  got:confidence "{c}"^^xsd:float ;')
            except Exception:
                pass

        if ev:
            lines.append(f'  got:evidenceText "{turtle_escape(ev)}"^^xsd:string ;')

        if ev_start is not None and str(ev_start) != "nan":
            try:
                es = int(ev_start)
                lines.append(f'  got:evidenceStart "{es}"^^xsd:integer ;')
            except Exception:
                pass

        if ev_end is not None and str(ev_end) != "nan":
            try:
                ee = int(ev_end)
                lines.append(f'  got:evidenceEnd "{ee}"^^xsd:integer ;')
            except Exception:
                pass

        # close axiom node
        if lines[-1].endswith(" ;"):
            lines[-1] = lines[-1][:-2] + " ."
        else:
            lines.append(" .")
        lines.append("")

    os.makedirs(os.path.dirname(out_ttl), exist_ok=True)
    with open(out_ttl, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    # Adjust paths if your folder layout differs
    write_graph(
        entities_csv="data/entities.csv",
        relations_csv="data/relations.csv",
        out_ttl="data/got_graph.ttl",
    )
    print("Wrote: data/got_graph.ttl")