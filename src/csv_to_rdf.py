from __future__ import annotations

"""
csv_to_rdf.py — CSV (entities + relations) -> RDF/Turtle ABox ONLY

⚠️ ABox-only version:
- Ne génère PLUS de déclarations d'ontologie (pas de owl:Class / owl:ObjectProperty / owl:DatatypeProperty)
- Écrit uniquement des individus, leurs types (classes déjà définies dans ontologie.ttl),
  les relations (propriétés déjà définies dans ontologie.ttl) et la provenance (reification rdf:Statement).

Entrées CSV (colonnes attendues) :
- entities: entity_id, canonical_text, entity_class, url (optionnel), ...
- relations: subject_id, object_id, relation, confidence (optionnel), evidence (optionnel),
             evidence_start_char (optionnel), evidence_end_char (optionnel), source_url (optionnel)

Sortie :
- TTL ABox (données) — à charger avec l’ontologie (TBox) séparément, ou via owl:imports (optionnel).
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


# -----------------------------
# RDF/Turtle helpers
# -----------------------------

def ttl_escape_literal(s: str) -> str:
    """Escape string for Turtle literal."""
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return s


def is_safe_local_name(s: str) -> bool:
    """Conservative check for Turtle prefixed-name local part."""
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*", s))


def slugify(s: str) -> str:
    """Generate a safe-ish identifier from label."""
    s = s.strip().lower()
    s = re.sub(r"['’]", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "entity"
    if re.match(r"^\d", s):
        s = "e_" + s
    return s


@dataclass
class Namespaces:
    base: str
    prefix: str


# -----------------------------
# Domain mappings (Queen's Gambit / Jeu de la Dame)
# -----------------------------

# Entity class mapping (entity_class -> Ontology class local name)
CLASS_MAP: Dict[str, str] = {
    # Core
    "CHARACTER": "Character",
    "PERSON": "Character",          # if extractor uses PERSON for fictional chars
    "REAL_PERSON": "RealPerson",    # real chess players / historical persons
    "WORK": "Work",
    "EPISODE": "Episode",
    "BOOK": "Book",
    "MOVIE": "Work",
    "SERIES": "Work",

    # Places / orgs / events
    "LOCATION": "Location",
    "PLACE": "Location",
    "CITY": "Location",
    "COUNTRY": "Location",
    "ORGANIZATION": "Organization",
    "ORG": "Organization",
    "TOURNAMENT": "Tournament",
    "EVENT": "Event",

    # Objects / concepts
    "CHESS_OPENING": "ChessOpening",
    "GAME": "Game",
    "TITLE": "Title",
}

# Relations mapping (relation -> Ontology object property local name)
REL_MAP: Dict[str, str] = {
    # Story / interactions
    "MET": "met",
    "HELPED": "helped",
    "SUPPORTED": "supported",
    "BEFRIENDED": "befriended",
    "LOVED": "loved",
    "MARRIED": "married",
    "DIVORCED": "divorced",
    "TRAINED_BY": "trainedBy",
    "COACHED_BY": "coachedBy",
    "TAUGHT_BY": "taughtBy",
    "LEARNED_FROM": "learnedFrom",

    # Chess-related
    "PLAYED": "played",
    "PLAYED_IN": "playedIn",
    "COMPETED_IN": "competedIn",
    "ENTERED": "entered",
    "WON": "won",
    "LOST": "lost",
    "DEFEATED": "defeated",
    "LOST_TO": "lostTo",
    "DREW_WITH": "drewWith",
    "USED_OPENING": "usedOpening",

    # Family
    "PARENT_OF": "parentOf",
    "CHILD_OF": "childOf",
    "SIBLING_OF": "siblingOf",
}


# -----------------------------
# Reading CSV
# -----------------------------

def read_csv_dicts(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}


def pick_id(row: dict, fallback_label_key: str = "canonical_text") -> str:
    """
    Pick an entity identifier.
    Priority: entity_id, id, canonical_id, else slugify(canonical_text).
    """
    for k in ("entity_id", "id", "canonical_id"):
        if row.get(k):
            return row[k]
    label = row.get(fallback_label_key) or "entity"
    return slugify(label)


def safe_curie(ns: Namespaces, local: str) -> Tuple[str, str]:
    """
    Return (term, iri) where term is either prefixed name (prefix:local) or <iri>.
    """
    if is_safe_local_name(local):
        return f"{ns.prefix}:{local}", ns.base + local
    local2 = slugify(local)
    if is_safe_local_name(local2):
        return f"{ns.prefix}:{local2}", ns.base + local2
    return f"<{ns.base}{local2}>", ns.base + local2


# -----------------------------
# Turtle serialization
# -----------------------------

def ttl_header(ns: Namespaces) -> str:
    return "\n".join(
        [
            f"@prefix {ns.prefix}: <{ns.base}> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "",
        ]
    )


def emit_triple(s: str, p: str, o: str) -> str:
    return f"{s} {p} {o} .\n"


def emit_literal(s: str, p: str, lit: str, lang: Optional[str] = None, dtype: Optional[str] = None) -> str:
    esc = ttl_escape_literal(lit)
    if lang:
        o = f"\"{esc}\"@{lang}"
    elif dtype:
        o = f"\"{esc}\"^^{dtype}"
    else:
        o = f"\"{esc}\""
    return emit_triple(s, p, o)


# -----------------------------
# Main conversion logic (ABox only)
# -----------------------------

def build_entity_index(entities_path: Path) -> Dict[str, dict]:
    """
    Map entity_id -> entity row.
    Ensures stable IDs even if CSV uses different id column names.
    """
    idx: Dict[str, dict] = {}
    for row in read_csv_dicts(entities_path):
        eid = pick_id(row)
        idx[eid] = row
        # Keep alias if original CSV has explicit entity_id different from computed one
        if row.get("entity_id") and row["entity_id"] != eid:
            idx[row["entity_id"]] = row
    return idx


def convert(
    entities_path: Path,
    relations_path: Path,
    out_path: Path,
    base: str,
    prefix: str,
    lang: str = "fr",
    ontology_import: Optional[str] = None,
    strict_relations: bool = True,
) -> None:
    """
    ABox-only conversion:
    - No class/property declarations
    - Unknown relations: skipped if strict_relations=True
    """
    ns = Namespaces(base=base, prefix=prefix)
    entity_idx = build_entity_index(entities_path)

    ttl: list[str] = []
    ttl.append(ttl_header(ns))

    # Optional: import ontology (purely informational in TTL; some tools ignore file: IRIs)
    if ontology_import:
        # represent the base namespace as an ontology node and import ontology_import
        base_iri = ns.base[:-1] if ns.base.endswith(("#", "/")) else ns.base
        ttl.append(emit_triple(f"<{base_iri}>", "rdf:type", "owl:Ontology"))
        ttl.append(emit_triple(f"<{base_iri}>", "owl:imports", f"<{ontology_import}>"))
        ttl.append("\n")

    # ----- Entities (Individuals) -----
    for row in read_csv_dicts(entities_path):
        eid = pick_id(row)
        label = row.get("canonical_text") or row.get("label") or eid
        eclass_raw = (row.get("entity_class") or row.get("class") or "").upper().strip()
        class_local = CLASS_MAP.get(eclass_raw, "Thing")  # must exist in ontology

        subj, _ = safe_curie(ns, eid)
        class_term, _ = safe_curie(ns, class_local)

        ttl.append(emit_triple(subj, "rdf:type", class_term))
        ttl.append(emit_literal(subj, "rdfs:label", label, lang=lang))

        url = row.get("url") or row.get("source_url") or row.get("wikipedia") or row.get("wikidata")
        if url:
            # property must exist in ontology (DatatypeProperty or ObjectProperty to IRI)
            p_source, _ = safe_curie(ns, "sourceUrl")
            ttl.append(emit_triple(subj, p_source, f"<{url}>"))

    # ----- Relations (Object Property Assertions) -----
    # Provenance properties (must exist in ontology; we DO NOT declare them here)
    prov_props = {
        "confidence": ("confidence", "xsd:decimal"),
        "evidence": ("evidence", None),
        "evidence_start_char": ("evidenceStartChar", "xsd:integer"),
        "evidence_end_char": ("evidenceEndChar", "xsd:integer"),
        "source_url": ("sourceUrl", None),
    }

    for row in read_csv_dicts(relations_path):
        sid = row.get("subject_id") or row.get("subject") or row.get("subj") or ""
        oid = row.get("object_id") or row.get("object") or row.get("obj") or ""
        rel_raw = (row.get("relation") or row.get("predicate") or "").upper().strip()

        if not sid or not oid or not rel_raw:
            continue

        # normalize ids if relations use canonical_text instead of ids
        sid2 = sid if sid in entity_idx else slugify(sid)
        oid2 = oid if oid in entity_idx else slugify(oid)

        s_term, _ = safe_curie(ns, sid2)
        o_term, _ = safe_curie(ns, oid2)

        prop_local = REL_MAP.get(rel_raw)
        if not prop_local:
            if strict_relations:
                continue
            # non-strict mode: create a predicate from relation label (still ABox-only, but may invent properties)
            prop_local = slugify(rel_raw)

        p_term, _ = safe_curie(ns, prop_local)

        ttl.append(emit_triple(s_term, p_term, o_term))

        # Attach provenance via reified statement if any extras exist
        extras = {k: row.get(k) for k in prov_props.keys() if row.get(k)}
        if extras:
            ttl.append("[] rdf:type rdf:Statement ;\n")
            ttl.append(f"   rdf:subject {s_term} ;\n")
            ttl.append(f"   rdf:predicate {p_term} ;\n")
            ttl.append(f"   rdf:object {o_term} ;\n")

            for k, v in extras.items():
                plocal, dtype = prov_props[k]
                pextra, _ = safe_curie(ns, plocal)

                if k == "confidence":
                    ttl.append(f"   {pextra} \"{ttl_escape_literal(v)}\"^^xsd:decimal ;\n")
                elif dtype:
                    ttl.append(f"   {pextra} \"{ttl_escape_literal(v)}\"^^{dtype} ;\n")
                elif plocal == "sourceUrl" and re.match(r"^https?://", v):
                    ttl.append(f"   {pextra} <{v}> ;\n")
                else:
                    ttl.append(f"   {pextra} \"{ttl_escape_literal(v)}\" ;\n")

            # replace last ; with .
            if ttl[-1].endswith(";\n"):
                ttl[-1] = ttl[-1][:-2] + " .\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(ttl), encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert entities + relations CSVs to RDF/Turtle (ABox-only).")
    p.add_argument("--entities", type=Path, default=Path("data/entities_dedup.csv"), help="Path to entities CSV.")
    p.add_argument("--relations", type=Path, default=Path("data/relations.csv"), help="Path to relations CSV.")
    p.add_argument("--out", type=Path, default=Path("data/queens_gambit_graph_abox.ttl"), help="Output TTL path.")
    p.add_argument("--base", type=str, default="http://example.org/qg#", help="Base IRI for your namespace.")
    p.add_argument("--prefix", type=str, default="qg", help="Prefix for your namespace.")
    p.add_argument("--lang", type=str, default="fr", help="Language tag for rdfs:label (default: fr).")
    p.add_argument(
        "--ontology-import",
        type=str,
        default=None,
        help="Optional IRI of ontology to import (e.g., file:ontologie.ttl or http://.../ontologie.ttl).",
    )
    p.add_argument(
        "--non-strict-relations",
        action="store_true",
        help="If set, unknown relations are NOT skipped (they will be slugified into predicates).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        entities_path=args.entities,
        relations_path=args.relations,
        out_path=args.out,
        base=args.base,
        prefix=args.prefix,
        lang=args.lang,
        ontology_import=args.ontology_import,
        strict_relations=not args.non_strict_relations,
    )
    print(f" Written Turtle (ABox-only) to: {args.out}")


if __name__ == "__main__":
    main()