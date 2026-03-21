#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E_08b_swrl_reasoning.py — SWRL Reasoning with OWLReady2

PART 1 : family.owl  — deux règles SWRL (oncle + grand-parent)
PART 2 : Queens Gambit KB — deux règles SWRL sur votre graphe réel

Usage:
    python src/E_08b_swrl_reasoning.py
    python src/E_08b_swrl_reasoning.py --family_owl data/family.owl
    python src/E_08b_swrl_reasoning.py --abox data/queens_gambit_graph_abox.ttl
    python src/E_08b_swrl_reasoning.py --tbox data/ontologie_tbox.ttl
    python src/E_08b_swrl_reasoning.py --out results/swrl_results.json

Si Pellet ne fonctionne pas, le script essaie HermiT en fallback.
Sous Windows, il configure explicitement Java pour OwlReady2.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import owlready2
from rdflib import Graph
from rdflib.namespace import OWL

try:
    from owlready2 import (
        get_ontology,
        ObjectProperty,
        Imp,
        sync_reasoner,
        sync_reasoner_pellet,
    )
except ImportError:
    print("[ERROR] owlready2 not found. Install it with: pip install owlready2")
    sys.exit(1)


QG_BASE = "http://example.org/qg#"


# ─────────────────────────────────────────────────────────────────────────────
# Java config for OwlReady2 / Windows
# ─────────────────────────────────────────────────────────────────────────────

def _configure_java() -> None:
    """
    Configure explicitement java.exe pour OwlReady2.
    Priorité :
      1) JAVA_HOME/bin/java(.exe)
      2) java trouvé dans le PATH
      3) quelques chemins Windows classiques
    """
    candidates = []

    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidates.append(Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java"))

    java_in_path = shutil.which("java")
    if java_in_path:
        candidates.append(Path(java_in_path))

    if os.name == "nt":
        candidates.extend([
            Path(r"C:\Program Files\Eclipse Adoptium\jdk-17\bin\java.exe"),
            Path(r"C:\Program Files\Eclipse Adoptium\jdk-21\bin\java.exe"),
            Path(r"C:\Program Files\Java\jdk-17\bin\java.exe"),
            Path(r"C:\Program Files\Java\jdk-21\bin\java.exe"),
            Path(r"C:\Program Files\Microsoft\jdk-17.0.*/bin/java.exe"),
            Path(r"C:\Program Files\Eclipse Adoptium\jdk-25.0.2.10-hotspot\bin\java.exe")
        ])

    for candidate in candidates:
        candidate = Path(str(candidate))
        if "*" in str(candidate):
            import glob
            matches = glob.glob(str(candidate))
            if matches:
                owlready2.JAVA_EXE = matches[0]
                break
        elif candidate.exists():
            owlready2.JAVA_EXE = str(candidate)
            break

    if not getattr(owlready2, "JAVA_EXE", None):
        raise RuntimeError(
            "Java introuvable pour OwlReady2. "
            "Installe Java 17 ou définis JAVA_HOME."
        )

    print(f"[INFO] OwlReady2 Java = {owlready2.JAVA_EXE}")

    try:
        result = subprocess.run(
            [owlready2.JAVA_EXE, "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"[DEBUG] Java test return code: {result.returncode}")
        version_text = (result.stderr or result.stdout).strip()
        if version_text:
            first_line = version_text.splitlines()[0]
            print(f"[DEBUG] {first_line}")
    except Exception as e:
        raise RuntimeError(f"Impossible de lancer Java depuis Python: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_label(ind) -> str:
    labels = getattr(ind, "label", [])
    if labels:
        return str(labels[0])

    name = getattr(ind, "name", None) or str(ind)
    if "_" in name:
        parts = name.split("_")
        if len(parts) > 2:
            inner = parts[1:-1]
            return " ".join(w.capitalize() for w in inner)
    return name


def _ttl_to_rdfxml(src_path: Path, remove_imports: bool = False) -> Path:
    """
    Convertit un fichier Turtle en RDF/XML temporaire, plus fiable pour OwlReady2.
    """
    g = Graph()
    g.parse(str(src_path), format="turtle")

    if remove_imports:
        for triple in list(g.triples((None, OWL.imports, None))):
            g.remove(triple)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".owl")
    tmp.close()

    g.serialize(destination=tmp.name, format="xml")
    return Path(tmp.name)


def _load_ontology_auto(path: Path, remove_imports: bool = False):
    """
    Charge une ontologie de manière robuste :
      - .ttl  -> conversion Turtle -> RDF/XML puis load(fileobj=...)
      - .owl/.rdf/.xml -> load(fileobj=...)
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    ext = path.suffix.lower()

    if ext == ".ttl":
        converted = _ttl_to_rdfxml(path, remove_imports=remove_imports)
        onto = get_ontology(path.resolve().as_uri())
        with open(converted, "rb") as f:
            onto.load(fileobj=f, only_local=True)
        return onto

    if ext in {".owl", ".rdf", ".xml"}:
        onto = get_ontology(path.resolve().as_uri())
        with open(path, "rb") as f:
            onto.load(fileobj=f, only_local=True)
        return onto

    raise ValueError(f"Unsupported ontology format: {path.suffix}")


def _run_reasoner(onto, title: str) -> None:
    """
    Essaie Pellet puis fallback HermiT.
    """
    print(f"\n-> Running reasoner on {title} ...")
    print(f"[DEBUG] owlready2.JAVA_EXE = {getattr(owlready2, 'JAVA_EXE', None)}")

    try:
        with onto:
            sync_reasoner_pellet(
                infer_property_values=True,
                infer_data_property_values=False,
            )
        print("  Pellet completed successfully.")
        return
    except Exception as e:
        print(f"  [WARN] Pellet failed: {e}")
        print("  Trying HermiT fallback...")

    try:
        with onto:
            sync_reasoner(
                infer_property_values=True,
            )
        print("  HermiT completed successfully.")
    except Exception as e:
        print(f"  [WARN] HermiT failed too: {e}")
        print("  Continuing with partial results.")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — family.owl
# ─────────────────────────────────────────────────────────────────────────────

def run_family_rules(family_owl_path: Path) -> Dict:
    print("\n" + "=" * 60)
    print("  PART 1 - family.owl SWRL Rules")
    print("=" * 60)
    print(f"[DEBUG] Loading family ontology from: {family_owl_path.resolve()}")

    onto = _load_ontology_auto(family_owl_path)
    ns = onto.get_namespace("http://www.owl-ontologies.com/unnamed.owl#")

    with onto:
        if not onto.search_one(iri="*isUncleOf"):
            class isUncleOf(ObjectProperty):
                namespace = ns
                domain = [ns.Male]
                range = [ns.Person]

        if not onto.search_one(iri="*isGrandParentOf"):
            class isGrandParentOf(ObjectProperty):
                namespace = ns
                domain = [ns.Person]
                range = [ns.Person]

    # R0 : infer brother from shared parent
    with onto:
        rule_brother = Imp()
        rule_brother.set_as_rule(
            "Male(?x), Person(?y), Person(?p), "
            "isChildOf(?x, ?p), isChildOf(?y, ?p) "
            "-> isBrotherOf(?x, ?y)"
        )

    # R1 : uncle
    with onto:
        rule_uncle = Imp()
        rule_uncle.set_as_rule(
            "Male(?x), Person(?y), Person(?z), "
            "isBrotherOf(?x, ?y), isChildOf(?z, ?y) "
            "-> isUncleOf(?x, ?z)"
        )

    # R2 : grandparent
    with onto:
        rule_gp = Imp()
        rule_gp.set_as_rule(
            "Person(?x), Person(?y), Person(?z), "
            "isParentOf(?x, ?y), isChildOf(?z, ?y) "
            "-> isGrandParentOf(?x, ?z)"
        )

    _run_reasoner(onto, "family.owl")

    uncle_results = []
    grandp_results = []

    for ind in onto.individuals():
        for u in getattr(ind, "isUncleOf", []):
            if ind != u:
                uncle_results.append({
                    "uncle": ind.name,
                    "nephew_niece": u.name,
                })

        for g in getattr(ind, "isGrandParentOf", []):
            if ind != g:
                grandp_results.append({
                    "grandparent": ind.name,
                    "grandchild": g.name,
                })

    print("\n[RULE 1 - Uncle]")
    print("  Male(?x), isBrotherOf(?x,?y), isChildOf(?z,?y) -> isUncleOf(?x,?z)")
    if uncle_results:
        for r in uncle_results:
            print(f"  INFERRED: {r['uncle']}  isUncleOf  {r['nephew_niece']}")
    else:
        print("  (no uncle triples inferred)")

    print("\n[RULE 2 - GrandParent]")
    print("  Person(?x), isParentOf(?x,?y), isChildOf(?z,?y) -> isGrandParentOf(?x,?z)")
    if grandp_results:
        for r in grandp_results:
            print(f"  INFERRED: {r['grandparent']}  isGrandParentOf  {r['grandchild']}")
    else:
        print("  (no grandparent triples inferred)")

    return {
        "ontology": str(family_owl_path),
        "rule_uncle": uncle_results,
        "rule_grandparent": grandp_results,
    }

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Queens Gambit KB
# ─────────────────────────────────────────────────────────────────────────────

def run_qg_rules(abox_path: Path, tbox_path: Path) -> Dict:
    print("\n" + "=" * 60)
    print("  PART 2 - Queens Gambit KB SWRL Rules")
    print("=" * 60)
    print(f"[DEBUG] Loading TBox from: {tbox_path.resolve()}")
    print(f"[DEBUG] Loading ABox from: {abox_path.resolve()}")

    onto = _load_ontology_auto(tbox_path)
    _ = _load_ontology_auto(abox_path, remove_imports=True)

    ns = onto.get_namespace(QG_BASE)

    with onto:
        if not onto.search_one(iri=QG_BASE + "hasRival"):
            class hasRival(ObjectProperty):
                namespace = onto
                domain = [ns.Character]
                range = [ns.Character]

        if not onto.search_one(iri=QG_BASE + "knownOpponent"):
            class knownOpponent(ObjectProperty):
                namespace = onto
                domain = [ns.Character]
                range = [ns.Character]

    with onto:
        rule_rival = Imp()
        rule_rival.set_as_rule(
            "Character(?x), Character(?y), Character(?z), "
            "defeated(?x, ?y), defeated(?y, ?z) "
            "-> hasRival(?x, ?z)"
        )

    with onto:
        rule_known = Imp()
        rule_known.set_as_rule(
            "Character(?x), Character(?y), Character(?z), "
            "met(?x, ?y), defeated(?y, ?z) "
            "-> knownOpponent(?x, ?z)"
        )

    _run_reasoner(onto, "Queens Gambit KB")

    rival_results = []
    known_results = []

    try:
        characters = list(ns.Character.instances())
    except Exception:
        characters = list(onto.individuals())

    for ind in characters:
        ind_name = _get_label(ind)

        for r in getattr(ind, "hasRival", []):
            rival_results.append({
                "subject": ind_name,
                "property": "hasRival",
                "object": _get_label(r),
            })

        for k in getattr(ind, "knownOpponent", []):
            known_results.append({
                "subject": ind_name,
                "property": "knownOpponent",
                "object": _get_label(k),
            })

    print("\n[RULE 1 - Indirect Rival]")
    print("  Character(?x), defeated(?x,?y), defeated(?y,?z) -> hasRival(?x,?z)")
    if rival_results:
        for r in rival_results:
            print(f"  INFERRED: {r['subject']}  hasRival  {r['object']}")
    else:
        print("  (no triples inferred)")

    print("\n[RULE 2 - Known Opponent by Proxy]")
    print("  Character(?x), met(?x,?y), defeated(?y,?z) -> knownOpponent(?x,?z)")
    if known_results:
        for r in known_results:
            print(f"  INFERRED: {r['subject']}  knownOpponent  {r['object']}")
    else:
        print("  (no triples inferred)")

    return {
        "ontology_tbox": str(tbox_path),
        "ontology_abox": str(abox_path),
        "rule_indirect_rival": rival_results,
        "rule_known_opponent": known_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SWRL reasoning -- family.owl + Queens Gambit KB"
    )
    p.add_argument("--family_owl", type=Path, default=Path("data/family.owl"))
    p.add_argument("--abox", type=Path, default=Path("data/queens_gambit_graph_abox.ttl"))
    p.add_argument("--tbox", type=Path, default=Path("data/ontologie_tbox.ttl"))
    p.add_argument("--out", type=Path, default=None, help="Optional path to save results as JSON")
    p.add_argument("--skip_family", action="store_true", help="Skip Part 1 (family.owl)")
    p.add_argument("--skip_qg", action="store_true", help="Skip Part 2 (Queens Gambit KB)")
    return p.parse_args()


def main() -> None:
    try:
        _configure_java()
    except Exception as e:
        print(f"[WARN] Java configuration issue: {e}")
        print("       The reasoner may fail if Java is not reachable by OwlReady2.")

    args = parse_args()
    all_results: Dict = {}

    if not args.skip_family:
        if not args.family_owl.exists():
            print(f"[ERROR] family.owl not found at: {args.family_owl}")
            print("        Use --family_owl to specify the correct path.")
            print("        Or --skip_family to skip Part 1.")
        else:
            all_results["part1_family"] = run_family_rules(args.family_owl)

    if not args.skip_qg:
        missing = []
        if not args.abox.exists():
            missing.append(f"ABox: {args.abox}")
        if not args.tbox.exists():
            missing.append(f"TBox: {args.tbox}")

        if missing:
            print("[ERROR] Missing files for Part 2:")
            for m in missing:
                print(f"        {m}")
        else:
            all_results["part2_queens_gambit"] = run_qg_rules(args.abox, args.tbox)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)


    if "part2_queens_gambit" in all_results:
        p2 = all_results["part2_queens_gambit"]
        print(f"  QG KB      -- hasRival triples         : {len(p2['rule_indirect_rival'])}")
        print(f"  QG KB      -- knownOpponent triples    : {len(p2['rule_known_opponent'])}")

    if args.out and all_results:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  Results saved -> {args.out}")


if __name__ == "__main__":
    main()