#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""
Z_rag_evaluation.py — Evaluate RAG pipeline on 5 questions (baseline vs RAG)
 
Usage:
    # Full evaluation (requires Ollama running)
    python src/Z_rag_evaluation.py
 
    # Template-only (no Ollama needed)
    python src/Z_rag_evaluation.py --provider template
 
    # Save results to JSON
    python src/Z_rag_evaluation.py --out data/rag/evaluation_results.json
 
Output:
    - Console: formatted comparison table
    - JSON: full results for report inclusion
"""
 
from __future__ import annotations
 
import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, List
 
# Add src/ to path so imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))
 
from F_19_rag_cli import generate_answer, clean_text
from G_20_rag_cli_llm import (
    run_pipeline,
    build_llm_prompt,
    call_ollama_llm,
    ollama_is_available,
)
 
# ---------------------------------------------------------------------------
# The 5 evaluation questions
# Each has a "reference" — the known correct answer used for qualitative scoring
# ---------------------------------------------------------------------------
EVAL_QUESTIONS = [
    {
        "id": "Q1",
        "question": "Who is Beth Harmon?",
        "category": "Character identity",
        "reference": (
            "Beth Harmon is the main fictional character of The Queen's Gambit. "
            "She is an orphaned chess prodigy who becomes a world-class player. "
            "She is portrayed by Anya Taylor-Joy in the Netflix miniseries."
        ),
    },
    {
        "id": "Q2",
        "question": "Who is William Shaibel?",
        "category": "Character identity",
        "reference": (
            "William Shaibel is the janitor at the Methuen Home orphanage who teaches "
            "Beth Harmon to play chess. He is her first mentor and opponent."
        ),
    },
    {
        "id": "Q3",
        "question": "What chess tournaments did Beth Harmon win?",
        "category": "Factual / KG",
        "reference": (
            "Beth Harmon won the Kentucky State Championship, the Cincinnati tournament, "
            "the Pittsburgh tournament, and the Moscow Invitational."
        ),
    },
    {
        "id": "Q4",
        "question": "What is happening in the episode Doubled Pawns?",
        "category": "Episode summary",
        "reference": (
            "Doubled Pawns is the third episode of The Queen's Gambit. "
            "Beth enters the US Open and faces stronger opponents. "
            "She struggles with her dependency on tranquilisers."
        ),
    },
    {
        "id": "Q5",
        "question": "Who portrays Beth Harmon in the Netflix series?",
        "category": "Real-world / casting",
        "reference": (
            "Beth Harmon is portrayed by Anya Taylor-Joy in the Netflix miniseries "
            "The Queen's Gambit."
        ),
    },
]
 
# ---------------------------------------------------------------------------
# Scoring rubric (qualitative, 0-3)
# ---------------------------------------------------------------------------
SCORE_LABELS = {
    0: "❌ Wrong / hallucinated",
    1: "⚠️  Partial / vague",
    2: "✅ Mostly correct",
    3: "✅✅ Complete & precise",
}
 
 
# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
 
def get_template_answer(
    question: str,
    chunks_path: Path,
    index_dir: Path,
    kg_path: Path,
) -> Dict:
    """Baseline: template generation, no LLM."""
    pack = run_pipeline(
        question=question,
        chunks_path=chunks_path,
        index_dir=index_dir,
        kg_path=kg_path,
    )
    result = generate_answer(question, pack)
    return {
        "answer": result["answer"],
        "n_text_evidence": len(pack.get("text_evidence", [])),
        "n_kg_evidence": len(pack.get("kg_evidence", [])),
        "matched_entities": [e["label"] for e in pack.get("matched_entities", [])],
    }
 
 
def get_rag_answer(
    question: str,
    chunks_path: Path,
    index_dir: Path,
    kg_path: Path,
    model: str,
    ollama_url: str,
    temperature: float,
) -> Dict:
    """Full RAG: Ollama LLM with KG+text evidence."""
    pack = run_pipeline(
        question=question,
        chunks_path=chunks_path,
        index_dir=index_dir,
        kg_path=kg_path,
    )
    prompt = build_llm_prompt(question, pack, short_mode=False)
    try:
        answer = call_ollama_llm(
            prompt=prompt,
            model=model,
            temperature=temperature,
            base_url=ollama_url,
        )
    except Exception as e:
        answer = f"[LLM unavailable: {e}]"
 
    return {
        "answer": answer,
        "n_text_evidence": len(pack.get("text_evidence", [])),
        "n_kg_evidence": len(pack.get("kg_evidence", [])),
        "matched_entities": [e["label"] for e in pack.get("matched_entities", [])],
    }
 
 
# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
 
def wrap(text: str, width: int = 80, indent: str = "    ") -> str:
    lines = textwrap.wrap(text, width=width - len(indent))
    return ("\n" + indent).join(lines)
 
 
def print_comparison_table(results: List[Dict]) -> None:
    SEP = "─" * 90
 
    print("\n")
    print("╔" + "═" * 88 + "╗")
    print("║" + "  RAG EVALUATION — Baseline (template) vs RAG (Ollama)".center(88) + "║")
    print("╚" + "═" * 88 + "╝")
 
    for r in results:
        q    = r["question_obj"]
        base = r["baseline"]
        rag  = r["rag"]
 
        print(f"\n{SEP}")
        print(f"  {q['id']} — {q['category']}")
        print(f"  Question : {q['question']}")
        print(SEP)
 
        print(f"\n  📖 REFERENCE ANSWER")
        print(f"    {wrap(q['reference'])}")
 
        print(f"\n  🔵 BASELINE (template, no LLM)")
        print(f"    Entities matched : {base['matched_entities']}")
        print(f"    KG facts used    : {base['n_kg_evidence']}  |  "
              f"Text chunks used : {base['n_text_evidence']}")
        print(f"    Answer :")
        print(f"    {wrap(base['answer'])}")
 
        print(f"\n  🟢 RAG (Ollama {r['model']})")
        print(f"    Entities matched : {rag['matched_entities']}")
        print(f"    KG facts used    : {rag['n_kg_evidence']}  |  "
              f"Text chunks used : {rag['n_text_evidence']}")
        print(f"    Answer :")
        print(f"    {wrap(rag['answer'])}")
 
        # Auto-score heuristic (keyword overlap with reference)
        ref_words = set(q["reference"].lower().split())
 
        def score(answer: str) -> int:
            ans_words = set(answer.lower().split())
            overlap = len(ref_words & ans_words) / max(len(ref_words), 1)
            if overlap > 0.35:
                return 3
            if overlap > 0.20:
                return 2
            if overlap > 0.08:
                return 1
            return 0
 
        sb = score(base["answer"])
        sr = score(rag["answer"])
 
        print(f"\n  Baseline score : {sb}/3 — {SCORE_LABELS[sb]}")
        print(f"  RAG score      : {sr}/3 — {SCORE_LABELS[sr]}")
 
    print(f"\n{SEP}")
 
 
def print_summary_table(results: List[Dict]) -> None:
    """Print the compact markdown-ready table for the report."""
 
    ref_words_list = [
        set(r["question_obj"]["reference"].lower().split())
        for r in results
    ]
 
    def score(answer: str, ref_words: set) -> int:
        ans_words = set(answer.lower().split())
        overlap = len(ref_words & ans_words) / max(len(ref_words), 1)
        if overlap > 0.35:
            return 3
        if overlap > 0.20:
            return 2
        if overlap > 0.08:
            return 1
        return 0
 
    print("\n")
    print("SUMMARY TABLE (copy into report)")
    print("─" * 90)
    print(f"{'ID':<4} {'Category':<24} {'Question':<38} {'Baseline':>9} {'RAG':>5}")
    print("─" * 90)
 
    total_base = 0
    total_rag  = 0
 
    for r, rw in zip(results, ref_words_list):
        q  = r["question_obj"]
        sb = score(r["baseline"]["answer"], rw)
        sr = score(r["rag"]["answer"], rw)
        total_base += sb
        total_rag  += sr
 
        q_short = q["question"][:36] + ".." if len(q["question"]) > 36 else q["question"]
        print(f"{q['id']:<4} {q['category']:<24} {q_short:<38} {sb}/3     {sr}/3")
 
    print("─" * 90)
    print(f"{'TOTAL':<66} {total_base}/{len(results)*3}    {total_rag}/{len(results)*3}")
    print()
 
    print("Scoring rubric:")
    for k, v in SCORE_LABELS.items():
        print(f"  {k}/3 = {v}")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline (baseline vs Ollama)")
    parser.add_argument(
        "--provider",
        choices=["ollama", "template"],
        default="ollama",
        help="LLM provider for the RAG column (default: ollama)",
    )
    parser.add_argument("--model",        type=str,   default="phi3:mini")
    parser.add_argument("--temperature",  type=float, default=0.2)
    parser.add_argument("--ollama_url",   type=str,   default="http://localhost:11434")
    parser.add_argument("--chunks_path",  type=Path,  default=Path("data/rag/chunks.jsonl"))
    parser.add_argument("--index_dir",    type=Path,  default=Path("data/rag/index"))
    parser.add_argument("--kg",           type=Path,  default=Path("data/expanded_kb.ttl"))
    parser.add_argument("--out",          type=Path,  default=None,
                        help="Optional path to save full results as JSON")
    args = parser.parse_args()
 
    # Check Ollama if needed
    if args.provider == "ollama":
        if not ollama_is_available(args.ollama_url):
            print(f"[ERROR] Ollama not running at {args.ollama_url}.")
            print("        Start it with: ollama serve")
            print("        Or use --provider template to skip the LLM.")
            sys.exit(1)
        print(f"Ollama available at {args.ollama_url} — model: {args.model}")
    else:
        print("Provider: template (no LLM, both columns use template generation)")
 
    # Run evaluation
    all_results = []
 
    for i, q_obj in enumerate(EVAL_QUESTIONS, start=1):
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] {q_obj['id']} — {q_obj['question']}")
 
        print("  → baseline (template)...")
        base = get_template_answer(
            question=q_obj["question"],
            chunks_path=args.chunks_path,
            index_dir=args.index_dir,
            kg_path=args.kg,
        )
 
        print(f"  → RAG ({args.provider})...")
        if args.provider == "ollama":
            rag = get_rag_answer(
                question=q_obj["question"],
                chunks_path=args.chunks_path,
                index_dir=args.index_dir,
                kg_path=args.kg,
                model=args.model,
                ollama_url=args.ollama_url,
                temperature=args.temperature,
            )
        else:
            # Both columns use template — shows retrieval quality difference
            rag = get_template_answer(
                question=q_obj["question"],
                chunks_path=args.chunks_path,
                index_dir=args.index_dir,
                kg_path=args.kg,
            )
 
        all_results.append({
            "question_obj": q_obj,
            "baseline": base,
            "rag": rag,
            "model": args.model if args.provider == "ollama" else "template",
        })
 
    # Display results
    print_comparison_table(all_results)
    print_summary_table(all_results)
 
    # Save JSON
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Make serialisable (remove non-JSON objects)
        clean = []
        for r in all_results:
            clean.append({
                "id":       r["question_obj"]["id"],
                "category": r["question_obj"]["category"],
                "question": r["question_obj"]["question"],
                "reference":r["question_obj"]["reference"],
                "baseline": r["baseline"],
                "rag":      r["rag"],
                "model":    r["model"],
            })
        args.out.write_text(
            json.dumps(clean, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nFull results saved → {args.out}")
 
 
if __name__ == "__main__":
    main()