#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import requests

# réutilise ton pipeline existant
from F_19_rag_cli import (
    text_retrieve,
    rerank_text_results,
    kg_retrieve,
    build_evidence_pack,
    generate_answer,   # fallback template
    clean_text,
)


def build_llm_prompt(question: str, pack: Dict, short_mode: bool = False) -> str:
    text_evidence = pack.get("text_evidence", [])[:4]
    kg_evidence = pack.get("kg_evidence", [])[:4]

    text_blocks = []
    for i, e in enumerate(text_evidence, start=1):
        text_blocks.append(
            f"[S{i}]\n"
            f"Title: {e.get('title')}\n"
            f"URL: {e.get('url')}\n"
            f"Snippet: {clean_text(e.get('text', ''))[:400]}"
        )

    kg_blocks = []
    for i, e in enumerate(kg_evidence, start=1):
        kg_blocks.append(
            f"[K{i}]\n"
            f"Fact: {e.get('fact_text')}"
        )

    style = "Answer in 2 to 4 sentences." if short_mode else "Answer in 4 to 6 sentences."

    prompt = f"""
You are a knowledge-grounded assistant.

Use ONLY the evidence below.
Do NOT invent facts.
If the evidence is insufficient, say so clearly.
Keep the answer concise and precise.
Each important claim must cite a source tag like [S1], [S2], [K1].

Return exactly this format:

ANSWER:
<answer with citations>

QUESTION:
{question}

TEXT SOURCES:
{chr(10).join(text_blocks) if text_blocks else "None"}

KG SOURCES:
{chr(10).join(kg_blocks) if kg_blocks else "None"}

{style}
"""
    return prompt


def ollama_is_available(base_url: str = "http://localhost:11434") -> bool:
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ensure_ollama_model(model: str, base_url: str = "http://localhost:11434") -> None:
    """
    Vérifie si le modèle existe dans Ollama.
    Sinon, lève une erreur claire.
    """
    url = base_url.rstrip("/") + "/api/tags"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    installed = {m.get("name", "") for m in data.get("models", [])}
    if model not in installed:
        raise RuntimeError(
            f"Ollama is running but the model '{model}' is not installed.\n"
            f"Install it with: ollama pull {model}"
        )


def call_ollama_llm(
    prompt: str,
    model: str = "llama3.1:8b",
    temperature: float = 0.2,
    base_url: str = "http://localhost:11434",
) -> str:
    if not ollama_is_available(base_url):
        raise RuntimeError(
            "Ollama is not running on http://localhost:11434.\n"
            "Start Ollama, then try again."
        )

    ensure_ollama_model(model, base_url=base_url)

    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    text = data.get("response", "")
    if not text or not text.strip():
        return "No response returned by Ollama."
    return text.strip()


def print_simple(answer: str) -> None:
    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print()
    print(answer)
    print()

def print_sources(pack: Dict) -> None:
    print("\nSources:")
    
    seen = set()
    i = 1

    for e in pack.get("text_evidence", []):
        url = e.get("url")
        if not url:
            continue
        if url in seen:
            continue

        seen.add(url)
        print(f"{i}. {url}")
        i += 1

    if i == 1:
        print("No sources available.")


def print_debug(question: str, llm_answer: str, pack: Dict, prompt: Optional[str] = None) -> None:
    print("\n==============================")
    print("QUESTION")
    print("==============================")
    print(question)

    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print(llm_answer)

    print("\n==============================")
    print("TEXT EVIDENCE")
    print("==============================")
    for e in pack.get("text_evidence", [])[:5]:
        print(f"[{e['rank']}] score={e['score']:.4f} | {e['title']}")
        print(f"url: {e['url']}")
        print(f"text: {clean_text(e['text'])[:350]}")
        print("-" * 80)

    print("\n==============================")
    print("KG EVIDENCE")
    print("==============================")
    for e in pack.get("kg_evidence", [])[:8]:
        print(f"[{e['rank']}] score={e['score']:.4f}")
        print(e["fact_text"])
        print("-" * 80)

    if prompt is not None:
        print("\n==============================")
        print("PROMPT")
        print("==============================")
        print(prompt[:4000])
        if len(prompt) > 4000:
            print("\n...[prompt truncated]...")


def run_pipeline(
    question: str,
    chunks_path: Path,
    index_dir: Path,
    kg_path: Path,
) -> Dict:
    text_results = text_retrieve(question, index_dir=index_dir, chunks_path=chunks_path, top_k=8)
    text_results = rerank_text_results(question, text_results)
    kg_results = kg_retrieve(question, kg_path=kg_path, top_k_entities=5, top_k_facts=10)
    pack = build_evidence_pack(question, text_results, kg_results)
    return pack


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid KG+Text RAG assistant without paid API (Ollama local or template fallback)"
    )
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")

    parser.add_argument("--mode", choices=["simple", "debug"], default="simple")
    parser.add_argument("--answer_mode", choices=["short", "normal"], default="normal")
    parser.add_argument("--show_sources", action="store_true",help="Display sources used to generate the answer")
    
    parser.add_argument("--provider", choices=["ollama", "template"], default="ollama")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434")

    parser.add_argument("--chunks_path", type=Path, default=Path("data/rag/chunks.jsonl"))
    parser.add_argument("--index_dir", type=Path, default=Path("data/rag/index"))
    parser.add_argument("--kg", type=Path, default=Path("data/expanded_kb.ttl"))

    parser.add_argument("--save_pack", type=Path, default=None)
    parser.add_argument("--save_answer", type=Path, default=None)

    args = parser.parse_args()

    default_model = {
        "ollama": "llama3.1:8b",
        "template": "template",
    }[args.provider]
    model_name = args.model or default_model

    def answer_one(question: str) -> None:
        pack = run_pipeline(
            question=question,
            chunks_path=args.chunks_path,
            index_dir=args.index_dir,
            kg_path=args.kg,
        )

        short_mode = args.answer_mode == "short"

        if args.provider == "template":
            result = generate_answer(question, pack)
            final_answer = result["answer"]
            prompt = None
        else:
            prompt = build_llm_prompt(question, pack, short_mode=short_mode)
            try:
                final_answer = call_ollama_llm(
                    prompt=prompt,
                    model=model_name,
                    temperature=args.temperature,
                    base_url=args.ollama_url,
                )
            except Exception as e:
                print(f"[WARN] Local LLM call failed: {e}")
                print("[WARN] Falling back to template generation.")
                result = generate_answer(question, pack)
                final_answer = result["answer"]
                prompt = None

        if args.mode == "simple":
            print_simple(final_answer)

            if args.show_sources:
                print_sources(pack)

        else:
            print_debug(question, final_answer, pack, prompt=prompt)

            if args.show_sources:
                print_sources(pack)

        if args.save_pack is not None:
            args.save_pack.parent.mkdir(parents=True, exist_ok=True)
            with args.save_pack.open("w", encoding="utf-8") as f:
                json.dump(pack, f, ensure_ascii=False, indent=2)

        if args.save_answer is not None:
            args.save_answer.parent.mkdir(parents=True, exist_ok=True)
            with args.save_answer.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "question": question,
                        "provider": args.provider,
                        "model": model_name,
                        "answer": final_answer,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    if args.interactive:
        print("Hybrid RAG assistant (local LLM / no paid API)")
        print("Type your question, or 'quit' to exit.\n")
        while True:
            q = input("Question> ").strip()
            if not q:
                continue
            if q.lower() in {"quit", "exit", "q"}:
                break
            answer_one(q)
    else:
        if not args.question:
            raise ValueError("Use --question or --interactive")
        answer_one(args.question)


if __name__ == "__main__":
    main()