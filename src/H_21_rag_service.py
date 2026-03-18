from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

from G_20_rag_cli_llm import (
    run_pipeline,
    build_llm_prompt,
    call_ollama_llm,
)
from F_19_rag_cli import generate_answer, clean_text


import re

def clean_llm_answer(text: str) -> str:
    text = re.sub(r"^\s*ANSWER\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bQUESTION\s*:.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()

def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def format_sources(pack: Dict) -> List[Dict]:
    sources = []
    seen_urls = set()

    text_rank = 1
    for e in pack.get("text_evidence", [])[:6]:
        url = (e.get("url") or "").strip()
        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        title = (e.get("title") or "Source sans titre").strip()
        snippet = clean_text(e.get("text", ""))[:220].strip()
        domain = _domain_from_url(url)
        score = float(e.get("score", 0.0))

        sources.append({
            "id": f"S{text_rank}",
            "type": "text",
            "title": title,
            "url": url,
            "domain": domain,
            "snippet": snippet,
            "score": round(score, 3),
        })
        text_rank += 1

    kg_rank = 1
    for e in pack.get("kg_evidence", [])[:4]:
        fact = (e.get("fact_text") or "").strip()
        if not fact:
            continue

        sources.append({
            "id": f"K{kg_rank}",
            "type": "kg",
            "title": "Knowledge Graph",
            "fact": fact,
            "score": round(float(e.get("score", 0.0)), 3),
        })
        kg_rank += 1

    return sources


def ask_rag(
    question: str,
    chunks_path: Path = Path("data/rag/chunks.jsonl"),
    index_dir: Path = Path("data/rag/index"),
    kg_path: Path = Path("data/expanded_kb.ttl"),
    provider: str = "ollama",
    model_name: str = "phi3:mini",
    temperature: float = 0.2,
    ollama_url: str = "http://localhost:11434",
    short_mode: bool = False,
) -> Dict:

    pack = run_pipeline(
        question=question,
        chunks_path=chunks_path,
        index_dir=index_dir,
        kg_path=kg_path,
    )

    if provider == "template":
        result = generate_answer(question, pack)
        final_answer = result["answer"]
    else:
        prompt = build_llm_prompt(question, pack, short_mode=short_mode)
        try:
            final_answer = call_ollama_llm(
                prompt=prompt,
                model=model_name,
                temperature=temperature,
                base_url=ollama_url,
            )
        except Exception:
            result = generate_answer(question, pack)
            final_answer = result["answer"]

    return {
        "question": question,
        "answer": final_answer,
        "sources": format_sources(pack),
        "raw_pack": pack,
    }