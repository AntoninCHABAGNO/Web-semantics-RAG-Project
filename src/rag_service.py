from pathlib import Path
from typing import Dict, List

from G_20_rag_cli_llm import (
    run_pipeline,
    build_llm_prompt,
    call_ollama_llm,
)
from F_19_rag_cli import generate_answer, clean_text


def format_sources(pack: Dict) -> List[Dict]:
    sources = []
    seen_urls = set()

    for i, e in enumerate(pack.get("text_evidence", [])[:4], start=1):
        url = e.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        sources.append({
            "id": f"S{i}",
            "type": "text",
            "title": e.get("title", "Untitled"),
            "url": url,
            "snippet": clean_text(e.get("text", ""))[:250]
        })

    for i, e in enumerate(pack.get("kg_evidence", [])[:4], start=1):
        sources.append({
            "id": f"K{i}",
            "type": "kg",
            "fact": e.get("fact_text", ""),
            "score": e.get("score", 0.0)
        })

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