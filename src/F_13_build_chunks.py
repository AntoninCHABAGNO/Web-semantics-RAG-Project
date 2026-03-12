#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Découpage simple en phrases.
    Suffisant pour un premier pipeline RAG.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    # Coupe après ., !, ? suivis d'un espace
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def chunk_by_sentences(
    text: str,
    chunk_size_words: int = 220,
    overlap_words: int = 50,
) -> List[str]:
    """
    Construit des chunks en agrégeant des phrases jusqu'à chunk_size_words.
    Puis applique un overlap approximatif en mots.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())

        if current_sentences and current_words + sent_words > chunk_size_words:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # overlap approximatif : on garde les derniers mots du chunk précédent
            if overlap_words > 0:
                previous_words = chunk_text.split()
                tail_words = previous_words[-overlap_words:] if len(previous_words) > overlap_words else previous_words
                overlap_text = " ".join(tail_words).strip()
                current_sentences = [overlap_text] if overlap_text else []
                current_words = len(overlap_text.split()) if overlap_text else 0
            else:
                current_sentences = []
                current_words = 0

        current_sentences.append(sent)
        current_words += sent_words

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)

    # nettoyage final
    cleaned_chunks = []
    for c in chunks:
        c = normalize_whitespace(c)
        if c:
            cleaned_chunks.append(c)

    return cleaned_chunks


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_page_id(obj: Dict, fallback_index: int) -> str:
    for key in ("page_id", "id", "url"):
        val = obj.get(key)
        if val:
            return str(val)
    return f"page_{fallback_index}"


def extract_entities_from_page(obj: Dict) -> List[str]:
    """
    Si tu as déjà une liste d'entités dans certaines pages, on peut les récupérer.
    Sinon retourne [].
    """
    entities = obj.get("entities")
    if isinstance(entities, list):
        return [str(e).strip() for e in entities if str(e).strip()]
    return []


def build_chunks(
    input_jsonl: Path,
    output_jsonl: Path,
    chunk_size_words: int,
    overlap_words: int,
    min_chunk_words: int,
) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_chunks = 0

    with output_jsonl.open("w", encoding="utf-8") as out:
        for i, page in enumerate(load_jsonl(input_jsonl)):
            total_pages += 1

            title = str(page.get("title", "") or "").strip()
            url = str(page.get("url", "") or "").strip()
            domain = str(page.get("domain", "") or "").strip()
            date = page.get("date")
            text = str(page.get("text", "") or "").strip()

            if not text:
                continue

            page_id = safe_page_id(page, i)
            page_entities = extract_entities_from_page(page)

            chunks = chunk_by_sentences(
                text=text,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
            )

            for j, chunk_text in enumerate(chunks):
                if len(chunk_text.split()) < min_chunk_words:
                    continue

                chunk_obj = {
                    "chunk_id": f"{page_id}::chunk_{j}",
                    "page_id": page_id,
                    "chunk_index": j,
                    "title": title,
                    "url": url,
                    "domain": domain,
                    "date": date,
                    "text": chunk_text,
                    "n_words": len(chunk_text.split()),
                    "entities": page_entities,
                }

                out.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Pages processed: {total_pages}")
    print(f"Chunks written: {total_chunks}")
    print(f"Output: {output_jsonl}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RAG chunks from cleaned pages.jsonl")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw_jsonl/pages.jsonl"),
        help="Input pages.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/rag/chunks.jsonl"),
        help="Output chunks.jsonl",
    )
    p.add_argument(
        "--chunk_size_words",
        type=int,
        default=220,
        help="Target chunk size in words",
    )
    p.add_argument(
        "--overlap_words",
        type=int,
        default=50,
        help="Overlap size in words between chunks",
    )
    p.add_argument(
        "--min_chunk_words",
        type=int,
        default=40,
        help="Discard chunks smaller than this",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_chunks(
        input_jsonl=args.input,
        output_jsonl=args.output,
        chunk_size_words=args.chunk_size_words,
        overlap_words=args.overlap_words,
        min_chunk_words=args.min_chunk_words,
    )


if __name__ == "__main__":
    main()