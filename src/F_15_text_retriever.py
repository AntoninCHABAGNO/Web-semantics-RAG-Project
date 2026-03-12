#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from sentence_transformers import SentenceTransformer


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def search_with_faiss(index, query_vec: np.ndarray, top_k: int):
    scores, indices = index.search(query_vec.astype(np.float32), top_k)
    return scores[0], indices[0]


def search_with_numpy(embeddings: np.ndarray, query_vec: np.ndarray, top_k: int):
    sims = embeddings @ query_vec[0]
    best_idx = np.argsort(-sims)[:top_k]
    best_scores = sims[best_idx]
    return best_scores, best_idx


def format_result(rank: int, score: float, meta: Dict, text: str) -> Dict:
    return {
        "rank": rank,
        "score": float(score),
        "chunk_id": meta.get("chunk_id"),
        "page_id": meta.get("page_id"),
        "chunk_index": meta.get("chunk_index"),
        "title": meta.get("title"),
        "url": meta.get("url"),
        "domain": meta.get("domain"),
        "date": meta.get("date"),
        "entities": meta.get("entities", []),
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve relevant RAG chunks from vector index")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="User question",
    )
    parser.add_argument(
        "--index_dir",
        type=Path,
        default=Path("data/rag/index"),
        help="Directory containing chunk embeddings, metadata, and FAISS index",
    )
    parser.add_argument(
        "--chunks_path",
        type=Path,
        default=Path("data/rag/chunks.jsonl"),
        help="Original chunks file to recover chunk texts",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Embedding model name; if omitted, use index_config.json",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to save retrieval results as JSON",
    )
    args = parser.parse_args()

    config_path = args.index_dir / "index_config.json"
    meta_path = args.index_dir / "chunk_metadata.json"
    emb_path = args.index_dir / "chunk_embeddings.npy"
    faiss_path = args.index_dir / "chunk_faiss.index"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {emb_path}")
    if not args.chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {args.chunks_path}")

    config = load_json(config_path)
    metadata: List[Dict] = load_json(meta_path)
    embeddings = np.load(emb_path)

    # recharge aussi les textes complets
    chunk_text_by_id: Dict[str, str] = {}
    with args.chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunk_text_by_id[str(obj["chunk_id"])] = str(obj.get("text", ""))

    model_name = args.model_name or config["model_name"]
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    query = args.question.strip()
    if not query:
        raise ValueError("Question is empty")

    print(f"Encoding query: {query}")
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    query_vec = l2_normalize(query_vec)

    if HAS_FAISS and faiss_path.exists():
        print("Using FAISS index")
        index = faiss.read_index(str(faiss_path))
        scores, indices = search_with_faiss(index, query_vec, args.top_k)
    else:
        print("FAISS unavailable or missing index file; using numpy fallback")
        embeddings = l2_normalize(embeddings.astype(np.float32))
        scores, indices = search_with_numpy(embeddings, query_vec, args.top_k)

    results: List[Dict] = []
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[int(idx)]
        chunk_id = str(meta.get("chunk_id"))
        text = chunk_text_by_id.get(chunk_id, "")
        results.append(format_result(rank, float(score), meta, text))

    print("\n=== TOP RETRIEVED CHUNKS ===\n")
    for r in results:
        print(f"[{r['rank']}] score={r['score']:.4f}")
        print(f"title: {r['title']}")
        print(f"url:   {r['url']}")
        if r.get("entities"):
            print(f"entities: {r['entities']}")
        print(f"text: {r['text'][:500]}")
        print("-" * 80)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "question": query,
                    "top_k": args.top_k,
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved results to: {args.output_json}")


if __name__ == "__main__":
    main()