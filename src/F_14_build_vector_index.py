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


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def build_faiss_index(embeddings: np.ndarray):
    """
    Index FAISS pour similarité cosinus via inner product
    après normalisation L2.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector index for RAG chunks")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/rag/chunks.jsonl"),
        help="Input chunks.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/rag/index"),
        help="Directory to save embeddings, metadata, and vector index",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Embedding batch size",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    texts = [str(r.get("text", "")).strip() for r in rows]
    metadata = []

    for r in rows:
        metadata.append(
            {
                "chunk_id": r.get("chunk_id"),
                "page_id": r.get("page_id"),
                "chunk_index": r.get("chunk_index"),
                "title": r.get("title"),
                "url": r.get("url"),
                "domain": r.get("domain"),
                "date": r.get("date"),
                "n_words": r.get("n_words"),
                "entities": r.get("entities", []),
            }
        )

    print(f"Loaded chunks: {len(rows)}")
    print(f"Loading embedding model: {args.model_name}")

    model = SentenceTransformer(args.model_name)

    print("Encoding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = embeddings.astype(np.float32)
    embeddings = l2_normalize(embeddings)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = args.output_dir / "chunk_embeddings.npy"
    meta_path = args.output_dir / "chunk_metadata.json"
    config_path = args.output_dir / "index_config.json"

    np.save(emb_path, embeddings)
    save_json(meta_path, metadata)

    config = {
        "input_chunks": str(args.input),
        "n_chunks": len(rows),
        "embedding_dim": int(embeddings.shape[1]),
        "model_name": args.model_name,
        "faiss_enabled": HAS_FAISS,
    }

    if HAS_FAISS:
        print("Building FAISS index...")
        index = build_faiss_index(embeddings)
        faiss_path = args.output_dir / "chunk_faiss.index"
        faiss.write_index(index, str(faiss_path))
        config["faiss_index"] = str(faiss_path)
    else:
        print("FAISS not available: only embeddings + metadata will be saved.")
        config["faiss_index"] = None

    save_json(config_path, config)

    print(f"Embeddings saved to: {emb_path}")
    print(f"Metadata saved to:   {meta_path}")
    print(f"Config saved to:     {config_path}")
    if HAS_FAISS:
        print(f"FAISS index saved to: {config['faiss_index']}")


if __name__ == "__main__":
    main()