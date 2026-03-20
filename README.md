# ♟️ The Queen's Gambit — Knowledge Graph & RAG Pipeline

> End-to-end semantic web engineering project: web crawling · NLP extraction · OWL ontology · Wikidata alignment · Knowledge Graph Embeddings · Hybrid RAG with local LLM

**Academic Year 2024–2025 · Web Mining & Semantic Web**

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Pipeline — Step by Step](#pipeline--step-by-step)
  - [A. Web Crawling](#a-web-crawling)
  - [B. NLP Extraction](#b-nlp-extraction)
  - [C. Knowledge Graph Construction](#c-knowledge-graph-construction)
  - [D. Wikidata Alignment & Expansion](#d-wikidata-alignment--expansion)
  - [E. Knowledge Graph Embeddings](#e-knowledge-graph-embeddings)
  - [F–H. RAG Pipeline](#fh-rag-pipeline)
- [RAG Demo](#rag-demo)
- [Key Results](#key-results)
- [Data Files](#data-files)

---

## Overview

This project builds a complete knowledge engineering pipeline over the Netflix miniseries *The Queen's Gambit*. Starting from raw web content crawled from the Fandom wiki, the system:

1. Extracts entities and relations using spaCy (transformer-based NER + dependency parsing)
2. Constructs an OWL 2 knowledge graph (238 entities, 17 classes, 33 properties)
3. Aligns the private graph to Wikidata via `owl:sameAs` links and predicate mappings
4. Expands the KB through multi-hop Wikidata traversal (25,231 triples total)
5. Trains three KGE models (TransE, DistMult, ComplEx) using PyKEEN
6. Serves a hybrid RAG assistant combining FAISS dense retrieval, KG fact lookup, and a local LLM (Ollama) through a Streamlit chat interface

---

## Project Structure

```
project-root/
├── src/
│   ├── A_01_crawl_clean.py          # BFS crawler (httpx + trafilatura)
│   ├── B_02_nlp_extract.py          # NER + relation extraction (spaCy)
│   ├── dedupe_entities.py           # Entity deduplication & alias resolution
│   ├── C_03_clean_before_rdf.py     # Manual entity/relation cleanup rules
│   ├── C_04_csv_to_rdf.py           # CSV → RDF/Turtle (ABox-only)
│   ├── D_05_predicate_alignment.py  # OWL/RDFS predicate alignment to Wikidata
│   ├── D_06_entity_linking.py       # Type-aware entity linking → Wikidata
│   ├── D_07_filter_entity_linking.py# Confidence filtering of sameAs links
│   ├── D_08_expand_kb.py            # Multi-hop Wikidata BFS expansion
│   ├── E_09_prepare_triples.py      # Extract URI triples from expanded KB
│   ├── E_10_clean_for_embedding.py  # Filter literals, deduplicate
│   ├── E_11_split_dataset.py        # Train/valid/test split (80/10/10)
│   ├── E_12_train_eval_visualize_kge.py  # PyKEEN training + t-SNE
│   ├── F_13_build_chunks.py         # Sentence-aware chunking with overlap
│   ├── F_14_build_vector_index.py   # FAISS index + sentence-transformers
│   ├── F_15_text_retriever.py       # Dense text retrieval
│   ├── F_16_kg_retriever.py         # One-hop KG fact retrieval (RDFLib)
│   ├── F_17_hybrid_retriever_reranked.py  # Hybrid retrieval + reranking
│   ├── F_18_generate_answer.py      # Template-based answer generation
│   ├── F_19_rag_cli.py              # Full RAG CLI (template mode)
│   ├── G_20_rag_cli_llm.py          # RAG CLI with Ollama LLM + fallback
│   ├── H_21_rag_service.py          # RAG service layer (API-ready)
│   └── H_22_app_streamlit.py        # Streamlit chat UI
│
├── data/
│   ├── raw_jsonl/
│   │   ├── pages.jsonl              # Crawled pages (text + metadata)
│   │   └── out_links.jsonl          # Extracted hyperlinks
│   ├── entities.csv                 # Raw extracted entities
│   ├── entities_dedup.csv           # Deduplicated canonical entities
│   ├── entity_aliases.csv           # Alias → canonical mapping
│   ├── entities_clean.csv           # Manually cleaned entities
│   ├── relations.csv                # Extracted relations with evidence
│   ├── relations_clean.csv          # Cleaned and validated relations
│   ├── entity_wikidata_mapping.csv  # Private → Wikidata entity links
│   ├── queens_gambit_graph_abox.ttl # Private ABox (RDF/Turtle)
│   ├── entity_sameas.ttl            # owl:sameAs alignment graph
│   ├── predicate_alignment.ttl      # Predicate alignment (OWL/RDFS)
│   ├── expanded_kb.ttl              # Expanded KB (private + Wikidata)
│   ├── kg_triples.txt               # Raw URI triples
│   ├── kg_clean.txt                 # Deduplicated URI-only triples
│   ├── train.txt                    # KGE training set
│   ├── valid.txt                    # KGE validation set
│   ├── test.txt                     # KGE test set
│   └── rag/
│       ├── chunks.jsonl             # Sentence chunks with metadata
│       └── index/
│           ├── chunk_embeddings.npy
│           ├── chunk_metadata.json
│           ├── chunk_faiss.index
│           └── index_config.json
│
├── kg_artifacts/
│   ├── ontologie_tbox.ttl           # OWL 2 ontology (TBox)
│   ├── predicate_alignment.ttl
│   └── entity_sameas.ttl
│
├── models/                          # Saved PyKEEN models
│   ├── transe/
│   ├── distmult/
│   └── complex/
│
├── results/                         # KGE evaluation outputs
│   ├── kge_metrics_comparison.csv
│   ├── *_entity_embeddings.csv
│   ├── *_nearest_neighbors.csv
│   └── *_tsne.png
│
├── reports/
│   └── final_report.pdf
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Storage | 5 GB free | 10 GB free |
| CPU | 4 cores | 8 cores |
| GPU | — | Optional (speeds up spaCy + KGE) |
| OS | Linux / macOS / Windows | Linux / macOS |

> **Note on KGE training:** `E_12_train_eval_visualize_kge.py` trains 3 models for 200 epochs each on ~20k triples. Expect **5–15 minutes** on CPU (faster with CUDA). ComplEx in particular benefits from GPU.

> **Note on Ollama:** The LLM inference runs locally via Ollama. `phi3:mini` requires ~2.5 GB of RAM. `llama3.1:8b` requires ~6 GB.

---

## Installation

### 1. Clone and create environment

```bash
git clone <your-repo-url>
cd <repo-root>

conda create -n semantic-web-data-mining-project python=3.12
conda activate semantic-web-data-mining-project

pip install -r requirements.txt
```

### 2. Download the spaCy model

```bash
python -m spacy download en_core_web_trf
```

### 3. Install and configure Ollama

Download Ollama from [https://ollama.com](https://ollama.com), then pull the model:

```bash
# Lightweight model (~2.5 GB) — used by default in Streamlit
ollama pull phi3:mini

# Alternatively, the larger model used in CLI examples
ollama pull llama3.1:8b
```

Start the Ollama server (runs in the background):

```bash
ollama serve
```

> The RAG pipeline includes an automatic fallback to template-based generation if Ollama is unavailable.

---

## Pipeline — Step by Step

Each module can be run independently. Run them in the order below for a full pipeline execution.

### A. Web Crawling

Crawls the Queen's Gambit Fandom wiki using a BFS crawler. Respects `robots.txt`, uses a 1-second politeness delay, and extracts text with trafilatura.

```bash
python src/A_01_crawl_clean.py
```

**Output:** `data/raw_jsonl/pages.jsonl`, `data/raw_jsonl/out_links.jsonl`

**Config (in script):** `max_pages=50`, `max_depth=2`, `min_words=200`

---

### B. NLP Extraction

Runs NER (spaCy `en_core_web_trf`) and dependency-based relation extraction. Produces canonical entities with deduplication and alias resolution.

```bash
python src/B_02_nlp_extract.py --report
```

**Output:** `data/entities_dedup.csv`, `data/relations.csv`, `data/entity_aliases.csv`

The `--report` flag prints entity distribution and relation examples to stdout.

---

### C. Knowledge Graph Construction

**Step 1 — Manual cleanup:** applies correction rules (entity drops, canonical name fixes, relation constraints).

```bash
python src/C_03_clean_before_rdf.py
```

**Step 2 — CSV → RDF/Turtle:** converts cleaned entities and relations to an ABox TTL file.

```bash
python src/C_04_csv_to_rdf.py \
  --entities data/entities_clean.csv \
  --relations data/relations_clean.csv \
  --out data/queens_gambit_graph_abox.ttl \
  --ontology-import kg_artifacts/ontologie_tbox.ttl
```

**Output:** `data/queens_gambit_graph_abox.ttl`

---

### D. Wikidata Alignment & Expansion

**Step 1 — Predicate alignment:** maps private predicates to Wikidata properties via `owl:equivalentProperty` / `rdfs:subPropertyOf`.

```bash
python src/D_05_predicate_alignment.py --mode emit --out data/predicate_alignment.ttl
```

**Step 2 — Entity linking:** links private entities to Wikidata QIDs using type-aware scoring and manual overrides.

```bash
python src/D_06_entity_linking.py \
  --abox data/queens_gambit_graph_abox.ttl \
  --out_csv data/entity_wikidata_mapping.csv \
  --out_ttl data/entity_sameas.ttl \
  --min_conf 0.70
```

**Step 3 — Filter links:**

```bash
python src/D_07_filter_entity_linking.py
```

**Step 4 — KB expansion:** multi-hop BFS over Wikidata EntityData API, guided by a property whitelist.

```bash
python src/D_08_expand_kb.py
```

**Output:** `data/expanded_kb.ttl` (~25,000 triples)

---

### E. Knowledge Graph Embeddings

**Prepare triples:**

```bash
python src/E_09_prepare_triples.py   # extract URI triples
python src/E_10_clean_for_embedding.py  # remove literals, deduplicate
python src/E_11_split_dataset.py     # 80/10/10 train/valid/test split
```

**Train and evaluate (TransE, DistMult, ComplEx):**

```bash
python src/E_12_train_eval_visualize_kge.py
```

**Output:** `results/kge_metrics_comparison.csv`, t-SNE plots, nearest neighbor CSVs.

| Model | MRR | Hits@10 | Training time |
|-------|-----|---------|---------------|
| TransE | 0.1387 | 0.3314 | ~86s |
| **DistMult** | **0.1986** | **0.4124** | ~102s |
| ComplEx | 0.0074 | 0.0090 | ~173s |

---

### F–H. RAG Pipeline

#### Build the text index

```bash
# Chunk pages into overlapping text segments
python src/F_13_build_chunks.py \
  --input data/raw_jsonl/pages.jsonl \
  --output data/rag/chunks.jsonl \
  --chunk_size_words 220 \
  --overlap_words 50

# Build FAISS vector index (sentence-transformers/all-MiniLM-L6-v2)
python src/F_14_build_vector_index.py \
  --input data/rag/chunks.jsonl \
  --output_dir data/rag/index
```

#### Run individual retrievers (optional, for debugging)

```bash
# Dense text retrieval only
python src/F_15_text_retriever.py \
  --question "Who is Beth Harmon?" \
  --top_k 5

# KG fact retrieval only
python src/F_16_kg_retriever.py \
  --question "Who is Beth Harmon?" \
  --kg data/expanded_kb.ttl
```

#### Run the full hybrid pipeline

```bash
# Hybrid retrieval + reranking → evidence pack
python src/F_17_hybrid_retriever_reranked.py \
  --question "Who is Beth Harmon?" \
  --output_json data/rag/evidence_beth.json

# Template-based answer (no LLM required)
python src/F_19_rag_cli.py \
  --question "Who is Beth Harmon?" \
  --mode debug
```

---

## RAG Demo

### CLI with local LLM (Ollama)

Make sure Ollama is running (`ollama serve`), then:

```bash
# Single question
python src/G_20_rag_cli_llm.py \
  --question "Who is Beth Harmon?" \
  --provider ollama \
  --model phi3:mini \
  --mode simple \
  --show_sources

# Interactive mode
python src/G_20_rag_cli_llm.py \
  --interactive \
  --provider ollama \
  --model phi3:mini

# Template fallback (no Ollama needed)
python src/G_20_rag_cli_llm.py \
  --question "Who is Beth Harmon?" \
  --provider template
```

**Example questions:**

```bash
python src/G_20_rag_cli_llm.py --question "What is the Queen's Gambit?" --provider ollama --model phi3:mini --mode simple
python src/G_20_rag_cli_llm.py --question "What is happening in the episode Doubled Pawns?" --provider ollama --model phi3:mini --mode simple
python src/G_20_rag_cli_llm.py --question "Who is Alice Harmon?" --provider ollama --model phi3:mini --mode debug
python src/G_20_rag_cli_llm.py --question "Who trained Beth Harmon?" --provider ollama --model phi3:mini --mode simple
python src/G_20_rag_cli_llm.py --question "What chess tournaments did Beth Harmon enter?" --provider ollama --model phi3:mini --mode simple
```

### Streamlit Web UI

```bash
streamlit run src/H_22_app_streamlit.py
```

Opens at `http://localhost:8501`. The chat interface shows the LLM answer alongside expandable source cards (text passages with scores and URLs, and KG fact triples).

> If Ollama is not available, the app automatically falls back to template-based generation.

---

## Key Results

### Knowledge Graph

| Metric | Value |
|--------|-------|
| Pages crawled | 50 (depth 2) |
| Entities (canonical) | 238 |
| OWL classes | 17 |
| Object properties | 28 |
| Relation instances | 21 |
| Wikidata entity links | ~40 (conf ≥ 0.70) |
| Predicate alignments | 12 (3 equivalent + 9 subproperty) |
| Total triples after expansion | 25,231 |
| Unique entities after expansion | 9,218 |

### KGE Evaluation

Best model: **DistMult** (MRR = 0.1986, Hits@10 = 0.4124)

### RAG Architecture

| Component | Technology |
|-----------|-----------|
| Chunking | Sentence-aware, 220 words, 50-word overlap |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector index | FAISS `IndexFlatIP` (cosine via L2-norm) |
| KG retrieval | RDFLib one-hop traversal + label matching |
| Reranking | Question-type-aware score fusion |
| LLM | Ollama `phi3:mini` (local, no API key) |
| Fallback | Template-based generation (always available) |
| UI | Streamlit chat with source attribution |

---

## Data Files

Large generated files are not committed to the repository. To reproduce them, run the pipeline from step A. The following files need to be generated locally:

| File | Generated by | Size (approx.) |
|------|-------------|----------------|
| `data/raw_jsonl/pages.jsonl` | A_01 | ~5 MB |
| `data/expanded_kb.ttl` | D_08 | ~15 MB |
| `data/rag/index/chunk_faiss.index` | F_14 | ~1 MB |
| `models/` | E_12 | ~200 MB |

A sample of 5 pages is included in `data/samples/pages_sample.jsonl` to allow testing the RAG pipeline without re-running the full crawl.
