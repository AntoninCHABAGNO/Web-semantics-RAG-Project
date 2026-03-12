#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pykeen.pipeline import pipeline
from sklearn.manifold import TSNE


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

TRAIN_PATH = DATA_DIR / "train.txt"
VALID_PATH = DATA_DIR / "valid.txt"
TEST_PATH = DATA_DIR / "test.txt"
ABOX_PATH = DATA_DIR / "queens_gambit_graph_abox.ttl"

MODEL_NAMES = ["TransE", "DistMult", "ComplEx"]

# Configuration commune pour une comparaison équitable
COMMON_CONFIG = {
    "embedding_dim": 200,
    "num_epochs": 200,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "random_seed": 42,
}

# Couleurs par classe locale
CLASS_COLORS = {
    "Character": "tab:blue",
    "RealPerson": "tab:orange",
    "Organization": "tab:green",
    "Location": "tab:red",
    "Event": "tab:purple",
    "Episode": "tab:pink",
    "Work": "tab:brown",
    "Book": "tab:gray",
}


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_train_entities_and_relations() -> Tuple[set[str], set[str]]:
    df = pd.read_csv(TRAIN_PATH, sep="\t", header=None, names=["h", "r", "t"])
    entities = set(df["h"]).union(set(df["t"]))
    relations = set(df["r"])
    return entities, relations


def load_entity_classes_from_abox(abox_path: Path) -> Dict[str, str]:
    """
    Lit grossièrement l'ABox TTL pour récupérer rdf:type par entité locale.
    On évite ici d'imposer rdflib si tu veux garder le script léger,
    mais rdflib peut aussi être utilisé si tu préfères.
    """
    try:
        from rdflib import Graph
        from rdflib.namespace import RDF

        g = Graph()
        g.parse(str(abox_path))

        entity_to_class: Dict[str, str] = {}
        for s, _, o in g.triples((None, RDF.type, None)):
            s_str = str(s)
            o_str = str(o)
            if "#" in o_str:
                cls = o_str.split("#", 1)[1]
            else:
                cls = o_str.rsplit("/", 1)[-1]
            if cls in CLASS_COLORS:
                entity_to_class[s_str] = cls
        return entity_to_class
    except Exception as e:
        print(f"[WARN] Impossible de charger les classes depuis l'ABox: {e}")
        return {}


def extract_main_metrics(metric_results) -> Dict[str, float]:
    mr = metric_results.to_dict()
    realistic = mr["both"]["realistic"]

    return {
        "MRR": float(realistic["inverse_harmonic_mean_rank"]),
        "Hits@1": float(realistic["hits_at_1"]),
        "Hits@3": float(realistic["hits_at_3"]),
        "Hits@10": float(realistic["hits_at_10"]),
        "MeanRank": float(realistic["arithmetic_mean_rank"]),
    }


def run_one_model(model_name: str):
    print(f"\n=== Training {model_name} ===")

    result = pipeline(
        training=str(TRAIN_PATH),
        validation=str(VALID_PATH),
        testing=str(TEST_PATH),
        model=model_name,
        random_seed=COMMON_CONFIG["random_seed"],
        model_kwargs=dict(
            embedding_dim=COMMON_CONFIG["embedding_dim"],
        ),
        training_kwargs=dict(
            num_epochs=COMMON_CONFIG["num_epochs"],
            batch_size=COMMON_CONFIG["batch_size"],
        ),
        optimizer_kwargs=dict(
            lr=COMMON_CONFIG["learning_rate"],
        ),
    )

    model_dir = MODELS_DIR / model_name.lower()
    result.save_to_directory(str(model_dir))

    metrics = extract_main_metrics(result.metric_results)

    metrics["model"] = model_name
    metrics["training_time_seconds"] = float(result.train_seconds or 0.0)
    metrics["evaluation_time_seconds"] = float(result.evaluate_seconds or 0.0)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return result, metrics


def save_metrics_table(all_metrics: List[Dict[str, float]]) -> None:
    df = pd.DataFrame(all_metrics)
    df = df[
        [
            "model",
            "MRR",
            "Hits@1",
            "Hits@3",
            "Hits@10",
            "MeanRank",
            "training_time_seconds",
            "evaluation_time_seconds",
        ]
    ]
    out_csv = RESULTS_DIR / "kge_metrics_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Metrics saved to {out_csv}")


def get_entity_embeddings_and_labels(result, entity_to_class: Dict[str, str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Retourne:
    - embeddings numpy
    - labels des entités
    - classes des entités
    """
    triples_factory = result.training
    id_to_entity = triples_factory.entity_id_to_label

    emb_tensor = result.model.entity_representations[0]()
    if isinstance(emb_tensor, tuple):
        emb_tensor = emb_tensor[0]
    emb = emb_tensor.detach().cpu().numpy()

    # ComplEx embeddings are complex numbers
    if np.iscomplexobj(emb):
        emb = np.concatenate([emb.real, emb.imag], axis=1)

    labels: List[str] = []
    classes: List[str] = []

    for i in range(len(id_to_entity)):
        ent = id_to_entity[i]
        labels.append(ent)
        classes.append(entity_to_class.get(ent, "Unknown"))

    return emb, labels, classes


def save_entity_embeddings_csv(
    model_name: str,
    embeddings: np.ndarray,
    labels: List[str],
    classes: List[str],
) -> None:
    dim = embeddings.shape[1]
    cols = [f"dim_{i}" for i in range(dim)]

    df = pd.DataFrame(embeddings, columns=cols)
    df.insert(0, "entity", labels)
    df.insert(1, "class", classes)

    out_csv = RESULTS_DIR / f"{model_name.lower()}_entity_embeddings.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Embeddings saved to {out_csv}")


def plot_tsne(
    model_name: str,
    embeddings: np.ndarray,
    labels: List[str],
    classes: List[str],
    max_points: int = 2500,
) -> None:
    """
    t-SNE peut être lent; on échantillonne si besoin.
    """
    n = len(labels)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_points, replace=False)
        embeddings = embeddings[idx]
        labels = [labels[i] for i in idx]
        classes = [classes[i] for i in idx]

    perplexity = min(30, max(5, len(labels) // 20))

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    unique_classes = sorted(set(classes))
    for cls in unique_classes:
        mask = [c == cls for c in classes]
        pts = coords[mask]
        color = CLASS_COLORS.get(cls, "black")
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            s=14,
            alpha=0.75,
            c=color,
            label=cls,
        )

    plt.title(f"{model_name} — t-SNE of entity embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()

    out_png = RESULTS_DIR / f"{model_name.lower()}_tsne.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] t-SNE plot saved to {out_png}")


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def save_nearest_neighbors(
    model_name: str,
    embeddings: np.ndarray,
    labels: List[str],
    targets: List[str],
    top_k: int = 10,
) -> None:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    rows = []

    emb_norm = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)

    for target in targets:
        if target not in label_to_idx:
            continue

        idx = label_to_idx[target]
        sims = emb_norm @ emb_norm[idx]
        best_idx = np.argsort(-sims)

        rank = 0
        for j in best_idx:
            if j == idx:
                continue
            rows.append(
                {
                    "target_entity": target,
                    "neighbor_rank": rank + 1,
                    "neighbor_entity": labels[j],
                    "cosine_similarity": float(sims[j]),
                }
            )
            rank += 1
            if rank >= top_k:
                break

    out_csv = RESULTS_DIR / f"{model_name.lower()}_nearest_neighbors.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Nearest neighbors saved to {out_csv}")


def pick_analysis_targets(labels: List[str]) -> List[str]:
    wanted_keywords = [
        "beth",
        "harmon",
        "anya",
        "netflix",
        "openings",
        "exchanges",
        "moscow",
        "paris",
        "walter",
        "frank",
    ]

    chosen = []
    for label in labels:
        low = label.lower()
        if any(k in low for k in wanted_keywords):
            chosen.append(label)

    # limite pour ne pas produire un CSV énorme
    return chosen[:12]


def main() -> None:
    ensure_dirs()

    entity_to_class = load_entity_classes_from_abox(ABOX_PATH)
    train_entities, train_relations = load_train_entities_and_relations()

    print(f"Train entities: {len(train_entities)}")
    print(f"Train relations: {len(train_relations)}")

    all_metrics: List[Dict[str, float]] = []

    for model_name in MODEL_NAMES:
        result, metrics = run_one_model(model_name)
        all_metrics.append(metrics)

        embeddings, labels, classes = get_entity_embeddings_and_labels(result, entity_to_class)

        save_entity_embeddings_csv(model_name, embeddings, labels, classes)
        plot_tsne(model_name, embeddings, labels, classes)

        targets = pick_analysis_targets(labels)
        save_nearest_neighbors(model_name, embeddings, labels, targets=targets, top_k=10)

    save_metrics_table(all_metrics)
    print("\n[DONE] Training, evaluation, and visualization completed for all models.")


if __name__ == "__main__":
    main()