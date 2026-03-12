import difflib
import hashlib
import re

import pandas as pd


AMBIGUOUS_SINGLE_TOKENS = {
    "unknown", "white", "black", "life", "christian",
    "american", "russian", "french", "english", "german", "hungarian"
}

def is_single_token_name(name: str) -> bool:
    return len(str(name).split()) == 1


def is_probably_initialism(name: str) -> bool:
    return bool(re.fullmatch(r"(?:[A-Z]\.?){2,}", str(name).strip()))


def has_disambiguation_marker(name: str) -> bool:
    s = str(name).strip()
    return "(" in s or ")" in s or ":" in s


def normalize_text_for_match(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize_name(s: str) -> list[str]:
    s = normalize_text_for_match(s)
    parts = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", s)
    return [p.lower() for p in parts]


def build_global_entity_table(ent_df: pd.DataFrame) -> pd.DataFrame:
    count_col = "mention_count" if "mention_count" in ent_df.columns else "count_in_page"

    g = ent_df.groupby(["entity_class", "canonical_text"], as_index=False).agg(
        total_count=(count_col, "sum"),
        urls=("url", lambda x: "|".join(sorted(set(v for v in x.dropna().astype(str) if v.strip())))),
        aliases=("aliases", lambda x: "|".join(sorted(set(
            part.strip()
            for val in x.dropna().astype(str)
            for part in val.split("|")
            if part.strip()
        )))),
        entity_label=("entity_label", "first"),
        page_context=("page_context", lambda x: "REAL_WORLD" if "REAL_WORLD" in set(x) else "IN_UNIVERSE"),
    )
    return g


def merge_typos_with_similarity(group_df: pd.DataFrame, threshold: float = 0.94):
    names = list(group_df["canonical_text"])
    parent = {n: n for n in names}
    counts = dict(zip(group_df["canonical_text"], group_df["total_count"]))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return

        ca = counts.get(ra, 0)
        cb = counts.get(rb, 0)

        if ca > cb:
            parent[rb] = ra
        elif cb > ca:
            parent[ra] = rb
        else:
            rep = ra if len(tokenize_name(ra)) >= len(tokenize_name(rb)) else rb
            other = rb if rep == ra else ra
            parent[other] = rep

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]

            if has_disambiguation_marker(a) or has_disambiguation_marker(b):
                continue

            if is_single_token_name(a) and len(a) < 5:
                continue
            if is_single_token_name(b) and len(b) < 5:
                continue

            if abs(len(a) - len(b)) > 3:
                continue

            sim = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            if sim >= threshold:
                union(a, b)

    return {n: find(n) for n in names}


def choose_best_full_name(candidates: list[str], counts: dict[str, int]) -> str:
    return max(candidates, key=lambda x: (counts.get(x, 0), len(tokenize_name(x)), len(x)))


def merge_single_token_names(group_df: pd.DataFrame, min_ratio: float = 4.0):
    rep = {n: n for n in group_df["canonical_text"].tolist()}
    counts = dict(zip(group_df["canonical_text"], group_df["total_count"]))

    full_names = [n for n in counts if len(tokenize_name(n)) >= 2]

    for n in counts:
        if not is_single_token_name(n):
            continue
        if is_probably_initialism(n):
            continue
        if has_disambiguation_marker(n):
            continue

        n_tokens = tokenize_name(n)
        if len(n_tokens) != 1:
            continue
        token = n_tokens[0]
        
        if token in AMBIGUOUS_SINGLE_TOKENS:
            continue

        candidates = []
        for fn in full_names:
            fn_tokens = tokenize_name(fn)
            if not fn_tokens:
                continue
            if token == fn_tokens[0] or token == fn_tokens[-1]:
                candidates.append(fn)

        if not candidates:
            continue

        best = choose_best_full_name(candidates, counts)
        best_count = counts.get(best, 0)

        # Sur un wiki, on préfère le nom complet même s'il est moins fréquent.
        if best_count >= 2:
            rep[n] = best

    return rep


def merge_partial_full_names(group_df: pd.DataFrame, min_ratio: float = 3.0):
    rep = {n: n for n in group_df["canonical_text"].tolist()}
    counts = dict(zip(group_df["canonical_text"], group_df["total_count"]))

    full_names = [n for n in counts if len(tokenize_name(n)) >= 2]

    for n in counts:
        n_tokens = tokenize_name(n)
        if len(n_tokens) != 1:
            continue
        if is_probably_initialism(n) or has_disambiguation_marker(n):
            continue

        token = n_tokens[0]

        candidates = []
        for fn in full_names:
            fn_tokens = tokenize_name(fn)
            if token in {fn_tokens[0], fn_tokens[-1]}:
                candidates.append(fn)

        if not candidates:
            continue

        best = choose_best_full_name(candidates, counts)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (counts.get(x, 0), len(tokenize_name(x)), len(x)),
            reverse=True
        )

        best_count = counts.get(best, 0)
        second_count = counts.get(sorted_candidates[1], 0) if len(sorted_candidates) > 1 else 0
        cur_count = counts.get(n, 0)

        if best_count >= max(2, cur_count * min_ratio) and best_count >= max(1, second_count * 1.5):
            rep[n] = best

    return rep


def apply_merges(global_df: pd.DataFrame, rep_maps: list[dict]) -> pd.DataFrame:
    def resolve(x):
        prev = None
        cur = x
        while prev != cur:
            prev = cur
            for m in rep_maps:
                cur = m.get(cur, cur)
        return cur

    global_df = global_df.copy()
    global_df["resolved_text"] = global_df["canonical_text"].apply(resolve)

    out = global_df.groupby(["entity_class", "resolved_text"], as_index=False).agg(
        total_count=("total_count", "sum"),
        entity_label=("entity_label", "first"),
        page_context=("page_context", lambda x: "REAL_WORLD" if "REAL_WORLD" in set(x) else "IN_UNIVERSE"),
        urls=("urls", lambda x: "|".join(sorted(set(
            part.strip()
            for val in x.dropna().astype(str)
            for part in val.split("|")
            if part.strip()
        )))),
        aliases=("aliases", lambda x: "|".join(sorted(set(
            part.strip()
            for val in x.dropna().astype(str)
            for part in val.split("|")
            if part.strip()
        )))),
    )

    out = out.rename(columns={"resolved_text": "canonical_text"})
    return out


def make_canonical_entity_id(entity_class: str, canonical_text: str) -> str:
    def slug(s):
        s = re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
        return s or "unknown"

    h = hashlib.sha1(f"{entity_class}:{canonical_text}".encode("utf-8")).hexdigest()[:6]
    return f"{entity_class}:{slug(canonical_text)}-{h}"


def dedupe_entities_csv(
    path_in="data/entities.csv",
    path_out="data/entities_dedup.csv",
    path_aliases="data/entity_aliases.csv"
):
    ent_df = pd.read_csv(path_in)
    global_df = build_global_entity_table(ent_df)

    all_rows = []
    alias_rows = []

    for ent_class, g in global_df.groupby("entity_class"):
        g = g.copy()

        rep_typos = merge_typos_with_similarity(g, threshold=0.94)

        g["canonical_text_tmp"] = g["canonical_text"].map(rep_typos)
        g2 = g.groupby(["entity_class", "canonical_text_tmp"], as_index=False).agg(
            total_count=("total_count", "sum"),
            entity_label=("entity_label", "first"),
            page_context=("page_context", lambda x: "REAL_WORLD" if "REAL_WORLD" in set(x) else "IN_UNIVERSE"),
            urls=("urls", lambda x: "|".join(sorted(set(
                part.strip()
                for val in x.dropna().astype(str)
                for part in val.split("|")
                if part.strip()
            )))),
            aliases=("aliases", lambda x: "|".join(sorted(set(
                part.strip()
                for val in x.dropna().astype(str)
                for part in val.split("|")
                if part.strip()
            )))),
        ).rename(columns={"canonical_text_tmp": "canonical_text"})

        rep_single = merge_single_token_names(g2, min_ratio=4.0)

        g2["canonical_text_tmp2"] = g2["canonical_text"].map(rep_single)
        g3 = g2.groupby(["entity_class", "canonical_text_tmp2"], as_index=False).agg(
            total_count=("total_count", "sum"),
            entity_label=("entity_label", "first"),
            page_context=("page_context", lambda x: "REAL_WORLD" if "REAL_WORLD" in set(x) else "IN_UNIVERSE"),
            urls=("urls", lambda x: "|".join(sorted(set(
                part.strip()
                for val in x.dropna().astype(str)
                for part in val.split("|")
                if part.strip()
            )))),
            aliases=("aliases", lambda x: "|".join(sorted(set(
                part.strip()
                for val in x.dropna().astype(str)
                for part in val.split("|")
                if part.strip()
            )))),
        ).rename(columns={"canonical_text_tmp2": "canonical_text"})

        rep_partial = merge_partial_full_names(g3, min_ratio=3.0)

        merged = apply_merges(g, [rep_typos, rep_single, rep_partial])

        merged["entity_id"] = merged.apply(
            lambda r: make_canonical_entity_id(r["entity_class"], r["canonical_text"]),
            axis=1
        )

        def resolve_full(x):
            prev = None
            cur = x
            while prev != cur:
                prev = cur
                cur = rep_typos.get(cur, cur)
                cur = rep_single.get(cur, cur)
                cur = rep_partial.get(cur, cur)
            return cur

        canon_to_id = dict(zip(merged["canonical_text"], merged["entity_id"]))

        for orig in g["canonical_text"].tolist():
            canon = resolve_full(orig)
            final_id = canon_to_id[canon]
            alias_rows.append({
                "alias": orig,
                "canonical_text": canon,
                "entity_id": final_id,
                "entity_class": ent_class,
            })

        all_rows.append(merged)

    out_df = pd.concat(all_rows, ignore_index=True)
    out_df = out_df.sort_values(["entity_class", "total_count"], ascending=[True, False])
    out_df.to_csv(path_out, index=False)

    alias_df = pd.DataFrame(alias_rows).drop_duplicates()
    alias_df.to_csv(path_aliases, index=False)

    print(f"Wrote: {path_out}")
    print(f"Wrote: {path_aliases}")