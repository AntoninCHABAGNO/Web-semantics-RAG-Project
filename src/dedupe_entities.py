import pandas as pd
import difflib
import re

def is_single_token_name(name: str) -> bool:
    return len(name.split()) == 1

def is_probably_initialism(name: str) -> bool:
    # ex: "D.L." or "D.L"
    return bool(re.fullmatch(r"[A-Z]\.(?:[A-Z]\.)+", name.strip()))

def build_global_entity_table(ent_df: pd.DataFrame) -> pd.DataFrame:
    count_col = "mention_count" if "mention_count" in ent_df.columns else "count_in_page"
    g = ent_df.groupby(["entity_class", "canonical_text"], as_index=False).agg(
        total_count=(count_col, "sum"),
        urls=("url", lambda x: "|".join(sorted(set(x)))),
        aliases=("aliases", lambda x: "|".join(sorted(set("|".join(x).split("|"))))),
        entity_label=("entity_label", "first"),
        page_context=("page_context", lambda x: "REAL_WORLD" if "REAL_WORLD" in set(x) else "IN_UNIVERSE"),
    )
    return g


def merge_typos_with_similarity(group_df: pd.DataFrame, threshold: float = 0.94):
    """
    Merge names extremely close (e.g. Michaelangelo vs Michaelanglo).
    Works inside a class (Character/RealPerson/etc).
    """
    names = list(group_df["canonical_text"])
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # keep the one with higher count as representative
        ca = group_df.loc[group_df["canonical_text"] == ra, "total_count"].iloc[0]
        cb = group_df.loc[group_df["canonical_text"] == rb, "total_count"].iloc[0]
        parent[rb if ca >= cb else ra] = ra if ca >= cb else rb

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if abs(len(a) - len(b)) > 3:
                continue
            sim = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            if sim >= threshold:
                union(a, b)

    # mapping canonical -> representative
    rep = {n: find(n) for n in names}
    return rep

def merge_firstnames(group_df: pd.DataFrame, min_ratio: float = 8.0):
    """
    For single-token names (Beth, Benny, Alma),
    attach to the most frequent full name that starts with it:
      Beth -> Beth Harmon
    only if that full name is much more frequent (ratio).
    """
    rep = {n: n for n in group_df["canonical_text"].tolist()}
    counts = dict(zip(group_df["canonical_text"], group_df["total_count"]))

    full_names = [n for n in counts if len(n.split()) >= 2]
    for n in counts:
        if not is_single_token_name(n):
            continue
        if is_probably_initialism(n):
            continue

        candidates = [fn for fn in full_names if fn.lower().startswith(n.lower() + " ")]
        if not candidates:
            continue

        # best candidate = highest frequency
        best = max(candidates, key=lambda x: counts.get(x, 0))

        # only merge if strong evidence
        if counts.get(best, 0) >= counts[n] * min_ratio:
            rep[n] = best

    return rep

def apply_merges(global_df: pd.DataFrame, rep_maps: list[dict]) -> pd.DataFrame:
    # combine maps (in order)
    def resolve(x):
        for m in rep_maps:
            x = m.get(x, x)
        return x

    global_df["resolved_text"] = global_df["canonical_text"].apply(resolve)

    # aggregate to resolved_text
    out = global_df.groupby(["entity_class", "resolved_text"], as_index=False).agg(
        total_count=("total_count", "sum"),
        entity_label=("entity_label", "first"),
        page_context=("page_context", "first"),
        urls=("urls", lambda x: "|".join(sorted(set("|".join(x).split("|"))))),
        aliases=("aliases", lambda x: "|".join(sorted(set("|".join(x).split("|"))))),
    )
    out = out.rename(columns={"resolved_text": "canonical_text"})
    return out

def make_canonical_entity_id(entity_class: str, canonical_text: str) -> str:
    # stable id after dedupe
    import hashlib, re
    def slug(s):
        s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
        return s or "unknown"
    h = hashlib.sha1(f"{entity_class}:{canonical_text}".encode("utf-8")).hexdigest()[:6]
    return f"{entity_class}:{slug(canonical_text)}-{h}"

def dedupe_entities_csv(path_in="data/entities.csv", path_out="data/entities_dedup.csv", path_aliases="data/entity_aliases.csv"):
    ent_df = pd.read_csv(path_in)
    global_df = build_global_entity_table(ent_df)

    all_rows = []
    alias_rows = []

    for ent_class, g in global_df.groupby("entity_class"):
        g = g.copy()

        # Pass A: typos
        rep_typos = merge_typos_with_similarity(g, threshold=0.94)

        # Apply on-the-fly then recompute for firstname merge
        g["canonical_text_tmp"] = g["canonical_text"].map(rep_typos)
        g2 = g.groupby(["entity_class", "canonical_text_tmp"], as_index=False).agg(
            total_count=("total_count", "sum"),
            entity_label=("entity_label", "first"),
            page_context=("page_context", "first"),
            urls=("urls", "first"),
            aliases=("aliases", lambda x: "|".join(sorted(set("|".join(x).split("|"))))),
        ).rename(columns={"canonical_text_tmp": "canonical_text"})

        # Pass B: firstnames
        rep_first = merge_firstnames(g2, min_ratio=8.0)

        # Apply both
        merged = apply_merges(g, [rep_typos, rep_first])

        # Create final IDs + alias table
        merged["entity_id"] = merged.apply(lambda r: make_canonical_entity_id(r["entity_class"], r["canonical_text"]), axis=1)

        # Build alias mapping: every original canonical that resolves to canonical_text
        def resolve_full(x):
            x = rep_typos.get(x, x)
            x = rep_first.get(x, x)
            return x

        for orig in g["canonical_text"].tolist():
            canon = resolve_full(orig)
            final_id = merged.loc[merged["canonical_text"] == canon, "entity_id"].iloc[0]
            alias_rows.append({"alias": orig, "canonical_text": canon, "entity_id": final_id, "entity_class": ent_class})

        all_rows.append(merged)

    out_df = pd.concat(all_rows, ignore_index=True)
    out_df = out_df.sort_values(["entity_class", "total_count"], ascending=[True, False])
    out_df.to_csv(path_out, index=False)

    alias_df = pd.DataFrame(alias_rows).drop_duplicates()
    alias_df.to_csv(path_aliases, index=False)

    print(f"Wrote: {path_out}")
    print(f"Wrote: {path_aliases}")