# src/crawl_clean.py
import json
import time
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
import re
import httpx
import trafilatura
from bs4 import BeautifulSoup

@dataclass
class CrawlConfig:
    user_agent: str = "WebMiningSemanticsLab/1.0 (contact: your_email@example.com)"
    timeout: float = 20.0
    delay_s: float = 1.0
    min_words: int = 500
    max_pages: int = 50
    same_domain_only: bool = True
    max_depth: int = 3  # <-- NEW: limite de profondeur (optionnel mais recommandé)

def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()

def robots_allowed(url: str, client: httpx.Client, user_agent: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        r = client.get(robots_url, headers={"User-Agent": user_agent})
        rp = RobotFileParser()
        rp.parse(r.text.splitlines())
        return rp.can_fetch(user_agent, url)
    except Exception:
        return False

def extract_main_text(html: str):
    downloaded = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
        output_format="json"
    )
    if not downloaded:
        return None
    data = json.loads(downloaded)
    title = data.get("title")
    text = data.get("text") or ""
    date = data.get("date")
    return title, text, date

def is_useful(text: str, min_words: int) -> bool:
    return len(text.split()) >= min_words



def extract_awoiaf_links_important(html: str, base_url: str, limit: int | None = None):
    soup = BeautifulSoup(html, "html.parser")

    # Zone principale du contenu
    content = soup.select_one("#mw-content-text .mw-parser-output") or soup.select_one("#mw-content-text")
    if not content:
        return []

    # Supprime des blocs très "bruit" DANS le contenu
    for sel in [
        "#toc",                    # table des matières
        ".infobox",                # infobox à droite
        ".navbox",                 # navbox
        ".vertical-navbox",
        ".mw-references-wrap",     # références
        "ol.references",
        ".reflist",
        ".catlinks",               # catégories en bas
        ".printfooter",
        ".metadata",
    ]:
        for tag in content.select(sel):
            tag.decompose()

    links = []
    seen = set()

    # Parcours des <a> dans l’ordre du document (important!)
    for a in content.select("a[href]"):
        href = a.get("href", "")
        if not href.startswith("/index.php/"):
            continue

        # Filtrage MediaWiki
        if any(x in href for x in ["action=", "oldid=", "diff=", "printable="]):
            continue
        if any(href.startswith(prefix) for prefix in [
            "/index.php/Special:",
            "/index.php/Category:",
            "/index.php/File:",
            "/index.php/Template:",
            "/index.php/Talk:",
            "/index.php/User:",
            "/index.php/Help:",
        ]):
            continue

        # Évite ancres et fragments
        href = href.split("#", 1)[0]
        full = urljoin(base_url, href)

        # Dédup en conservant l’ordre
        if full not in seen:
            seen.add(full)
            links.append(full)

        if limit is not None and len(links) >= limit:
            break

    return links

def crawl_seeds_level_order(
    seeds,
    out_pages_jsonl_path: str,
    out_links_jsonl_path: str,
    cfg: CrawlConfig
):
    seen = set()
    seed_domains = {get_domain(s) for s in seeds}

    # frontier = URLs à visiter au niveau courant
    frontier = list(seeds)
    depth = 0

    with httpx.Client(timeout=cfg.timeout, follow_redirects=True) as client, \
         open(out_pages_jsonl_path, "w", encoding="utf-8") as f_pages, \
         open(out_links_jsonl_path, "w", encoding="utf-8") as f_links:

        while frontier and len(seen) < cfg.max_pages and depth <= cfg.max_depth:
            print(f"\n=== DEPTH {depth} | frontier size = {len(frontier)} ===")

            next_frontier = []
            # (optionnel) pour éviter beaucoup de doublons dans le prochain niveau
            next_frontier_set = set()

            for url in frontier:
                if len(seen) >= cfg.max_pages:
                    break

                if cfg.same_domain_only and get_domain(url) not in seed_domains:
                    continue
                if url in seen:
                    continue
                seen.add(url)

                if not robots_allowed(url, client, cfg.user_agent):
                    continue

                try:
                    resp = client.get(url, headers={"User-Agent": cfg.user_agent})
                    if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                        continue

                    # 1) EXTRAIRE + EXPORTER TOUS LES LIENS DE CETTE PAGE
                    if "awoiaf.westeros.org" in get_domain(url):
                        out_links = extract_awoiaf_links_important(resp.text, url, limit=80)
                    else:
                        out_links = []  # ici tu peux brancher un extracteur générique si tu veux

                    f_links.write(json.dumps({
                        "url": url,
                        "domain": get_domain(url),
                        "depth": depth,
                        "out_links": out_links
                    }, ensure_ascii=False) + "\n")

                    # 2) CONSTRUIRE LE NIVEAU SUIVANT (SANS Y ALLER TOUT DE SUITE)
                    for u in out_links:
                        if u in seen:
                            continue
                        if cfg.same_domain_only and get_domain(u) not in seed_domains:
                            continue
                        if u not in next_frontier_set:
                            next_frontier_set.add(u)
                            next_frontier.append(u)

                    # 3) EXTRAIRE TEXTE ET EXPORTER LA PAGE (comme avant)
                    extracted = extract_main_text(resp.text)
                    if extracted:
                        title, text, date = extracted
                        if is_useful(text, cfg.min_words):
                            f_pages.write(json.dumps({
                                "url": url,
                                "domain": get_domain(url),
                                "title": title,
                                "date": date,
                                "text": text,
                                "depth": depth
                            }, ensure_ascii=False) + "\n")

                    time.sleep(cfg.delay_s)

                except Exception:
                    continue

            # FIN du niveau courant -> on descend d'un niveau
            frontier = next_frontier
            depth += 1

if __name__ == "__main__":
    seeds = [
        "https://awoiaf.westeros.org/index.php/A_Game_of_Thrones"
        #"https://awoiaf.westeros.org/index.php/Westeros",
        #"https://awoiaf.westeros.org/index.php/A_Song_of_Ice_and_Fire",
        #"https://awoiaf.westeros.org/index.php/Daenerys_Targaryen",
        #"https://awoiaf.westeros.org/index.php/Arya_Stark"
    ]
    cfg = CrawlConfig(min_words=300, max_pages=20, delay_s=1.0, max_depth=1)
    crawl_seeds_level_order(
        seeds,
        out_pages_jsonl_path="data/raw_jsonl/pages.jsonl",
        out_links_jsonl_path="data/raw_jsonl/out_links.jsonl",
        cfg=cfg
    )