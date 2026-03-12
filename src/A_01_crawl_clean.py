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

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Upgrade-Insecure-Requests": "1",
}

def robots_allowed(url: str, client: httpx.Client, user_agent: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        r = client.get(robots_url)
        rp = RobotFileParser()
        rp.parse(r.text.splitlines())
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True
    
    
def extract_main_text(html: str):
    downloaded = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
        output_format="json"
    )

    title = None
    text = ""
    date = None

    if downloaded:
        data = json.loads(downloaded)
        title = data.get("title")
        text = data.get("text") or ""
        date = data.get("date")

    # Fallback titre si trafilatura ne l'a pas trouvé
    if not title:
        soup = BeautifulSoup(html, "html.parser")

        # Fandom / MediaWiki
        h1 = soup.select_one("#firstHeading")
        if h1:
            title = h1.get_text(" ", strip=True)

        # Fallback générique
        if not title:
            tag = soup.find("title")
            if tag:
                title = tag.get_text(" ", strip=True)

    if not text:
        return None

    return title, text, date

def is_useful(text: str, min_words: int) -> bool:
    return len(text.split()) >= min_words



def extract_fandom_links_important(html: str, base_url: str, limit: int | None = 80):
    soup = BeautifulSoup(html, "html.parser")

    # Contenu principal MediaWiki/Fandom
    content = soup.select_one("#mw-content-text .mw-parser-output") or soup.select_one("#mw-content-text")
    if not content:
        return []

    # Enlève les zones "bruit" fréquentes sur Fandom
    for sel in [
        "#toc",                    # table des matières
        ".portable-infobox",       # infobox Fandom (PortableInfobox) :contentReference[oaicite:1]{index=1}
        ".infobox",
        ".navbox",
        ".mw-references-wrap",
        "ol.references",
        ".reflist",
        ".catlinks",
        ".printfooter",
        ".metadata",
    ]:
        for tag in content.select(sel):
            tag.decompose()

    links = []
    seen = set()
    base_domain = urlparse(base_url).netloc.lower()

    # Parcours dans l’ordre du DOM => ordre stable
    for a in content.select("a[href]"):
        href = a.get("href", "")

        # On garde les pages wiki internes: /wiki/...
        if not href.startswith("/wiki/"):
            continue

        # Ignore ancres et pages spéciales
        href_no_frag = href.split("#", 1)[0]
        if any(href_no_frag.startswith(p) for p in [
            "/wiki/Special:",
            "/wiki/Category:",
            "/wiki/File:",
            "/wiki/Template:",
            "/wiki/Talk:",
            "/wiki/User:",
            "/wiki/Help:",
        ]):
            continue

        full = urljoin(base_url, href_no_frag)

        # Sécurité: rester sur le même domaine
        if urlparse(full).netloc.lower() != base_domain:
            continue

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

    with httpx.Client(timeout=cfg.timeout, follow_redirects=True,headers=BROWSER_HEADERS) as client, \
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

                print(f"\n[URL] {url}")

                if cfg.same_domain_only and get_domain(url) not in seed_domains:
                    print("  -> skipped: not in seed domains")
                    continue
                if url in seen:
                    print("  -> skipped: already seen")
                    continue
                seen.add(url)

                allowed = robots_allowed(url, client, cfg.user_agent)
                print(f"  -> robots allowed: {allowed}")
                if not allowed:
                    continue

                try:
                    resp = client.get(url, headers={"User-Agent": cfg.user_agent})
                    print(f"  -> status: {resp.status_code}")
                    print(f"  -> content-type: {resp.headers.get('content-type')}")

                    if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                        print("  -> skipped: not a valid HTML page")
                        continue

                    if "the-queens-gambit.fandom.com" in get_domain(url):
                        out_links = extract_fandom_links_important(resp.text, url, limit=80)
                    else:
                        out_links = []

                    print(f"  -> extracted links: {len(out_links)}")

                    f_links.write(json.dumps({
                        "url": url,
                        "domain": get_domain(url),
                        "depth": depth,
                        "out_links": out_links
                    }, ensure_ascii=False) + "\n")

                    for u in out_links:
                        if u in seen:
                            continue
                        if cfg.same_domain_only and get_domain(u) not in seed_domains:
                            continue
                        if u not in next_frontier_set:
                            next_frontier_set.add(u)
                            next_frontier.append(u)

                    extracted = extract_main_text(resp.text)
                    print(f"  -> extracted main text: {extracted is not None}")
                    if extracted:
                        title, text, date = extracted
                        print(f"  -> word count: {len(text.split())}")
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

                except Exception as e:
                    print(f"  -> ERROR {type(e).__name__}: {e}")
                    continue

            # FIN du niveau courant -> on descend d'un niveau
            frontier = next_frontier
            depth += 1

if __name__ == "__main__":
    seeds = [
        "https://the-queens-gambit.fandom.com/wiki/Beth_Harmon",
        "https://the-queens-gambit.fandom.com/wiki/The_Queen%27s_Gambit",
        "https://the-queens-gambit.fandom.com/wiki/Category:Characters",
        "https://the-queens-gambit.fandom.com/wiki/Category:Episodes"
    ]   
    cfg = CrawlConfig(min_words=200, max_pages=50, delay_s=1.0, max_depth=2)
    crawl_seeds_level_order(
        seeds,
        out_pages_jsonl_path="data/raw_jsonl/pages.jsonl",
        out_links_jsonl_path="data/raw_jsonl/out_links.jsonl",
        cfg=cfg
    )