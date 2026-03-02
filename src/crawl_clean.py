# src/crawl_clean.py
import json
import time
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
import re
import httpx
import trafilatura

@dataclass
class CrawlConfig:
    user_agent: str = "WebMiningSemanticsLab/1.0 (contact: your_email@example.com)"
    timeout: float = 20.0
    delay_s: float = 1.0
    min_words: int = 500
    max_pages: int = 50
    same_domain_only: bool = True

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
        # en cas d'erreur robots, on peut être conservateur (False) ou permissif (True).
        # Pour le lab, je conseille conservateur:
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
    words = text.split()
    return len(words) >= min_words

def extract_awoiaf_links(html: str, base_url: str):
    # Capture les liens internes de type /index.php/...
    raw = set(re.findall(r'href="(/index\.php/[^"#]*)"', html))

    cleaned = []
    for link in raw:
        # exclusions utiles (fichiers, catégories, pages spéciales mediawiki, etc.)
        if any(x in link for x in ["action=", "oldid=", "diff=", "printable="]):
            continue
        if any(prefix in link for prefix in [
            "/index.php/Special:",
            "/index.php/Category:",
            "/index.php/File:",
            "/index.php/Template:",
            "/index.php/Talk:",
            "/index.php/User:",
            "/index.php/Help:",
            ]):
            continue


        cleaned.append(urljoin(base_url, link))

    return cleaned


def crawl_seeds(seeds, out_jsonl_path: str, cfg: CrawlConfig):
    seen = set()
    to_visit = list(seeds)
    seed_domains = {get_domain(s) for s in seeds}

    with httpx.Client(timeout=cfg.timeout, follow_redirects=True) as client, open(out_jsonl_path, "w", encoding="utf-8") as f:
        while to_visit and len(seen) < cfg.max_pages:
            url = to_visit.pop(0)
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

                
                if "awoiaf.westeros.org" in get_domain(url):
                    new_links = extract_awoiaf_links(resp.text, url)
                    for u in new_links:
                        if u not in seen:
                            to_visit.append(u)

                extracted = extract_main_text(resp.text)
                if not extracted:
                    continue

                title, text, date = extracted
                if not is_useful(text, cfg.min_words):
                    continue

                record = {
                    "url": url,
                    "domain": get_domain(url),
                    "title": title,
                    "date": date,
                    "text": text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                time.sleep(cfg.delay_s)

            except Exception:
                continue

if __name__ == "__main__":
    seeds = [
        
        "https://awoiaf.westeros.org/index.php/Westeros",
        "https://awoiaf.westeros.org/index.php/A_Song_of_Ice_and_Fire",
        "https://awoiaf.westeros.org/index.php/Daenerys_Targaryen",
        "https://awoiaf.westeros.org/index.php/Arya_Stark"
    ]
    cfg = CrawlConfig(min_words=300, max_pages=15, delay_s=1.0)
    crawl_seeds(seeds, "data/raw_jsonl/pages.jsonl", cfg)
