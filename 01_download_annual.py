"""
01_download_filings.py
wDonload latest 10-K (or whatever form in config.yaml) for each ticker and extract raw text
— using config.yaml for ALL settings (no hardcoded constants).

Requires:
  - .env with:
      SEC_API_KEY=...            # or whatever key name you set in config.yaml
      CONTACT_EMAIL=you@domain   # used in SEC-friendly User-Agent
  - config.yaml with keys shown below (examples):
      tickers: ["AAPL", "MSFT"]
      form_type: "10-K"
      output_dirs:
        html: "docs"
        raw_txt: "docs_txt"
        clean_txt: "docs_txt_clean"
      sec_api:
        key_env_var: "SEC_API_KEY"
      sec_headers:
        contact_email_env_var: "CONTACT_EMAIL"
        user_agent_template: "FinancialChatbot/1.0 ({email})"
      timing:
        request_delay_seconds: 0.8
      parse:
        bs4_parser: "html.parser"   # or "lxml"
"""

import os
import re
import time
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs, urljoin

import yaml
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from sec_api import QueryApi, ExtractorApi

# --------------- config ---------------

def load_cfg(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# --------------- SEC-friendly HTTP ---------------

def make_sec_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    retries = Retry(
        total=5,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://",  HTTPAdapter(max_retries=retries))
    return s

# --------------- URL helpers ---------------

def abs_sec_url(base_url: str, href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return "https://www.sec.gov" + href
    return urljoin(base_url, href)

def looks_like_ixviewer_html(html: str) -> bool:
    h = html[:6000].lower()
    return ("inline xbrl viewer" in h) or ("edgar inline xbrl viewer" in h) or ("/ixviewer" in h)

def extract_doc_from_ixviewer_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "doc" in qs and qs["doc"]:
        doc = qs["doc"][0]
        if doc.startswith("/"):
            return "https://www.sec.gov" + doc
        return "https://www.sec.gov/" + doc
    return None

# --------------- SEC-API helpers ---------------

def get_latest_detail_url(query_api: QueryApi, ticker: str, form_type: str) -> Optional[str]:
    res = query_api.get_filings({
        "query": f'ticker:{ticker} AND formType:"{form_type}"',
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}]
    })
    filings = res.get("filings", [])
    if not filings:
        return None
    return filings[0]["linkToHtml"]

def resolve_primary_doc_url(detail_html: str, detail_url: str, bs4_parser: str) -> Optional[str]:
    """
    Prefer the 'Document Format Files' row where Type == target form (e.g., 10-K).
    Fallbacks:
      - any /Archives/edgar/data/.../*.htm (non-ixviewer)
      - ixviewer links → extract ?doc=
    """
    soup = BeautifulSoup(detail_html, bs4_parser)

    candidates: list[Tuple[int, str]] = []
    for table in soup.find_all("table"):
        header = table.get_text(" ", strip=True).lower()
        if "document format files" in header or ("description" in header and "document" in header and "type" in header):
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 4:
                    continue
                link = tds[2].find("a", href=True)
                typ  = tds[3].get_text(" ", strip=True).strip().lower()
                if not link:
                    continue
                href = abs_sec_url(detail_url, link["href"])
                if "/ixviewer" in href.lower():
                    # keep as candidate; resolve ?doc= later if needed
                    score = 2 if typ in {"10-k", "10q", "10-q"} else 1
                    candidates.append((score, href))
                else:
                    if href.lower().endswith(".htm") and "/archives/edgar/data/" in href.lower():
                        score = 3 if typ in {"10-k", "10q", "10-q"} else (2 if "10" in href.lower() else 1)
                        candidates.append((score, href))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # any Archives .htm on the page (non-ixviewer)
    for a in soup.find_all("a", href=True):
        href = abs_sec_url(detail_url, a["href"])
        if "/archives/edgar/data/" in href.lower() and href.lower().endswith(".htm") and "/ixviewer" not in href.lower():
            return href

    # ixviewer → doc=
    for a in soup.find_all("a", href=True):
        href = abs_sec_url(detail_url, a["href"])
        if "/ixviewer" in href.lower():
            doc = extract_doc_from_ixviewer_url(href)
            if doc:
                return doc

    return None

# --------------- Download + verify HTML ---------------

def download_text(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def fetch_real_filing_html(session: requests.Session, url: str, bs4_parser: str) -> str:
    # First hop
    r = session.get(url, timeout=30)
    r.raise_for_status()
    html = r.text

    # If viewer shell, extract ?doc= and refetch
    if "/ixviewer" in r.url.lower() or looks_like_ixviewer_html(html):
        doc_url = extract_doc_from_ixviewer_url(r.url)
        if not doc_url:
            soup = BeautifulSoup(html, bs4_parser)
            for a in soup.find_all("a", href=True):
                if "doc=" in a["href"].lower():
                    doc_url = extract_doc_from_ixviewer_url(abs_sec_url(r.url, a["href"]))
                    if doc_url:
                        break
        if doc_url:
            r2 = session.get(doc_url, timeout=30)
            r2.raise_for_status()
            html2 = r2.text
            if looks_like_ixviewer_html(html2):
                return ""  # signal fallback
            return html2
        return ""

    return html  # not a viewer shell

# --------------- Text extraction ---------------

def html_to_text_with_headings(html: str, bs4_parser: str) -> str:
    soup = BeautifulSoup(html, bs4_parser)

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n")

    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        h.insert_before("\n\n## " + h.get_text(" ", strip=True) + "\n")

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

# --------------- Fallback: ExtractorApi sections ---------------

def save_fallback_sections(extractor: ExtractorApi, detail_url: str, ticker: str, form_type: str, out_txt_dir: str) -> bool:
    """
    If raw HTML can’t be fetched, save Item 1A and 7 via ExtractorApi.
    """
    try:
        risk = extractor.get_section(detail_url, "1A", "text")
    except Exception:
        risk = ""
    try:
        mda  = extractor.get_section(detail_url, "7", "text")
    except Exception:
        mda  = ""

    saved_any = False
    os.makedirs(out_txt_dir, exist_ok=True)
    if risk:
        with open(os.path.join(out_txt_dir, f"{ticker}_{form_type}_Item1A.txt"), "w", encoding="utf-8") as f:
            f.write(risk)
        saved_any = True
    if mda:
        with open(os.path.join(out_txt_dir, f"{ticker}_{form_type}_Item7.txt"), "w", encoding="utf-8") as f:
            f.write(mda)
        saved_any = True
    return saved_any

# --------------- Per-ticker flow ---------------

def process_ticker(
    query_api: QueryApi,
    extractor: ExtractorApi,
    session: requests.Session,
    ticker: str,
    form_type: str,
    out_html_dir: str,
    out_txt_dir: str,
    delay: float,
    bs4_parser: str
):
    print(f"\n==> {ticker}: locating latest {form_type}")
    
    # Check if files already exist
    html_path = os.path.join(out_html_dir, f"{ticker}_{form_type}.html")
    txt_path = os.path.join(out_txt_dir, f"{ticker}_{form_type}.txt")
    
    if os.path.exists(html_path) and os.path.exists(txt_path):
        print(f"   Files already exist, skipping {ticker}")
        return
    
    detail_url = get_latest_detail_url(query_api, ticker, form_type)
    if not detail_url:
        print(f"[WARN] No {form_type} detail page for {ticker}")
        return

    time.sleep(delay)
    detail_html = download_text(session, detail_url)

    primary_url = resolve_primary_doc_url(detail_html, detail_url, bs4_parser)
    if not primary_url:
        print(f"[WARN] Could not resolve a primary {form_type} HTML for {ticker}")
        if save_fallback_sections(extractor, detail_url, ticker, form_type, out_txt_dir):
            print(f"[FALLBACK] Saved sections for {ticker} via ExtractorApi")
        else:
            print(f"[FAIL] {ticker}: no sections saved")
        return

    print(f"   primary doc: {primary_url}")
    time.sleep(delay)
    filing_html = fetch_real_filing_html(session, primary_url, bs4_parser)

    # If still viewer shell or empty → fallback to ExtractorApi
    if not filing_html or looks_like_ixviewer_html(filing_html):
        print(f"[INFO] {ticker}: raw filing returned viewer shell. Using ExtractorApi fallback.")
        if save_fallback_sections(extractor, detail_url, ticker, form_type, out_txt_dir):
            print(f"[FALLBACK] Saved Item 1A / Item 7 for {ticker}")
        else:
            print(f"[FAIL] {ticker}: fallback sections not available")
        return

    # Save raw HTML
    os.makedirs(out_html_dir, exist_ok=True)
    html_path = os.path.join(out_html_dir, f"{ticker}_{form_type}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(filing_html)
    print(f"   saved HTML → {html_path}")

    # Save raw TXT
    os.makedirs(out_txt_dir, exist_ok=True)
    raw_txt = html_to_text_with_headings(filing_html, bs4_parser)
    txt_path = os.path.join(out_txt_dir, f"{ticker}_{form_type}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(raw_txt)
    print(f"   saved TXT  → {txt_path}")

# --------------- Entry ---------------

def main():
    # load config + env
    cfg = load_cfg("config.yaml")
    load_dotenv()

    # pull config
    tickers   = cfg.get("tickers", ["AAPL", "MSFT"])
    form_type = cfg.get("form_type", "10-K")

    out_html_dir = cfg.get("output_dirs", {}).get("sec_html_annual", "sec_data/annual")
    out_txt_dir  = cfg.get("output_dirs", {}).get("sec_txt_annual", "sec_txt/annual")
    # clean_txt dir is created by your cleaner; we don't use it here

    sec_key_env = cfg.get("sec_api", {}).get("key_env_var", "SEC_API_KEY")
    email_env   = cfg.get("sec_headers", {}).get("contact_email_env_var", "CONTACT_EMAIL")
    ua_tpl      = cfg.get("sec_headers", {}).get("user_agent_template", "FinancialChatbot/1.0 ({email})")

    delay       = float(cfg.get("timing", {}).get("request_delay_seconds", 0.8))
    bs4_parser  = cfg.get("parse", {}).get("bs4_parser", "html.parser")

    # secrets
    sec_api_key   = os.getenv(sec_key_env)
    contact_email = os.getenv(email_env, "you@example.com")
    if not sec_api_key:
        raise ValueError(f"{sec_key_env} missing (.env)")
    if contact_email == "you@example.com":
        print("[INFO] Set CONTACT_EMAIL in .env for a proper SEC User-Agent.")

    user_agent = ua_tpl.format(email=contact_email)

    # clients
    session   = make_sec_session(user_agent)
    query_api = QueryApi(api_key=sec_api_key)
    extractor = ExtractorApi(api_key=sec_api_key)

    # run
    for t in tickers:
        try:
            process_ticker(query_api, extractor, session, t, form_type, out_html_dir, out_txt_dir, delay, bs4_parser)
            time.sleep(delay)
        except requests.HTTPError as e:
            print(f"[HTTP ERROR] {t}: {e}")
        except Exception as e:
            print(f"[ERROR] {t}: {e}")

if __name__ == "__main__":
    main()
