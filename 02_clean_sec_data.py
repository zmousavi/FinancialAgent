"""
Clean raw TXT (with viewer/XBRL junk) into narrative-only text.
- Uses config.yaml output_dirs
- Single-file mode (explicit --in/--out) or batch mode (--batch)

Usage:
  # single file
  python 02_clean_filing_text.py --in docs_txt/MSFT_10K.txt --out docs_txt_clean/MSFT_10K.clean.txt

  # batch all tickers in config.yaml
  python 02_clean_filing_text.py --batch
"""

import os
import re
import argparse
import yaml

VIEWER_NOISE_PATTERNS = [
    r"^XBRL\s+Viewer\s*$",
    r"^Please enable JavaScript to use the EDGAR Inline XBRL Viewer\.\s*$",
    r"^This page uses Javascript\.",
    r"^Your browser either doesn't support Javascript or you have it turned off\.",
]

START_ANCHORS = [
    r"UNITED\s+STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION",
    r"\bFORM\s+10[\-\u2011\u2013]?\s*K\b",
    r"\bANNUAL\s+REPORT\b",
]

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_text(p): 
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_text(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)

def strip_viewer_noise(s: str) -> str:
    lines = s.splitlines()
    keep = []
    for line in lines:
        if any(re.search(pat, line.strip(), flags=re.IGNORECASE) for pat in VIEWER_NOISE_PATTERNS):
            continue
        keep.append(line)
    return "\n".join(keep)

def find_start(s: str) -> int:
    s_up = s.upper()
    best = len(s)
    for pat in START_ANCHORS:
        m = re.search(pat, s_up, flags=re.IGNORECASE)
        if m:
            best = min(best, m.start())
    return 0 if best == len(s) else best

def normalize(s: str) -> str:
    s = re.sub(r"\[/?TABLE\]", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n\s*(ITEM\s+\d+[A-Z]?(?:\.[^\n]*)?)\s*\n", r"\n\n\1\n", s, flags=re.IGNORECASE)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = "\n".join(ln.rstrip() for ln in s.splitlines())
    return s.strip()

def clean_text(raw: str) -> str:
    raw = strip_viewer_noise(raw)
    start = find_start(raw)
    kept = raw[start:] if start < len(raw) else raw
    return normalize(kept)

def clean_file(inp: str, outp: str):
    raw = read_text(inp)
    cleaned = clean_text(raw)
    write_text(outp, cleaned)
    print(f"Cleaned → {outp}")

def batch_clean_annual(cfg: dict):
    """Clean annual 10-K files"""
    raw_dir = cfg["output_dirs"]["sec_txt_annual"]
    clean_dir = cfg["output_dirs"]["sec_txt_clean_annual"]
    form    = cfg["form_type"]
    tickers = cfg["tickers"]

    print("=== Cleaning Annual 10-K Files ===")
    for t in tickers:
        inp  = os.path.join(raw_dir, f"{t}_{form}.txt")
        outp = os.path.join(clean_dir, f"{t}_{form}.clean.txt")
        
        # Skip if output already exists
        if os.path.exists(outp):
            print(f" {outp} already exists, skipping")
            continue
            
        if os.path.exists(inp):
            clean_file(inp, outp)
        else:
            print(f"[SKIP] {inp} not found")

def batch_clean_quarterly(cfg: dict):
    """Clean quarterly 10-Q files"""
    raw_dir = cfg["output_dirs"]["sec_txt_quarterly"]
    clean_dir = cfg["output_dirs"]["sec_txt_clean_quarterly"]
    
    print("\n=== Cleaning Quarterly 10-Q Files ===")
    
    # Find all quarterly txt files
    if not os.path.exists(raw_dir):
        print(f"[SKIP] Quarterly directory {raw_dir} not found")
        return
    
    quarterly_files = []
    for filename in os.listdir(raw_dir):
        if filename.endswith("_10-Q.txt"):
            quarterly_files.append(filename)
    
    if not quarterly_files:
        print(f"[SKIP] No quarterly 10-Q files found in {raw_dir}")
        return
    
    print(f"Found {len(quarterly_files)} quarterly files to clean")
    
    for filename in sorted(quarterly_files):
        # Input: AAPL_2025_Q1_10-Q.txt
        # Output: AAPL_2025_Q1_10-Q.clean.txt
        inp = os.path.join(raw_dir, filename)
        outp = os.path.join(clean_dir, filename.replace(".txt", ".clean.txt"))
        
        # Skip if output already exists
        if os.path.exists(outp):
            print(f" {outp} already exists, skipping")
            continue
            
        clean_file(inp, outp)

def batch_clean(cfg: dict):
    """Clean both annual and quarterly files"""
    batch_clean_annual(cfg)
    batch_clean_quarterly(cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="path to raw txt")
    ap.add_argument("--out", dest="outp", help="path to write clean txt")
    ap.add_argument("--batch", action="store_true", help="clean all tickers in config.yaml")
    args = ap.parse_args()

    cfg = load_cfg("config.yaml")

    if args.batch:
        batch_clean(cfg)
        return

    if not args.inp or not args.outp:
        ap.error("Provide --in and --out, or use --batch")

    clean_file(args.inp, args.outp)

if __name__ == "__main__":
    main()
