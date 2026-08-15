#!/usr/bin/env python3
"""
recover_failed.py — recover the CSRD reports that phase1.py couldn't download
from the sandbox (WAF/IP-blocked, timeouts, and JS landing pages).

RUN THIS ON YOUR LOCAL MACHINE (normal home/office network) — not in the
sandbox. Most of the failures are only because the sandbox's datacenter IP is
blocked; from a residential IP they just work.

------------------------------------------------------------------------------
SETUP (VS Code terminal)
------------------------------------------------------------------------------
    pip install requests pandas playwright
    playwright install chromium          # one-time browser download

    # Put this file in the same folder as:
    #   - download_log.csv            (written by phase1.py; failures are picked out of it)
    #   - pdfs/                       (existing folder; recovered PDFs land here)
    # If pdfs/ doesn't exist locally, it will be created.

    python recover_failed.py                 # requests + browser fallback
    python recover_failed.py --no-browser    # requests only (faster, weaker)
    python recover_failed.py --only error:http_403     # one failure category
------------------------------------------------------------------------------

Output:
    pdfs/<Company>_<ISIN>.pdf   for every recovered report
    recover_log.csv             one row per attempt (recovered / still_failed)

After running, copy the new pdfs/ back next to phase1.py and run:
    python3 phase1.py --skip-download
to extract text/tables + detect language for the newly recovered PDFs.
"""

import argparse
import os
import re
import sys
import time
from urllib.parse import urljoin, urlsplit, parse_qs, unquote

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
FAILED_CSV = "download_log.csv"   # phase1.py writes this; failures are filtered out of it
PDF_DIR = "pdfs"
RECOVER_LOG = "recover_log.csv"
TIMEOUT = 60
MAX_RETRIES = 3
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def headers_for(url):
    origin = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}/"
    return {
        "User-Agent": UA,
        "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": origin,
    }


def unwrap_redirect(url):
    """google.com/url?...&url=<real> wrappers -> the real target."""
    p = urlsplit(url)
    if p.netloc.endswith("google.com") and p.path == "/url":
        q = parse_qs(p.query)
        real = q.get("url") or q.get("q")
        if real:
            return unquote(real[0])
    return url


def is_pdf(content_type, data):
    ct = (content_type or "").lower()
    return "pdf" in ct or "octet-stream" in ct or data[:4] == b"%PDF"


def find_pdf_links_in_html(html, base_url):
    """Return candidate PDF URLs found in an HTML page (best-first)."""
    out = []
    # meta refresh redirect
    m = re.search(r'http-equiv=["\']refresh["\'][^>]*content=["\'][^"\']*url=([^"\'>]+)',
                  html, re.IGNORECASE)
    if m:
        out.append(urljoin(base_url, m.group(1).strip()))
    # href/src ending in .pdf
    for href in re.findall(r'(?:href|src)=["\']([^"\']*\.pdf(?:\?[^"\']*)?)["\']',
                           html, re.IGNORECASE):
        out.append(urljoin(base_url, href))
    # prefer links that look like annual/sustainability reports
    kw = ("annual", "report", "sustainability", "csrd", "esg", "integrated")
    out.sort(key=lambda u: 0 if any(k in u.lower() for k in kw) else 1)
    # de-dup, keep order
    seen, uniq = set(), []
    for u in out:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq


# ----------------------------------------------------------------------------
# strategy 1: plain requests (with SSL fallback + retries + HTML link follow)
# ----------------------------------------------------------------------------
def fetch(url, timeout=TIMEOUT, retries=MAX_RETRIES):
    for attempt in range(1, retries + 1):
        for verify in (True, False):
            try:
                if not verify:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                r = requests.get(url, headers=headers_for(url), timeout=timeout, verify=verify)
                r.raise_for_status()
                return r
            except requests.exceptions.SSLError:
                if verify:
                    continue
                return None
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                break
            except requests.RequestException:
                return None
        if attempt < retries:
            time.sleep(2 ** attempt)
    return None


def try_requests(url):
    url = unwrap_redirect(url)
    r = fetch(url)
    if r is None:
        return None
    if is_pdf(r.headers.get("Content-Type", ""), r.content):
        return r.content
    # HTML — look for an embedded PDF link and follow it (one level)
    if "html" in r.headers.get("Content-Type", "").lower():
        for cand in find_pdf_links_in_html(r.content.decode("utf-8", "ignore"), url)[:4]:
            r2 = fetch(cand)
            if r2 and is_pdf(r2.headers.get("Content-Type", ""), r2.content):
                return r2.content
    return None


# ----------------------------------------------------------------------------
# strategy 2: real browser (Playwright) — defeats WAFs and renders JS pages
# ----------------------------------------------------------------------------
def try_browser(url):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    url = unwrap_redirect(url)
    origin = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}/"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True,
                                    args=["--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(user_agent=UA, accept_downloads=True,
                                  ignore_https_errors=True)
        page = ctx.new_page()
        try:
            # warm up: visit homepage so the WAF sets cookies / clears JS challenge
            try:
                page.goto(origin, wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(2500)
            except Exception:
                pass

            # A) direct PDF: fetch within the warmed, cookie'd browser context
            resp = ctx.request.get(url, timeout=60000)
            body = resp.body()
            if resp.ok and is_pdf(resp.headers.get("content-type", ""), body):
                return body

            # B) landing/JS page: render it, harvest .pdf links, fetch the best
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(2500)
                hrefs = page.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => e.href)")
                cands = find_pdf_links_in_html(
                    "".join(f'<a href="{h}"></a>' for h in hrefs), url)
                for cand in cands[:5]:
                    r = ctx.request.get(cand, timeout=60000)
                    b = r.body()
                    if r.ok and is_pdf(r.headers.get("content-type", ""), b):
                        return b
            except Exception:
                pass
        finally:
            browser.close()
    return None


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=FAILED_CSV)
    ap.add_argument("--no-browser", action="store_true",
                    help="skip the Playwright fallback (requests only)")
    ap.add_argument("--only", help="only rows with this failure status, e.g. error:http_403")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"missing {args.csv} — run phase1.py first, or pass --csv")
    os.makedirs(PDF_DIR, exist_ok=True)

    df = pd.read_csv(args.csv)
    # phase1's download_log.csv carries every attempt; keep only the failures.
    # "reason" is the column name the old standalone worklist used.
    if "reason" not in df.columns and "status" in df.columns:
        df = df[~df["status"].isin(["success", "skipped"])].copy()
        df["reason"] = df["status"]
    df = df.drop_duplicates(subset=["url"])
    if args.only:
        df = df[df["reason"] == args.only]
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        print(f"Nothing to recover from {args.csv}.")
        return

    rows, ok = [], 0
    for i, row in df.reset_index(drop=True).iterrows():
        company, url = str(row["company"]), str(row["url"])
        dest = row.get("dest") or row.get("expected_file") or os.path.join(
            PDF_DIR, re.sub(r"\s+", "_", re.sub(r"[^\w\s-]", "", company)) + f"_{row['isin']}.pdf")
        print(f"[{i+1}/{len(df)}] {company} ({row.get('reason','?')})")

        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print("   already have it"); rows.append((company, url, "skipped", dest)); continue

        data = try_requests(url)
        how = "requests"
        if data is None and not args.no_browser:
            data = try_browser(url); how = "browser"

        if data and data[:4] == b"%PDF" and len(data) > 10_000:
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            print(f"   RECOVERED via {how} -> {dest} ({len(data)//1024} KB)")
            rows.append((company, url, f"recovered:{how}", dest)); ok += 1
        else:
            print("   still failed")
            rows.append((company, url, "still_failed", dest))
        time.sleep(1)

    pd.DataFrame(rows, columns=["company", "url", "result", "dest"]).to_csv(RECOVER_LOG, index=False)
    print(f"\nDone. Recovered {ok}/{len(df)}. Log: {RECOVER_LOG}")
    print(f"PDFs are in {PDF_DIR}/ — copy them next to phase1.py and run: "
          f"python3 phase1.py --skip-download")


if __name__ == "__main__":
    main()
