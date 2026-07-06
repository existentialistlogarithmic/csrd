#!/usr/bin/env python3
# 1st stage

import argparse
import csv
import json
import logging
import os
import re
import time

import fitz          # pip install pymupdf
import pandas as pd
import requests
from urllib.parse import urljoin, urlsplit, parse_qs, unquote

from langdetect import detect, DetectorFactory, LangDetectException  # pip install langdetect
DetectorFactory.seed = 0  # make detection deterministic

# config
EXCEL_FILE = "SRN-CSRD_report_archive.xlsx"
SHEET_NAME = "csrd"
HEADER_ROW = 2  # 0-indexed, so row 3 in the spreadsheet

PDF_DIR = "pdfs"
TEXT_DIR = "extracted_text"
TABLE_DIR = "extracted_tables"
DOWNLOAD_LOG = "download_log.csv"
SUMMARY_CSV = "extraction_summary.csv"

DELAY = 1.0      # seconds between downloads
TIMEOUT = int(os.environ.get("DL_TIMEOUT", "60"))   # seconds per request
MAX_RETRIES = int(os.environ.get("DL_RETRIES", "3"))  # retries for timeout/connection errors
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0.0.0 Safari/537.36")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def clean_filename(name):
    """Remove special chars, replace spaces with underscores."""
    name = re.sub(r"[^\w\s-]", "", name)
    return re.sub(r"\s+", "_", name.strip())


def _unwrap_redirect(url):
    """A handful of spreadsheet links are Google search redirect wrappers
    (google.com/url?...&url=<real target>&...) instead of the real link."""
    parts = urlsplit(url)
    if parts.netloc.endswith("google.com") and parts.path == "/url":
        qs = parse_qs(parts.query)
        real = qs.get("url") or qs.get("q")
        if real:
            return unquote(real[0])
    return url


def _build_headers(url):
    origin = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}/"
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": origin,
    }


# 1 read thru excel n filter---

def load_reports(path):
    log.info("Reading %s ...", path)
    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW)

    # some column names have newlines in them, clean that up
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]

    # download everything that has a usable link; page ranges are optional
    # (rows without start/end PDF get the whole document extracted)
    mask = df["link"].astype(str).str.strip().str.startswith("http")
    valid = df[mask].copy()

    # keep start/end as nullable ints: NaN means "no page range given"
    valid["start PDF"] = pd.to_numeric(valid["start PDF"], errors="coerce").astype("Int64")
    valid["end PDF"] = pd.to_numeric(valid["end PDF"], errors="coerce").astype("Int64")
    no_range = valid["start PDF"].isna() | valid["end PDF"].isna()

    log.info("Total rows: %d | With link: %d (no page range: %d, will extract full document)",
             len(df), len(valid), int(no_range.sum()))
    return valid.reset_index(drop=True)


# 2 downloads pdfs ---
def _fetch_with_retries(url, headers, timeout=TIMEOUT, max_retries=MAX_RETRIES):
    """Fetch a URL with SSL fallback and retry on timeout/connection errors."""
    last_err = None
    origin = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}/"
    for attempt in range(1, max_retries + 1):
        for verify in (True, False):
            try:
                if not verify:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    log.info("  Retrying without SSL (attempt %d/%d)...", attempt, max_retries)
                resp = requests.get(url, headers=headers, timeout=timeout, verify=verify)
                if resp.status_code == 403 and attempt == 1 and origin != url:
                    # some WAFs block bare GETs but allow a session that first
                    # visited the homepage (looks less like a bot / picks up cookies)
                    try:
                        s = requests.Session()
                        s.get(origin, headers=headers, timeout=timeout, verify=verify)
                        warm_resp = s.get(url, headers=headers, timeout=timeout, verify=verify)
                        if warm_resp.status_code != 403:
                            log.info("  Recovered 403 via homepage warm-up: %s", url)
                            resp = warm_resp
                    except requests.RequestException:
                        pass
                resp.raise_for_status()
                return resp
            except requests.exceptions.SSLError:
                if verify:
                    continue  # try again without SSL
                log.error("  SSL error persists: %s", url)
                return None
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                if verify:
                    break  # no point retrying without SSL for timeout
                break
            except requests.RequestException as e:
                log.error("  Download failed: %s", e)
                return None
        if attempt < max_retries:
            wait = 2 ** attempt
            log.info("  Timeout/connection error, retrying in %ds (attempt %d/%d)...",
                     wait, attempt, max_retries)
            time.sleep(wait)
    log.error("  Download failed after %d retries: %s", max_retries, last_err)
    return None


def _find_pdf_link_in_html(html_bytes, base_url):
    """Try to find a PDF download link in an HTML page."""
    try:
        html = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return None

    # some report pages are just a redirect wrapper:
    # <meta http-equiv="refresh" content="0;url=...">
    refresh = re.search(
        r'http-equiv=["\']refresh["\'][^>]*content=["\'][^"\']*url=([^"\'>]+)',
        html, re.IGNORECASE,
    )
    if refresh:
        target = urljoin(base_url, refresh.group(1).strip())
        if target.lower() != base_url.lower():
            log.info("  Found meta-refresh redirect: %s", target)
            return target

    # look for links ending in .pdf (common pattern for report download pages)
    pdf_links = re.findall(r'href=["\']([^"\']*\.pdf(?:\?[^"\']*)?)["\']', html, re.IGNORECASE)
    if not pdf_links:
        return None
    # prefer links that look like annual/sustainability reports
    report_keywords = ["annual", "report", "sustainability", "csrd", "esg"]
    best = None
    for link in pdf_links:
        absolute = urljoin(base_url, link)
        if any(kw in link.lower() for kw in report_keywords):
            best = absolute
            break
    if best is None:
        best = urljoin(base_url, pdf_links[0])
    log.info("  Found PDF link in HTML page: %s", best)
    return best


def download_pdf(url, dest):
    """Try to download a PDF. Returns a status string."""
    if os.path.exists(dest):
        log.info("  Already exists, skipping")
        return "skipped"

    unwrapped = _unwrap_redirect(url)
    if unwrapped != url:
        log.info("  Unwrapped redirect link: %s", unwrapped)
        url = unwrapped

    headers = _build_headers(url)

    resp = _fetch_with_retries(url, headers)
    if resp is None:
        return "error:download_failed"

    data = resp.content
    content_type = resp.headers.get("Content-Type", "").lower()

    # check if it's actually a PDF
    looks_like_pdf = "pdf" in content_type or "octet-stream" in content_type
    starts_with_pdf = data[:4] == b"%PDF"

    if not looks_like_pdf and not starts_with_pdf:
        # if we got HTML, try to find a PDF link in the page
        if "html" in content_type:
            pdf_url = _find_pdf_link_in_html(data, url)
            if pdf_url:
                resp2 = _fetch_with_retries(pdf_url, _build_headers(pdf_url))
                if resp2 is not None:
                    data2 = resp2.content
                    ct2 = resp2.headers.get("Content-Type", "").lower()
                    if "pdf" in ct2 or "octet-stream" in ct2 or data2[:4] == b"%PDF":
                        data = data2
                    else:
                        log.warning("  Linked file is not a PDF either (%s): %s", ct2, pdf_url)
                        return "not_pdf"
                else:
                    log.warning("  Could not download linked PDF: %s", pdf_url)
                    return "not_pdf"
            else:
                log.warning("  Not a PDF (%s) and no PDF link found in page: %s", content_type, url)
                return "not_pdf"
        else:
            log.warning("  Not a PDF (%s): %s", content_type, url)
            return "not_pdf"

    # save it
    with open(dest, "wb") as f:
        f.write(data)

    # reject tiny files (probably error pages)
    if os.path.getsize(dest) < 10_000:
        log.warning("  Too small (%d bytes), removing", os.path.getsize(dest))
        os.remove(dest)
        return "error:file_too_small"

    return "success"


def download_all(df):
    statuses = {}
    is_new = not os.path.exists(DOWNLOAD_LOG)

    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["index", "company", "isin", "url", "status", "dest"])

        for pos, (idx, row) in enumerate(df.iterrows()):
            company = str(row["company"])
            isin = str(row["isin"])
            url = str(row["link"])
            dest = os.path.join(PDF_DIR, f"{clean_filename(company)}_{isin}.pdf")

            log.info("[%d/%d] %s", pos + 1, len(df), company)
            status = download_pdf(url, dest)
            statuses[idx] = status
            writer.writerow([idx, company, isin, url, status, dest])
            f.flush()

            if status == "success":
                log.info("  saved to %s", dest)
            if status != "skipped":
                time.sleep(DELAY)

    return statuses


def dedup_download_log():
    """Keep only the latest status per row index (retry runs append new rows)."""
    if not os.path.exists(DOWNLOAD_LOG):
        return
    d = pd.read_csv(DOWNLOAD_LOG)
    d = d.drop_duplicates(subset="index", keep="last").sort_values("index")
    d.to_csv(DOWNLOAD_LOG, index=False)


# -3 get txt ffrom tables (this will return it bin...)

def extract_text(pdf_path, start, end):
    """Get text from pages start to end (1-indexed, inclusive).
    start/end of None means the whole document."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.error("  Can't open %s: %s", pdf_path, e)
        return None

    if start is None:
        start = 1
    if end is None:
        end = len(doc)

    pages = []
    for p in range(start - 1, min(end, len(doc))):
        pages.append(doc[p].get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES))
    doc.close()
    return "\n\n".join(pages)


def extract_tables(pdf_path, start, end):
    """Find tables using pymupdf's built-in table finder."""
    found = []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return found

    if start is None:
        start = 1
    if end is None:
        end = len(doc)

    for p in range(start - 1, min(end, len(doc))):
        try:
            for table in doc[p].find_tables().tables:
                rows = table.extract()
                if rows:
                    # replace None cells with empty string
                    cleaned = [
                        [cell if cell is not None else "" for cell in row]
                        for row in rows
                    ]
                    found.append(cleaned)
        except Exception as e:
            log.debug("Table extraction failed on page %d: %s", p + 1, e)

    doc.close()
    return found
# up until here

def detect_language(text):
    """Detect the language of extracted text ('en', 'de', ...) or 'unknown'.

    Samples from the start, middle and end so a translated cover page
    can't fool the detector."""
    if not text or not text.strip():
        return "unknown"
    n = len(text)
    sample = text[:4000]
    if n > 12000:
        mid = n // 2
        sample += "\n" + text[mid:mid + 4000] + "\n" + text[-4000:]
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


def process_one_pdf(pdf_path, company, isin, start, end):
    """Extract text + tables from one PDF. Returns a dict with metadata."""
    result = {
        "text_file": None,
        "table_file": None,
        "word_count": 0,
        "pages_extracted": 0,
        "tables_found": 0,
        "language": "unknown",
        "extraction_status": "success",
    }

    stem = f"{clean_filename(company)}_{isin}"

    # get the text
    text = extract_text(pdf_path, start, end)
    if text is None:
        result["extraction_status"] = "failed"
        return result

    # no. how many pages actually extracted
    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
    except Exception:
        total_pages = end or 0

    eff_start = start if start is not None else 1
    eff_end = end if end is not None else total_pages
    result["pages_extracted"] = max(0, min(eff_end, total_pages) - (eff_start - 1))
    result["word_count"] = len(text.split())
    result["language"] = detect_language(text)

    # save the text
    text_file = os.path.join(TEXT_DIR, f"{stem}.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)
    result["text_file"] = text_file

    # get tables and save them
    tables = extract_tables(pdf_path, start, end)
    result["tables_found"] = len(tables)
    if tables:
        table_file = os.path.join(TABLE_DIR, f"{stem}_tables.json")
        with open(table_file, "w", encoding="utf-8") as f:
            json.dump(tables, f, ensure_ascii=False, indent=2)
        result["table_file"] = table_file

    return result


# 4 summary

def build_summary(df, download_statuses, extraction_results):
    # find SASB column names (ok)
    industry_col = ""
    sector_col = ""
    for c in df.columns:
        if "SASB industry" in c:
            industry_col = c
        if "SASB sector" in c:
            sector_col = c

    rows = []
    for idx, row in df.iterrows():
        ext = extraction_results.get(idx, {})
        rows.append({
            "company": str(row["company"]),
            "isin": str(row["isin"]),
            "country": row.get("country", ""),
            "industry": row.get(industry_col, ""),
            "sector": row.get(sector_col, ""),
            "download_status": download_statuses.get(idx, "unknown"),
            "extraction_status": ext.get("extraction_status", "not_attempted"),
            "language": ext.get("language", "unknown"),
            "pages_extracted": ext.get("pages_extracted", 0),
            "word_count": ext.get("word_count", 0),
            "tables_found": ext.get("tables_found", 0),
            "text_file": ext.get("text_file", ""),
            "table_file": ext.get("table_file", ""),
        })

    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    log.info("Summary saved to %s (%d rows)", SUMMARY_CSV, len(rows))


# main

def main():
    parser = argparse.ArgumentParser(description="CSRD PDF scraping pipeline")
    parser.add_argument("--excel-file", default=EXCEL_FILE, help="Input Excel file")
    parser.add_argument("--limit", type=int, help="Only process first N reports")
    parser.add_argument("--start-from", type=int, default=0, help="Resume from this index")
    parser.add_argument("--retry-failed", action="store_true",
                         help="Only re-attempt rows whose last recorded download status "
                              "wasn't success/skipped; reuse everything else from the "
                              "existing download_log.csv / extraction_summary.csv")
    args = parser.parse_args()

    # create output folders
    for d in [PDF_DIR, TEXT_DIR, TABLE_DIR]:
        os.makedirs(d, exist_ok=True)

    # load and filter
    df = load_reports(args.excel_file)

    if args.start_from > 0:
        df = df.iloc[args.start_from:]
        log.info("Starting from index %d (%d left)", args.start_from, len(df))
    if args.limit is not None:
        df = df.iloc[:args.limit]
        log.info("Limiting to %d reports", len(df))
    # reset index after slicing so download_all and extraction loop use 0-based indices
    df = df.reset_index(drop=True)
    if df.empty:
        log.warning("Nothing to process.")
        return

    prev_summary_by_key = {}
    retry_idx = set(df.index)
    if args.retry_failed:
        prev_log = None
        if os.path.exists(DOWNLOAD_LOG):
            prev_log = pd.read_csv(DOWNLOAD_LOG).drop_duplicates(subset="index", keep="last").set_index("index")
        if os.path.exists(SUMMARY_CSV):
            for _, r in pd.read_csv(SUMMARY_CSV).iterrows():
                prev_summary_by_key[(str(r["company"]), str(r["isin"]))] = r
        if prev_log is not None:
            prev_status = prev_log["status"].to_dict()
            retry_idx = {i for i in df.index if prev_status.get(i, "missing") not in ("success", "skipped")}
        log.info("Retry mode: %d/%d rows need another attempt", len(retry_idx), len(df))

    # download (only the rows that still need it, in retry mode)
    log.info("=" * 50)
    log.info("STEP 2: Downloading PDFs")
    log.info("=" * 50)
    sub_df = df.loc[sorted(retry_idx)] if args.retry_failed else df
    if sub_df.empty:
        log.info("Nothing left to (re-)download.")
    else:
        download_all(sub_df)
    dedup_download_log()

    full_log = pd.read_csv(DOWNLOAD_LOG).drop_duplicates(subset="index", keep="last").set_index("index")
    download_statuses = {i: full_log["status"].get(i, "unknown") for i in df.index}

    # extract
    log.info("=" * 50)
    log.info("STEP 3: Extracting text and tables")
    log.info("=" * 50)
    extraction_results = {}

    for idx, row in df.iterrows():
        company = str(row["company"])
        isin = str(row["isin"])
        pdf_path = os.path.join(PDF_DIR, f"{clean_filename(company)}_{isin}.pdf")
        start = None if pd.isna(row["start PDF"]) else int(row["start PDF"])
        end = None if pd.isna(row["end PDF"]) else int(row["end PDF"])

        status = download_statuses.get(idx, "unknown")

        # row wasn't retried this run and we already have its extraction result: reuse it
        if idx not in retry_idx and (company, isin) in prev_summary_by_key:
            r = prev_summary_by_key[(company, isin)]
            extraction_results[idx] = {
                "text_file": r.get("text_file") or None,
                "table_file": r.get("table_file") or None,
                "word_count": int(r.get("word_count") or 0),
                "pages_extracted": int(r.get("pages_extracted") or 0),
                "tables_found": int(r.get("tables_found") or 0),
                "language": r.get("language", "unknown"),
                "extraction_status": r.get("extraction_status", "not_attempted"),
            }
            continue

        if status not in ("success", "skipped"):
            log.info("[%d/%d] Skipping %s (download: %s)", idx+1, len(df), company, status)
            continue
        if not os.path.exists(pdf_path):
            log.warning("[%d/%d] PDF missing: %s", idx+1, len(df), company)
            continue

        if start is not None and end is not None:
            log.info("[%d/%d] Extracting %s (pages %d-%d)", idx+1, len(df), company, start, end)
        else:
            log.info("[%d/%d] Extracting %s (full document)", idx+1, len(df), company)
        extraction_results[idx] = process_one_pdf(pdf_path, company, isin, start, end)
        r = extraction_results[idx]
        log.info("  %d pages, %d words, %d tables, language: %s",
                 r["pages_extracted"], r["word_count"], r["tables_found"], r["language"])

    # summary
    log.info("=" * 50)
    log.info("STEP 4: Building summary")
    log.info("=" * 50)
    build_summary(df, download_statuses, extraction_results)

    # final count
    success_count = 0
    skip_count = 0
    fail_count = 0
    for s in download_statuses.values():
        if s == "success":
            success_count += 1
        elif s == "skipped":
            skip_count += 1
        else:
            fail_count += 1
    log.info("Done! success: %d, skipped: %d, failed: %d", success_count, skip_count, fail_count)


if __name__ == "__main__":
    main()
