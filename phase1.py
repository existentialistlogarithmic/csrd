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
from urllib.parse import urljoin

from langdetect import detect, DetectorFactory, LangDetectException  # pip install langdetect
DetectorFactory.seed = 0  # make detection deterministic

# config
EXCEL_FILE = "SRN-CSRD_report_archive.xlsx"
SHEET_NAME = "csrd"
HEADER_ROW = 2  # 0-indexed, so row 3 in the spreadsheet

# SRN (Sustainability Reporting Navigator, srnav.com) API — hosts the report
# PDFs itself, so downloads don't depend on 500 different company websites.
# Endpoint variants are probed in order; the first that answers JSON wins.
SRN_API_BASES = [
    "https://api.srnav.com/api",
    "https://www.srnav.com/api",
    "https://api.sustainabilityreportingnavigator.com/api",
]

PDF_DIR = "pdfs"
TEXT_DIR = "extracted_text"
TABLE_DIR = "extracted_tables"
DOWNLOAD_LOG = "download_log.csv"
SUMMARY_CSV = "extraction_summary.csv"

DELAY = 1.5      # seconds between downloads
TIMEOUT = 60     # seconds per request
MAX_RETRIES = 3  # retries for timeout/connection errors
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


# 1b SRN API source ---

def _pick(d, *keys, default=None):
    """First non-empty value among several possible key spellings."""
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return v
    return default


def _srn_get_json(path):
    """GET a JSON payload from the first SRN API base that answers."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for base in SRN_API_BASES:
        url = base.rstrip("/") + path
        resp = _fetch_with_retries(url, headers, max_retries=1)
        if resp is None:
            continue
        try:
            return resp.json(), base
        except ValueError:
            log.debug("Non-JSON response from %s", url)
    return None, None


def _srn_items(payload):
    """API may return a bare list or wrap it in data/items/results."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("data", "items", "results", "documents", "companies"):
            if isinstance(payload.get(k), list):
                return payload[k]
    return []


def load_reports_srn(years):
    """Build the same dataframe shape as load_reports, but from the SRN API.

    Every report gets a report_year; page ranges are unknown, so the full
    document is extracted downstream."""
    log.info("Querying SRN API for report documents (years: %s) ...",
             ", ".join(str(y) for y in years))

    companies_raw, base = _srn_get_json("/companies")
    if companies_raw is None:
        log.error("Could not reach any SRN API endpoint (%s). "
                  "Check network access / endpoint list.", ", ".join(SRN_API_BASES))
        return pd.DataFrame()
    log.info("SRN API base: %s (%d companies)", base, len(_srn_items(companies_raw)))

    companies = {}
    for c in _srn_items(companies_raw):
        cid = _pick(c, "id", "company_id", "uuid")
        companies[cid] = {
            "company": _pick(c, "name", "company", "company_name", default=""),
            "isin": _pick(c, "isin", "ISIN", default=""),
            "country": _pick(c, "country", "country_name", default=""),
            "sector": _pick(c, "sector", "sics_sector", "SASB sector", default=""),
            "industry": _pick(c, "industry", "sics_industry", "SASB industry", default=""),
        }

    docs_raw, _ = _srn_get_json("/documents")
    docs = _srn_items(docs_raw) if docs_raw is not None else []
    if not docs:
        log.error("SRN API returned no documents.")
        return pd.DataFrame()

    types_seen = sorted({str(_pick(d, "type", "doc_type", "document_type", default="?"))
                         for d in docs})
    log.info("SRN document types seen: %s", ", ".join(types_seen))

    rows = []
    for d in docs:
        year = _pick(d, "year", "fiscal_year", "reporting_year")
        try:
            year = int(str(year)[:4])
        except (TypeError, ValueError):
            continue
        if year not in years:
            continue
        dtype = str(_pick(d, "type", "doc_type", "document_type", default="")).lower()
        # keep annual/sustainability/CSRD reports, skip presentations etc.
        if dtype and not any(t in dtype for t in ("ar", "annual", "sr", "sustain", "csrd", "report")):
            continue
        did = _pick(d, "id", "document_id", "uuid")
        href = _pick(d, "href", "url", "download_url", "link",
                     default=f"{base.rstrip('/')}/documents/{did}/download")
        meta = companies.get(_pick(d, "company_id", "company", "companyId"), {})
        rows.append({
            "company": meta.get("company") or _pick(d, "company_name", "name", default=str(did)),
            "isin": meta.get("isin", ""),
            "country": meta.get("country", ""),
            "SASB industry": meta.get("industry", ""),
            "SASB sector": meta.get("sector", ""),
            "link": href,
            "start PDF": pd.NA,
            "end PDF": pd.NA,
            "report_year": year,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["start PDF"] = df["start PDF"].astype("Int64")
        df["end PDF"] = df["end PDF"].astype("Int64")
        per_year = df["report_year"].value_counts().sort_index().to_dict()
        log.info("SRN reports selected: %d (%s)", len(df),
                 ", ".join(f"{y}: {n}" for y, n in per_year.items()))
    return df.reset_index(drop=True)


# 2 downloads pdfs ---
def _fetch_with_retries(url, headers, timeout=TIMEOUT, max_retries=MAX_RETRIES):
    """Fetch a URL with SSL fallback and retry on timeout/connection errors."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        for verify in (True, False):
            try:
                if not verify:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    log.info("  Retrying without SSL (attempt %d/%d)...", attempt, max_retries)
                resp = requests.get(url, headers=headers, timeout=timeout, verify=verify)
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

    headers = {"User-Agent": USER_AGENT}

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
                resp2 = _fetch_with_retries(pdf_url, headers)
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


def report_stem(row):
    """Filename stem for one report; includes the year when known so a
    company's 2024 and 2025 reports don't overwrite each other."""
    stem = f"{clean_filename(str(row['company']))}_{row['isin']}"
    year = row.get("report_year")
    if year is not None and not pd.isna(year):
        stem += f"_{int(year)}"
    return stem


def download_all(df):
    statuses = {}
    is_new = not os.path.exists(DOWNLOAD_LOG)

    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["index", "company", "isin", "url", "status", "dest"])

        for idx, row in df.iterrows():
            company = str(row["company"])
            isin = str(row["isin"])
            url = str(row["link"])
            dest = os.path.join(PDF_DIR, f"{report_stem(row)}.pdf")

            log.info("[%d/%d] %s", idx + 1, len(df), company)
            status = download_pdf(url, dest)
            statuses[idx] = status
            writer.writerow([idx, company, isin, url, status, dest])
            f.flush()

            if status == "success":
                log.info("  saved to %s", dest)
            if status != "skipped":
                time.sleep(DELAY)

    return statuses


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
        # prefix each page with a [[page:N]] marker so downstream ESRS
        # extraction (phase3) can cite page numbers; N is the 1-indexed PDF page
        body = doc[p].get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
        pages.append(f"[[page:{p + 1}]]\n{body}")
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


def process_one_pdf(pdf_path, stem, start, end):
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
    # count words on the marker-free text so [[page:N]] markers don't inflate it
    result["word_count"] = len(re.sub(r"\[\[page:\d+\]\]", " ", text).split())
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
            "report_year": row.get("report_year", ""),
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
    parser.add_argument("--source", choices=["excel", "srn"], default="excel",
                        help="excel: links from the SRN archive spreadsheet; "
                             "srn: report list + PDFs from the SRN API (srnav.com)")
    parser.add_argument("--years", default="2024,2025",
                        help="Fiscal years to fetch with --source srn (comma-separated)")
    parser.add_argument("--excel-file", default=EXCEL_FILE, help="Input Excel file")
    parser.add_argument("--limit", type=int, help="Only process first N reports")
    parser.add_argument("--start-from", type=int, default=0, help="Resume from this index")
    args = parser.parse_args()

    # create output folders
    for d in [PDF_DIR, TEXT_DIR, TABLE_DIR]:
        os.makedirs(d, exist_ok=True)

    # load and filter
    if args.source == "srn":
        years = {int(y) for y in args.years.split(",") if y.strip()}
        df = load_reports_srn(years)
    else:
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

    # download
    log.info("=" * 50)
    log.info("STEP 2: Downloading PDFs")
    log.info("=" * 50)
    download_statuses = download_all(df)

    # extract
    log.info("=" * 50)
    log.info("STEP 3: Extracting text and tables")
    log.info("=" * 50)
    extraction_results = {}

    for idx, row in df.iterrows():
        company = str(row["company"])
        stem = report_stem(row)
        pdf_path = os.path.join(PDF_DIR, f"{stem}.pdf")
        start = None if pd.isna(row["start PDF"]) else int(row["start PDF"])
        end = None if pd.isna(row["end PDF"]) else int(row["end PDF"])

        status = download_statuses.get(idx, "unknown")
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
        extraction_results[idx] = process_one_pdf(pdf_path, stem, start, end)
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
