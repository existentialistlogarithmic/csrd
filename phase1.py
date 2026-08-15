#!/usr/bin/env python3
"""Phase 1 — build the CSRD corpus: index -> download -> text + tables.

Source of truth is the **live** SRN CSRD archive at
https://www.srnav.com/reports?referrer=google-sheet (see srn_client.py), so the
run always reflects what the site is publishing today rather than a spreadsheet
snapshot. Every report comes with the page range of its sustainability
statement, which is what makes this fast: a 125 MB annual report is parsed over
its ~90 relevant pages instead of all 400.

Speed comes from three places:
  * downloads run in a thread pool, throttled **per host** rather than globally,
    so ~1 500 distinct company web servers are hit concurrently while no single
    server sees more than one request at a time;
  * PDFs are streamed to disk, sniffed for the %PDF magic on the first chunk and
    abandoned early if the server answers with an HTML error page;
  * extraction runs in a process pool (PyMuPDF is CPU-bound and holds the GIL),
    opening each document exactly once for text, tables and page count.

Everything is resumable: an already-downloaded, valid PDF is skipped, and so is
a report whose text file is already newer than its PDF.

    python3 phase1.py                       # full live corpus
    python3 phase1.py --limit 20            # smoke test
    python3 phase1.py --years 2025          # one fiscal year
    python3 phase1.py --workers 24          # more download concurrency
    python3 phase1.py --no-tables           # text only (roughly 3x faster)
    python3 phase1.py --source excel        # legacy: links from the spreadsheet
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import json
import logging
import os
import re
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:                                  # PyMuPDF >= 1.24 prefers the new name
    import pymupdf
except ImportError:                   # pragma: no cover - older installs
    import fitz as pymupdf

from srn_client import USER_AGENT, fetch_csrd_reports

# --- config ------------------------------------------------------------------

EXCEL_GLOB = "SRN-CSRD_report_archive*.xlsx"
SHEET_NAME = "csrd"
HEADER_ROW = 2  # 0-indexed, so row 3 in the spreadsheet

PDF_DIR = "pdfs"
TEXT_DIR = "extracted_text"
TABLE_DIR = "extracted_tables"
DOWNLOAD_LOG = "download_log.csv"
SUMMARY_CSV = "extraction_summary.csv"
INDEX_CSV = "reports_index.csv"

DOWNLOAD_WORKERS = 12    # concurrent downloads across all hosts
HOST_DELAY = 1.0         # min seconds between two requests to the same host
TIMEOUT = (15, 120)      # (connect, read) seconds
MAX_RETRIES = 3
MIN_PDF_BYTES = 10_000   # anything smaller is an error page, not a report
PDF_SNIFF_BYTES = 1024   # the %PDF header must appear within this many bytes
MAX_PDF_MB = 500         # refuse absurd payloads rather than fill the disk
CHUNK = 1 << 18          # 256 KiB streaming chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def clean_filename(name):
    """Remove special chars, replace spaces with underscores."""
    name = re.sub(r"[^\w\s-]", "", str(name))
    return re.sub(r"\s+", "_", name.strip())[:80]


def _as_int(value):
    try:
        s = str(value).strip()
        return int(float(s)) if s and s.lower() not in ("nan", "none", "<na>") else None
    except (TypeError, ValueError):
        return None


# --- 1. index ----------------------------------------------------------------

def load_reports_live(years=None):
    """The live CSRD archive, normalised to one row per report."""
    rows = []
    for d in fetch_csrd_reports():
        year = _as_int(d.get("year"))
        if years and year not in years:
            continue
        company = d.get("company") or {}
        rows.append({
            "doc_id": d.get("id", ""),
            "company": company.get("name") or "unknown",
            "isin": company.get("isin") or "",
            "lei": company.get("lei") or "",
            "country": company.get("country") or "",
            "SASB sector": company.get("sector") or "",
            "SASB industry": company.get("industry") or "",
            "report_year": year,
            "doc_type": d.get("type") or "",
            "csrd_compliant": d.get("csrd_compliant") or "",
            "csrd_report_number": d.get("csrd_report_number") or "",
            "auditor": d.get("auditor") or "",
            "publication_date": d.get("publication_date") or "",
            "link": (d.get("original_link") or "").strip(),
            "start PDF": _as_int(d.get("pdfpage_sust_start")),
            "end PDF": _as_int(d.get("pdfpage_sust_end")),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["link"].str.startswith("http")].copy()
    # the same PDF can back two rows (e.g. a combined 2024/2025 filing)
    before = len(df)
    df = df.drop_duplicates(subset=["link", "start PDF", "end PDF"]).reset_index(drop=True)
    ranged = int((df["start PDF"].notna() & df["end PDF"].notna()).sum())
    log.info("Live index: %d reports with a link (%d deduped), %d with a page range",
             len(df), before - len(df), ranged)
    log.info("  by year: %s", ", ".join(
        f"{y}: {n}" for y, n in sorted(df["report_year"].value_counts().items())))
    return df


def load_reports_excel(path=None, years=None):
    """Legacy source: links from a downloaded copy of the archive spreadsheet."""
    if not path:
        matches = sorted(glob.glob(EXCEL_GLOB))
        if not matches:
            raise SystemExit(f"No spreadsheet matching {EXCEL_GLOB} — use the default "
                             f"--source live, or pass --excel-file")
        path = matches[-1]
    log.info("Reading %s ...", path)
    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW)
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]

    df = df[df["link"].astype(str).str.strip().str.startswith("http")].copy()
    df["link"] = df["link"].astype(str).str.strip()
    df["start PDF"] = df["start PDF"].map(_as_int)
    df["end PDF"] = df["end PDF"].map(_as_int)
    if "report_year" not in df.columns:
        for candidate in ("year", "Year", "fiscal year"):
            if candidate in df.columns:
                df["report_year"] = df[candidate].map(_as_int)
                break
        else:
            df["report_year"] = None
    if years:
        df = df[df["report_year"].isin(years) | df["report_year"].isna()]
    # the spreadsheet predates most of the live index's columns; fill the gaps
    # so the rest of the pipeline sees one shape regardless of source
    for col in ("doc_id", "lei", "doc_type", "csrd_compliant", "csrd_report_number",
                "auditor", "publication_date", "country", "SASB sector", "SASB industry"):
        if col not in df.columns:
            df[col] = ""
    log.info("Spreadsheet rows with a link: %d", len(df))
    return df.reset_index(drop=True)


def report_stem(row):
    """Filename stem; the year keeps a company's 2024 and 2025 filings apart."""
    stem = f"{clean_filename(row['company'])}_{row.get('isin') or 'noisin'}"
    year = row.get("report_year")
    if year not in (None, "") and not pd.isna(year):
        stem += f"_{int(year)}"
    start, end = _as_int(row.get("start PDF")), _as_int(row.get("end PDF"))
    if start and end:
        stem += f"_p{start}-{end}"
    return stem


# --- 2. download -------------------------------------------------------------

class HostThrottle:
    """One in-flight request per host, spaced by at least ``delay`` seconds.

    Politeness lives at the host level, not globally: hammering one company's
    web server is rude, but 1 500 different servers in parallel is not.
    """

    def __init__(self, delay=HOST_DELAY):
        self.delay = delay
        self._locks = defaultdict(threading.Lock)
        self._last = defaultdict(float)
        self._guard = threading.Lock()

    def lock_for(self, url):
        host = (urlparse(url).hostname or "").lower()
        with self._guard:
            return host, self._locks[host]

    def wait(self, host):
        gap = self.delay - (time.monotonic() - self._last[host])
        if gap > 0:
            time.sleep(gap)

    def done(self, host):
        self._last[host] = time.monotonic()


_thread_local = threading.local()


def get_session():
    """One pooled, retrying Session per worker thread."""
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": USER_AGENT,
            # some CDNs (Akamai/Cloudflare) 403 requests that look non-browser
            "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        retry = Retry(total=MAX_RETRIES, backoff_factor=1.5,
                      status_forcelist=(408, 425, 429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET", "HEAD"]),
                      respect_retry_after_header=True)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _thread_local.session = s
    return s


def _get(url, verify=True, referer=None):
    headers = {"Referer": referer} if referer else None
    return get_session().get(url, timeout=TIMEOUT, stream=True, headers=headers,
                             verify=verify, allow_redirects=True)


def _get_tolerant(url, verify=True):
    """GET, retrying a 401/403 once with a same-origin Referer.

    Several IR sites sit behind a WAF that rejects a bare request for a PDF but
    serves it happily when it looks like a click from the company's own page.
    """
    resp = _get(url, verify=verify)
    if resp.status_code in (401, 403):
        parts = urlparse(url)
        resp.close()
        resp = _get(url, verify=verify, referer=f"{parts.scheme}://{parts.netloc}/")
    return resp


def _read_capped(resp, limit):
    """First ``limit`` bytes of a streamed body, decoded as text."""
    buf = bytearray()
    for chunk in resp.iter_content(CHUNK):
        buf += chunk
        if len(buf) >= limit:
            break
    return buf.decode(resp.encoding or "utf-8", errors="ignore")


def _find_pdf_link_in_html(html, base_url):
    """Some links point at a landing page; find the report PDF on it."""
    links = re.findall(r'href=["\']([^"\']*\.pdf(?:\?[^"\']*)?)["\']', html, re.IGNORECASE)
    if not links:
        return None
    keywords = ("annual", "report", "sustainability", "csrd", "esg", "nachhaltig")
    for link in links:
        if any(kw in link.lower() for kw in keywords):
            return urljoin(base_url, link)
    return urljoin(base_url, links[0])


def _stream_to_file(resp, dest):
    """Write the body to ``dest`` via a .part file. Returns (bytes, error)."""
    part = dest + ".part"
    limit = MAX_PDF_MB * 1024 * 1024
    total = 0
    head, sniffed = b"", False
    try:
        with open(part, "wb") as f:
            for chunk in resp.iter_content(CHUNK):
                if not chunk:
                    continue
                if not sniffed:
                    # the header sits in the first KB (the spec tolerates a
                    # preamble); no %PDF there means an error page, so bail
                    # before pulling the other 99 MB
                    head += chunk[:PDF_SNIFF_BYTES]
                    if len(head) >= PDF_SNIFF_BYTES:
                        if b"%PDF" not in head:
                            return 0, "not_pdf"
                        sniffed = True
                total += len(chunk)
                if total > limit:
                    return total, f"too_large_over_{MAX_PDF_MB}mb"
                f.write(chunk)
        if b"%PDF" not in head:
            return total, "not_pdf"
        if total < MIN_PDF_BYTES:
            return total, "file_too_small"
        os.replace(part, dest)
        return total, None
    finally:
        if os.path.exists(part):
            os.remove(part)


def _valid_pdf_on_disk(path):
    try:
        if os.path.getsize(path) < MIN_PDF_BYTES:
            return False
        with open(path, "rb") as f:
            return b"%PDF" in f.read(PDF_SNIFF_BYTES)
    except OSError:
        return False


def download_one(url, dest, throttle, refresh=False):
    """Fetch one report PDF. Returns (status, bytes, final_url)."""
    if not refresh and _valid_pdf_on_disk(dest):
        return "skipped", os.path.getsize(dest), url

    host, lock = throttle.lock_for(url)
    with lock:
        throttle.wait(host)
        try:
            try:
                resp = _get_tolerant(url)
            except requests.exceptions.SSLError:
                # a surprising number of IR sites have broken chains
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                resp = _get_tolerant(url, verify=False)

            with resp:
                if resp.status_code >= 400:
                    return f"error:http_{resp.status_code}", 0, resp.url
                ctype = resp.headers.get("Content-Type", "").lower()

                if "html" in ctype:
                    pdf_url = _find_pdf_link_in_html(_read_capped(resp, 400_000), resp.url)
                    if not pdf_url:
                        return "not_pdf", 0, resp.url
                    resp2 = _get_tolerant(pdf_url)
                    with resp2:
                        if resp2.status_code >= 400:
                            return f"error:http_{resp2.status_code}", 0, pdf_url
                        size, err = _stream_to_file(resp2, dest)
                        return (f"error:{err}" if err else "success"), size, pdf_url

                size, err = _stream_to_file(resp, dest)
                return (f"error:{err}" if err else "success"), size, resp.url
        except requests.exceptions.Timeout:
            return "error:timeout", 0, url
        except requests.exceptions.ConnectionError as e:
            return f"error:connection:{type(e).__name__}", 0, url
        except requests.RequestException as e:
            return f"error:{type(e).__name__}", 0, url
        finally:
            throttle.done(host)


def _interleave_by_host(jobs):
    """Round-robin the queue across hosts so workers rarely queue on one lock."""
    buckets = defaultdict(deque)
    for job in jobs:
        buckets[(urlparse(job["url"]).hostname or "").lower()].append(job)
    order, queues = [], list(buckets.values())
    while queues:
        queues = [q for q in queues if q]
        for q in list(queues):
            order.append(q.popleft())
    return order


def download_all(jobs, workers=DOWNLOAD_WORKERS, refresh=False, host_delay=HOST_DELAY):
    """Download every job's PDF concurrently. Returns {stem: (status, bytes)}."""
    throttle = HostThrottle(host_delay)
    results = {}
    started = time.monotonic()
    is_new = not os.path.exists(DOWNLOAD_LOG)

    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["stem", "company", "isin", "url", "final_url",
                             "status", "bytes", "dest"])
        log_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download_one, job["url"], job["dest"], throttle, refresh): job
                for job in _interleave_by_host(jobs)
            }
            for n, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                try:
                    status, size, final_url = future.result()
                except Exception as e:               # noqa: BLE001 - never kill the run
                    status, size, final_url = f"error:{type(e).__name__}", 0, job["url"]
                    log.debug("%s: %s", job["stem"], e)
                results[job["stem"]] = (status, size)

                with log_lock:
                    writer.writerow([job["stem"], job["company"], job["isin"], job["url"],
                                     final_url, status, size, job["dest"]])
                    f.flush()
                if status == "success":
                    log.info("[%d/%d] %s (%.1f MB)", n, len(jobs), job["company"],
                             size / 1e6)
                elif status == "skipped":
                    log.debug("[%d/%d] %s (cached)", n, len(jobs), job["company"])
                else:
                    log.warning("[%d/%d] %s -> %s", n, len(jobs), job["company"], status)

    elapsed = time.monotonic() - started
    ok = sum(1 for s, _ in results.values() if s in ("success", "skipped"))
    log.info("Downloads finished in %.1f min: %d/%d usable (%.1f reports/min)",
             elapsed / 60, ok, len(jobs), len(jobs) / max(elapsed / 60, 1e-9))
    return results


# --- 3. extraction (text + tables) -------------------------------------------

# Function words per language — matched as whole words, not substrings, which
# is the whole trick: "e" is Italian but also lives inside every English word.
# Enough to separate the 20-odd languages in the archive without a native
# dependency; langdetect is used instead whenever it is installed.
_LANG_MARKERS = {
    "en": "the and of to in for with that this are was our we is on as by which",
    "de": "der die das und von zu für mit im den des ist auf sich nicht werden",
    "fr": "le la les des et de pour dans que est aux sur par une nos être",
    "nl": "de het een en van voor met dat is op zijn niet aan door worden",
    "es": "el la los las de y para con que es en por una del sus",
    "it": "il la le di e per con che sono del nel una degli anche",
    "sv": "och att det som en av för med den till inte har vi är",
    "da": "og at det som en af for med den til ikke har vi er",
    "no": "og at det som en av for med den til ikke har vi er",
    "fi": "ja on ei että se voi ovat myös kuin sen mutta oli sekä",
    "pt": "de que os as do da para com uma não em pelo seu",
    "pl": "i w z na do nie się jest oraz przez które lub dla",
    "cs": "a v na se je že pro do od které nebo být jsou",
    "el": "και το της των στο με για από που είναι στην τον",
}
_LANG_SETS = {lang: frozenset(words.split()) for lang, words in _LANG_MARKERS.items()}

try:
    from langdetect import DetectorFactory, LangDetectException, detect as _langdetect
    DetectorFactory.seed = 0
except ImportError:                    # pragma: no cover - optional dependency
    _langdetect = None


def detect_language(text):
    """'en', 'de', ... or 'unknown'.

    Samples the start, middle and end so a translated cover page can't fool the
    detector. Falls back to stopword frequency when langdetect isn't installed —
    which is also what makes this cheap enough to run on 2 000 reports.
    """
    if not text or not text.strip():
        return "unknown"
    n = len(text)
    sample = text[:4000]
    if n > 12000:
        mid = n // 2
        sample += "\n" + text[mid:mid + 4000] + "\n" + text[-4000:]

    if _langdetect is not None:
        try:
            return _langdetect(sample)
        except LangDetectException:
            return "unknown"

    words = re.findall(r"[^\W\d_]+", sample.lower(), re.UNICODE)
    if len(words) < 40:
        return "unknown"
    counts = Counter(words)
    total = len(words)
    # share of the sample made up of each language's function words
    scores = {lang: sum(counts[w] for w in markers) / total
              for lang, markers in _LANG_SETS.items()}
    lang, share = max(scores.items(), key=lambda kv: kv[1])
    # a real match runs 15-30%; below 4% we are just seeing coincidences
    return lang if share >= 0.04 else "unknown"


_WORD_CELL = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def _useful_table(rows):
    """Filter out the table finder's false positives.

    ``find_tables`` reads chart axes and page furniture as grids, which produces
    a lot of 4x4 blocks of bare tick labels. A real disclosure table has at
    least one word in it, and more than one row and column.
    """
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return False
    cells = [c for row in rows for c in row if c]
    if len(cells) < 4:
        return False
    return any(_WORD_CELL.search(c) for c in cells)


def extract_one(task):
    """Text + tables for one PDF, with the table finder's stdout hints muted.

    Runs in a worker process, so it takes and returns plain data only — and
    since nothing here legitimately prints, swallowing stdout is safe.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return _extract_one(task)


def _extract_one(task):
    """The document is opened **once** and each page is visited **once** for
    both text and tables — the old two-pass version paid the render cost twice.
    """
    pdf_path = task["pdf_path"]
    stem = task["stem"]
    result = {
        "stem": stem, "text_file": None, "table_file": None, "word_count": 0,
        "pages_extracted": 0, "pdf_pages": 0, "tables_found": 0,
        "language": "unknown", "extraction_status": "success", "error": "",
    }

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:                            # noqa: BLE001
        result.update(extraction_status="failed", error=f"open: {e}")
        return result

    try:
        total = doc.page_count
        result["pdf_pages"] = total
        start = task["start"] or 1
        end = task["end"] or total
        # a page range from the archive can outrun a re-issued PDF
        start = max(1, min(start, total))
        end = max(start, min(end, total))

        chunks, tables = [], []
        for p in range(start - 1, end):
            page = doc[p]
            # [[page:N]] markers let phase 3 cite the PDF page an answer came from
            chunks.append(f"[[page:{p + 1}]]\n"
                          + page.get_text("text", flags=pymupdf.TEXT_PRESERVE_LIGATURES))
            if task["tables"]:
                try:
                    for table in page.find_tables().tables:
                        rows = [[c if c is not None else "" for c in row]
                                for row in table.extract()]
                        if rows and (task["all_tables"] or _useful_table(rows)):
                            tables.append({"page": p + 1, "rows": rows})
                except Exception:                     # noqa: BLE001
                    pass                              # a bad page shouldn't sink the doc
    except Exception as e:                            # noqa: BLE001
        doc.close()
        result.update(extraction_status="failed", error=f"parse: {e}")
        return result
    doc.close()

    text = "\n\n".join(chunks)
    result["pages_extracted"] = end - start + 1
    # count words on marker-free text so [[page:N]] doesn't inflate the total
    result["word_count"] = len(re.sub(r"\[\[page:\d+\]\]", " ", text).split())
    result["language"] = detect_language(text)

    text_file = os.path.join(task["text_dir"], f"{stem}.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)
    result["text_file"] = text_file

    result["tables_found"] = len(tables)
    if tables:
        table_file = os.path.join(task["table_dir"], f"{stem}_tables.json")
        with open(table_file, "w", encoding="utf-8") as f:
            json.dump(tables, f, ensure_ascii=False)
        result["table_file"] = table_file
    return result


def extract_all(tasks, workers):
    """Run extraction across a process pool; returns {stem: result}."""
    results = {}
    if not tasks:
        return results
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, t): t for t in tasks}
        for n, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                r = future.result()
            except Exception as e:                    # noqa: BLE001
                r = {"stem": task["stem"], "extraction_status": "failed",
                     "error": str(e), "language": "unknown", "pages_extracted": 0,
                     "word_count": 0, "tables_found": 0, "pdf_pages": 0,
                     "text_file": None, "table_file": None}
            results[r["stem"]] = r
            if r["extraction_status"] == "success":
                log.info("[%d/%d] %s — %d pages, %d words, %d tables, %s",
                         n, len(tasks), r["stem"], r["pages_extracted"],
                         r["word_count"], r["tables_found"], r["language"])
            else:
                log.warning("[%d/%d] %s — extraction failed: %s",
                            n, len(tasks), r["stem"], r.get("error", ""))
    log.info("Extraction finished in %.1f min (%d documents, %d workers)",
             (time.monotonic() - started) / 60, len(tasks), workers)
    return results


# --- 4. summary --------------------------------------------------------------

def build_summary(df, downloads, extractions, path=SUMMARY_CSV):
    rows = []
    for _, row in df.iterrows():
        stem = row["_stem"]
        status, size = downloads.get(stem, ("not_attempted", 0))
        ext = extractions.get(stem, {})
        rows.append({
            "company": row["company"],
            "isin": row.get("isin", ""),
            "report_year": row.get("report_year", ""),
            "country": row.get("country", ""),
            "industry": row.get("SASB industry", ""),
            "sector": row.get("SASB sector", ""),
            "doc_type": row.get("doc_type", ""),
            "csrd_compliant": row.get("csrd_compliant", ""),
            "auditor": row.get("auditor", ""),
            "publication_date": row.get("publication_date", ""),
            "page_start": row.get("start PDF") if pd.notna(row.get("start PDF")) else "",
            "page_end": row.get("end PDF") if pd.notna(row.get("end PDF")) else "",
            "source_url": row.get("link", ""),
            "download_status": status,
            "pdf_bytes": size,
            "extraction_status": ext.get("extraction_status", "not_attempted"),
            "language": ext.get("language", "unknown"),
            "pdf_pages": ext.get("pdf_pages", 0),
            "pages_extracted": ext.get("pages_extracted", 0),
            "word_count": ext.get("word_count", 0),
            "tables_found": ext.get("tables_found", 0),
            "text_file": ext.get("text_file") or "",
            "table_file": ext.get("table_file") or "",
        })
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    log.info("Summary saved to %s (%d rows)", path, len(out))
    return out


def report_totals(summary):
    dl = summary["download_status"].map(
        lambda s: "ok" if s in ("success", "skipped") else "failed").value_counts()
    log.info("Downloads : %d ok, %d failed", dl.get("ok", 0), dl.get("failed", 0))
    ex = summary["extraction_status"].value_counts()
    log.info("Extraction: %s", ", ".join(f"{k}={v}" for k, v in ex.items()))
    done = summary[summary["extraction_status"] == "success"]
    if not done.empty:
        log.info("Corpus    : %s words over %s pages, %s tables",
                 f"{int(done['word_count'].sum()):,}",
                 f"{int(done['pages_extracted'].sum()):,}",
                 f"{int(done['tables_found'].sum()):,}")
        langs = done["language"].value_counts().head(6)
        log.info("Languages : %s", ", ".join(f"{k}={v}" for k, v in langs.items()))
        failed = summary[~summary["download_status"].isin(("success", "skipped"))]
        if not failed.empty:
            top = failed["download_status"].value_counts().head(5)
            log.info("Top download failures: %s",
                     ", ".join(f"{k}={v}" for k, v in top.items()))


# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CSRD phase 1 — download and parse the corpus")
    ap.add_argument("--source", choices=["live", "excel"], default="live",
                    help="live: srnav.com/reports (default); excel: archive spreadsheet")
    ap.add_argument("--excel-file", help=f"Spreadsheet for --source excel (default: {EXCEL_GLOB})")
    ap.add_argument("--years", default="2024,2025",
                    help="Fiscal years to keep, comma-separated ('all' for every year)")
    ap.add_argument("--limit", type=int, help="Only process the first N reports")
    ap.add_argument("--start-from", type=int, default=0, help="Skip the first N reports")
    ap.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS,
                    help="Concurrent downloads")
    ap.add_argument("--extract-workers", type=int, default=max(1, (os.cpu_count() or 2)),
                    help="Extraction processes (default: CPU count)")
    ap.add_argument("--host-delay", type=float, default=HOST_DELAY,
                    help="Min seconds between requests to the same host")
    ap.add_argument("--no-tables", action="store_true", help="Skip table detection")
    ap.add_argument("--all-tables", action="store_true",
                    help="Keep every detected table, including chart-axis false positives")
    ap.add_argument("--full-pages", action="store_true",
                    help="Extract whole documents, ignoring the sustainability page range")
    ap.add_argument("--refresh", action="store_true", help="Re-download PDFs already on disk")
    ap.add_argument("--re-extract", action="store_true",
                    help="Re-parse PDFs whose text file already exists")
    ap.add_argument("--skip-download", action="store_true",
                    help="Only extract from PDFs already in pdfs/")
    args = ap.parse_args()

    for d in (PDF_DIR, TEXT_DIR, TABLE_DIR):
        os.makedirs(d, exist_ok=True)

    years = None
    if args.years.strip().lower() != "all":
        years = {int(y) for y in args.years.split(",") if y.strip()}

    # 1. index
    log.info("=" * 60)
    log.info("STEP 1: Building the report index (source: %s)", args.source)
    log.info("=" * 60)
    df = (load_reports_live(years) if args.source == "live"
          else load_reports_excel(args.excel_file, years))
    if df.empty:
        log.error("No reports to process.")
        return

    if args.start_from:
        df = df.iloc[args.start_from:]
    if args.limit is not None:
        df = df.iloc[:args.limit]
    df = df.reset_index(drop=True)
    df["_stem"] = [report_stem(r) for _, r in df.iterrows()]
    # two rows can still collide on the stem (same company, year and range)
    df = df.drop_duplicates(subset=["_stem"]).reset_index(drop=True)
    df.drop(columns=["_stem"]).to_csv(INDEX_CSV, index=False)
    log.info("Processing %d reports (index written to %s)", len(df), INDEX_CSV)

    # 2. download
    log.info("=" * 60)
    log.info("STEP 2: Downloading PDFs (%d workers, %.1fs per host)",
             args.workers, args.host_delay)
    log.info("=" * 60)
    jobs = [{"stem": r["_stem"], "company": r["company"], "isin": r.get("isin", ""),
             "url": r["link"], "dest": os.path.join(PDF_DIR, f"{r['_stem']}.pdf")}
            for _, r in df.iterrows()]
    if args.skip_download:
        downloads = {j["stem"]: (("skipped", os.path.getsize(j["dest"]))
                                 if _valid_pdf_on_disk(j["dest"])
                                 else ("error:missing", 0)) for j in jobs}
        log.info("Skipping downloads; %d PDFs already on disk",
                 sum(1 for s, _ in downloads.values() if s == "skipped"))
    else:
        downloads = download_all(jobs, args.workers, args.refresh, args.host_delay)

    # 3. extract
    log.info("=" * 60)
    log.info("STEP 3: Extracting text%s (%d processes)",
             "" if args.no_tables else " and tables", args.extract_workers)
    log.info("=" * 60)
    tasks, cached = [], {}
    for _, row in df.iterrows():
        stem = row["_stem"]
        if downloads.get(stem, ("", 0))[0] not in ("success", "skipped"):
            continue
        pdf_path = os.path.join(PDF_DIR, f"{stem}.pdf")
        if not os.path.exists(pdf_path):
            continue
        text_path = os.path.join(TEXT_DIR, f"{stem}.txt")
        if not args.re_extract and os.path.exists(text_path) \
                and os.path.getmtime(text_path) >= os.path.getmtime(pdf_path):
            table_path = os.path.join(TABLE_DIR, f"{stem}_tables.json")
            body = open(text_path, encoding="utf-8").read()
            n_tables = 0
            if os.path.exists(table_path):
                with open(table_path, encoding="utf-8") as f:
                    n_tables = len(json.load(f))
            cached[stem] = {
                "stem": stem, "extraction_status": "success", "error": "",
                "language": detect_language(body),
                "pages_extracted": body.count("[[page:"),
                "pdf_pages": 0,
                "word_count": len(re.sub(r"\[\[page:\d+\]\]", " ", body).split()),
                "tables_found": n_tables, "text_file": text_path,
                "table_file": table_path if n_tables else None,
            }
            continue
        tasks.append({
            "pdf_path": pdf_path, "stem": stem,
            "start": None if args.full_pages else _as_int(row.get("start PDF")),
            "end": None if args.full_pages else _as_int(row.get("end PDF")),
            "tables": not args.no_tables, "all_tables": args.all_tables,
            "text_dir": TEXT_DIR, "table_dir": TABLE_DIR,
        })
    if cached:
        log.info("%d document(s) already parsed, reusing", len(cached))
    extractions = extract_all(tasks, args.extract_workers)
    extractions.update(cached)

    # 4. summary
    log.info("=" * 60)
    log.info("STEP 4: Summary")
    log.info("=" * 60)
    summary = build_summary(df, downloads, extractions)
    report_totals(summary)


if __name__ == "__main__":
    main()
