#!/usr/bin/env python3
"""Live client for the SRN CSRD report archive (srnav.com).

The archive that phase1 used to read out of a downloaded spreadsheet is served
live at

    https://www.srnav.com/reports?referrer=google-sheet

That page is a SvelteKit app, so the report table is not in the HTML as markup —
it arrives as a hydration payload. Two ways to get it, tried in order:

  1. ``/reports/__data.json`` — the SvelteKit data endpoint for the same route.
     Returns the payload in *devalue* form (a flat array where every value is an
     index into that array), decoded by :func:`devalue_unflatten` below.
  2. The ``data:[...]`` literal embedded in the page's bootstrap ``<script>``,
     used as a fallback if the data endpoint ever moves or changes shape.

Either way the result is one record per CSRD report::

    {"id", "year", "type", "csrd_compliant", "csrd_report_number", "active",
     "pdfpage_sust_start", "pdfpage_sust_end", "original_link",
     "publication_date", "auditor",
     "company": {"id", "lei", "isin", "name", "sector", "country", "industry"}}

which is strictly richer than the spreadsheet: it carries the sustainability
statement page range (1987 of 1999 reports at time of writing), the CSRD
compliance flag, the assurance provider and the publication date.

Run standalone to inspect what the site is serving right now::

    python3 srn_client.py                 # counts by year / compliance / country
    python3 srn_client.py --dump out.json # full payload to disk
"""
from __future__ import annotations

import json
import logging
import re

import requests

REPORTS_URL = "https://www.srnav.com/reports?referrer=google-sheet"
REPORTS_DATA_URL = "https://www.srnav.com/reports/__data.json?referrer=google-sheet"

# Legacy read-only API (companies/documents across all years, not CSRD-scoped).
# Only the third host still answers; the srnav.com ones 404.
SRN_API_BASES = [
    "https://api.sustainabilityreportingnavigator.com/api",
    "https://api.srnav.com/api",
    "https://www.srnav.com/api",
]

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0.0.0 Safari/537.36")

log = logging.getLogger(__name__)


# --- devalue -----------------------------------------------------------------

# devalue's reserved negative indices (see sveltejs/devalue)
_DEVALUE_CONST = {
    -1: None,            # UNDEFINED
    -2: None,            # HOLE
    -3: float("nan"),
    -4: float("inf"),
    -5: float("-inf"),
    -6: -0.0,
}


def devalue_unflatten(values):
    """Rebuild a devalue-flattened payload.

    ``values`` is a flat list; index 0 is the root and every int is a pointer
    into the list, which is how SvelteKit dedupes the thousands of repeated
    company objects in the CSRD table. Memoised so shared nodes stay shared and
    self-referential payloads terminate.
    """
    if not isinstance(values, list) or not values:
        return None
    memo = {}

    def hydrate(idx):
        if not isinstance(idx, int) or isinstance(idx, bool):
            return idx
        if idx < 0:
            return _DEVALUE_CONST.get(idx)
        if idx in memo:
            return memo[idx]
        value = values[idx]

        if isinstance(value, list):
            # ["Date"|"Set"|"Map"|"BigInt"|..., ...] are devalue's tagged types
            tag = value[0] if value and isinstance(value[0], str) else None
            if tag == "Date":
                memo[idx] = value[1]
                return memo[idx]
            if tag == "Set":
                out = []
                memo[idx] = out
                out.extend(hydrate(i) for i in value[1:])
                return out
            if tag == "Map":
                out = {}
                memo[idx] = out
                for k, v in zip(value[1::2], value[2::2]):
                    out[hydrate(k)] = hydrate(v)
                return out
            if tag in ("BigInt", "RegExp", "Object", "null"):
                memo[idx] = hydrate(value[1]) if len(value) > 1 else None
                return memo[idx]
            out = []
            memo[idx] = out          # register before recursing (cycles)
            out.extend(hydrate(i) for i in value)
            return out

        if isinstance(value, dict):
            out = {}
            memo[idx] = out
            for key, i in value.items():
                out[key] = hydrate(i)
            return out

        memo[idx] = value
        return value

    return hydrate(0)


# --- fetching ----------------------------------------------------------------

def _session(session=None):
    if session is not None:
        return session
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _documents_from_nodes(payload):
    """Pull the ``documents`` list out of a decoded SvelteKit route payload."""
    for node in payload.get("nodes") or []:
        if not isinstance(node, dict) or node.get("type") != "data":
            continue
        data = devalue_unflatten(node.get("data"))
        if isinstance(data, dict) and isinstance(data.get("documents"), list):
            return data["documents"]
    return None


def _documents_from_html(html):
    """Fallback: the bootstrap script carries the same payload as a JS literal.

    It is JS, not JSON (bare keys, single quotes), so rather than parse it we
    slice out the ``documents:[ ... ]`` array and repair it into JSON.
    """
    start = html.find("documents:[")
    if start < 0:
        return None
    start += len("documents:")
    depth, end = 0, None
    for i, ch in enumerate(html[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    literal = html[start:end]
    # bare object keys -> quoted keys; `null`/`true`/`false` already match JSON
    literal = re.sub(r"([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', literal)
    try:
        return json.loads(literal)
    except json.JSONDecodeError as e:
        log.debug("HTML fallback parse failed: %s", e)
        return None


def fetch_csrd_reports(session=None, timeout=90):
    """Every CSRD report the SRN reports page is serving right now.

    Returns a list of report dicts (see module docstring). Raises
    :class:`RuntimeError` if neither the data endpoint nor the HTML fallback
    yields a report list.
    """
    s = _session(session)

    log.info("Fetching live CSRD report index from %s ...", REPORTS_DATA_URL)
    try:
        resp = s.get(REPORTS_DATA_URL, timeout=timeout,
                     headers={"Accept": "application/json"})
        resp.raise_for_status()
        docs = _documents_from_nodes(resp.json())
        if docs:
            log.info("Live index: %d CSRD reports", len(docs))
            return docs
        log.warning("Data endpoint returned no 'documents' node, falling back to HTML")
    except (requests.RequestException, ValueError) as e:
        log.warning("Data endpoint failed (%s), falling back to HTML", e)

    resp = s.get(REPORTS_URL, timeout=timeout)
    resp.raise_for_status()
    docs = _documents_from_html(resp.text)
    if not docs:
        raise RuntimeError(
            f"Could not extract the report list from {REPORTS_URL}. "
            "The site's payload format probably changed — check srn_client.py.")
    log.info("Live index (HTML fallback): %d CSRD reports", len(docs))
    return docs


def fetch_api_json(path, session=None, timeout=60):
    """GET a JSON payload from the first legacy SRN API base that answers."""
    s = _session(session)
    for base in SRN_API_BASES:
        url = base.rstrip("/") + path
        try:
            resp = s.get(url, timeout=timeout, headers={"Accept": "application/json"})
            resp.raise_for_status()
            return resp.json(), base
        except (requests.RequestException, ValueError) as e:
            log.debug("SRN API %s: %s", url, e)
    return None, None


def mirror_key(url):
    """Normalise a report URL for matching against SRN's cached copies."""
    url = re.sub(r"^https?://", "", (url or "").strip(), flags=re.IGNORECASE)
    return url.split("?")[0].rstrip("/").lower()


def fetch_mirror_index(session=None):
    """Map ``mirror_key(original_url) -> SRN mirror download URL``.

    SRN keeps its own cached copy of the documents in its older
    ``/api/documents`` table and serves them from
    ``/api/documents/{id}/download``, which sidesteps a dead company link or an
    IP-blocking WAF.

    Keyed on the **source URL**, deliberately, not on ``(isin, year)``: the
    CSRD archive's page ranges are measured against one specific file, and a
    same-company-same-year match could easily be a different document, which
    would silently extract the wrong pages. Matching the URL means the cached
    copy is the same file the range was measured against. That is a smaller
    net — roughly 150 of 2 000 reports — but it cannot mislead.

    Returns ``{}`` if the API can't be reached; the mirror is a bonus, never a
    requirement.
    """
    documents, base = fetch_api_json("/documents", session)
    if not documents:
        log.warning("SRN mirror index unavailable — continuing without it")
        return {}

    mirror = {}
    for d in documents:
        if not isinstance(d, dict) or not d.get("id"):
            continue
        key = mirror_key(d.get("href"))
        if key:
            mirror.setdefault(key, f"{base.rstrip('/')}/documents/{d['id']}/download")
    log.info("SRN mirror index: %d cached documents", len(mirror))
    return mirror


def _main():
    import argparse
    from collections import Counter

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    ap = argparse.ArgumentParser(description="Inspect the live SRN CSRD report index")
    ap.add_argument("--dump", help="Write the full JSON payload here")
    args = ap.parse_args()

    docs = fetch_csrd_reports()
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        log.info("Wrote %s", args.dump)

    linked = sum(1 for d in docs if str(d.get("original_link") or "").startswith("http"))
    ranged = sum(1 for d in docs if d.get("pdfpage_sust_start") and d.get("pdfpage_sust_end"))
    print(f"reports          {len(docs)}")
    print(f"with a link      {linked}")
    print(f"with page range  {ranged}")
    for label, key in (("year", "year"), ("csrd_compliant", "csrd_compliant"),
                       ("type", "type"), ("auditor", "auditor")):
        counts = Counter(str(d.get(key)) for d in docs).most_common(8)
        print(f"{label:16} " + ", ".join(f"{k}={v}" for k, v in counts))
    countries = Counter(str((d.get("company") or {}).get("country")) for d in docs)
    print(f"countries        {len(countries)}: "
          + ", ".join(f"{k}={v}" for k, v in countries.most_common(8)))


if __name__ == "__main__":
    _main()
