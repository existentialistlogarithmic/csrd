#!/usr/bin/env python3
"""Phase 3 — ESRS disclosure-requirement extraction (Anthropic API).

Maps each English report in the corpus onto the ESRS Set 1 taxonomy defined in
``esrs_system_prompt.md``: the materiality assessment, a status per disclosure
requirement (reported / not-material / missing / phase-in), quantitative
datapoints with units and baselines, and quality-and-greenwashing flags — every
finding grounded in a quoted evidence span with the PDF page it came from
(phase 1 writes ``[[page:N]]`` markers into the text for exactly this).

This module is also the shared spine for the other phase-3 backends:
``phase3_local.py`` (Ollama / vLLM / Groq / any OpenAI-compatible server) and
``phase2b_hf.py`` (local Hugging Face scoring) import the report selection, text
loading, JSON repair and flattening helpers from here so all three write the
same shape of output.

    export ANTHROPIC_API_KEY=...
    python3 phase3_esrs.py --limit 5
    python3 phase3_esrs.py --model claude-sonnet-5
    python3 phase3_esrs.py --chunk-chars 300000     # split very long reports
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time

import pandas as pd

SUMMARY_CSV = "extraction_summary.csv"
TEXT_DIR = "extracted_text"
OUT_DIR = "esrs_output"
OUT_CSV = "esrs_coverage.csv"
FAIL_CSV = "esrs_failures.csv"
PROMPT_FILE = "esrs_system_prompt.md"

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 16000

# the five statuses the system prompt's step 4 allows
STATUSES = (
    "reported",
    "not_material_omitted",
    "material_not_reported",
    "not_addressed",
    "phase_in_deferred",
)

# ESRS 2 is mandatory for everyone, so its coverage is the one hard pass/fail
ESRS2_CODES = frozenset({
    "BP-1", "BP-2",
    "GOV-1", "GOV-2", "GOV-3", "GOV-4", "GOV-5",
    "SBM-1", "SBM-2", "SBM-3",
    "IRO-1", "IRO-2",
})

TOPICAL_STANDARDS = ("E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# --- shared helpers (imported by phase2b_hf and phase3_local) ----------------

def load_prompt(path=PROMPT_FILE):
    if not os.path.exists(path):
        raise SystemExit(f"System prompt not found: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_text(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def robust_json(raw):
    """Parse a model's answer into a dict, tolerating the usual sloppiness.

    Models wrap JSON in ```json fences, prepend "Here is the analysis:", or run
    out of tokens mid-object. Try the strict parse, then the fenced block, then
    the outermost braces, then a trailing-comma / truncation repair.
    """
    if isinstance(raw, dict):
        return raw
    if not raw or not raw.strip():
        raise ValueError("empty model response")
    text = raw.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            text = fenced.group(1).strip()

    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    candidate = text[start:text.rfind("}") + 1] or text[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # last resort: drop trailing commas, then close whatever is still open
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    depth_curly = repaired.count("{") - repaired.count("}")
    depth_square = repaired.count("[") - repaired.count("]")
    if repaired.rstrip().endswith(","):
        repaired = repaired.rstrip().rstrip(",")
    repaired += "]" * max(0, depth_square) + "}" * max(0, depth_curly)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(f"unparseable model response ({e}): {text[:200]!r}") from e


def flatten(meta, result):
    """One CSV row per report: metadata + coverage counters.

    Keeps the per-status counts, the per-standard reported counts, the ESRS 2
    completeness flag and the quality-flag tally, so ``esrs_coverage.csv`` is
    directly pivotable without re-reading the per-report JSON.
    """
    disclosures = result.get("disclosures") or []
    materiality = result.get("materiality_assessment") or {}
    coverage = result.get("coverage_summary") or {}
    doc_meta = result.get("document_meta") or {}

    row = dict(meta)
    row["n_disclosures"] = len(disclosures)
    for status in STATUSES:
        row[f"n_{status}"] = sum(1 for d in disclosures if d.get("status") == status)

    reported = {d.get("dr_code") for d in disclosures if d.get("status") == "reported"}
    row["esrs2_reported"] = len(ESRS2_CODES & reported)
    row["esrs2_total"] = len(ESRS2_CODES)
    row["esrs2_complete"] = bool(coverage.get("esrs2_complete",
                                              ESRS2_CODES.issubset(reported)))
    for standard in TOPICAL_STANDARDS:
        row[f"{standard}_reported"] = sum(
            1 for d in disclosures
            if d.get("status") == "reported"
            and str(d.get("standard") or d.get("dr_code") or "").startswith(standard))

    row["material_topics"] = "|".join(materiality.get("material_topics") or [])
    row["non_material_topics"] = "|".join(materiality.get("non_material_topics") or [])
    row["materiality_method_described"] = bool(materiality.get("method_described"))
    row["phase_ins_invoked"] = "|".join(str(x) for x in (coverage.get("phase_ins_invoked") or []))

    row["n_datapoints"] = sum(len(d.get("datapoints") or []) for d in disclosures)
    flags = [f for d in disclosures for f in (d.get("quality_flags") or [])]
    row["n_quality_flags"] = len(flags)
    row["quality_flags"] = "|".join(sorted(set(flags)))

    confidences = [d["confidence"] for d in disclosures
                   if isinstance(d.get("confidence"), (int, float))]
    row["avg_confidence"] = round(sum(confidences) / len(confidences), 3) if confidences else ""
    row["assurance"] = doc_meta.get("assurance", "unknown")
    row["iro2_index_present"] = bool(doc_meta.get("iro2_index_present"))
    return row


def select_reports(args):
    """The reports still to process, as jobs for whichever backend is running.

    Each job is ``{"stem", "text_path", "out_path", "meta"}``. Only successfully
    extracted English reports are eligible — the ESRS prompt is English-only —
    and anything already written to ``--out-dir`` is skipped unless
    ``--overwrite`` is set.
    """
    df = pd.read_csv(args.summary)
    total = len(df)

    if "extraction_status" in df.columns:
        df = df[df["extraction_status"] == "success"]
    if "language" in df.columns:
        df = df[df["language"] == "en"]
    else:
        log.warning("No 'language' column in %s — re-run phase1.py for language "
                    "detection; processing every report", args.summary)
    df = df[df["text_file"].astype(str).str.strip() != ""]
    log.info("%d of %d rows are extracted English reports", len(df), total)

    text_dir = getattr(args, "text_dir", TEXT_DIR)
    out_dir = getattr(args, "out_dir", OUT_DIR)
    overwrite = getattr(args, "overwrite", False)

    jobs = []
    for _, row in df.iterrows():
        text_path = str(row["text_file"])
        if not os.path.exists(text_path):
            # summary may carry a path from another machine; fall back to the dir
            text_path = os.path.join(text_dir, os.path.basename(text_path))
            if not os.path.exists(text_path):
                log.debug("Missing text file for %s", row.get("company"))
                continue
        stem = os.path.splitext(os.path.basename(text_path))[0]
        out_path = os.path.join(out_dir, f"{stem}.json")
        if not overwrite and os.path.exists(out_path):
            continue
        jobs.append({
            "stem": stem,
            "text_path": text_path,
            "out_path": out_path,
            "meta": {
                "stem": stem,
                "company": row.get("company", ""),
                "isin": row.get("isin", ""),
                "report_year": row.get("report_year", ""),
                "country": row.get("country", ""),
                "sector": row.get("sector", ""),
                "industry": row.get("industry", ""),
                "auditor": row.get("auditor", ""),
                "csrd_compliant": row.get("csrd_compliant", ""),
                "word_count": row.get("word_count", ""),
            },
        })

    limit = getattr(args, "limit", None)
    if limit:
        jobs = jobs[:limit]
    return jobs


def chunk_text(text, max_chars):
    """Split on ``[[page:N]]`` markers, greedily packing chunks under max_chars,
    so a 500-page report fits a bounded context. Returns ``[text]`` if it fits."""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?=\[\[page:\d+\]\])", text)
    chunks, current = [], ""
    for part in parts:
        if current and len(current) + len(part) > max_chars:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks


def merge_results(parts):
    """Union per-chunk results: best-confidence disclosure per DR wins, meta and
    materiality come from the chunk that actually carried the assessment, and
    the coverage summary is recomputed over the merged set."""
    parts = [p for p in parts if p]
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]

    merged = {"document_meta": {}, "materiality_assessment": {},
              "disclosures": [], "coverage_summary": {}}
    best, extras = {}, []
    for result in parts:
        materiality = result.get("materiality_assessment") or {}
        if materiality.get("material_topics") and not merged["materiality_assessment"].get("material_topics"):
            merged["materiality_assessment"] = materiality
            merged["document_meta"] = result.get("document_meta") or {}
        for d in (result.get("disclosures") or []):
            code = d.get("dr_code")
            if not code:
                extras.append(d)
                continue
            current = best.get(code)
            if current is None or (d.get("confidence") or 0) > (current.get("confidence") or 0):
                best[code] = d
    if not merged["document_meta"]:
        merged["document_meta"] = parts[0].get("document_meta", {})
    if not merged["materiality_assessment"]:
        merged["materiality_assessment"] = parts[0].get("materiality_assessment", {})

    merged["disclosures"] = list(best.values()) + extras
    reported = {d.get("dr_code") for d in merged["disclosures"] if d.get("status") == "reported"}
    merged["coverage_summary"] = {
        "esrs2_complete": ESRS2_CODES.issubset(reported),
        "material_drs_reported": len(reported),
        "material_drs_missing": sum(1 for d in merged["disclosures"]
                                    if d.get("status") == "material_not_reported"),
        "phase_ins_invoked": [d.get("dr_code") for d in merged["disclosures"]
                              if d.get("status") == "phase_in_deferred"],
    }
    return merged


def write_outputs(rows, failures, out_csv=OUT_CSV, fail_csv=FAIL_CSV):
    """Append coverage rows to the CSV and report the corpus-level averages."""
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        log.info("Coverage summary: %d rows -> %s", len(df), out_csv)
        log.info("--- ESRS coverage across %d reports ---", len(df))
        for status in STATUSES:
            col = f"n_{status}"
            if col in df:
                log.info("  avg %-22s %.1f per report", status, df[col].mean())
        if "esrs2_complete" in df:
            log.info("  ESRS 2 complete: %d of %d reports",
                     int(df["esrs2_complete"].sum()), len(df))
    if failures:
        pd.DataFrame(failures).to_csv(fail_csv, index=False)
        log.warning("%d report(s) failed -> %s", len(failures), fail_csv)


# --- Anthropic backend -------------------------------------------------------

def make_client(api_key=None):
    try:
        from anthropic import Anthropic
    except ImportError:
        raise SystemExit("The 'anthropic' package is required: pip install -r requirements.txt")
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("Set $ANTHROPIC_API_KEY or pass --api-key")
    return Anthropic(api_key=key)


def call_model(client, model, prompt, report_text, max_retries=4):
    """One extraction call, retrying on rate limits and transient API errors.

    The system prompt is marked for caching: it is ~13 KB and identical for
    every report, so after the first call it is billed at the cache-read rate.
    """
    delay = 4.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                temperature=0,
                system=[{"type": "text", "text": prompt,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": report_text}],
            )
            return robust_json("".join(b.text for b in resp.content if b.type == "text"))
        except Exception as e:                        # noqa: BLE001
            transient = any(w in str(e).lower() for w in
                            ("rate limit", "overloaded", "timeout", "429", "500", "502", "529"))
            if attempt == max_retries or not transient:
                raise
            log.warning("  %s — retrying in %.0fs (%d/%d)", type(e).__name__, delay,
                        attempt, max_retries)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def extract_one(client, model, prompt, text, chunk_chars):
    return merge_results([call_model(client, model, prompt, chunk)
                          for chunk in chunk_text(text, chunk_chars)])


def main():
    ap = argparse.ArgumentParser(description="CSRD ESRS extraction — Anthropic API")
    ap.add_argument("--summary", default=SUMMARY_CSV)
    ap.add_argument("--text-dir", default=TEXT_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", help="Overrides $ANTHROPIC_API_KEY")
    ap.add_argument("--chunk-chars", type=int, default=0,
                    help="Split reports longer than this many chars (0 = never)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(args.summary):
        log.error("Summary CSV not found: %s — run phase1.py first", args.summary)
        return

    prompt = load_prompt(PROMPT_FILE)
    jobs = select_reports(args)
    if not jobs:
        log.warning("Nothing to do (all reports already extracted? use --overwrite to redo).")
        return

    client = make_client(args.api_key)
    log.info("%d report(s) to extract with %s", len(jobs), args.model)

    rows, failures = [], []
    for i, job in enumerate(jobs, 1):
        log.info("[%d/%d] %s", i, len(jobs), job["stem"])
        try:
            result = extract_one(client, args.model, prompt,
                                 read_text(job["text_path"]), args.chunk_chars)
        except Exception as e:                        # noqa: BLE001 - log and keep going
            log.error("  failed: %s", e)
            failures.append({"stem": job["stem"], "error": str(e)})
            continue
        with open(job["out_path"], "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        rows.append(flatten(job["meta"], result))
        coverage = result.get("coverage_summary", {})
        log.info("  %d disclosures, %s material DRs reported",
                 len(result.get("disclosures", [])),
                 coverage.get("material_drs_reported", "?"))

    write_outputs(rows, failures, args.out_csv, FAIL_CSV)


if __name__ == "__main__":
    main()
