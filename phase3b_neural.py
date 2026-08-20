#!/usr/bin/env python3
"""Phase 3b — ESRS disclosure extraction with neural models only, no LLM.

Same job as phase3_esrs.py and the same output schema, but nothing generative
and nothing paid: three small encoder models running locally on CPU.

    retrieve   sentence-transformers embeds every sentence of the report and
               every DR in the ESRS Set 1 taxonomy, and cosine similarity picks
               the passages that could be about each DR.
    classify   a natural-language-inference model scores whether those passages
               actually *entail* "this report discloses <DR title>", which is
               what separates a genuine disclosure from a passing mention or an
               index-table entry.
    extract    an extractive question-answering model pulls the number out of
               the passage for quantitative DRs, and the answer span doubles as
               the evidence quote.

The trade against an LLM is honest and worth stating. This cannot reason about
a disclosure it has not retrieved, it cannot normalise units it has not seen,
and it produces a status plus a grounded quote rather than an argued judgement.
What it does give you is a run over the whole corpus at zero cost, on your own
machine, deterministic apart from int8 batching noise, and with every finding
traceable to a sentence and a page.

Output matches phase 3 exactly, so esrs_coverage.csv rows from either backend
are directly comparable:

    python3 phase3b_neural.py --limit 5
    python3 phase3b_neural.py                    # whole corpus
    python3 phase3b_neural.py --no-qa            # statuses only, ~30% faster
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time

import numpy as np

from phase3_esrs import (
    ESRS2_CODES, PROMPT_FILE, SUMMARY_CSV, TEXT_DIR,
    flatten, read_text, select_reports, write_outputs,
)

OUT_DIR = "esrs_neural_output"
OUT_CSV = "esrs_neural_coverage.csv"
FAIL_CSV = "esrs_neural_failures.csv"

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
QA_MODEL = "deepset/tinyroberta-squad2"

MAX_SENTENCES = 1200      # retrieval pool per report
TOP_K = 4                 # passages considered per DR
# Measured over the corpus, best-similarity per DR runs p5=0.46 / median=0.64,
# because sustainability prose is uniformly on-topic — similarity is a decent
# retriever and a poor judge. So the floor is set only low enough to skip the
# NLI pass on hopeless DRs, and entailment makes the actual call.
SIM_FLOOR = 0.45
ENTAIL_REPORTED = 0.55    # entailment needed to call a DR "reported"
QA_MIN_SCORE = 0.35       # below this the span is a guess, not an answer
MATERIAL_MIN_DRS = 2      # DRs a topical standard must actually report to count
MAX_DRS_PER_SENTENCE = 3  # one sentence answering more than this is boilerplate
MAX_SENTENCE_CHARS = 800  # every model here has a 512-token window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# --- taxonomy ----------------------------------------------------------------

def load_taxonomy(path=PROMPT_FILE):
    """The ESRS Set 1 DR list, read from the same file the LLM backend uses.

    One source of truth: change the taxonomy in esrs_system_prompt.md and both
    backends pick it up.
    """
    standard, drs = None, []
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("### "):
            match = re.search(r"ESRS\s+(E\d|S\d|G\d|2|1)\b", line)
            standard = match.group(1) if match else ("MDR" if "Cross-cutting" in line else None)
            if standard == "2":
                standard = "ESRS2"
            continue
        match = re.match(r"^-\s+`([^`]+)`\s+—\s+(.+)$", line.strip())
        if match and standard and standard not in ("1", "MDR"):
            # MDR-P/A/M/T recur inside every topical standard rather than
            # standing alone, and ESRS 1 is architectural, not a DR list
            drs.append({"standard": standard, "dr_code": match.group(1),
                        "dr_title": match.group(2)})
    return drs


# Questions for the DRs that carry a number worth pulling out. Anything not
# listed falls back to a question built from the DR title.
QA_QUESTIONS = {
    "E1-5": "What was the total energy consumption?",
    "E1-6": "What were the gross Scope 1 greenhouse gas emissions?",
    "E1-7": "How many carbon credits were purchased?",
    "E1-8": "What is the internal carbon price per tonne?",
    "E2-4": "What quantity of pollutants was emitted?",
    "E3-4": "What was the total water consumption?",
    "E5-4": "What was the total mass of materials used?",
    "E5-5": "What was the total waste generated?",
    "S1-6": "How many employees does the company have?",
    "S1-14": "How many work-related injuries were recorded?",
    "S1-16": "What is the gender pay gap?",
    "S1-9": "What percentage of the workforce is female?",
    "G1-4": "How many incidents of corruption or bribery were recorded?",
    "G1-6": "What is the average payment period to suppliers?",
}

# A value like "124,500 tCO2e", "124 500 tCO2e", "42.3 %" or "EUR 1.2 million".
# A space only counts as a thousands separator when it really separates groups
# of three — otherwise "37 2024" (a count next to a year) glues into 372024.
_VALUE_RE = re.compile(
    r"(?P<value>-?\d{1,3}(?:[\s\u00a0]\d{3})+(?![\d])(?:[.,]\d+)?|-?\d[\d.,]*)\s*"
    r"(?P<unit>%|tco2e?|co2e?|mwh|gwh|kwh|gj|tj|"
    r"tonnes?|kt|mt|m3|employees?|days?|hours?|eur|usd|million|billion)?",
    re.IGNORECASE)

# An ESRS content index repeats every DR title verbatim next to a page number,
# so it out-scores real prose on any title-similarity retrieval. The system
# prompt calls this out for the LLM backend ("a high-value map, not evidence of
# substantive disclosure"); here it has to be filtered explicitly.
_DR_CODE_RE = re.compile(
    r"\b(?:BP-[12]|GOV-[1-5]|SBM-[1-3]|IRO-[12]|MDR-[PAMT]|"
    r"E[1-5]-\d{1,2}|S[1-4]-\d{1,2}|G1-\d)\b")
_PAGE_REF_RE = re.compile(
    r"(?:\bp{1,2}\.?\s*\d{1,4}\b|\b\d{1,4}\s*ff\.|\b\d{1,4}\s*[–-]\s*\d{1,4}\s*$|"
    r"\b\d{1,4}\s*$)")
_INDEX_PHRASE_RE = re.compile(
    r"disclosure requirement.{0,40}(?:page|content|section)|content index|"
    r"list of datapoints|index of disclosure", re.IGNORECASE)


def _looks_like_index(sentence):
    """True for a row of the ESRS content index rather than a disclosure.

    Deliberately conservative: a heading that carries its DR code and then
    continues into real prose must survive, so a row only counts as index when
    it is short, names a DR, and ends in a page pointer — or announces itself
    with an index phrase.
    """
    if _INDEX_PHRASE_RE.search(sentence):
        return True
    codes = _DR_CODE_RE.findall(sentence)
    if len(set(codes)) >= 2 and len(sentence) < 400:
        return True
    return bool(codes and len(sentence) < 160
                and _PAGE_REF_RE.search(sentence.strip()))


_PHASE_IN_RE = re.compile(
    r"phase[- ]in|phased in|transitional (?:relief|provision)|first year of "
    r"(?:reporting|application)|exempt(?:ion)? (?:for|from) the first",
    re.IGNORECASE)
_NOT_MATERIAL_RE = re.compile(
    r"(?:not|non)[- ]material|assessed as immaterial|deemed immaterial|"
    r"no material impacts", re.IGNORECASE)


# --- text prep ---------------------------------------------------------------

def sentences_with_pages(text, cap=MAX_SENTENCES, min_len=40,
                         max_chars=MAX_SENTENCE_CHARS):
    """[(sentence, page)] from a phase-1 text file.

    The [[page:N]] markers phase 1 writes are what let a finding cite the PDF
    page it came from, so they are consumed here rather than stripped. Long
    runs (a table dumped as one line) are truncated to stay inside the models'
    512-token window.
    """
    out, page = [], None
    for chunk in re.split(r"\[\[page:(\d+)\]\]", text):
        if chunk.isdigit() and len(chunk) < 6:
            page = int(chunk)
            continue
        flat = re.sub(r"\s+", " ", chunk)
        for raw in re.split(r"(?<=[.!?])\s+", flat):
            raw = raw.strip()
            if len(raw) >= min_len:
                out.append((raw[:max_chars], page))
    if cap and len(out) > cap:
        step = len(out) / cap
        out = [out[int(i * step)] for i in range(cap)]
    return out


# --- models ------------------------------------------------------------------

def _quantize(model, name):
    """Dynamic int8 on the linear layers — the standard CPU inference win."""
    try:
        import torch
        return torch.ao.quantization.quantize_dynamic(
            model.eval(), {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as e:                            # noqa: BLE001
        log.warning("int8 unavailable for %s (%s); running fp32", name, e)
        return model


class Models:
    """The three encoders, loaded once and reused across the corpus."""

    def __init__(self, use_qa=True, quantize=True, threads=None):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import (AutoModelForQuestionAnswering,
                                  AutoModelForSequenceClassification, AutoTokenizer)
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)

        log.info("Loading encoders (CPU, int8=%s) ...", quantize)
        self.st = SentenceTransformer(ST_MODEL, device="cpu",
                                      model_kwargs={"dtype": torch.float32})
        self.st.max_seq_length = min(getattr(self.st, "max_seq_length", 256) or 256, 256)

        self.nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        # transformers 5 honours the checkpoint's fp16 dtype by default, which
        # CPU kernels (and dynamic int8) cannot use — pin fp32 explicitly
        nli = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL, dtype=torch.float32).eval()
        self.nli = _quantize(nli, "NLI") if quantize else nli
        # label order differs between NLI checkpoints; find entailment by name
        self.entail_idx = next(
            (i for i, lbl in nli.config.id2label.items() if "entail" in str(lbl).lower()
             and "not" not in str(lbl).lower()), 0)

        self.qa = self.qa_tok = None
        if use_qa:
            self.qa_tok = AutoTokenizer.from_pretrained(QA_MODEL)
            qa = AutoModelForQuestionAnswering.from_pretrained(
                QA_MODEL, dtype=torch.float32).eval()
            self.qa = _quantize(qa, "QA") if quantize else qa

    # -- retrieval
    def embed(self, texts, batch_size=64):
        return self.st.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)

    # -- entailment
    def entailment(self, pairs, batch_size=16):
        """P(entailment) for each (premise, hypothesis), longest batch first."""
        if not pairs:
            return []
        order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][0]))
        scores = [0.0] * len(pairs)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self.nli_tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx],
                               return_tensors="pt", padding=True, truncation=True,
                               max_length=512)
            with self.torch.no_grad():
                logits = self.nli(**enc).logits
            probs = self.torch.softmax(logits, dim=-1)[:, self.entail_idx]
            for slot, p in zip(idx, probs.tolist()):
                scores[slot] = float(p)
        return scores

    # -- extraction
    def answer(self, question, context):
        """Best extractive span for a question, with its confidence."""
        if self.qa is None:
            return None, 0.0
        enc = self.qa_tok(question, context, return_tensors="pt",
                          truncation="only_second", max_length=384)
        with self.torch.no_grad():
            out = self.qa(**enc)
        start = self.torch.softmax(out.start_logits[0], -1)
        end = self.torch.softmax(out.end_logits[0], -1)
        i, j = int(start.argmax()), int(end.argmax())
        if j < i or j - i > 30:
            return None, 0.0
        span = self.qa_tok.decode(enc["input_ids"][0][i:j + 1]).strip()
        return (span or None), float((start[i] * end[j]) ** 0.5)


# --- per-report extraction ----------------------------------------------------

def _parse_value(span):
    """Split an answer span into a numeric value and its unit, if it has one."""
    if not span:
        return None, None
    match = _VALUE_RE.search(span)
    if not match or not match.group("value"):
        return None, None
    raw = match.group("value").replace(" ", "").rstrip(".,")
    # 124,500 is a thousands separator; 42,3 is a decimal comma
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", raw):
        raw = raw.replace(",", "")
    elif re.fullmatch(r"-?\d+,\d{1,2}", raw):
        raw = raw.replace(",", ".")
    else:
        raw = raw.replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None, None
    return (int(value) if value.is_integer() else value), (match.group("unit") or None)


def extract_report(text, models, taxonomy, dr_embeddings, use_qa=True):
    """One report -> the phase-3 JSON schema, using retrieval + NLI + QA."""
    pairs = sentences_with_pages(text)
    # the content index is a map of where disclosures live, not a disclosure;
    # left in the pool it wins every DR on title similarity alone
    index_rows = sum(1 for s, _ in pairs if _looks_like_index(s))
    pairs = [(s, p) for s, p in pairs if not _looks_like_index(s)]
    if not pairs:
        return _empty_result()
    sentences = [s for s, _ in pairs]
    pages = [p for _, p in pairs]

    sims = models.embed(sentences) @ dr_embeddings.T        # [n_sent, n_dr]

    # 1. shortlist passages per DR, and skip the NLI pass entirely for DRs with
    #    nothing that even looks relevant — that is most of them, and it is
    #    what keeps a 104-DR taxonomy affordable on a CPU
    shortlist = {}
    for j, dr in enumerate(taxonomy):
        column = sims[:, j]
        top = np.argsort(column)[::-1][:TOP_K]
        shortlist[dr["dr_code"]] = [(int(i), float(column[i])) for i in top
                                    if column[i] >= SIM_FLOOR]

    # A sentence that is the best answer to a dozen different DRs is an
    # omnibus row that slipped past the index filter, not a dozen disclosures.
    # Keep it for the DRs it fits best and drop it from the rest.
    claims = {}
    for code, hits in shortlist.items():
        if hits:
            claims.setdefault(hits[0][0], []).append((hits[0][1], code))
    banned = set()
    for i, holders in claims.items():
        if len(holders) > MAX_DRS_PER_SENTENCE:
            for _, code in sorted(holders, reverse=True)[MAX_DRS_PER_SENTENCE:]:
                banned.add((code, i))
    shortlist = {code: [(i, v) for i, v in hits if (code, i) not in banned]
                 for code, hits in shortlist.items()}

    queue = []
    for dr in taxonomy:
        hypothesis = f"This report discloses {dr['dr_title']}."
        for i, _ in shortlist[dr["dr_code"]][:2]:
            queue.append((sentences[i], hypothesis, dr["dr_code"], i))

    entail_scores = models.entailment([(p, h) for p, h, _, _ in queue])
    best_entail = {}
    for (_, _, code, i), score in zip(queue, entail_scores):
        if score > best_entail.get(code, (0.0, None))[0]:
            best_entail[code] = (score, i)

    # 2. a topical standard counts as material when the report actually
    #    discloses against it, which is a firmer signal than topic similarity
    per_standard = {}
    for dr in taxonomy:
        entail, _ = best_entail.get(dr["dr_code"], (0.0, None))
        per_standard.setdefault(dr["standard"], []).append(entail)
    material, non_material = [], []
    for standard, scores in per_standard.items():
        if standard == "ESRS2":
            continue                                  # always mandatory
        strong = sum(1 for e in scores if e >= ENTAIL_REPORTED)
        (material if strong >= MATERIAL_MIN_DRS else non_material).append(standard)

    # 3. a status and evidence per DR
    disclosures = []
    for dr in taxonomy:
        code = dr["dr_code"]
        hits = shortlist[code]
        entail, best_i = best_entail.get(code, (0.0, None))
        evidence, quote = [], ""
        if best_i is not None:
            quote = sentences[best_i]
            evidence = [{"quote": quote[:300], "page": pages[best_i]}]
        elif hits:
            i = hits[0][0]
            quote = sentences[i]
            evidence = [{"quote": quote[:300], "page": pages[i]}]

        mandatory = dr["standard"] == "ESRS2" or code in ESRS2_CODES
        if entail >= ENTAIL_REPORTED:
            status = "reported"
        elif quote and _PHASE_IN_RE.search(quote):
            status = "phase_in_deferred"
        elif not hits:
            status = "not_addressed"
        elif dr["standard"] in non_material and not mandatory:
            status = "not_material_omitted"
        else:
            status = "material_not_reported"

        top_sim = hits[0][1] if hits else 0.0
        confidence = round(min(1.0, 0.45 * top_sim / 0.6 + 0.55 * entail), 3)

        datapoints, flags = [], []
        if use_qa and status == "reported" and _is_quantitative(dr):
            context = " ".join(sentences[i] for i, _ in hits[:TOP_K])
            question = QA_QUESTIONS.get(code) or f"What is the reported {dr['dr_title']}?"
            span, qa_score = models.answer(question, context)
            value, unit = _parse_value(span)
            if not _plausible_datapoint(value, unit, span):
                value = None
                if span:
                    flags.append("value_rejected_implausible")
            if value is not None and qa_score >= QA_MIN_SCORE:
                datapoints.append({
                    "name": dr["dr_title"], "value": value, "unit": unit,
                    "scope_or_breakdown": None, "period": None,
                    "baseline_year": _find_year(context, r"baseline"),
                    "target_year": _find_year(context, r"target|by"),
                    "prior_year_value": None, "qa_confidence": round(qa_score, 3),
                })
            elif span:
                flags.append("value_not_parsed")
        if status == "reported" and code == "E1-6" and "scope 3" not in quote.lower():
            flags.append("scope_3_not_in_evidence")
        if status == "reported" and _is_target_dr(dr) and not _find_year(quote, r"baseline"):
            flags.append("target_without_baseline")

        disclosures.append({
            "standard": dr["standard"], "dr_code": code, "dr_title": dr["dr_title"],
            "status": status, "confidence": confidence, "evidence": evidence,
            "datapoints": datapoints, "quality_flags": flags, "notes": None,
        })

    reported = {d["dr_code"] for d in disclosures if d["status"] == "reported"}
    return {
        "document_meta": {
            "company": None, "reporting_period": None,
            "framework_version": "ESRS Set 1 (2023/2772)",
            "consolidation_scope": None, "assurance": "unknown",
            "iro2_index_present": bool(re.search(r"IRO-2|content index", text, re.I)),
            "extraction_method": "neural: retrieval + NLI + extractive QA (no LLM)",
            "index_rows_filtered": index_rows,
        },
        "materiality_assessment": {
            "method_described": bool(re.search(r"double materiality", text, re.I)),
            "material_topics": sorted(material),
            "non_material_topics": sorted(non_material),
            "undetermined_topics": [],
        },
        "disclosures": disclosures,
        "coverage_summary": {
            "esrs2_complete": ESRS2_CODES.issubset(reported),
            "material_drs_reported": len(reported),
            "material_drs_missing": sum(1 for d in disclosures
                                        if d["status"] == "material_not_reported"),
            "phase_ins_invoked": [d["dr_code"] for d in disclosures
                                  if d["status"] == "phase_in_deferred"],
        },
    }


_QUANT_WORDS = ("emission", "energy", "consumption", "water", "waste", "material",
                "employee", "injuries", "pay gap", "incidents", "payment", "credits",
                "pricing", "number", "characteristics", "diversity", "workforce")
_TARGET_WORDS = ("target", "tracking effectiveness")


def _plausible_datapoint(value, unit, span):
    """Reject the answers extractive QA gives when it has nothing to point at.

    Without a unit the model happily returns a section number, a four-digit
    year, or two numbers run together ("372024"), so a bare integer has to
    look like a quantity rather than a label to be believed.
    """
    if value is None:
        return False
    if unit:
        return True
    if isinstance(value, int):
        if 1900 <= value <= 2100:                     # a year, not a quantity
            return False
        if value < 10:                                # a heading or list number
            return False
    # no unit anywhere in the span is weak evidence that this is a measurement
    return bool(re.search(r"[%€$]|\b(?:t|kt|mt|kg|m3|mwh|gwh|kwh|gj|tj|eur|usd|"
                          r"tonnes?|employees?|days?|hours?|incidents?)\b",
                          span or "", re.IGNORECASE))


def _is_quantitative(dr):
    return dr["dr_code"] in QA_QUESTIONS or any(
        w in dr["dr_title"].lower() for w in _QUANT_WORDS)


def _is_target_dr(dr):
    return any(w in dr["dr_title"].lower() for w in _TARGET_WORDS)


def _find_year(text, near):
    """A year appearing just after a cue word, e.g. "baseline year 2019".

    ``near`` is an alternation, so it has to be grouped — bare "target|by"
    would let "target" match on its own with no year behind it.
    """
    match = re.search(rf"(?:{near})[^.]{{0,40}}?((?:19|20)\d{{2}})", text or "",
                      re.IGNORECASE)
    return int(match.group(1)) if match else None


def _empty_result():
    return {"document_meta": {}, "materiality_assessment": {}, "disclosures": [],
            "coverage_summary": {}}


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="CSRD ESRS extraction — neural, local, no LLM")
    ap.add_argument("--summary", default=SUMMARY_CSV)
    ap.add_argument("--text-dir", default=TEXT_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--no-qa", action="store_true",
                    help="Statuses only, skip datapoint extraction (~30%% faster)")
    ap.add_argument("--no-quantize", action="store_true", help="Run the models in fp32")
    ap.add_argument("--threads", type=int, help="torch CPU threads (default: all)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(args.summary):
        log.error("Summary CSV not found: %s — run phase1.py first", args.summary)
        return

    taxonomy = load_taxonomy()
    log.info("ESRS taxonomy: %d disclosure requirements across %d standards",
             len(taxonomy), len({d["standard"] for d in taxonomy}))

    jobs = select_reports(args)
    if not jobs:
        log.warning("Nothing to do (all reports already extracted? use --overwrite).")
        return

    models = Models(use_qa=not args.no_qa, quantize=not args.no_quantize,
                    threads=args.threads)
    dr_embeddings = models.embed([f"{d['dr_code']}: {d['dr_title']}" for d in taxonomy])
    log.info("%d report(s) to extract", len(jobs))

    rows, failures = [], []
    started = time.monotonic()
    for n, job in enumerate(jobs, 1):
        try:
            result = extract_report(read_text(job["text_path"]), models, taxonomy,
                                    dr_embeddings, use_qa=not args.no_qa)
        except Exception as e:                        # noqa: BLE001 - log and continue
            log.error("[%d/%d] %s failed: %s", n, len(jobs), job["stem"], e)
            failures.append({"stem": job["stem"], "error": str(e)})
            continue
        with open(job["out_path"], "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        rows.append(flatten(job["meta"], result))
        cov = result["coverage_summary"]
        rate = n / max(time.monotonic() - started, 1e-9)
        log.info("[%d/%d] %s — %d reported, %d missing, %d datapoints (%.1f rep/min)",
                 n, len(jobs), job["stem"], cov["material_drs_reported"],
                 cov["material_drs_missing"],
                 sum(len(d["datapoints"]) for d in result["disclosures"]), rate * 60)

    write_outputs(rows, failures, args.out_csv, FAIL_CSV)


if __name__ == "__main__":
    main()
