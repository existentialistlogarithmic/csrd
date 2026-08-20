#!/usr/bin/env python3
"""Phase 2b — Hugging Face NLP layer (free, local, no LLM).

A stronger measurement layer than phase 2's keyword counting, sitting between the
keywords (phase 2) and the LLM ESRS mapping (phase 3). For each English report it
produces, running entirely on your own machine (CPU works; GPU/ADA faster):

  1. Semantic ESRS scoring — sentence-transformers embeddings scored against each
     ESRS topic description (cosine similarity). Catches paraphrases the keyword
     lists miss ("cut our carbon footprint" scores on E1 without the word
     "emission"). Continuous relevance per standard + per E/S/G pillar.
  2. FinBERT-ESG classification — a trained classifier tags each sentence
     Environmental / Social / Governance / None; report-level shares.
  3. ClimateBERT greenwashing signal — detect climate sentences, then classify
     them commitment vs. vague. greenwashing_index is high when a report talks
     about climate a lot but makes few concrete commitments.

Outputs one JSON per report to hf_output/ and an aggregate hf_esg_scores.csv.

    pip install -r requirements.txt          # core deps
    pip install -r requirements-hf.txt       # torch + transformers + sentence-transformers (heavy)

    python3 phase2b_hf.py --limit 5                     # test on a few
    python3 phase2b_hf.py                               # all English reports
    python3 phase2b_hf.py --max-sentences 0             # use every sentence (slower)
    python3 phase2b_hf.py --no-finbert --no-climatebert # embeddings only (lightest)
"""
import argparse
import json
import logging
import os
import re

import numpy as np
import pandas as pd

# reuse report selection + text reading from phase 3 (imports anthropic lazily,
# so this module needs only numpy/pandas + the HF stack)
from phase3_esrs import select_reports, read_text, SUMMARY_CSV, TEXT_DIR

OUT_DIR = "hf_output"
OUT_CSV = "hf_esg_scores.csv"

# Short semantic anchor per ESRS Set 1 topical standard — what each DR family is
# "about", used as the query text the report sentences are scored against.
ESRS_TOPICS = {
    "E1": "Climate change: greenhouse gas emissions, scope 1 2 3, energy mix, decarbonisation, net zero targets, transition plan, carbon",
    "E2": "Pollution of air, water and soil, substances of concern, emissions of pollutants",
    "E3": "Water and marine resources, water consumption and withdrawal, water stress",
    "E4": "Biodiversity and ecosystems, land use, deforestation, species and habitat impacts",
    "E5": "Resource use and circular economy, material inflows and outflows, recycling, waste",
    "S1": "Own workforce: employees, working conditions, health and safety, diversity, pay gap, training",
    "S2": "Workers in the value chain, supply chain labour rights, human rights due diligence",
    "S3": "Affected communities, indigenous peoples, community engagement and impacts",
    "S4": "Consumers and end-users, product safety, data privacy, responsible marketing",
    "G1": "Business conduct, corporate culture, anti-corruption and bribery, lobbying, supplier payment practices",
}
PILLAR = {
    "E1": "environment", "E2": "environment", "E3": "environment",
    "E4": "environment", "E5": "environment",
    "S1": "social", "S2": "social", "S3": "social", "S4": "social",
    "G1": "governance",
}

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINBERT_MODEL = "yiyanghkust/finbert-esg"
CLIMATE_DETECTOR = "climatebert/distilroberta-base-climate-detector"
CLIMATE_COMMITMENT = "climatebert/distilroberta-base-climate-commitment"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# --- text prep --------------------------------------------------------------

# Every model here has a 512-token window. A report's text is not all prose:
# a table dumped by the PDF parser arrives as one run with no full stop, and
# splitting on [.!?] then yields a 9,600-character "sentence" that overflows
# the position embeddings. Cap it — 800 characters is ~200 tokens, comfortably
# inside the window and longer than any real sentence.
MAX_SENTENCE_CHARS = 800


def split_sentences(text, min_len=30, cap=400, max_chars=MAX_SENTENCE_CHARS):
    """Sentences from the report, page markers stripped, short fragments dropped.
    cap>0 evenly subsamples to keep runtime bounded while covering the whole doc."""
    text = re.sub(r"\[\[page:\d+\]\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    sents = []
    for raw in re.split(r"(?<=[.!?])\s+", text):
        raw = raw.strip()
        if len(raw) < min_len:
            continue
        # keep the head of an over-long run rather than dropping it: for a
        # table dump that is the header row, which is the informative part
        sents.append(raw[:max_chars] if len(raw) > max_chars else raw)
    if cap and len(sents) > cap:
        step = len(sents) / cap
        sents = [sents[int(i * step)] for i in range(cap)]
    return sents


# --- 1. semantic ESRS scoring ----------------------------------------------

def semantic_scores(st_model, sentences, topk=15):
    """Mean cosine similarity of the top-k most relevant sentences to each ESRS
    topic; rolled up to E/S/G pillars. Embeddings are L2-normalised so the dot
    product is cosine similarity."""
    topics = list(ESRS_TOPICS)
    t_emb = st_model.encode([ESRS_TOPICS[t] for t in topics],
                            normalize_embeddings=True, convert_to_numpy=True)
    if not sentences:
        per_topic = {t: 0.0 for t in topics}
    else:
        s_emb = st_model.encode(sentences, batch_size=64, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        sims = s_emb @ t_emb.T  # [n_sent, n_topics]
        per_topic = {}
        for j, t in enumerate(topics):
            top = np.sort(sims[:, j])[::-1][:topk]
            per_topic[t] = round(float(top.mean()), 4) if top.size else 0.0

    pillar = {"environment": [], "social": [], "governance": []}
    for t, v in per_topic.items():
        pillar[PILLAR[t]].append(v)
    pillar = {p: round(float(np.mean(v)), 4) if v else 0.0 for p, v in pillar.items()}
    dominant = max(pillar, key=pillar.get) if any(pillar.values()) else "unknown"
    return per_topic, pillar, dominant


# --- 2. FinBERT-ESG classification -----------------------------------------

def _classify(pipe, sentences, batch_size=32):
    """Run a text-classification pipeline, results in the input order.

    Sentences are fed longest-first: a batch is padded to its longest member,
    so mixing a 10-word sentence with a 200-word one makes the short one cost
    as much as the long one. Grouping similar lengths together removes most of
    that wasted compute for free.
    """
    if not sentences:
        return []
    order = sorted(range(len(sentences)), key=lambda i: -len(sentences[i]))
    ordered = [sentences[i] for i in order]
    results = list(pipe(ordered, batch_size=batch_size))
    out = [None] * len(sentences)
    for slot, result in zip(order, results):
        out[slot] = result
    return out


def finbert_shares(pipe, sentences):
    from collections import Counter
    c = Counter()
    if sentences:
        for r in _classify(pipe, sentences):
            lab = str(r["label"]).lower()
            if "environ" in lab:
                c["environmental"] += 1
            elif "social" in lab:
                c["social"] += 1
            elif "govern" in lab:
                c["governance"] += 1
            else:
                c["none"] += 1
    total = sum(c.values()) or 1
    return {f"finbert_{k}_share": round(c[k] / total, 4)
            for k in ("environmental", "social", "governance", "none")}


# --- 3. ClimateBERT greenwashing signal ------------------------------------

def _is_positive(label):
    """True for the 'yes'/'commitment' class of ClimateBERT's binary heads.
    Label strings vary by model release; adjust here if a model uses other names."""
    l = str(label).lower()
    return l in {"yes", "commitment", "1", "true", "label_1"} or "commit" in l or l.endswith("_1")


def climate_signals(detector, committer, sentences):
    n = len(sentences) or 1
    climate = [s for s, r in zip(sentences, _classify(detector, sentences))
               if _is_positive(r["label"])] if sentences else []
    commitments = 0
    if climate:
        commitments = sum(1 for r in _classify(committer, climate)
                          if _is_positive(r["label"]))
    cs = len(climate)
    commit_rate = round(commitments / cs, 4) if cs else 0.0
    climate_share = round(cs / n, 4)
    # high = lots of climate talk, few concrete commitments
    greenwashing = round(climate_share * (1 - commit_rate), 4)
    return {
        "climate_sentences": cs,
        "climate_share": climate_share,
        "climate_commitments": commitments,
        "commitment_rate": commit_rate,
        "greenwashing_index": greenwashing,
    }


# --- per-report orchestration ----------------------------------------------

def process_report(text, models, args):
    sentences = split_sentences(text, cap=args.max_sentences)
    detail = {"n_sentences": len(sentences)}
    row = {"n_sentences": len(sentences)}

    per_topic, pillar, dominant = semantic_scores(models["st"], sentences)
    detail["semantic_per_topic"] = per_topic
    row.update({f"sem_{p}": pillar[p] for p in pillar})
    row["sem_dominant_pillar"] = dominant

    if models.get("finbert") is not None:
        fb = finbert_shares(models["finbert"], sentences)
        row.update(fb)
        detail.update(fb)
    if models.get("climate_detector") is not None and models.get("climate_committer") is not None:
        cs = climate_signals(models["climate_detector"], models["climate_committer"], sentences)
        row.update(cs)
        detail.update(cs)
    return row, detail


# --- model loading ----------------------------------------------------------

def pick_device(pref):
    if pref:
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _quantize(model, name):
    """Dynamic int8 on the linear layers — the standard CPU inference win.

    These are classification heads over frozen encoders, so the quantisation
    error lands well below the decision boundary; what it buys is roughly a
    2x speedup and a quarter of the memory, which is what makes scoring a
    2,000-report corpus on a CPU a couple of hours instead of a couple of days.
    """
    try:
        import torch
        return torch.ao.quantization.quantize_dynamic(
            model.eval(), {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as e:  # noqa: BLE001 — speed is optional, correctness is not
        log.warning("int8 quantisation unavailable for %s (%s); running fp32", name, e)
        return model


def load_models(args):
    device = pick_device(args.device)
    log.info("Loading models on device=%s ...", device)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("Hugging Face stack missing. Install it: pip install -r requirements-hf.txt")
        raise SystemExit(1)
    st = SentenceTransformer(args.st_model, device=device)
    st.max_seq_length = min(getattr(st, "max_seq_length", 256) or 256, 256)
    models = {"st": st}
    dev_idx = 0 if device == "cuda" else -1
    quantize = device == "cpu" and not args.no_quantize

    if not args.no_finbert or not args.no_climatebert:
        from transformers import pipeline

    def build(name, model_id):
        """A text-classification pipeline with truncation pinned at the
        tokenizer, not passed per call — transformers 5 ignores the call kwarg
        and a single over-long input then blows up the whole report."""
        pipe = pipeline("text-classification", model=model_id, device=dev_idx,
                        truncation=True, max_length=512)
        pipe.tokenizer.model_max_length = 512
        pipe.tokenizer.truncation_side = "right"
        if quantize:
            pipe.model = _quantize(pipe.model, name)
        return pipe

    if not args.no_finbert:
        try:
            models["finbert"] = build("FinBERT-ESG", FINBERT_MODEL)
        except Exception as e:  # noqa: BLE001 — degrade gracefully
            log.warning("FinBERT-ESG unavailable (%s); skipping ESG classification", e)
    if not args.no_climatebert:
        try:
            models["climate_detector"] = build("ClimateBERT-detector", CLIMATE_DETECTOR)
            models["climate_committer"] = build("ClimateBERT-commitment", CLIMATE_COMMITMENT)
        except Exception as e:  # noqa: BLE001
            log.warning("ClimateBERT unavailable (%s); skipping greenwashing signal", e)
    return models


# --- main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CSRD Hugging Face NLP layer — phase 2b")
    parser.add_argument("--summary", default=SUMMARY_CSV)
    parser.add_argument("--text-dir", default=TEXT_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--out-csv", default=OUT_CSV)
    parser.add_argument("--st-model", default=DEFAULT_ST_MODEL, help="sentence-transformers model")
    parser.add_argument("--max-sentences", type=int, default=400, help="Subsample cap per report (0 = all)")
    parser.add_argument("--device", help="cuda|cpu (auto-detected if omitted)")
    parser.add_argument("--no-finbert", action="store_true", help="Skip FinBERT-ESG classification")
    parser.add_argument("--no-climatebert", action="store_true", help="Skip ClimateBERT greenwashing signal")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip int8 quantisation of the classifiers (CPU only; ~2x slower)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(args.summary):
        log.error("Summary CSV not found: %s — run phase1.py first", args.summary)
        return
    jobs = select_reports(args)
    if not jobs:
        log.warning("Nothing to do (all reports already scored? use --overwrite to redo).")
        return

    models = load_models(args)
    log.info("%d report(s) to score", len(jobs))

    rows = []
    for i, job in enumerate(jobs, 1):
        log.info("[%d/%d] %s", i, len(jobs), job["stem"])
        try:
            row, detail = process_report(read_text(job["text_path"]), models, args)
        except Exception as e:  # noqa: BLE001
            log.error("  failed: %s", e)
            continue
        row = {**job["meta"], **row}
        with open(job["out_path"], "w", encoding="utf-8") as f:
            json.dump({**job["meta"], **detail}, f, ensure_ascii=False, indent=2)
        rows.append(row)
        log.info("  dominant=%s  greenwashing_index=%s",
                 row.get("sem_dominant_pillar"), row.get("greenwashing_index", "n/a"))

    if rows:
        header = not os.path.exists(args.out_csv)
        pd.DataFrame(rows).to_csv(args.out_csv, mode="a", header=header, index=False)
        log.info("Scores: %d rows -> %s", len(rows), args.out_csv)
        df = pd.DataFrame(rows)
        for p in ("environment", "social", "governance"):
            col = f"sem_{p}"
            if col in df:
                log.info("  avg semantic %-12s %.3f", p, df[col].mean())
        if "greenwashing_index" in df:
            log.info("  avg greenwashing_index: %.3f", df["greenwashing_index"].mean())


if __name__ == "__main__":
    main()
