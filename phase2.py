#!/usr/bin/env python3
# phase 2
#analyzed results

#!/usr/bin/env python3

import argparse
import json
import logging
import math
import os
import re
from collections import Counter

import pandas as pd

import matplotlib
matplotlib.use("Agg")  # no display needed, we only save PNGs
import matplotlib.pyplot as plt

# _________________step 1 starting
TEXT_DIR     = "extracted_text"
SUMMARY_CSV  = "extraction_summary.csv"
NLP_OUT_CSV  = "nlp_results.csv"
NLP_OUT_DIR  = "nlp_output"
CHART_DIR    = os.path.join(NLP_OUT_DIR, "charts")

TOP_N_KEYWORDS = 20
MAX_METRICS    = 200

# chart colors (validated categorical palette; pillar color is fixed per pillar)
SURFACE       = "#fcfcfb"
INK           = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED         = "#898781"
GRIDLINE      = "#e1e0d9"
BASELINE      = "#c3c2b7"
PILLAR_COLORS = {
    "environment": "#2a78d6",
    "social":      "#1baf7a",
    "governance":  "#eda100",
}

# ordered years get an ordinal ramp (light = earlier), validated steps
YEAR_RAMP = ["#86b6ef", "#2a78d6", "#184f95"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

#_________________step 2 ESG metrics
THEMES = {
    "environment": [
        "emission", "co2", "carbon", "ghg", "greenhouse", "climate",
        "energy", "renewable", "solar", "wind", "fossil", "waste",
        "recycl", "water", "biodiversity", "deforestation", "scope 1",
        "scope 2", "scope 3", "net zero", "decarboni", "pollution",
    ],
    "social": [
        "employee", "worker", "staff", "workforce", "gender", "diversity",
        "inclusion", "safety", "health", "injury", "training", "education",
        "human rights", "community", "supply chain", "labour", "labor",
        "wage", "pay gap", "wellbeing", "parental",
    ],
    "governance": [
        "board", "director", "governance", "audit", "compliance",
        "risk management", "ethics", "anti-corruption", "bribery",
        "whistleblow", "transparency", "remuneration", "executive pay",
        "policy", "code of conduct", "shareholder", "accountability",
    ],
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "that", "this", "these",
    "those", "it", "its", "we", "our", "they", "their", "as", "not", "no",
    "more", "also", "which", "who", "all", "any", "one", "can", "into",
    "than", "such", "both", "other", "each", "report", "company", "group",
    "year", "per", "total", "including", "following", "within", "through",
    "page", "table", "figure", "section", "overview", "accordance",
}

#_________________theme keyword regexes
# a keyword only counts when it STARTS a word: "emission" still matches
# "emissions", but "board" no longer matches "onboarding"/"cardboard",
# "wage" no longer matches "sewage", "wind" no longer matches "windows"
_EXACT_WORDS = {"wind"}  # prefix matching would catch "window"/"windfall"

def _kw_pattern(kw):
    esc = re.escape(kw)
    if kw in _EXACT_WORDS:
        return rf"\b{esc}s?\b"
    return rf"\b{esc}\w*"

_THEME_RE = {
    pillar: re.compile("|".join(_kw_pattern(k) for k in kws), re.IGNORECASE)
    for pillar, kws in THEMES.items()
}


def count_theme_hits(text: str, pillar: str) -> int:
    return len(_THEME_RE[pillar].findall(text))


#_________________ ESRS topical-standard coverage
# The real research question for CSRD reports is *which ESRS topical standards a
# report substantively addresses*, not which generic words are frequent. Each
# standard is detected by its ESRS code (E1, S1-2, G1 ...) plus its defining
# concepts. `pillar` ties the standard to E / S / G for colouring & grouping.
ESRS_STANDARDS = {
    "E1 Climate change": ("environment", [
        r"\bE1\b", r"\bE1-\d", r"climate change", r"greenhouse gas", r"\bghg\b",
        r"scope [123]", r"transition plan", r"decarboni", r"net[- ]zero", r"carbon footprint"]),
    "E2 Pollution": ("environment", [
        r"\bE2\b", r"\bE2-\d", r"pollution", r"pollutant", r"microplastic",
        r"substances of concern", r"air emissions"]),
    "E3 Water & marine": ("environment", [
        r"\bE3\b", r"\bE3-\d", r"water consumption", r"water withdrawal",
        r"water discharge", r"marine resource"]),
    "E4 Biodiversity": ("environment", [
        r"\bE4\b", r"\bE4-\d", r"biodiversity", r"ecosystem", r"deforestation",
        r"land[- ]use change", r"invasive species"]),
    "E5 Circular economy": ("environment", [
        r"\bE5\b", r"\bE5-\d", r"circular economy", r"resource use",
        r"resource inflow", r"resource outflow", r"material efficiency"]),
    "S1 Own workforce": ("social", [
        r"\bS1\b", r"\bS1-\d", r"own workforce", r"collective bargaining",
        r"gender pay gap", r"work[- ]life balance", r"occupational health"]),
    "S2 Value-chain workers": ("social", [
        r"\bS2\b", r"\bS2-\d", r"workers in the value chain",
        r"value chain workers", r"supply chain (?:labour|labor|workers)"]),
    "S3 Affected communities": ("social", [
        r"\bS3\b", r"\bS3-\d", r"affected communities", r"indigenous peoples",
        r"local communities"]),
    "S4 Consumers & end-users": ("social", [
        r"\bS4\b", r"\bS4-\d", r"consumers and end[- ]users", r"end[- ]users",
        r"product safety", r"consumer health"]),
    "G1 Business conduct": ("governance", [
        r"\bG1\b", r"\bG1-\d", r"business conduct", r"anti[- ]corruption",
        r"anti[- ]bribery", r"bribery", r"whistleblow", r"corporate culture",
        r"political (?:engagement|contributions)", r"lobbying"]),
}
_ESRS_RE = {std: re.compile("|".join(pats), re.IGNORECASE)
            for std, (pillar, pats) in ESRS_STANDARDS.items()}
ESRS_PILLAR = {std: pillar for std, (pillar, _) in ESRS_STANDARDS.items()}

# a standard counts as "covered" only above this many matches, so a single
# passing reference in a contents list doesn't count as real coverage.
ESRS_COVER_THRESHOLD = 5


#_________________ CSRD / ESRS cross-cutting concepts (adoption = present at all)
CSRD_CONCEPTS = {
    "Double materiality":        r"double materiality|impact materiality|financial materiality",
    "Materiality assessment":    r"materiality assessment|material (?:topics?|impacts?|matters?)",
    "EU Taxonomy":               r"eu taxonomy|taxonomy[- ](?:aligned|eligible)",
    "Scope 3 emissions":         r"scope 3",
    "Transition plan":           r"transition plan",
    "Science-based targets":     r"science[- ]based target|\bsbti\b",
    "IROs":                      r"impacts,? risks and opportunities|\biros?\b",
    "Value chain":               r"value chain",
    "Due diligence":             r"due diligence",
    "External assurance":        r"limited assurance|reasonable assurance",
}
_CONCEPT_RE = {c: re.compile(p, re.IGNORECASE) for c, p in CSRD_CONCEPTS.items()}


def esrs_coverage(text: str) -> dict:
    """Match count per ESRS topical standard."""
    return {std: len(rx.findall(text)) for std, rx in _ESRS_RE.items()}


def concept_flags(text: str) -> dict:
    """True/False presence of each cross-cutting CSRD concept."""
    return {c: bool(rx.search(text)) for c, rx in _CONCEPT_RE.items()}


#_________________numeric metric regexes
# a real figure = at least one digit, optional thousands/decimals, then a UNIT
# of measure. GHG/CO2 on their own are categories, not units, so they are not
# accepted here (they produced noise like "3 GHG"); "tCO2e" (tonnes CO2-eq) is.
_NUM = r"\d[\d,]*(?:\.\d+)?"
_METRIC_PATTERNS = [
    # emissions & energy: "136,630.5 tCO2e", "952 GWh", "10.5 Mt CO2e"
    rf"{_NUM}\s*(?:mt|kt|t)?\s?co2e?\b",
    rf"{_NUM}\s*(?:tco2e?|ktco2e?|mtco2e?|tonnes?\s+co2e?)\b",
    rf"{_NUM}\s*(?:twh|gwh|mwh|kwh|gj|tj|pj|mj)\b",
    # percentages: "42.5%"  (optionally trailing context word)
    rf"{_NUM}\s*%\s*(?:of|reduction|increase|renewable|female|women|recycl\w+)?",
    # headcount: "3,200 employees" / "1,250 FTE"
    rf"{_NUM}\s*(?:employees?|workers?|fte)\b",
    # money: "€1.2m", "$500k", "EUR 3.4 billion"
    rf"(?:€|\$|£|eur|usd|gbp)\s*{_NUM}\s*(?:m|bn|k|million|billion|thousand)?\b",
]
_METRIC_RE = [re.compile(p, re.IGNORECASE) for p in _METRIC_PATTERNS]

def _clean_metric(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" .,").strip()


#_________________ step 3

def clean_text(raw: str) -> str:
    text = raw.replace("\f", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def tokenise(text: str) -> list:
    """Lowercase letter-only tokens ≥ 3 chars, no stopwords."""
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS]


def top_keywords(text: str, n: int = TOP_N_KEYWORDS) -> list:
    """Return [(word, count)] for the n most frequent content words."""
    return Counter(tokenise(text)).most_common(n)


def corpus_tfidf(doc_tfs, df_counter, exclude=None, top_n=25):
    """
    Rank corpus terms by TF-IDF instead of raw frequency.

    Raw counts surface boilerplate that appears in every report ("management",
    "business", "financial"). TF-IDF multiplies a term's total frequency by
    log(N / document-frequency), so words that appear in *every* report get an
    IDF near zero and drop out, leaving the terms that actually distinguish
    reports from one another. `exclude` drops entity noise (company names /
    tickers) that would otherwise top the list because a firm names itself a
    lot. Returns [(term, score)] high-to-low.
    """
    n_docs = len(doc_tfs)
    if n_docs == 0:
        return []
    exclude = exclude or set()
    # ignore ultra-rare terms (company-specific noise): require a term to appear
    # in a meaningful share of reports, and never in literally all of them.
    min_df = max(3, int(n_docs * 0.08))
    total_tf = Counter()
    for tf in doc_tfs:
        total_tf.update(tf)
    scores = {}
    for term, tf_sum in total_tf.items():
        if term in exclude or len(term) < 4:
            continue
        dfq = df_counter.get(term, 0)
        if dfq < min_df or dfq >= n_docs:      # too rare, or in literally every doc
            continue
        idf = math.log(n_docs / dfq)
        scores[term] = math.log1p(tf_sum) * idf
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]


#_________________ step 4

def score_themes(text: str) -> dict:
    """
    Count keyword hits per ESG pillar, then compute density (share of total hits).
    Returns flat dict: environment, social, governance, *_density.
    """
    raw = {p: count_theme_hits(text, p) for p in THEMES}
    total = max(sum(raw.values()), 1)
    density = {f"{p}_density": round(raw[p] / total, 3) for p in raw}
    return {**raw, **density}


#_________________ step 5

def extract_metrics(text: str, max_hits: int = MAX_METRICS) -> list:
    """Pull numeric+unit snippets from text (deduplicated, capped at max_hits)."""
    seen, hits = set(), []
    for pattern in _METRIC_RE:
        for m in pattern.finditer(text):
            snippet = _clean_metric(m.group(0))
            # drop degenerate captures like a bare unit or a lone digit
            if not re.search(r"\d", snippet) or len(snippet) < 3:
                continue
            if snippet.lower() not in seen:
                seen.add(snippet.lower())
                hits.append(snippet)
            if len(hits) >= max_hits:
                return hits
    return hits


#_________________ key

# disclosure-index / table-dump noise: ESRS datapoint codes, EFRAG IDs, MDR tags
_BOILERPLATE_RE = re.compile(
    r"ESRS|EFRAG|MDR-|E1-\d|E[1-5]-\d|\bIRO-1\b|\bAR \d|datapoint", re.IGNORECASE)

def _looks_like_prose(s: str) -> bool:
    """True if the sentence reads like real prose, not a code/table fragment."""
    words = s.split()
    if len(words) < 6:
        return False
    if _BOILERPLATE_RE.search(s):
        return False
    # real prose is mostly letters; table dumps are full of digits/brackets/codes
    alpha = sum(c.isalpha() or c.isspace() for c in s)
    if alpha / max(len(s), 1) < 0.75:
        return False
    # need a few lowercase 'connective' words to look like a sentence
    lc = [w for w in words if w.islower() and len(w) > 2]
    return len(lc) >= 4


def key_sentences(text: str, pillar: str, max_sents: int = 3) -> list:
    """Return up to max_sents prose sentences with the most keyword hits for a pillar."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    scored = []
    for s in sents:
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) < 40 or not _looks_like_prose(s):
            continue
        hits = count_theme_hits(s, pillar)
        if hits > 0:
            scored.append((hits, s))
    scored.sort(reverse=True)
    return [s for _, s in scored[:max_sents]]


#_________________analysis

def process_file(txt_path: str, meta: dict):
    """
    NLP on one .txt file.
    Returns (flat_row_dict, detail_dict) or None on read error.
    """
    try:
        with open(txt_path, encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        log.warning("Cannot read %s: %s", txt_path, e)
        return None

    text   = clean_text(raw)
    themes = score_themes(text)
    mets   = extract_metrics(text)
    esrs   = esrs_coverage(text)
    concepts = concept_flags(text)
    tokens = tokenise(text)                     # for corpus TF-IDF (done in main)
    tf     = Counter(tokens)

    pillars  = {p: themes[p] for p in THEMES}
    dominant = max(pillars, key=pillars.get) if any(pillars.values()) else "unknown"

    covered = [std for std, n in esrs.items() if n >= ESRS_COVER_THRESHOLD]

    flat = {
        "company":         meta.get("company", ""),
        "isin":            meta.get("isin", ""),
        "report_year":     meta.get("report_year", ""),
        "country":         meta.get("country", ""),
        "industry":        meta.get("industry", ""),
        "sector":          meta.get("sector", ""),
        "word_count":      len(text.split()),
        "dominant_pillar": dominant,
        **themes,
        # ESRS topical-standard coverage
        "esrs_standards_covered": len(covered),
        "esrs_covered":           "; ".join(covered),
        **{f"esrs::{std}": n for std, n in esrs.items()},
        # CSRD cross-cutting concept flags
        **{f"concept::{c}": int(v) for c, v in concepts.items()},
        "metric_hits":     len(mets),
    }

    detail = {
        **flat,
        "esrs_full":         esrs,
        "concepts":          concepts,
        "metrics":           mets,
        "env_sentences":     key_sentences(text, "environment"),
        "soc_sentences":     key_sentences(text, "social"),
        "gov_sentences":     key_sentences(text, "governance"),
    }

    # token freqs are returned out-of-band so main() can build corpus TF-IDF
    return flat, detail, tf


# _________________help

def out_df_companies(rows):
    """All company names seen this run (for entity-token exclusion in TF-IDF)."""
    return [r.get("company", "") for r in rows]


def _stem_from_meta(row: dict) -> str:
    """Reconstruct the filename stem that phase1 would have used."""
    name = re.sub(r"[^\w\s-]", "", str(row.get("company", "")))
    name = re.sub(r"\s+", "_", name.strip())
    stem = f"{name}_{row.get('isin', '')}"
    year = row.get("report_year", "")
    if str(year).strip() not in ("", "nan"):
        stem += f"_{int(float(year))}"
    return stem

# _________________charts

def _new_axes(width=8, height=4.5):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    log.info("Chart saved: %s", path)


def chart_dominant_pillar(out_df, path):
    """How many reports lean environment vs social vs governance."""
    counts = out_df["dominant_pillar"].value_counts()
    pillars = [p for p in THEMES if p in counts.index]
    if not pillars:
        return
    values = [int(counts[p]) for p in pillars]

    fig, ax = _new_axes(6.5, 4)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    bars = ax.bar(pillars, values, width=0.55,
                  color=[PILLAR_COLORS[p] for p in pillars])
    ax.bar_label(bars, padding=3, color=INK_SECONDARY, fontsize=9)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("Dominant ESG pillar across reports",
                 loc="left", color=INK, fontsize=12)
    ax.set_ylabel("Number of reports", color=INK_SECONDARY, fontsize=9)
    _save(fig, path)


def chart_pillar_hits(out_df, path):
    """Average keyword hits per pillar per report."""
    pillars = list(THEMES)
    values = [out_df[p].mean() for p in pillars]

    fig, ax = _new_axes(6.5, 4)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    bars = ax.bar(pillars, values, width=0.55,
                  color=[PILLAR_COLORS[p] for p in pillars])
    ax.bar_label(bars, fmt="%.0f", padding=3, color=INK_SECONDARY, fontsize=9)
    ax.set_title("Average ESG keyword hits per report",
                 loc="left", color=INK, fontsize=12)
    ax.set_ylabel("Avg keyword hits", color=INK_SECONDARY, fontsize=9)
    _save(fig, path)


def chart_density_by_country(out_df, path, top_n=10):
    """ESG focus mix (share of keyword hits) for the countries with most reports."""
    if "country" not in out_df.columns or out_df["country"].isna().all():
        return
    top = out_df["country"].value_counts().head(top_n).index.tolist()
    if len(top) < 2:
        return
    sub = out_df[out_df["country"].isin(top)]
    mix = sub.groupby("country")[[f"{p}_density" for p in THEMES]].mean()
    mix = mix.div(mix.sum(axis=1), axis=0).loc[top[::-1]]  # normalise, most reports on top

    fig, ax = _new_axes(8, 0.5 * len(top) + 1.6)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    left = pd.Series(0.0, index=mix.index)
    for p in THEMES:
        vals = mix[f"{p}_density"]
        ax.barh(mix.index, vals, left=left, height=0.55, label=p,
                color=PILLAR_COLORS[p], edgecolor=SURFACE, linewidth=1.6)
        # direct-label segments that are wide enough to hold the number
        for y, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.08:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center",
                        color=SURFACE, fontsize=8)
        left = left + vals
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title("ESG focus mix by country (share of keyword hits)",
                 loc="left", color=INK, fontsize=12, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _save(fig, path)


def chart_esrs_coverage(out_df, path):
    """Average disclosure intensity per ESRS topical standard.

    Coverage (presence) is near-universal because CSRD mandates every standard,
    so that view is flat. Intensity — the mean number of matches per report —
    shows where reporting actually goes deep (E1 Climate) vs stays thin
    (S3 Affected communities)."""
    n = len(out_df)
    if n == 0:
        return
    stds = list(ESRS_STANDARDS)
    intensity = [out_df[f"esrs::{s}"].mean() for s in stds]
    order = sorted(range(len(stds)), key=lambda i: intensity[i])  # biggest on top
    stds = [stds[i] for i in order]
    intensity = [intensity[i] for i in order]
    colors = [PILLAR_COLORS[ESRS_PILLAR[s]] for s in stds]

    fig, ax = _new_axes(7.5, 0.42 * len(stds) + 1.7)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    bars = ax.barh(stds, intensity, height=0.62, color=colors)
    ax.bar_label(bars, fmt="%.0f", padding=3, color=INK_SECONDARY, fontsize=8.5)
    ax.set_title("ESRS disclosure intensity by topical standard",
                 loc="left", color=INK, fontsize=12)
    ax.set_xlabel(f"Avg keyword/code matches per report (n = {n})",
                  color=INK_SECONDARY, fontsize=9)
    _save(fig, path)


def chart_concept_adoption(out_df, path):
    """Share of reports mentioning each cross-cutting CSRD concept."""
    n = len(out_df)
    if n == 0:
        return
    concepts = list(CSRD_CONCEPTS)
    pct = [out_df[f"concept::{c}"].mean() * 100 for c in concepts]
    order = sorted(range(len(concepts)), key=lambda i: pct[i])
    concepts = [concepts[i] for i in order]
    pct = [pct[i] for i in order]

    fig, ax = _new_axes(7.5, 0.42 * len(concepts) + 1.6)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    bars = ax.barh(concepts, pct, height=0.62, color=PILLAR_COLORS["governance"])
    ax.bar_label(bars, fmt="%.0f%%", padding=3, color=INK_SECONDARY, fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.set_title("CSRD reporting concepts — adoption rate",
                 loc="left", color=INK, fontsize=12)
    ax.set_xlabel(f"Share of the {n} reports mentioning the concept",
                  color=INK_SECONDARY, fontsize=9)
    _save(fig, path)


def chart_distinctive_terms(tfidf_ranked, path, top_n=15):
    """Corpus-distinctive terms by TF-IDF (filters out ubiquitous filler)."""
    common = tfidf_ranked[:top_n]
    if not common:
        return
    words = [w for w, _ in common][::-1]
    scores = [s for _, s in common][::-1]

    fig, ax = _new_axes(7, 0.36 * len(words) + 1.6)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    bars = ax.barh(words, scores, height=0.58, color=PILLAR_COLORS["social"])
    ax.set_title("Most distinctive terms across the corpus (TF-IDF)",
                 loc="left", color=INK, fontsize=12)
    ax.set_xlabel("TF-IDF weight — frequent, but not ubiquitous",
                  color=INK_SECONDARY, fontsize=9)
    ax.tick_params(axis="x", labelbottom=False)
    _save(fig, path)

def chart_year_comparison(out_df, path):
    """Average ESG density per pillar, one bar group per report year."""
    years = sorted({int(y) for y in pd.to_numeric(out_df.get("report_year"),
                                                  errors="coerce").dropna().unique()})
    if len(years) < 2:
        return
    pillars = list(THEMES)
    width = 0.8 / len(years)
    fig, ax = _new_axes(7, 4.2)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    sub_year = pd.to_numeric(out_df["report_year"], errors="coerce")
    for i, year in enumerate(years):
        sub = out_df[sub_year == year]
        vals = [sub[f"{p}_density"].mean() for p in pillars]
        xs = [j + (i - (len(years) - 1) / 2) * width for j in range(len(pillars))]
        bars = ax.bar(xs, vals, width=width * 0.9, label=str(year),
                      color=YEAR_RAMP[i % len(YEAR_RAMP)])
        ax.bar_label(bars, fmt="%.2f", padding=3, color=INK_SECONDARY, fontsize=8)
    ax.set_xticks(range(len(pillars)))
    ax.set_xticklabels(pillars)
    ax.set_title("ESG focus by report year (avg share of keyword hits)",
                 loc="left", color=INK, fontsize=12, pad=30)
    ax.set_ylabel("Avg density", color=INK_SECONDARY, fontsize=9)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(years),
              frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _save(fig, path)

def make_charts(out_df, tfidf_ranked, charts_dir):
    os.makedirs(charts_dir, exist_ok=True)
    chart_dominant_pillar(out_df, os.path.join(charts_dir, "dominant_pillar.png"))
    chart_pillar_hits(out_df, os.path.join(charts_dir, "pillar_hits.png"))
    chart_density_by_country(out_df, os.path.join(charts_dir, "esg_mix_by_country.png"))
    chart_esrs_coverage(out_df, os.path.join(charts_dir, "esrs_coverage.png"))
    chart_concept_adoption(out_df, os.path.join(charts_dir, "csrd_concept_adoption.png"))
    chart_distinctive_terms(tfidf_ranked, os.path.join(charts_dir, "distinctive_terms.png"))


# _________________main

def main():
    parser = argparse.ArgumentParser(description="CSRD NLP pipeline — phase 2")
    parser.add_argument("--summary",  default=SUMMARY_CSV,  help="extraction_summary.csv from phase 1")
    parser.add_argument("--text-dir", default=TEXT_DIR,     help="Folder with extracted .txt files")
    parser.add_argument("--out-csv",  default=NLP_OUT_CSV,  help="Output CSV path")
    parser.add_argument("--out-dir",  default=NLP_OUT_DIR,  help="Folder for per-company JSON files")
    parser.add_argument("--charts-dir", default=CHART_DIR,  help="Folder for chart PNGs")
    parser.add_argument("--limit",    type=int,             help="Only process first N documents")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.summary):
        log.error("Summary CSV not found: %s — run phase1.py first", args.summary)
        return

    df = pd.read_csv(args.summary)
    df = df[df["extraction_status"] == "success"].reset_index(drop=True)
    log.info("Loaded %d successfully extracted documents", len(df))

    # only analyse English reports (language detected in phase 1)
    if "language" in df.columns:
        n_before = len(df)
        df = df[df["language"] == "en"].reset_index(drop=True)
        log.info("English filter: kept %d of %d documents", len(df), n_before)
    else:
        log.warning("No 'language' column in %s — re-run phase1.py to get "
                    "language detection; analysing all documents", args.summary)

    if args.limit:
        df = df.iloc[:args.limit]
        log.info("Limiting to %d documents", args.limit)

    rows = []
    doc_tfs = []          # per-document term frequencies, for corpus TF-IDF
    df_counter = Counter()  # document frequency: in how many docs each term appears
    for _, row in df.iterrows():
        txt_path = str(row.get("text_file", ""))
        if not txt_path or not os.path.exists(txt_path):
            txt_path = os.path.join(args.text_dir, f"{_stem_from_meta(row.to_dict())}.txt")

        if not os.path.exists(txt_path):
            log.warning("Text file missing for %s, skipping", row.get("company"))
            continue

        log.info("Processing %s", row.get("company"))
        result = process_file(txt_path, row.to_dict())
        if result is None:
            continue

        flat, detail, tf = result
        rows.append(flat)
        doc_tfs.append(tf)
        df_counter.update(tf.keys())

        safe_name = re.sub(r"[^\w]", "_", flat["company"])[:60]
        json_path = os.path.join(args.out_dir, f"{safe_name}_{flat['isin']}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(detail, jf, ensure_ascii=False, indent=2)

    if not rows:
        log.warning("No results produced.")
        return

    # exclude company-name tokens + a few report-specific acronyms so TF-IDF
    # surfaces themes, not entity names (axa, bnp, santander ...)
    entity_tokens = set()
    for name in out_df_companies(rows):
        for tok in re.findall(r"[a-z]{3,}", str(name).lower()):
            entity_tokens.add(tok)
    entity_tokens |= {"nfis", "rse", "csr", "gri", "sasb", "tcfd", "ifrs"}
    tfidf_ranked = corpus_tfidf(doc_tfs, df_counter, exclude=entity_tokens)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    log.info("NLP results saved to %s (%d rows)", args.out_csv, len(rows))

    #_________________ visual output
    make_charts(out_df, tfidf_ranked, args.charts_dir)

    #_________________ research summary
    n = len(out_df)
    log.info("=" * 60)
    log.info("CSRD / ESRS ANALYSIS — %d English reports", n)
    log.info("=" * 60)
    log.info("Dominant ESG pillar:\n%s", out_df["dominant_pillar"].value_counts().to_string())
    log.info("Avg ESG keyword hits — env %.0f | soc %.0f | gov %.0f",
             out_df["environment"].mean(), out_df["social"].mean(), out_df["governance"].mean())

    log.info("--- ESRS topical-standard coverage (share of reports, >=%d matches) ---",
             ESRS_COVER_THRESHOLD)
    cov = {s: (out_df[f"esrs::{s}"] >= ESRS_COVER_THRESHOLD).mean() for s in ESRS_STANDARDS}
    for s, v in sorted(cov.items(), key=lambda kv: -kv[1]):
        log.info("  %-26s %5.1f%%", s, v * 100)
    log.info("  avg ESRS standards covered per report: %.1f of %d",
             out_df["esrs_standards_covered"].mean(), len(ESRS_STANDARDS))

    log.info("--- CSRD concept adoption (share of reports mentioning) ---")
    adopt = {c: out_df[f"concept::{c}"].mean() for c in CSRD_CONCEPTS}
    for c, v in sorted(adopt.items(), key=lambda kv: -kv[1]):
        log.info("  %-26s %5.1f%%", c, v * 100)

    log.info("--- most distinctive corpus terms (TF-IDF) ---")
    log.info("  %s", ", ".join(w for w, _ in tfidf_ranked[:15]))


if __name__ == "__main__":    main()
