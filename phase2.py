#!/usr/bin/env python3
# phase 2
#analyzed results

#!/usr/bin/env python3

import argparse
import json
import logging
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


#_________________numeric metric regexes
_METRIC_PATTERNS = [
    # e.g. "12,500 tCO2e"  or  "12.5 MWh"
    r"[\d,\.]+\s*(?:tco2e?|t co2e?|co2e?|ghg|mwh|gwh|gj|tj|mj|kwh|tonnes?|mt|kt)",
    # percentage with optional context word
    r"[\d,\.]+\s*%\s*(?:of|reduction|increase|renewable|female|women|recycl\w+)?",
    # headcount: "3,200 employees"
    r"[\d,\.]+\s*(?:employees?|workers?|staff|fte)",
    # monetary: "€ 1.2m"  or  "$500k"
    r"(?:€|\$|eur|usd|gbp)\s*[\d,\.]+\s*(?:m|bn|k|million|billion|thousand)?",
]
_METRIC_RE = [re.compile(p, re.IGNORECASE) for p in _METRIC_PATTERNS]


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
            snippet = m.group(0).strip()
            if snippet not in seen:
                seen.add(snippet)
                hits.append(snippet)
            if len(hits) >= max_hits:
                return hits
    return hits


#_________________ key

def key_sentences(text: str, pillar: str, max_sents: int = 3) -> list:
    """Return up to max_sents sentences with the most keyword hits for a pillar."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    scored = []
    for s in sents:
        s = s.strip()
        if len(s) < 30:
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
    kws    = top_keywords(text)
    mets   = extract_metrics(text)

    pillars  = {p: themes[p] for p in THEMES}
    dominant = max(pillars, key=pillars.get) if any(pillars.values()) else "unknown"

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
        "top_keywords":    ", ".join(w for w, _ in kws[:10]),
        "metric_hits":     len(mets),
    }

    detail = {
        **flat,
        "top_keywords_full": kws,
        "metrics":           mets,
        "env_sentences":     key_sentences(text, "environment"),
        "soc_sentences":     key_sentences(text, "social"),
        "gov_sentences":     key_sentences(text, "governance"),
    }

    return flat, detail


# _________________help

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


def chart_top_keywords(corpus_counter, path, top_n=15):
    """Most frequent content words across the whole corpus."""
    common = corpus_counter.most_common(top_n)
    if not common:
        return
    words = [w for w, _ in common][::-1]
    counts = [c for _, c in common][::-1]

    fig, ax = _new_axes(7, 0.35 * len(words) + 1.6)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    bars = ax.barh(words, counts, height=0.55, color=PILLAR_COLORS["environment"])
    ax.bar_label(bars, fmt="%d", padding=3, color=INK_SECONDARY, fontsize=8)
    ax.set_title("Top keywords across all English reports",
                 loc="left", color=INK, fontsize=12)
    ax.set_xlabel("Occurrences (within per-report top keywords)",
                  color=INK_SECONDARY, fontsize=9)
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


def make_charts(out_df, corpus_counter, charts_dir):
    os.makedirs(charts_dir, exist_ok=True)
    chart_dominant_pillar(out_df, os.path.join(charts_dir, "dominant_pillar.png"))
    chart_pillar_hits(out_df, os.path.join(charts_dir, "pillar_hits.png"))
    chart_density_by_country(out_df, os.path.join(charts_dir, "esg_mix_by_country.png"))
    chart_top_keywords(corpus_counter, os.path.join(charts_dir, "top_keywords.png"))
    chart_year_comparison(out_df, os.path.join(charts_dir, "esg_by_year.png"))


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
    corpus_counter = Counter()
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

        flat, detail = result
        rows.append(flat)
        corpus_counter.update(dict(detail["top_keywords_full"]))


        safe_name = re.sub(r"[^\w]", "_", flat["company"])[:60]
        json_path = os.path.join(args.out_dir, f"{safe_name}_{flat['isin']}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(detail, jf, ensure_ascii=False, indent=2)

    if not rows:
        log.warning("No results produced.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    log.info("NLP results saved to %s (%d rows)", args.out_csv, len(rows))

    #_________________ visual output
    make_charts(out_df, corpus_counter, args.charts_dir)

    #_________________ summary stats
    log.info("--- ESG hit averages ---")
    for p in THEMES:
        log.info("  %-15s avg hits: %.1f", p, out_df[p].mean())
    log.info("  dominant pillar distribution:\n%s",
             out_df["dominant_pillar"].value_counts().to_string())


if __name__ == "__main__":    main()
