# CSRD

A pipeline over the [Sustainability Reporting Navigator](https://www.srnav.com)
CSRD archive: pull the live report index, download every PDF, extract the
sustainability statement (text **and** tables), then analyse it against the ESRS
Set 1 taxonomy.

The index is read **live** from
`https://www.srnav.com/reports?referrer=google-sheet` on every run — no
spreadsheet to keep in sync. At the time of writing that is **1 999 CSRD reports**
(1 040 for FY2024, 959 for FY2025) across 34 countries, 1 990 with a working PDF
link and 1 987 with the page range of the sustainability statement.

## Requirements

Python 3.10+

```bash
pip install -r requirements.txt
```

`langdetect` needs a C toolchain on some platforms. If it won't install, skip it —
phase 1 falls back to a built-in function-word detector that handles the 14
languages in the archive.

## Pipeline

### Phase 1 — index, download, extract

```bash
python3 phase1.py                    # the whole live corpus
python3 phase1.py --limit 20         # smoke test
python3 phase1.py --years 2025       # one fiscal year ('all' for every year)
python3 phase1.py --workers 24       # more download concurrency
python3 phase1.py --no-tables        # text only, roughly 3x faster
python3 phase1.py --source excel     # legacy: links from the archive spreadsheet
```

Outputs:

| file | contents |
| --- | --- |
| `reports_index.csv` | the live index as fetched, one row per report |
| `pdfs/` | downloaded PDFs, `<Company>_<ISIN>_<year>_p<start>-<end>.pdf` |
| `extracted_text/` | one `.txt` per report, with `[[page:N]]` markers |
| `extracted_tables/` | one `_tables.json` per report, each table tagged with its page |
| `download_log.csv` | one row per download attempt, with the final URL and status |
| `extraction_summary.csv` | the master table phases 2 and 3 read |

`extraction_summary.csv` carries the SRN metadata (country, sector, industry,
auditor, `csrd_compliant`, publication date, sustainability page range) next to
the extraction results (language, pages, word count, tables found, file paths).

**Why it's fast.** Three things, worth roughly two orders of magnitude together
against a naive serial loop:

* **Page ranges.** SRN publishes where the sustainability statement sits inside
  each annual report, so a 125 MB / 400-page filing is parsed over its ~90
  relevant pages. This is the single biggest win, and it also sharpens the
  downstream analysis — no financial-statement boilerplate in the corpus.
* **Per-host concurrency.** Downloads run in a thread pool but the throttle is
  *per host*, so ~1 500 different company web servers are hit in parallel while
  no single server ever sees two concurrent requests. Measured at ~130
  reports/min against the old fixed 1.5 s sleep between every download.
* **A process pool for parsing.** PyMuPDF is CPU-bound and holds the GIL, so
  extraction is spread across cores. Each document is opened once and each page
  visited once for text *and* tables.

Everything is resumable. A valid PDF already on disk is not re-downloaded, and a
text file newer than its PDF is not re-parsed. Use `--refresh` / `--re-extract`
to force either, and `--skip-download` to parse what's already in `pdfs/`.

Downloads are streamed to a `.part` file and checked for the `%PDF` magic on the
first chunk, so an HTML error page is abandoned before it costs bandwidth. A
401/403 is retried once with a same-origin `Referer`, which gets past most of the
WAFs. Expect a residual few per cent of failures from IP-blocking datacenter
ranges — `recover_failed.py` retries those from a normal network with a real
browser:

```bash
pip install playwright && playwright install chromium
python recover_failed.py                        # reads download_log.csv
python recover_failed.py --only error:http_403  # one failure category
python3 phase1.py --skip-download               # then parse what it recovered
```

### Phase 2 — keyword ESG analysis

English reports only (language is detected in phase 1).

```bash
python3 phase2.py
python3 phase2.py --limit 5
```

Outputs `nlp_results.csv`, per-company JSON in `nlp_output/`, and charts in
`nlp_output/charts/`: `dominant_pillar.png`, `pillar_hits.png`,
`esg_mix_by_country.png`, `top_keywords.png`, and `esg_by_year.png` when both
fiscal years are present.

### Phase 2b — Hugging Face scoring (free, local, no LLM)

A stronger measurement than keyword counting: semantic ESRS scoring with
sentence-transformers, FinBERT-ESG classification, and a ClimateBERT
`greenwashing_index`.

```bash
pip install -r requirements-hf.txt     # heavy: torch + transformers, ~2 GB
python3 phase2b_hf.py --limit 5
python3 phase2b_hf.py
python3 phase2b_hf.py --no-finbert --no-climatebert   # embeddings only
```

Outputs per-report JSON in `hf_output/` plus `hf_esg_scores.csv`.

### Phase 3 — ESRS disclosure-requirement extraction

Maps each report onto the ESRS Set 1 DR taxonomy in `esrs_system_prompt.md`: the
materiality assessment, a status per DR (reported / not-material / missing /
phase-in), quantitative datapoints with units and baselines, and
quality/greenwashing flags — each grounded in a quoted evidence span and the PDF
page it came from.

Outputs one JSON per report in `esrs_output/` plus `esrs_coverage.csv`.

**Free / local** — runs an open model, nothing leaves the machine:

```bash
# 1. install Ollama: https://ollama.com/download
# 2. ollama pull llama3.1:8b        # or qwen2.5:14b / qwen2.5:32b for better quality
python3 phase3_local.py --limit 5
python3 phase3_local.py --model qwen2.5:32b            # bigger local model
python3 phase3_local.py --chunk-chars 120000           # split long reports for small contexts
python3 phase3_local.py --provider groq --model llama-3.3-70b-versatile   # free API tier
```

**Hosted (Anthropic)** — higher quality, costs money. The system prompt is
prompt-cached, so it's billed once rather than per report:

```bash
export ANTHROPIC_API_KEY=...
python3 phase3_esrs.py --limit 5
```

Both backends share `phase3_esrs.py` for report selection, prompt loading, JSON
repair, chunking and coverage flattening, so their output is directly
comparable.

## Files

| file | role |
| --- | --- |
| `srn_client.py` | live SRN client; decodes the site's SvelteKit payload. Run it standalone to see what the archive currently holds |
| `phase1.py` | index → download → text + tables |
| `phase2.py` | keyword ESG analysis + charts |
| `phase2b_hf.py` | local Hugging Face semantic scoring |
| `phase3_esrs.py` | ESRS extraction (Anthropic) + shared helpers |
| `phase3_local.py` | ESRS extraction (Ollama / vLLM / Groq / OpenRouter / OpenAI) |
| `recover_failed.py` | browser-based retry for WAF-blocked downloads |
| `esrs_system_prompt.md` | the ESRS Set 1 reference taxonomy and output schema |

## Troubleshooting

**SSL certificate errors.** `pip install --upgrade certifi`. Phase 1 also falls
back to an unverified connection when a site's chain is broken.

**The site changed its payload format.** `srn_client.py` tries the SvelteKit data
endpoint first and falls back to parsing the bootstrap script in the HTML. If
both fail it raises with the URL it tried — that module is the only place that
needs updating.
