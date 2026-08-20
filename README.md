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
python3 phase1.py --no-tables        # text only, ~20x faster parsing
python3 phase1.py --source excel     # legacy: links from the archive spreadsheet
python3 phase1.py --discard-pdfs     # delete each PDF once parsed (GBs, not tens)
python3 phase1.py --max-pending 24   # cap fetched-but-unparsed PDFs (disk backpressure)
```

A full run over all ~1,990 reports is about 15 minutes of downloading, fully
overlapped with roughly three hours of parsing on four cores. Table detection is
essentially the whole of that cost, so `--no-tables` turns the three hours into
minutes when you only need the text.

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

**Why it's fast.**

* **Page ranges.** SRN publishes where the sustainability statement sits inside
  each annual report, so a 125 MB / 400-page filing is parsed over its ~90
  relevant pages. This is the single biggest win, and it also sharpens the
  downstream analysis — no financial-statement boilerplate in the corpus.
* **Per-host concurrency.** Downloads run in a thread pool but the throttle is
  *per host*, so ~1 500 different company web servers are hit in parallel while
  no single server ever sees two concurrent requests. Measured at ~175
  reports/min against the old fixed 1.5 s sleep between every download.
* **Fetching and parsing overlap.** They are not two phases. A PDF goes to the
  process pool the moment its bytes land, so the cores are busy while the next
  batch is still in flight and the run costs about as long as the slower half
  rather than the sum of both.
* **A process pool for parsing.** PyMuPDF is CPU-bound and holds the GIL. Each
  document is opened once and each page visited once for text *and* tables, and
  pages with no vector drawings skip the table finder entirely — it cannot find
  a ruled table where nothing is drawn, and it costs ~40x a text extraction.

**Backpressure.** Downloading is roughly 20x faster than parsing, so left alone
the fetchers would put the whole ~40 GB of PDFs on disk long before the parsers
reached them. A downloader takes one of `--max-pending` slots before it writes
and gets it back when that PDF has been parsed, which holds disk flat at about
`max-pending` × the average report. With `--discard-pdfs` each PDF is deleted
the moment it is parsed, so a full run needs gigabytes rather than tens of them
— the corpus you keep is the text, not the sources.

Everything is resumable, including after `--discard-pdfs`: extraction is reused
from the text file, which does not need the PDF to still exist. Use `--refresh`
/ `--re-extract` to force either, and `--skip-download` to parse what's already
in `pdfs/`.

### How failed downloads are handled

Roughly one link in seven does not hand over a PDF on the first plain request.
Phase 1 works through a ladder of fallbacks before giving up on one:

1. **The link is repaired first.** The archive carries a few malformed URLs —
   a doubled scheme (`hhttps://`), scheme-relative links, `google.com/url?…`
   wrappers, stray spaces. `clean_url` fixes those at index time, so they never
   reach the network broken.
2. **A rejected request is re-dressed and re-asked**, cheapest first: a
   same-origin `Referer`, then `Accept: */*` (which is what a 406 is really
   complaining about), then a full browser header set with cookies picked up
   from the host's own home page. Only statuses that mean *"not to a robot"*
   escalate — 401, 403, 406, 409, 418, 429, 451. A 404 does not: the file is
   genuinely missing and re-asking only wastes the host's time.
3. **Landing pages are mined for the real link.** Up to four candidates per
   page, ranked, drawn from anchors, `<iframe>`/`<embed>` sources, meta-refresh
   redirects, and extensionless `/download` endpoints.
4. **ZIP payloads are unpacked.** ESEF reporting packages — normal in Poland
   and the Baltics — arrive as a zipped bundle; the report PDF is pulled out of
   the archive.
5. **SRN's own cached copy is the last resort.** Where SRN has cached the exact
   same URL, `/api/documents/{id}/download` serves it, which sidesteps a dead
   company link entirely. Matched on the **source URL**, never on
   company-and-year: the page ranges are measured against one specific file, and
   a same-company-same-year guess could silently extract the wrong pages.
   Those rows are marked `success:srn_mirror` in `download_log.csv`. Disable
   with `--no-mirror`.

Measured over all 1,989 links in the live archive, a plain request gets 1,704
of them (85.7%). The ladder above recovers 41 more of the 285 failures — most
of the ZIP packages (7 of 10), 20 of the 55 landing pages, both 406s, both
418s, and a handful of WAFs — for 1,745 (87.7%).

What remains is mostly not a code problem. 93 of the 244 are bot walls
(Cloudflare's JS challenge and Akamai IP denials), 59 are reports genuinely
withdrawn from the company site, and 31 are landing pages that build their
download link in JavaScript. `recover_failed.py` retries exactly those from a
normal network with a real browser:

```bash
pip install playwright && playwright install chromium
python recover_failed.py                        # reads download_log.csv
python recover_failed.py --only error:http_403  # one failure category
python3 phase1.py --skip-download               # then parse what it recovered
```

`download_log.csv` records one row per attempt with the status vocabulary:

| status | meaning |
| --- | --- |
| `success` | PDF downloaded from the company link |
| `success:srn_mirror` | served from SRN's cached copy after the company link failed |
| `skipped` | a valid PDF was already on disk |
| `error:http_<code>` | the server refused, after the full escalation ladder |
| `not_pdf` / `html_no_pdf_link` | a page, not a report, and no usable link on it |
| `error:zip_without_pdf` | an ESEF package containing only XHTML/iXBRL |
| `error:cloudflare_challenge` | a "Just a moment…" JS proof-of-work wall |
| `error:imperva_challenge` | an Incapsula/Imperva interstitial |
| `error:akamai_denied` | Akamai refusing the client's IP range outright |
| `error:timeout` / `error:connection` | the host never answered |
| `error:connection_dropped` | the host hung up mid-handshake — TLS fingerprinting |
| `error:retries_exhausted` | repeated 5xx/429 through the retry budget |

The last five are the categories no amount of header work can fix, and they are
reported as what they are rather than as a generic 403 so you can act on them.
A JS challenge has to be executed, not negotiated; a TLS fingerprint block dies
before a single HTTP header is sent; an IP-range denial needs a different IP.
Recognising a wall also ends the escalation early instead of spending three
more requests on a host that has already made up its mind. All of them go to
the browser path:

```bash
python recover_failed.py --only error:cloudflare_challenge
python recover_failed.py --only error:connection_dropped
```

### Phase 2 — keyword ESG analysis

English reports only (language is detected in phase 1).

```bash
python3 phase2.py
python3 phase2.py --limit 5
python3 phase2.py --workers 8     # scoring processes (default: CPU count)
```

Scoring is pure CPU over the text and every report is independent, so it fans
out across cores; each worker writes its own detail JSON and returns only the
summary row.

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

### Phase 3b — ESRS extraction with neural models, no LLM (free)

The same ESRS output as phase 3, produced by three small encoders on CPU
instead of a generative model. Nothing paid, nothing leaves the machine.

```bash
pip install -r requirements-hf.txt
python3 phase3b_neural.py --limit 5
python3 phase3b_neural.py                 # whole corpus, ~4 h on 4 cores
python3 phase3b_neural.py --no-qa         # statuses only, ~30% faster
```

How it works, and why each model is there:

| stage | model | job |
| --- | --- | --- |
| retrieve | `all-MiniLM-L6-v2` | embed every sentence and every DR, shortlist passages per DR |
| classify | `deberta-v3-xsmall-zeroshot` | does the passage *entail* "this report discloses &lt;DR&gt;"? |
| extract | `tinyroberta-squad2` | pull the number out, the answer span doubling as evidence |

Similarity alone is a poor judge here — sustainability prose is uniformly
on-topic, so best-similarity per DR runs p5=0.46 / median=0.64 across the
corpus. It is used only to shortlist; entailment makes the call.

Two corrections that mattered more than any threshold:

* **The ESRS content index is filtered out.** It repeats every DR title
  verbatim next to a page number, so on title similarity it beats real prose
  every time — in every report sampled, the most-reused top hit was an index
  row. The system prompt warns the LLM about exactly this ("a high-value map,
  not evidence of substantive disclosure"); here it has to be explicit.
* **A sentence may serve at most three DRs.** Cross-reference boilerplate
  ("for further information, please refer to …") otherwise answers a dozen.

**Validation.** Over a 70-report sample, coverage tracks SRN's independent,
human-assigned `csrd_compliant` flag: reports marked *full* score 61.5 DRs
reported against 48.5 for *partial* (Mann-Whitney one-sided, **p = 0.0092**).
It is not simply measuring length — correlation with word count is r = 0.23.

**What it cannot do**, against the LLM backend: it cannot reason about a
disclosure it failed to retrieve, cannot normalise units it has not seen, and
gives a status with a grounded quote rather than an argued judgement.
Datapoint extraction is deliberately conservative — a span with no unit that
looks like a year, a section number, or two numbers run together is rejected
rather than reported — so expect few datapoints per report and trust the ones
you get. Output goes to `esrs_neural_output/` and `esrs_neural_coverage.csv`,
in the same schema as phase 3, so the two backends are directly comparable.

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
python3 phase3_local.py --concurrency 4    # keep at or below the server's parallel slots
```

**Hosted (Anthropic)** — higher quality, costs money. The system prompt is
prompt-cached, so it's billed once rather than per report:

```bash
export ANTHROPIC_API_KEY=...
python3 phase3_esrs.py --limit 5
python3 phase3_esrs.py --concurrency 12    # reports in flight at once
```

Every second of an extraction is spent waiting on the model, so both backends
run reports concurrently rather than one at a time — `--concurrency` is the
single knob that decides whether a full corpus takes an hour or a day. Results
are written per report as they land, so an interrupted run resumes from what is
already in `esrs_output/`.

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
| `phase3b_neural.py` | ESRS extraction with local encoders, no LLM, free |
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
