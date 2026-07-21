# CSRD

# where_do_we_go_when_we_fall_asleep

the excel of csrd has to be updated to the newer version

## Requirements

Python 3.10+

```bash
pip install -r requirements.txt

# Run after phase1.py has completed
python3 phase2.py

# Test on first 5 documents
python3 phase2.py --limit 5

# Custom paths
python3 phase2.py --summary path/to/extraction_summary.csv --text-dir path/to/texts/

pip3 install --upgrade certifi --> this will be probably asked once u enter certifi ->>  pip install --upgrade pip

**Pipeline
**
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Phase 1 (downloads ALL PDFs with a valid link + extracts text/tables
#    + detects the language of every report -> "language" column in extraction_summary.csv)
python3 phase1.py

# 2b. OR fetch both fiscal years straight from the SRN API (srnav.com) -
#     SRN hosts the PDFs itself, so no dead company links; adds report_year everywhere
python3 phase1.py --source srn --years 2024,2025

# 3. Run Phase 2 (NLP analysis - English reports only)
#    Word output:  nlp_results.csv + per-company JSON in nlp_output/
#    Visual output: charts in nlp_output/charts/
#      - dominant_pillar.png       (reports per dominant ESG pillar)
#      - pillar_hits.png           (avg keyword hits per pillar)
#      - esg_mix_by_country.png    (ESG focus mix by country)
#      - top_keywords.png          (top keywords across the corpus)
#      - esg_by_year.png           (2024 vs 2025 ESG focus, when both years present)
python3 phase2.py

# 3b. Run Phase 2b (Hugging Face NLP layer - free, local, no LLM)
#     A stronger measurement than keyword counting: semantic ESRS scoring
#     (sentence-transformers), FinBERT-ESG classification, and a ClimateBERT
#     greenwashing_index. Output: per-report JSON in hf_output/ + hf_esg_scores.csv
pip install -r requirements-hf.txt          # heavy: torch + transformers (~2 GB)
python3 phase2b_hf.py --limit 5                       # test on a few
python3 phase2b_hf.py                                 # all English reports
python3 phase2b_hf.py --no-finbert --no-climatebert  # embeddings only (lightest)

# 4. Run Phase 3 (ESRS disclosure-requirement extraction with an LLM)
#    Maps each English report to the ESRS Set 1 DR taxonomy (esrs_system_prompt.md):
#    materiality assessment, per-DR status (reported/not-material/missing/phase-in),
#    quantitative datapoints with units + baselines, and quality/greenwashing flags,
#    each grounded in a quoted evidence span.
#    Output: one JSON per report in esrs_output/ + esrs_coverage.csv
#
# 4a. FREE / LOCAL — runs an open model
#       1. install Ollama (https://ollama.com/download)
#       2. ollama pull llama3.1:8b       # or qwen2.5:14b / qwen2.5:32b for better quality
#       3. run:
python3 phase3_local.py --limit 5                       # local Ollama, first few
python3 phase3_local.py --model qwen2.5:32b             # bigger local model (e.g. on ADA)
python3 phase3_local.py --chunk-chars 120000           # split long reports for small models
python3 phase3_local.py --provider groq --model llama-3.3-70b-versatile   # free API tier


# If all goes well since they have updated their website with direct pdf links phase_extra_error_pdfs can be discarded.
