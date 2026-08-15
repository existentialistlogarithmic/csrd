#!/usr/bin/env python3
"""Phase 3 (free) — ESRS disclosure-requirement extraction with a local / open model.

Same job as phase3_esrs.py, but instead of the paid Anthropic API it talks to any
OpenAI-compatible endpoint. By default that's a **local** Ollama server, so the
whole analysis runs on your own machine (or on ADA) at zero cost and with no rate
limits, nothing is outsourced. The exact same ESRS system prompt
(esrs_system_prompt.md) and JSON output schema are used; only the model backend
changes.

Providers (choose with --provider, or set --base-url / --model yourself):
  ollama      local Ollama          http://localhost:11434/v1   (FREE, default)
  vllm        local vLLM server     http://localhost:8000/v1     (FREE)
  hf          HF Inference (hosted) https://router.huggingface.co/v1  (free tier, rate-limited)
  groq        Groq free tier        https://api.groq.com/openai/v1
  openrouter  OpenRouter            https://openrouter.ai/api/v1
  openai      OpenAI / ChatGPT      https://api.openai.com/v1

# opoenai i am not sure what version is allowed for free

Local setup (free path):
  1. Install Ollama:  https://ollama.com/download
  2. Pull a model:    ollama pull llama3.1:8b   (or qwen2.5:14b / qwen2.5:32b for better quality)
  3. Run:             python3 phase3_local.py --limit 5

    pip install -r requirements.txt          # installs the openai client

    python3 phase3_local.py --limit 5                      # local Ollama, first few
    python3 phase3_local.py --model qwen2.5:32b            # bigger local model (e.g. on ADA)
    python3 phase3_local.py --chunk-chars 120000          # split long reports for small-context models
    python3 phase3_local.py --provider groq --model llama-3.3-70b-versatile   # free API tier
"""
import argparse
import logging
import os

# reuse the report selection, prompt loading, JSON parsing, chunking, merging
# and coverage flattening — only the model backend differs from phase3_esrs
from phase3_esrs import (
    chunk_text, load_prompt, merge_results, robust_json,
    run_jobs, select_reports, write_outputs,
    FAIL_CSV, OUT_CSV, OUT_DIR, PROMPT_FILE, SUMMARY_CSV, TEXT_DIR,
)

# provider -> (base_url, api-key env var, dummy key when the server needs none)
PROVIDERS = {
    "ollama":     ("http://localhost:11434/v1", "OLLAMA_API_KEY", "ollama"),
    "vllm":       ("http://localhost:8000/v1", "VLLM_API_KEY", "vllm"),
    "hf":         ("https://router.huggingface.co/v1", "HF_TOKEN", None),
    "groq":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY", None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", None),
    "openai":     ("https://api.openai.com/v1", "OPENAI_API_KEY", None),
}
DEFAULT_MODEL = "llama3.1:8b"
MAX_TOKENS = 8000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def make_client(args):
    try:
        from openai import OpenAI
    except ImportError:
        log.error("The 'openai' package is required: pip install -r requirements.txt")
        raise SystemExit(1)
    base_url, key_env, dummy = PROVIDERS[args.provider]
    if args.base_url:
        base_url = args.base_url
    api_key = args.api_key or os.environ.get(key_env) or dummy
    if not api_key:
        log.error("%s requires an API key: set $%s or pass --api-key", args.provider, key_env)
        raise SystemExit(1)
    return OpenAI(base_url=base_url, api_key=api_key)


def call_model(client, model, prompt, report_text, use_json_mode=True):
    """One OpenAI-compatible chat completion; returns parsed dict."""
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": report_text},
        ],
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:  # some servers/models reject response_format — retry without it
        if use_json_mode and "response_format" in str(e).lower():
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return robust_json(resp.choices[0].message.content)


def extract_one(client, model, prompt, text, chunk_chars, json_mode):
    chunks = chunk_text(text, chunk_chars)
    parts = [call_model(client, model, prompt, c, json_mode) for c in chunks]
    return merge_results(parts)


def main():
    parser = argparse.ArgumentParser(description="CSRD ESRS extraction — free / local (OpenAI-compatible)")
    parser.add_argument("--summary", default=SUMMARY_CSV)
    parser.add_argument("--text-dir", default=TEXT_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--out-csv", default=OUT_CSV)
    parser.add_argument("--provider", default="ollama", choices=list(PROVIDERS))
    parser.add_argument("--base-url", help="Override the provider's base URL")
    parser.add_argument("--api-key", help="API key/token (overrides the provider's env var)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name (e.g. llama3.1:8b, qwen2.5:32b)")
    parser.add_argument("--chunk-chars", type=int, default=0,
                        help="Split reports longer than this many chars (for small-context models); 0 = never")
    parser.add_argument("--no-json-mode", action="store_true",
                        help="Don't request response_format=json_object (some models/servers reject it)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Reports extracted in parallel. Keep this at or below "
                             "the server's parallel slots (Ollama defaults to 4)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(args.summary):
        log.error("Summary CSV not found: %s — run phase1.py first", args.summary)
        return
    prompt = load_prompt(PROMPT_FILE)
    jobs = select_reports(args)
    if not jobs:
        log.warning("Nothing to do (all reports already extracted? use --overwrite to redo).")
        return

    client = make_client(args)
    base_url = args.base_url or PROVIDERS[args.provider][0]
    log.info("%d report(s) to extract via %s (%s), model=%s, %d at a time",
             len(jobs), args.provider, base_url, args.model, args.concurrency)

    rows, failures = run_jobs(
        jobs,
        lambda text: extract_one(client, args.model, prompt, text,
                                 args.chunk_chars, not args.no_json_mode),
        args.concurrency)
    write_outputs(rows, failures, args.out_csv, FAIL_CSV)


if __name__ == "__main__":
    main()
