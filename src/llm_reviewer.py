"""LLM-based dataset quality review.

Supports two backends (auto-detected from available env vars / args):
  1. OpenRouter  — set OPENROUTER_API_KEY  (uses urllib, no extra package needed)
  2. Anthropic   — set ANTHROPIC_API_KEY   (requires: pip install anthropic)

OpenRouter is tried first when its key is present.

Adds an optional ``llm_quality`` check that samples records and asks a model to rate:
  - clarity   : How clear and specific is the instruction / prompt?
  - quality   : How high-quality, accurate, and complete is the output?
  - coherence : Does the output actually address the instruction?

Each dimension is scored 0–10.  The check integrates cleanly with the existing
scoring framework and skips gracefully when no key is available.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional Anthropic SDK (fallback backend)
# ---------------------------------------------------------------------------
try:
    import anthropic as _anthropic  # type: ignore
    _ANTHROPIC_SDK = True
except ImportError:
    _ANTHROPIC_SDK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_OR_MODEL  = "anthropic/claude-haiku-4-5"   # OpenRouter model name
_DEFAULT_SDK_MODEL = "claude-haiku-4-5-20251001"     # Anthropic SDK model id
_DEFAULT_SAMPLE    = 20
_DEFAULT_BATCH     = 10

_SYSTEM_PROMPT = """\
You are a dataset quality evaluator for LLM fine-tuning.
Your job is to assess the quality of instruction–output pairs.
Respond ONLY with a valid JSON array — no markdown, no explanation."""

_USER_TEMPLATE = """\
Assess each of the following fine-tuning records.
For each record provide three integer scores from 0 to 10:
  - clarity:   How clear, specific, and well-phrased is the instruction?
               (0 = completely unintelligible, 10 = crystal clear)
  - quality:   How accurate, complete, and well-written is the output?
               (0 = wrong/empty, 10 = excellent)
  - coherence: How well does the output actually address the instruction?
               (0 = completely off-topic, 10 = perfectly on-topic)

Also add a brief "flag" string (≤15 words) only when there is a notable issue;
leave it as null otherwise.

Respond with a JSON array with one object per record, in the same order:
[
  {{"clarity": <int>, "quality": <int>, "coherence": <int>, "flag": <str|null>}},
  ...
]

Records:
{records_json}"""


# ---------------------------------------------------------------------------
# Record text extraction (format-aware)
# ---------------------------------------------------------------------------

def _get_instruction(record: Dict[str, Any]) -> str:
    if "instruction" in record:
        instr = record.get("instruction") or ""
        extra = record.get("input") or ""
        return f"{instr}\n{extra}".strip() if extra else str(instr)
    if "prompt" in record:
        return str(record["prompt"] or "")
    for msg in record.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    for turn in record.get("conversations", []):
        if isinstance(turn, dict) and turn.get("from") == "human":
            return str(turn.get("value") or "")
    for f in ("text", "question", "input"):
        if f in record and record[f]:
            return str(record[f])
    return ""


def _get_output(record: Dict[str, Any]) -> str:
    for f in ("output", "completion", "answer", "response"):
        if f in record and record[f]:
            return str(record[f])
    for msg in reversed(record.get("messages", [])):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    for turn in reversed(record.get("conversations", [])):
        if isinstance(turn, dict) and turn.get("from") in ("gpt", "assistant"):
            return str(turn.get("value") or "")
    return ""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_records_json(records: List[Dict[str, Any]]) -> str:
    items = []
    for i, rec in enumerate(records, 1):
        instr = _get_instruction(rec)
        out   = _get_output(rec)
        items.append({
            "index": i,
            "instruction": instr[:400] if instr else "(none)",
            "output":      out[:400]   if out   else "(none)",
        })
    return json.dumps(items, ensure_ascii=False, indent=2)


def _parse_llm_response(text: str, n: int) -> List[Dict[str, Any]]:
    """Extract JSON array from LLM response, tolerating markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed[:n]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))[:n]
        except json.JSONDecodeError:
            pass
    return [{"clarity": 5, "quality": 5, "coherence": 5, "flag": None}] * n


# ---------------------------------------------------------------------------
# Backend: OpenRouter (urllib — zero extra dependencies)
# ---------------------------------------------------------------------------

def _call_openrouter(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str,
    timeout: int = 60,
) -> str:
    """Call the OpenRouter chat-completions endpoint and return the response text."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
    }).encode("utf-8")

    req = urllib.request.Request(
        _OPENROUTER_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/dakshjain-1616/Fine-tune-Dataset-Quality-Scorer",
            "X-Title": "Dataset Quality Scorer",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenRouter API error {e.code}: {e.read().decode('utf-8', errors='replace')}")


# ---------------------------------------------------------------------------
# Backend: Anthropic SDK
# ---------------------------------------------------------------------------

def _call_anthropic_sdk(
    system: str,
    user_content: str,
    api_key: str,
    model: str,
) -> str:
    client = _anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text if response.content else "[]"


# ---------------------------------------------------------------------------
# Unified call dispatcher
# ---------------------------------------------------------------------------

def _call_llm(
    system: str,
    user_content: str,
    openrouter_key: Optional[str],
    anthropic_key: Optional[str],
    model: str,
) -> str:
    """Call the best available LLM backend and return the response text."""
    if openrouter_key:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ]
        return _call_openrouter(messages, openrouter_key, model)

    if anthropic_key and _ANTHROPIC_SDK:
        return _call_anthropic_sdk(system, user_content, anthropic_key, model)

    raise RuntimeError(
        "No LLM backend available.\n"
        "  • Set OPENROUTER_API_KEY  for OpenRouter (no extra packages needed)\n"
        "  • Or set ANTHROPIC_API_KEY and pip install anthropic"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def review_sample(
    data: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    sample_size: int = _DEFAULT_SAMPLE,
    batch_size: int = _DEFAULT_BATCH,
    seed: int = 42,
) -> Tuple[bool, str, float, Dict[str, Any]]:
    """Run LLM-based quality review on a random sample of records.

    Returns the standard check tuple: ``(passed, message, score, details)``
    compatible with the rest of the quality-check framework.

    Backend selection (first match wins):
      1. ``api_key`` argument (assumes OpenRouter if it starts with ``sk-or-``, else Anthropic)
      2. ``OPENROUTER_API_KEY`` env var  → OpenRouter
      3. ``ANTHROPIC_API_KEY``  env var  → Anthropic SDK

    Skips gracefully (returns a neutral pass) when no key is available.
    """
    # ── Resolve API keys ────────────────────────────────────────────────────
    or_key  = os.environ.get("OPENROUTER_API_KEY")
    sdk_key = os.environ.get("ANTHROPIC_API_KEY") if _ANTHROPIC_SDK else None

    # An explicit api_key argument overrides env vars
    if api_key:
        if api_key.startswith("sk-or-"):
            or_key = api_key
        else:
            sdk_key = api_key

    if not or_key and not sdk_key:
        return (
            True,
            "LLM review skipped — set OPENROUTER_API_KEY (or ANTHROPIC_API_KEY + pip install anthropic)",
            1.0,
            {"skipped": True, "reason": "no API key"},
        )

    # ── Choose model based on backend ───────────────────────────────────────
    if model is None:
        resolved_model = _DEFAULT_OR_MODEL if or_key else _DEFAULT_SDK_MODEL
    else:
        resolved_model = model

    if not data:
        return False, "Empty dataset", 0.0, {}

    # ── Sample records ───────────────────────────────────────────────────────
    rng = random.Random(seed)
    sample_indices = sorted(rng.sample(range(len(data)), min(sample_size, len(data))))
    sample_records = [data[i] for i in sample_indices]

    all_scores: List[Dict[str, Any]] = []

    for batch_start in range(0, len(sample_records), batch_size):
        batch = sample_records[batch_start: batch_start + batch_size]
        records_json = _build_records_json(batch)
        user_content = _USER_TEMPLATE.format(records_json=records_json)

        try:
            raw = _call_llm(_SYSTEM_PROMPT, user_content, or_key, sdk_key, resolved_model)
        except Exception as exc:
            return (
                True,
                f"LLM review encountered an error: {exc}",
                1.0,
                {"skipped": True, "reason": str(exc)},
            )

        all_scores.extend(_parse_llm_response(raw, len(batch)))

    # ── Aggregate ────────────────────────────────────────────────────────────
    def _avg(key: str) -> float:
        vals = [s.get(key, 5) for s in all_scores if isinstance(s.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 5.0

    avg_clarity   = _avg("clarity")
    avg_quality   = _avg("quality")
    avg_coherence = _avg("coherence")
    avg_overall   = (avg_clarity + avg_quality + avg_coherence) / 3

    flagged: List[Dict[str, Any]] = []
    for orig_idx, scores in zip(sample_indices, all_scores):
        if scores.get("flag"):
            flagged.append({
                "row":       orig_idx + 1,
                "clarity":   scores.get("clarity"),
                "quality":   scores.get("quality"),
                "coherence": scores.get("coherence"),
                "flag":      scores["flag"],
            })

    normalised_score = avg_overall / 10.0
    passed = normalised_score >= 0.70    # tightened from 0.60 — require 7/10 to pass

    backend = "OpenRouter" if or_key else "Anthropic"
    details: Dict[str, Any] = {
        "backend":         backend,
        "model":           resolved_model,
        "records_sampled": len(sample_records),
        "avg_clarity":     round(avg_clarity,   2),
        "avg_quality":     round(avg_quality,   2),
        "avg_coherence":   round(avg_coherence, 2),
        "avg_overall_10":  round(avg_overall,   2),
        "flagged_count":   len(flagged),
        "flagged_rows":    flagged[:20],
    }

    label = "PASS" if passed else "FAIL"
    msg = (
        f"LLM review {label} via {backend}/{resolved_model}: "
        f"clarity={avg_clarity:.1f}, quality={avg_quality:.1f}, "
        f"coherence={avg_coherence:.1f} /10  (sample={len(sample_records)})"
    )
    return passed, msg, normalised_score, details
