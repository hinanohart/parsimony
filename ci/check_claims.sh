#!/usr/bin/env bash
# Honest-marketing + determinism claim gate for parsimony.            # honest:ok
# This script intentionally lists forbidden phrases so it can forbid  # honest:ok
# them everywhere else; it never scans itself.                        # honest:ok
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
fail=0

# Files to scan: prose + source. Never this script.
scan=()
for p in README.md docs src; do [ -e "$p" ] && scan+=("$p"); done

# --- Stage 1: forbidden hype phrases (ERE, case-insensitive) ---     # honest:ok
banned='state-of-the-art|outperforms|best-in-class|world-class|production-ready|revolutionary|guaranteed optimal|fully automatic|100% accurate|never fails|blazing fast'  # honest:ok
if [ "${#scan[@]}" -gt 0 ]; then
  if grep -rniE "$banned" "${scan[@]}"; then
    echo "FAIL stage1: forbidden hype phrase found above."
    fail=1
  else
    echo "PASS stage1: no forbidden hype phrases."
  fi
fi

# --- Source guards: zero-LLM, torch-free, deterministic engine ---
core="src/parsimony"
if grep -rnE 'import[[:space:]]+torch' "$core"; then
  echo "FAIL: torch import in core (core must be torch-free)."
  fail=1
else
  echo "PASS: torch-free core."
fi

if grep -rnE 'import[[:space:]]+(openai|anthropic|litellm|cohere|google\.generativeai)' "$core"; then
  echo "FAIL: LLM SDK import in core (policy path must be LLM-call-free)."
  fail=1
else
  echo "PASS: no LLM SDK in core."
fi

engine=()
for f in admission dedup eviction compression objective policy; do
  [ -e "$core/$f.py" ] && engine+=("$core/$f.py")
done
if [ "${#engine[@]}" -gt 0 ]; then
  if grep -rnE 'np\.random|[^_A-Za-z]random\.(random|shuffle|choice|randint|sample|uniform)' "${engine[@]}"; then
    echo "FAIL: nondeterministic RNG in policy engine."
    fail=1
  else
    echo "PASS: policy engine is RNG-free (deterministic)."
  fi
fi

# --- Stage 2: provenance gate (active once benchmark results are committed) ---
if [ -f bench/results.json ]; then
  if ! grep -qiE 'longmemeval|synthetic' README.md; then
    echo "FAIL stage2: results committed but README declares no data provenance."
    fail=1
  else
    echo "PASS stage2: data provenance declared."
  fi
  # Each reported number must carry a source tag.
  if grep -qE '"value"' bench/results.json && ! grep -qE '"source"' bench/results.json; then
    echo "FAIL stage2: results contain values without a 'source' provenance tag."
    fail=1
  fi
else
  echo "SKIP stage2: no bench/results.json yet (provenance gate dormant)."
fi

if [ "$fail" -ne 0 ]; then
  echo "claim-check: FAILED"
  exit 1
fi
echo "claim-check: OK"
