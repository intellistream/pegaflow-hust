#!/usr/bin/env python3
"""
run_perf_t3_hitrate.py — T3: hit-rate sensitivity.

Controls cross-instance hit rate by prefix ratio: the warmup seed stores the
full prompt; the Q0 test prompt shares only the first `ratio` fraction of the
system block (rest is a distinct suffix), so prefetch hits ~ratio of blocks.

Preregistered (perf plan §T3):
  - ratios: 50 / 75 / 90 / 100 (%)
  - gate: measured Q0 hit_blocks/76 must land within +/-5pp of target,
    else the run is INVALID (controls the manipulation actually worked)
  - deliverable: break-even vs hit rate; the crossover hit-rate threshold
    where prefill_saved <= dma_cost

Usage:
  python scripts/run_perf_t3_hitrate.py [--ratios 50,75,90,100] [--dry-run]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_perf_base import Experiment, run_experiment, SYSTEM_PROMPT

# One system block ~= 250 tokens; full seed = 38 blocks (~10k tokens).
_BLOCKS = 38
BLOCK_SIZE = 128  # vLLM block_size for Qwen3-8B

RATIOS = [50, 75, 90, 100]


def _make_ratio_prompts(ratio: int):
    """Return (warmup_prompt, per-npu test_prompts).

    Warmup seeds the full system block (~10k tokens). Each instance's Q0
    keeps the first `ratio`% of the system block identical (those blocks
    hit) and replaces the remainder with a per-instance DISTINCT filler
    (those blocks miss) — so an earlier instance's save can never make a
    later instance's Q0 a full hit. Total length stays ~10k so prefill
    cost is comparable while only the hit fraction varies.
    """
    full = SYSTEM_PROMPT
    suffix = "\n\nUser: What is the capital of France?\n\nAssistant:"
    cut = int(len(full) * ratio / 100)
    filler_len = len(full) - cut
    # Filler must tokenize at the SAME density as the system block, or the
    # total token count drifts and the hit fraction misses its target.
    # Shuffling the block's own words keeps density identical while the
    # content is distinct (never prefix-hits).
    import random
    words = SYSTEM_PROMPT.split()
    tests = []
    for npu in range(8):
        rng = random.Random(npu + 1)
        shuffled = list(words)
        rng.shuffle(shuffled)
        filler = (" ".join(shuffled) * (filler_len // (len(" ".join(shuffled))) + 1))
        filler = filler[:filler_len]
        tests.append(full[:cut] + filler + suffix)
    return full + suffix, tests


def _hitrate_gate(target: int):
    def gate(all_records, merge_result):
        # Cross-instance hits only: vLLM-local prefix-cache hits (local_hit)
        # are 100% by construction and would bias the mean.
        shared = [r for r in all_records
                  if r.get("phase") == "shared" and r.get("ok")
                  and not r.get("producer") and r.get("query_idx") == 0
                  and not r.get("local_hit")]
        if not shared:
            return ["T3 gate: no cross-instance shared Q0 records"]
        # The FIRST Q0 is the warmup instance's own repeat request (its
        # connector query only covers the unseeded tail; vLLM local-prefix
        # behavior makes it non-representative of cross-instance hits).
        # Exclude it regardless of which NPU it landed on (admission order
        # varies). Prereg deviation D, 2026-08-19.
        first_idx = min(r.get("req_idx", 10**9) for r in shared)
        shared = [r for r in shared if r.get("req_idx") != first_idx]
        if not shared:
            return ["T3 gate: no cross-instance shared Q0 records"]
        # Hit rate = hit_blocks / actual blocks (num_tokens/block_size),
        # not a fixed expected block count — prompt lengths vary.
        ratios = []
        for r in shared:
            blocks = r.get("num_tokens_total", 0) / BLOCK_SIZE
            if blocks > 0:
                ratios.append(r.get("hit_blocks", 0) / blocks)
        if not ratios:
            return ["T3 gate: no blocks to measure"]
        measured = (sum(ratios) / len(ratios)) * 100.0
        if abs(measured - target) > 5.0:
            return [f"T3 gate FAILED: measured hit rate {measured:.1f}% "
                    f"vs target {target}% (deviation > 5pp)"]
        return []
    return gate


def main():
    ratios = RATIOS
    args = []
    ratio_arg = None
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--ratios":
            ratio_arg = argv[i + 1]
            i += 2
            continue
        if a.startswith("--ratios="):
            ratio_arg = a.split("=", 1)[1]
            i += 1
            continue
        args.append(a)
        i += 1
    if ratio_arg:
        ratios = [int(r) for r in ratio_arg.split(",")]

    for ratio in ratios:
        warmup_p, tests = _make_ratio_prompts(ratio)

        def prompt_fn(q, qi, npu, _r=ratio, _w=warmup_p, _tests=tests):
            # qi == -1 is the warmup seed (full prompt); Q0 uses the
            # per-instance test prompt (shared prefix + distinct filler).
            if qi == -1:
                return _w
            if qi == 0:
                return _tests[npu % len(_tests)]
            return f"{SYSTEM_PROMPT}\n\nUser: {q}\n\nAssistant:"

        exp = Experiment(
            id=f"t3-h{ratio}",
            title=f"T3: hit rate {ratio}% (prefix manipulation)",
            cycles=1,
            requests_per_phase=1,   # only Q0 matters for hit rate
            warmup_first=True,
            prompt_fn=prompt_fn,
            reproduce_args=["--ratios", str(ratio)],
            extra_gates=[("hitrate", _hitrate_gate(ratio))],
            extra_metrics=[("total_s", "Total latency (full response)")],
        )
        print(f"\n{'='*70}\nT3 ratio: {ratio}%\n{'='*70}")
        run_experiment(exp, argv=args)


if __name__ == "__main__":
    main()
