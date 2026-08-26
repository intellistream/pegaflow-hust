#!/usr/bin/env python3
"""
run_perf_t4_len.py — T4: prompt-length gradient.

Scales the system block to target token counts; measures how prefill-saved
and DMA cost scale with prompt length (the break-even ratio).

Preregistered (perf plan §T4):
  - lengths: 1k / 4k / 8k / 16k tokens
  - length proxy: chars / 4 (English text ~4 chars/token); the system
    block is ~250 tokens per repetition, repeated to the target
  - deliverable: prefill_saved vs DMA cost per length; confirm the
    break-even holds across lengths

Usage:
  python scripts/run_perf_t4_len.py [--lengths 1000,4000,8000,16000] [--dry-run]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_perf_base import Experiment, run_experiment, SYSTEM_PROMPT

LENGTHS = [1000, 4000, 8000, 16000]

# One SYSTEM_PROMPT block is ~265 tokens; we scale by repeating blocks.
_BLOCK = "You are an expert AI assistant with deep knowledge. "


def _prompt_for_tokens(target: int) -> str:
    chars = target * 4
    reps = chars // len(_BLOCK) + 1
    return (_BLOCK * reps)[:chars]


def main():
    lengths = LENGTHS
    args = []
    len_arg = None
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--lengths":
            len_arg = argv[i + 1]
            i += 2
            continue
        if a.startswith("--lengths="):
            len_arg = a.split("=", 1)[1]
            i += 1
            continue
        args.append(a)
        i += 1
    if len_arg:
        lengths = [int(x) for x in len_arg.split(",")]

    for target in lengths:
        prompt = _prompt_for_tokens(target)

        def prompt_fn(q, qi, npu, _p=prompt):
            return f"{_p}\n\nUser: {q}\n\nAssistant:"

        exp = Experiment(
            id=f"t4-l{target // 1000}k",
            title=f"T4: prompt length {target // 1000}k tokens",
            cycles=1,
            requests_per_phase=3,
            warmup_first=True,
            prompt_fn=prompt_fn,
            reproduce_args=["--lengths", str(target)],
            extra_metrics=[("total_s", "Total latency (full response)")],
        )
        print(f"\n{'='*70}\nT4 length: {target // 1000}k tokens\n{'='*70}")
        run_experiment(exp, argv=args)


if __name__ == "__main__":
    main()
