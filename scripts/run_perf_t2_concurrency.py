#!/usr/bin/env python3
"""
run_perf_t2_concurrency.py — T2: concurrency gradient sweep.

Finds the DMA contention boundary: shared-arm benefit should degrade as
concurrent request submission rises (PCIe/NPU DMA contention). Each combo
is one cycle (AB/BA arm pair) with semaphore-limited concurrent sends at a
fixed batch interval.

Preregistered (perf plan §T2):
  - combos: concurrency {1, 4, 8} × batch interval {0, 50, 200} ms
  - metric: Q0 prefill-saved vs DMA cost per combo; contention point =
    first combo where shared Q0 TTFT median stops beating isolated
  - single cycle per combo (scan, not estimate); verdicts per combo are
    informational, the operating-region boundary is the deliverable

Usage:
  python scripts/run_perf_t2_concurrency.py [--combos c1:i1,c2:i2] [--dry-run]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_perf_base import Experiment, run_experiment

COMBOS = [
    (1, 0.0), (1, 0.05), (1, 0.2),
    (4, 0.0), (4, 0.05), (4, 0.2),
    (8, 0.0), (8, 0.05), (8, 0.2),
]


def main():
    combos = COMBOS
    args = []
    combo_arg = None
    i = 0
    argv = sys.argv[1:]
    while i < len(argv):
        a = argv[i]
        if a == "--combos":
            combo_arg = argv[i + 1]
            i += 2
            continue
        if a.startswith("--combos="):
            combo_arg = a.split("=", 1)[1]
            i += 1
            continue
        args.append(a)
        i += 1
    if combo_arg:
        combos = []
        for c in combo_arg.split(","):
            conc, iv = c.split(":")
            combos.append((int(conc), float(iv) / 1000.0))
    for concurrency, interval_s in combos:
        exp_id = f"t2-c{concurrency}-i{int(interval_s * 1000)}"
        exp = Experiment(
            id=exp_id,
            title=f"T2: concurrency={concurrency} interval={int(interval_s*1000)}ms",
            cycles=1,          # scan, not estimate
            requests_per_phase=3,
            concurrency=concurrency,
            batch_interval_s=interval_s,
            warmup_first=True,
            reproduce_args=[
                "--combos", f"{concurrency}:{int(interval_s * 1000)}"
            ],
            extra_metrics=[("total_s", "Total latency (full response)")],
        )
        print(f"\n{'='*70}\nT2 combo: concurrency={concurrency} "
              f"interval={int(interval_s*1000)}ms\n{'='*70}")
        run_experiment(exp, argv=args)


if __name__ == "__main__":
    main()
