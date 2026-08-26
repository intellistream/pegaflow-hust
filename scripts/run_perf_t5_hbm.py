#!/usr/bin/env python3
"""
run_perf_t5_hbm.py — T5: shared-resource pressure, 2x2 design.

Two orthogonal dimensions (prereg 2026-08-19, replaces the earlier
occupancy-only design which pressured the wrong cards):
  - external load (cross-process): 4 placeholder vLLM instances on NPU 4-7
    running continuous short-prompt inference (consumes NPU compute +
    PCIe + CPU) while the experiment runs on NPU 0-3.
  - memory headroom (vertical): experiment instances at gmu 0.85 (ample)
    vs 0.95 (KV pool near its cap).

Combos: (0.85, off) baseline / (0.95, off) headroom / (0.85, on) external /
        (0.95, on) full pressure. 4 experiment instances x 3 cycles.

Preregistered:
  - gmu 0.95 arm that fails to start is recorded INVALID (platform
    constraint) — never silently downgraded.
  - standard fail-close gates + TBT stability gate per combo.
  - deliverable: 4-combo comparison of saved / DMA GB/s / TBT p95 /
    failure rate (reported, not gated).

Usage:
  python scripts/run_perf_t5_hbm.py [--combos a,c,b,d] [--dry-run]
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_perf_base import (Experiment, run_experiment, start_vllm, stop_proc,
                           get_npu_free_memory, send_one_streaming,
                           DEFAULT_MODEL, DEFAULT_SERVER_PORT,
                           DEFAULT_CONDA_ENV, DEFAULT_CONDA_ROOT)

PLACEHOLDER_NPUS = [4, 5, 6, 7]
PLACEHOLDER_PORT_BASE = 19100
EXPERIMENT_NPUS = 4
MIN_FREE_MB = 28 * 1024

COMBO_ORDER = [("a", 0.85, False), ("c", 0.95, False),
               ("b", 0.85, True), ("d", 0.95, True)]


def _start_placeholders(log_dir: Path, conda_root: Path, conda_env: str,
                        model_path: str):
    procs = []
    for i, npu in enumerate(PLACEHOLDER_NPUS):
        port = PLACEHOLDER_PORT_BASE + i
        proc = start_vllm(
            port, "read_write", f"t5-ph-{npu}", npu, f"t5_ph_{npu}",
            model_path=model_path, log_dir=log_dir,
            server_port=DEFAULT_SERVER_PORT, conda_root=conda_root,
            conda_env=conda_env,
            gpu_memory_utilization=0.85, use_pegaflow=False,
        )
        procs.append(proc)
        print(f"    → placeholder NPU{npu} ready (port {port})")
    return procs


def _run_load(procs, stop_event: threading.Event, model_path: str):
    """Continuous short-prompt inference against placeholders."""
    ports = [PLACEHOLDER_PORT_BASE + i for i in range(len(procs))]
    prompt = "What is the capital of France?"

    def _loop(port):
        while not stop_event.is_set():
            try:
                send_one_streaming(port, prompt, model_path, max_tokens=32,
                                   timeout=120)
            except Exception:
                time.sleep(1)

    threads = [threading.Thread(target=_loop, args=(p,), daemon=True)
               for p in ports]
    for t in threads:
        t.start()
    return threads


def _arg_value(argv: list[str], name: str, default: str) -> str:
    """Read a common option without consuming it from the base harness argv."""
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return default


def main():
    combos = COMBO_ORDER
    args = []
    combo_arg = None
    argv = sys.argv[1:]
    i = 0
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
        combos = [c for c in COMBO_ORDER if c[0] in combo_arg.split(",")]

    dry = "--dry-run" in args

    if dry:
        for key, gmu, ext in combos:
            exp = Experiment(
                id=f"t5-{key}",
                title=f"T5 combo {key}: gmu={gmu} external={ext} (dry)",
                cycles=1, requests_per_phase=3, num_instances=EXPERIMENT_NPUS,
                gmu=gmu, reproduce_args=["--combos", key])
            run_experiment(exp, argv=args)
        return

    conda_root = Path(_arg_value(
        args, "--conda-root", str(DEFAULT_CONDA_ROOT)
    )).expanduser().resolve()
    conda_env = _arg_value(args, "--conda-env", DEFAULT_CONDA_ENV)
    model_path = _arg_value(args, "--model", DEFAULT_MODEL)
    log_dir = Path("/tmp/perf-t5-placeholders")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Pre-check experiment cards (0-3) have >= 28GB free.
    free = get_npu_free_memory()
    for npu in range(EXPERIMENT_NPUS):
        if free.get(npu, -1) < MIN_FREE_MB:
            print(f"[INVALID] NPU{npu} free {free.get(npu, -1)}MB < 28GB — T5 ABORTED")
            sys.exit(1)

    ph_procs = []
    load_stop = threading.Event()
    load_threads = []

    try:
        for key, gmu, external in combos:
            if external:
                if not ph_procs:
                    print("Starting placeholder instances...")
                    ph_procs = _start_placeholders(
                        log_dir, conda_root, conda_env, model_path)
                    time.sleep(30)
                print("Starting placeholder load...")
                load_stop = threading.Event()
                load_threads = _run_load(ph_procs, load_stop, model_path)
            elif load_threads:
                print("Stopping placeholder load...")
                load_stop.set()
                load_threads = []

            free = get_npu_free_memory()
            print(f"  Free MB on experiment cards: "
                  f"{ {i: free.get(i, -1) for i in range(EXPERIMENT_NPUS)} }")

            exp = Experiment(
                id=f"t5-{key}",
                title=f"T5 combo {key}: gmu={gmu} external={external}",
                cycles=3,
                requests_per_phase=3,
                num_instances=EXPERIMENT_NPUS,
                gmu=gmu,
                reproduce_args=["--combos", key],
                extra_metrics=[("total_s", "Total latency (full response)")],
            )
            print(f"\n{'='*70}\nT5 combo {key}: gmu={gmu} external={external}\n{'='*70}")
            run_experiment(exp, argv=args)
    finally:
        load_stop.set()
        for proc in ph_procs:
            stop_proc(proc)


if __name__ == "__main__":
    main()
