# PegaFlow connector equivalence real-run gate

This runbook distinguishes readiness evidence from measured system results.

## Evidence labels

- `preflight-only`: validates imports, model files, executable, ports, NPU
  memory, output uniqueness, command, Python, platform, and Git revision. It
  launches no service and proves no connector behavior.
- `real-online`: the runner launched every PegaFlow and vLLM process, waited for
  health, issued live requests, saved raw logs, and stopped only those process
  groups it created.
- Historical reports and regenerated summaries remain historical or derived
  artifacts; they are not a rerun of this gate.

## Safety invariants

The runner never invokes `pkill` and never terminates a process discovered by
name. Every child is registered immediately after `Popen`, placed in its own
process group, and stopped through the retained handle on failure, phase exit,
normal shutdown, or interpreter exit. Existing root-owned or shared services
are never reused as a clean baseline.

The output directory must not exist. The runner writes `preflight.json` before
starting a service and refuses to continue unless every preflight check passes.

## Preflight

```bash
python run_bench_8inst.py \
  --preflight-only \
  --project-root /workspace/HUST/pegaflow-hust \
  --model /workspace/HUST/models/Qwen3-8B \
  --output-dir /new/path/tied-to-the-run
```

Confirm the record reports `ready_for_real_online: true`. A false result is a
blocked preflight, not a negative performance result.

## Required equivalence matrix

The current runner covers the legacy `PegaKVConnector` module-path mode only.
It must not be used to claim typed Bundle v1 equivalence until the following
matched pairs use the same commit, model, prompts, devices, ports, cache budget,
request order, and server implementation:

1. legacy module-path configuration;
2. typed single-connector selection with an admitted PegaFlow manifest;
3. rollback to the legacy configuration after removing typed selection.

For every pair retain raw request records, scheduler/worker/API telemetry,
cache-hit evidence, failures, server logs, resolved configuration, and exact
revisions. Compare outputs, hit/miss choices, save/load behavior, TTFT,
throughput, failure policy, and shutdown cleanup. Static imports and CPU mocks
do not satisfy this gate.

## Current host observation

The 2026-08-29 host-112 preflight found all eight 910B2 devices and the selected
Qwen3-8B model available, with the requested ports free. It correctly refused
to run because the PegaFlow server binary and controlled runtime environment
were unavailable to the invoking user. This is `preflight-only` evidence.
