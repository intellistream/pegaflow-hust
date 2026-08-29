# PegaFlow connector equivalence real-run gate

This runbook distinguishes readiness evidence from measured system results.

## Evidence labels

- `preflight-only`: validates imports, model files, executable, ports, NPU
  memory, output uniqueness, exact core/provider sources, runnable CLI,
  command, Python, platform, and both Git revisions. It launches no service and
  proves no connector behavior.
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
  --core-root /workspace/HUST/vllm-hust \
  --python /controlled/env/bin/python \
  --model /workspace/HUST/models/Qwen3-8B \
  --output-dir /new/path/tied-to-the-run
```

Confirm the record reports `ready_for_real_online: true`. A false result is a
blocked preflight, not a negative performance result.

The controlled interpreter is also used for every vLLM service; the runner no
longer activates a hard-coded root-owned Conda environment through a shell.
Preflight requires `vllm`, its engine arguments, `pegaflow`, and its connector
to resolve from the selected core/provider trees, then constructs the actual
vLLM CLI. It also requires an available `torch.npu` runtime and verifies that
the server binary resolves the controlled interpreter's exact Python shared
library. A namespace-only import, a CUDA-only Torch environment, an unresolved
Python ABI, or a different checkout fails closed.

The PegaFlow server embeds Python through PyO3, so it must be built against the
same controlled interpreter. Its virtual-environment site packages, executable
directory, prefix, and Python shared-library directory are also materialized
into the server process environment; selecting the interpreter only for vLLM
subprocesses is insufficient.

## Reproducible Ascend server build

The protobuf compiler is vendored by `pegaflow-proto`; a host-wide `protoc`
installation is not required. Build the debug binary consumed by this runner
with the controlled interpreter and the host's CANN environment:

```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
source /usr/local/Ascend/cann/set_env.sh
export PATH="$HOME/.cargo/bin:$PATH"
export PYO3_PYTHON=/controlled/env/bin/python
cargo build -p pegaflow-server \
  --no-default-features --features ascend --bin pegaflow-server
```

After the build, run preflight with the same `--python` path. Before claiming
`real-online`, also start the binary with a small pool on an otherwise unused
device and require a successful `/health` response; this catches Python ABI,
site-package, CANN-load, and dynamic-library mismatches that `--help` cannot.

## Required equivalence matrix

The runner accepts `--connector-config-mode legacy|typed`. Typed mode selects
the packaged split-role manifest and explicitly allows its declared
`device_access`, `ipc`, and `network_egress` permissions. The following matched
pairs must use the same commit, model, prompts, devices, ports, cache budget,
request order, and server implementation:

1. legacy module-path configuration;
2. typed single-connector selection with an admitted PegaFlow manifest;
3. rollback to the legacy configuration after removing typed selection.

Run each mode into a different fresh output directory. The rollback run must be
a new process start with `legacy`, no typed selection, and no extension manifest
environment. A successful typed startup alone is not rollback evidence.

For every pair retain raw request records, scheduler/worker/API telemetry,
cache-hit evidence, failures, server logs, resolved configuration, and exact
revisions. Compare outputs, hit/miss choices, save/load behavior, TTFT,
throughput, failure policy, and shutdown cleanup. Static imports and CPU mocks
do not satisfy this gate.

## Current host observation

The 2026-08-29 host-112 rerun used an explicit controlled interpreter and core
checkout. It verified exact vLLM source and engine imports, exact PegaFlow and
connector source, the runnable vLLM CLI, all eight 910B2 devices, the selected
model, typed manifest, free ports, and a locally built Ascend server binary.
The initial typed preflight reported `ready_for_real_online: true`, but its
platform gate was incomplete. Server probes initialized ACL and detected runtime
1.17.0, then exposed first a system-Python build and, after a controlled-Python
rebuild, a CUDA-only Torch environment. Host-installed NPU environments also
failed to load `torch_npu` because the CANN installation has no `libhccl.so`.
Preflight now rejects those states through explicit NPU-runtime and Python-ABI
checks. These probes are startup-failure evidence, not `real-online`; a complete
CANN/HCCL environment, successful health probe, and fresh passing preflight are
required before the equivalence matrix starts.

A later read-only audit found all eight 910B2 devices idle with approximately
62 GiB free HBM each, so accelerator capacity is currently available. The
remaining gate is an ABI/package-set problem: the host identifies the CANN
installation as `cann-9.0.0` with package version `26.0.rc1`, exposes
`libhccl_v2.so` and related HCCL v2 libraries but no `libhccl.so`, while the
installed torch-npu `2.10.0` and `2.10.0.post2` extension binaries both retain
a direct `libhccl.so` dependency. After sourcing the shipped CANN environment
and adding the matching Torch library directory, `ldd` reports only
`libhccl.so` unresolved. Do not manufacture a compatibility symlink: install a
vendor-supported complete CANN/HCCL and torch-npu package set, then repeat the
import probe, server health check, and fresh preflight.
