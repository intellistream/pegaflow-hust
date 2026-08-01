# Trace Audit Preregistration: Ascend KV Transfer Break-Even

**Status**: Host-only (not yet executed on NPU)
**Date**: 2026-08-01
**Prior artifact**: `results/trace-audit-INVALID/` (methodological asymmetry — retained as negative example)

---

## 1. Hypothesis

PegaFlow shared KV cache pool reduces TTFT for cross-instance cache hits
compared to isolated namespaces under matched-lifecycle conditions.
The break-even condition: `prefill_saved > dma_cost`, measured per query class.

---

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Arm A (Shared) | Arm B (Isolated) |
|---|---|---|
| PegaFlow connector | PegaKVConnector | PegaKVConnector |
| Connector mode | read_write | read_write |
| Namespace | `audit-shared-c{cycle}` (all 8 instances same) | `audit-iso-c{cycle}-{i}` (each instance unique) |
| Warmup | 1 request on first-ready instance, 30s seal wait | 1 request on first-ready instance, 30s seal wait |
| Instance count | 8 | 8 |

**Both arms use PegaFlow with `read_write` mode.** The only difference is
namespace sharing. Both arms get fresh per-cycle namespace strings.
The 30s seal wait is calibrated to exceed the ~20-25s write-pipeline sealing
latency observed empirically — earlier 15s runs showed the first 1-2 timed
requests missing while later requests hit, indicating a race with block sealing.

**Both arms use PegaFlow.** The only difference is namespace sharing.
This isolates the cache-sharing variable without confounding PegaFlow vs no-PegaFlow.

### 2.2 Arm Order (AB/BA Alternation)

```
Cycle 1: A(shared) → B(isolated)
Cycle 2: B(isolated) → A(shared)
Cycle 3: A(shared) → B(isolated)
```

### 2.3 Independent Service Lifecycle

Each arm:
1. Start fresh pegaflow-server (`--pool-size 4096mb --devices 0,1,2,3,4,5,6,7`)
2. Start 8 vLLM instances (ThreadPool parallel)
3. Run warmup (1 request, 30s seal wait)
4. Run timed phase (3 queries × 8 instances = 24 requests, sequential round-robin, 0.5s gap, streaming SSE)
5. Shut down all 8 vLLM instances
6. Kill pegaflow-server

Server is NOT reused across arms or cycles.

### 2.4 Fixed Parameters

| Parameter | Value |
|---|---|
| Model | Qwen3-8B, FP16, 32 layers, 8 KV heads |
| Prompt | ~10,000 tokens shared system prompt + distinct user query |
| max-model-len | 16384 |
| max-num-seqs | 4 |
| max_tokens output | 64 |
| gpu-memory-utilization | dynamic: `max(0.15, min(0.85, (free_mb - 4096) / 65536))` |
| PYTHONHASHSEED | 0 |
| ASCEND_RT_VISIBLE_DEVICES | per-instance single NPU binding |
| Transfer backend | ascend_direct |
| Cycles | 3 |
| Requests per phase | 3 |

### 2.5 Hardware

- 8× Ascend 910B2 (64 GB HBM each, PCIe 4.0)
- Kunpeng 920, 512 GB RAM, 8 NUMA nodes
- CANN 9.0.0

---

## 3. Data Collection

### 3.1 Per-Request Raw Fields

| Field | Source | Description |
|---|---|---|
| `cycle` | script | Cycle number (1-3) |
| `phase` | script | "shared" or "isolated" |
| `req_idx` | script | Global request index |
| `query_idx` | script | Query index (0, 1, 2) |
| `instance` | script | Instance label |
| `npu` | script | Physical NPU ID |
| `ttft_s` | HTTP stream | Time to first token (seconds) |
| `total_s` | HTTP stream | Time to all tokens (seconds) |
| `hit_blocks` | vLLM connector log | Blocks matched from PegaFlow |
| `hit_tokens` | vLLM connector log | Tokens matched from PegaFlow |
| `missing_blocks` | server prefetch log | Blocks not in cache |
| `dma_bytes` | server log | DMA transfer bytes |
| `dma_ms` | server log | DMA transfer milliseconds |
| `dma_gbps` | server log | DMA effective bandwidth |

### 3.2 DMA Binding Method

Server log parsed line-by-line with timestamps.
Each prefetch `req_id` matched to nearest subsequent DMA completion on same device.
DMA entry consumed (one DMA per prefetch hit).

---

## 4. Analysis Plan (Preregistered)

### 4.1 Per-Query-Class Paired Comparison

TTFT compared within each query class (Q0, Q1, Q2) separately.
Mixed-class aggregation (mean of Q0+Q1+Q2) is **not** a valid metric.

For each query class:
- Shared median TTFT
- Isolated median TTFT
- Paired delta: `(isolated_median - shared_median) / isolated_median`
- DMA cost: mean of per-request DMA bound to this query class

### 4.2 Per-Cycle Consistency

For each cycle, compute shared mean TTFT and isolated mean TTFT.
Report gain consistency across cycles.

### 4.3 Statistical Reporting

- **Median** (not mean) as primary location statistic
- **IQR** (Q3-Q1) as dispersion
- **95% CI** via bootstrap (percentile method, 1000 resamples)
- Per-cycle breakdown to assess lifecycle independence

### 4.4 Break-Even Criterion

```
For each query class q:
  prefill_saved[q] = isolated_median[q] - shared_median[q]
  dma_cost[q] = mean(per-request DMA for q)

  if prefill_saved[q] > dma_cost[q] AND paired delta statistically significant:
    GO for query class q
  else:
    BREAK-EVEN for query class q
```

---

## 5. Artifact Binding

Each run records:
- `git rev-parse HEAD`
- `git rev-parse --abbrev-ref HEAD`
- Model `config.json` MD5
- `npu-smi info` output
- `PYTHONHASHSEED`, `ASCEND_RT_VISIBLE_DEVICES`, `PEGAFLOW_*` env vars
- Timestamp

All bound to `trace_audit.json` → `_env` key.

---

## 6. Exclusion Rules (Preregistered)

No post-hoc data removal. All data collected under the matched contract above
is included in the report. Conditions that produce negative or null results
(e.g., burst concurrent, MLA low-headroom) are preserved as structured
negative examples with explicit root-cause annotation.

---

## 7. Deliverables

| Artifact | Path |
|---|---|
| Preregistration (this doc) | `docs/trace_preregistration.md` |
| Trace runner (host-only) | `run_trace_audit.py` |
| Prior invalid trace (negative example) | `results/trace-audit-INVALID/` |
| Validated trace (TBD after run) | `results/trace-audit/trace_audit.json` |
| Summary (TBD after run) | `results/trace-audit/trace_summary.md` |

---

## 8. Prior Invalid Artifact

`results/trace-audit-INVALID/` contains the first trace attempt with
the following methodological violations (all corrected in this preregistration):

1. Arm order not alternated (always shared→isolated)
2. Warmup asymmetric (shared had warmup, isolated did not)
3. Namespace reuse asymmetry (shared reused namespace, isolated fresh each cycle)
4. Server lifecycle shared across all cycles (not independent)
5. Mixed-class mean reported as unified prefill saving (conflated Q0/Q1/Q2)
6. DMA global mean, not per-request bound
6. "Burst unrealistic" as post-hoc exclusion

This artifact is preserved as a methodological negative example.
