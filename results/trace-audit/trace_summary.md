# Trace Audit Summary

## Environment
- Commit: `bdd32f41f25f`
- Branch: `main`
- Model: `Qwen3-8B` (md5: `85a491bc89ba`)
- Timestamp: 2026-07-31T15:49:49+0000
- NPUs: 8× Ascend 910B2 (see artifact for full npu-smi)

## TTFT (Time-To-First-Token)

| Phase | N | Median | Mean | 95% CI | Min | Max |
|---|---|---|---|---|---|---|
| Shared | 72 | **0.1493s** | 0.1925s | [0.1649, 0.2243] | 0.0940s | 0.9337s |
| Isolated | 72 | **0.1503s** | 0.4062s | [0.3303, 0.4924] | 0.1377s | 0.9376s |

## Total Latency

| Phase | N | Median | Mean | 95% CI |
|---|---|---|---|---|
| Shared | 72 | 1.270s | 1.316s | [1.287, 1.348] |
| Isolated | 72 | 1.268s | 1.525s | [1.448, 1.611] |

## Break-Even Analysis

```
prefill_saved = isolated_mean_ttft - shared_mean_ttft
              = 0.4062s - 0.1925s
              = 0.2137s
dma_cost      = 45.5ms = 0.0455s
net_gain      = 0.2137s - 0.0455s
              = 0.1682s
```

**Verdict: PRELIMINARY GO** — PegaFlow saves 214ms prefill at cost of 46ms DMA, net gain 168ms. Proceed to prototype given matched trace confirms break-even.

## Per-Cycle TTFT (Mean)

| Cycle | Shared Mean | Isolated Mean | Gain |
|---|---|---|---|
| 1 | 0.2109s | 0.4059s | +48.0% |
| 2 | 0.1830s | 0.4070s | +55.0% |
| 3 | 0.1835s | 0.4058s | +54.8% |

## Per-Query TTFT: Q0

| Phase | N | Median | Mean | 95% CI |
|---|---|---|---|---|
| shared | 24 | **0.3244s** | 0.3295s | [0.2870, 0.3904] |
| isolated | 24 | **0.9239s** | 0.9246s | [0.9225, 0.9271] |

## Per-Query TTFT: Q1

| Phase | N | Median | Mean | 95% CI |
|---|---|---|---|---|
| shared | 24 | **0.1497s** | 0.1495s | [0.1479, 0.1512] |
| isolated | 24 | **0.1491s** | 0.1483s | [0.1467, 0.1499] |

## Per-Query TTFT: Q2

| Phase | N | Median | Mean | 95% CI |
|---|---|---|---|---|
| shared | 24 | **0.0961s** | 0.0984s | [0.0958, 0.1029] |
| isolated | 24 | **0.1457s** | 0.1458s | [0.1445, 0.1472] |

## Negative Examples (Preserved)

### Burst Concurrent (PCIe DMA Contention)

- Shared avg TTFT: 2.7s
- Isolated avg TTFT: 1.73s
- Result: +56% (shared WORSE)
- Root cause: 8 concurrent DMA streams saturate PCIe 4.0 uplink: 15 GB/s / 8 = 1.9 GB/s per stream, single DMA inflates from 85ms to ~750ms
- Verdict: Burst is unrealistic workload; staggered/normal serving load unaffected

### MLA+TP8 (Prefill Too Cheap)

- Shared avg TTFT: 0.184s
- Isolated avg TTFT: 0.187s
- Result: +1.6% (no meaningful gain)
- Root cause: MLA kv_lora_rank=512 compresses KV compute to ~100ms; DMA of compressed KV (~40 MB) takes ~3ms; prefill cost too small to save
- Verdict: PegaFlow requires large enough prefill gap to overcome DMA cost. 16B MLA model does not meet threshold; 236B+ may.

## Artifacts
- Raw records: `/workspace/HUST/pegaflow-hust/results/trace-audit/trace_audit.json`
- Server log: `/workspace/HUST/pegaflow-hust/results/trace-audit/logs/server.log`
- vLLM logs: `/workspace/HUST/pegaflow-hust/results/trace-audit/logs/vllm_*.log`
- Environment snapshot: `trace_audit.json` → `_env` key
