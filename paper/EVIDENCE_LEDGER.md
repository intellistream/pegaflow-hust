# PegaFlow paper evidence ledger

This ledger is fail-closed. An execution label describes how an artifact was
produced; it does not make the artifact paper-admissible. Issue #18 records that
15 of 17 historical PegaFlow commits referenced by tracked artifacts were not
reachable from current refs at review time. Those artifacts remain
exploratory until the student owners restore durable code/runtime custody.

| Artifact family | Evidence label | Current use |
|---|---|---|
| `results/trace-audit/**` | real-online trace, exploratory | Qualitative motivation and negative-boundary discovery only; no numeric paper claim. |
| `results/perf-t1*/**` | real-online trace audit, mixed VALID/INVALID | Exploratory; fail-closed audit status is retained, but code/runtime custody blocks citation. |
| `results/perf-t2*/**` | real-online trace audit, mixed VALID/INVALID | Exploratory for contention dimensions; not a formal result. |
| `results/perf-t3*/**` | real-online trace audit, mixed VALID/INVALID | Exploratory hit-coverage boundary; not a formal result. |
| `results/perf-t4*/**` | real-online trace audit, VALID manifests present | Exploratory prompt-length boundary; not a formal result. |
| `results/perf-t5*/**` | real-online trace audit, VALID manifests present | Exploratory multi-tenant/resource boundary; not a formal result. |
| `_archived_docs/benchmark_report.md` | derived narrative | Historical summary only; cannot stand in for raw evidence. |
| `docs/trace_preregistration.md` | contract | Baseline/treatment/oracle and stopping-rule source, not a result. |
| `python/tests/test_npu_multi_instance_simulation.py` and similar tests | simulation/probe | Local semantic qualification only; no serving or performance extrapolation. |
| Future matched matrix | pending | Must bind reachable PegaFlow/vLLM/vLLM-Ascend heads, official runtime identity, allocation, commands, raw receipts, oracle, and result manifest. |

## Claim gate

A result may enter the paper only when all referenced commits are reachable,
raw logs and hashes are present, execution commands are portable, lifecycle
receipts conserve identity, output correctness is 100%, and the result is
mapped to a preregistered baseline/treatment comparison. Invalid and negative
artifacts stay visible and close only the tested mechanism and regime.

## September 2026 storyline gate

Issue #23 narrows the next paper-bearing question to joint P/D host-memory
placement and RDMA-tail control. No existing artifact is treated as evidence
that unordered or bursty RDMA traffic causes request p99 inflation. The next
admissible result must bind object-level lifecycle and transfer receipts to the
same request-level TTFT/TPOT observations under fixed total CPU-memory capacity.
Effective TTL is derived from actual residency; a configured TTL is not a
result.
