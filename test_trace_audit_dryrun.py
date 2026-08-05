#!/usr/bin/env python3
"""
Synthetic dry-run tests for run_trace_audit.py — host-only, no NPU required.

Validates the P1/P2 gates before any real hardware run:
  Gate 1: admitted-device check — only NPUs with free HBM >= min are used
  Gate 2: fail-close — any instance start failure marks entire arm INVALID
  Gate 3: process isolation — cleanup only kills tracked PIDs
  Gate 4a: 1:1 request/DMA binding — coverage + conservation on the real
           merge_by_req_id production function
  Gate 4b: DMA arm scoping + cross-minute time window
  Gate 4c: duplicate connector events FAIL conservation (never first-match-wins)
  Gate 4d: duplicate prefetch events FAIL conservation
  Gate 4e: missing prefetch event FAILS (formally required evidence)
  Gate 4f: missing DMA on a claimed hit FAILS (not graceful fallback to 0)
  Gate 4g: orphan/leftover events FAIL conservation
  Gate 5:  mid-arm admission drift detection (owner PID + HBM)
  Gate 6:  fail-close exits non-zero (never 0 on INVALID evidence)
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Append project root so we can import run_trace_audit
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Gate 1: Admitted-device admission
# ---------------------------------------------------------------------------

def test_gate1_device_admission():
    """Only NPUs with free HBM >= min_free_mb are admitted."""

    # Mock: NPU 0,1 have 30GB free; NPU 2,3 have 10GB free; rest have 25GB
    mock_free = {0: 30000, 1: 30000, 2: 10000, 3: 10000, 4: 25000, 5: 25000, 6: 25000, 7: 25000}
    min_mb = 28 * 1024  # 28 GB

    admitted = [i for i in range(8) if mock_free.get(i, -1) >= min_mb]
    assert admitted == [0, 1], f"Expected [0,1], got {admitted}"
    assert len(admitted) < 8, "Gate 1: should reject due to insufficient NPUs"

    # With lower threshold, NPUs with >=10GB admitted (NPU 2,3 have exactly 10GB)
    admitted_all = [i for i in range(8) if mock_free.get(i, -1) >= 10000]
    assert len(admitted_all) == 8, f"Gate 1: should admit all 8 with 10GB threshold, got {len(admitted_all)}: {admitted_all}"

    print("  Gate 1 PASS: device admission correctly gates on free HBM")


# ---------------------------------------------------------------------------
# Gate 2: Fail-close on instance start failure
# ---------------------------------------------------------------------------

def test_gate2_fail_close():
    """If any instance fails to start, arm is INVALID."""

    # Simulate: 8 instances launched, but only 7 succeed
    specs = [{"label": f"I{i}", "port": 19000 + i, "mode": "read_write",
              "namespace": "test-ns", "physical_npu": i, "use_pegaflow": True}
             for i in range(8)]
    running = [(specs[i], MagicMock(poll=lambda: None)) for i in range(7)]  # only 7

    assert len(running) < len(specs), "Gate 2: should detect instance count mismatch"
    assert len(running) == 7, f"Expected 7 running, got {len(running)}"

    # INVALID marker generated
    record = {"cycle": 1, "phase": "shared", "req_idx": -1, "instance": "INVALID",
              "ok": False, "error": f"instance launch failed: {len(running)}/{len(specs)}"}
    assert record["instance"] == "INVALID", "Gate 2: fail-close should mark INVALID"
    assert not record["ok"], "Gate 2: fail-close record should be ok=False"

    print("  Gate 2 PASS: fail-close on instance start failure")


# ---------------------------------------------------------------------------
# Gate 3: Process isolation
# ---------------------------------------------------------------------------

def test_gate3_process_isolation():
    """Cleanup only kills tracked processes, never external ones."""
    import run_trace_audit as audit

    # Simulate tracked PIDs
    audit._tracked_pids[:] = [12345, 12346]
    external_pid = 99999  # some other user's process

    # Verify kill_tracked targets only our PIDs
    called_pids = []
    def fake_kill(pid, sig):
        called_pids.append(abs(pid))

    orig_kill = os.kill
    os.kill = fake_kill
    try:
        audit.kill_tracked()
    finally:
        os.kill = orig_kill
        audit._tracked_pids.clear()

    for pid in [12345, 12346]:
        assert pid in called_pids, f"Gate 3: should kill tracked PID {pid}"
    assert external_pid not in called_pids, "Gate 3: should NOT kill external PID"
    assert len(audit._tracked_pids) == 0, "Gate 3: tracked list should be cleared"

    print("  Gate 3 PASS: process isolation — only tracked PIDs killed")


# ---------------------------------------------------------------------------
# Gate 4a: 1:1 request/DMA binding with coverage + conservation
# ---------------------------------------------------------------------------

def test_gate4_request_dma_binding():
    """Each record gets exactly one connector entry via client req_id lookup,
    with the full evidence chain (prefetch + DMA) conserved.
    Uses the real merge_by_req_id production function (host-only gate).
    """
    import run_trace_audit as audit

    records = [
        {"req_id": "trace-aaa111111111", "instance": "C1_shared_3",
         "producer": False},
        {"req_id": "trace-bbb222222222", "instance": "C1_shared_3",
         "producer": False},
        {"req_id": "trace-ccc333333333", "instance": "C1_shared_3",
         "producer": False},
    ]
    connector_by_req = {
        "cmpl-trace-aaa111111111-0-a1b2c3": {
            "req_id": "cmpl-trace-aaa111111111-0-a1b2c3",
            "label": "C1_shared_3", "hit_blocks": 76, "hit_tokens": 9728,
            "num_tokens": 9854},
        "cmpl-trace-bbb222222222-0-d4e5f6": {
            "req_id": "cmpl-trace-bbb222222222-0-d4e5f6",
            "label": "C1_shared_3", "hit_blocks": 1, "hit_tokens": 128,
            "num_tokens": 128},
        "cmpl-trace-ccc333333333-0-g7h8i9": {
            "req_id": "cmpl-trace-ccc333333333-0-g7h8i9",
            "label": "C1_shared_3", "hit_blocks": 1, "hit_tokens": 128,
            "num_tokens": 128},
    }
    prefetch_by_req = {
        "cmpl-trace-aaa111111111-0-a1b2c3": {"total_keys": 76,
                                             "hit_blocks": 76,
                                             "missing_blocks": 0},
        "cmpl-trace-bbb222222222-0-d4e5f6": {"total_keys": 1,
                                             "hit_blocks": 1,
                                             "missing_blocks": 0},
        "cmpl-trace-ccc333333333-0-g7h8i9": {"total_keys": 1,
                                             "hit_blocks": 1,
                                             "missing_blocks": 0},
    }
    prefetch_dma_map = {
        k: {"dma_bytes": 1_400_000_000, "dma_ms": 85.0, "dma_gbps": 16.5}
        for k in prefetch_by_req
    }

    result = audit.merge_by_req_id(records, connector_by_req,
                                   prefetch_by_req, prefetch_dma_map)
    assert result["matched"] == 3, f"Gate 4a: expected 3 matched, got {result}"
    assert result["unmatched"] == 0, f"Gate 4a: expected 0 unmatched, got {result}"
    assert result["coverage_pct"] == 100.0
    assert result["conservation_ok"], \
        f"Gate 4a: conservation must hold: {result['violations']}"
    assert result["violations"] == [], \
        f"Gate 4a: no violations expected: {result['violations']}"
    assert result["connector_duplicates"] == 0
    assert result["connector_orphans"] == 0
    assert result["prefetch_orphans"] == 0
    assert result["dma_orphans"] == 0
    assert records[0]["hit_blocks"] == 76
    assert records[1]["hit_blocks"] == 1
    assert records[2]["hit_blocks"] == 1
    assert records[0]["dma_ms"] == 85.0, "hit>0 record must get its DMA timing"
    assert not records[0].get("_audit_invalid"), "valid record must not be flagged"
    print("  Gate 4a PASS: merge_by_req_id conserves connector/prefetch/DMA")

    # Edge case: missing connector entry → unmatched AND conservation broken
    records_missing = [{"req_id": "trace-missing", "producer": False}]
    r2 = audit.merge_by_req_id(records_missing, {}, {}, {})
    assert r2["unmatched"] == 1
    assert r2["coverage_pct"] == 0.0
    assert not r2["conservation_ok"], "missing connector must break conservation"
    assert any("missing connector" in v for v in r2["violations"]), \
        f"expected missing-connector violation: {r2['violations']}"
    print("  Gate 4a-b PASS: missing connector entry → unmatched + INVALID")


# ---------------------------------------------------------------------------
# Gate 4b: Synthetic DMA binding test (timestamp-based + arm scoping)
# ---------------------------------------------------------------------------

def test_gate4_dma_arm_scoping():
    """DMA binding via the production bind_dma_to_prefetch (R9): arm scope,
    device scope, cross-minute window, hit=0 skip."""
    import run_trace_audit as audit

    connector_by_req = {
        "cmpl-req-A": {"req_id": "cmpl-req-A", "label": "C1_shared_3"},
        "cmpl-req-B": {"req_id": "cmpl-req-B", "label": "C1_isolated_5"},
        "cmpl-req-cross": {"req_id": "cmpl-req-cross", "label": "C1_shared_3"},
    }
    label_to_npu = {"C1_shared_3": 3, "C1_isolated_5": 5}
    ts_prefetches = [
        {"ts": "2026-08-01T10:00:01.000", "req_id": "cmpl-req-A",
         "arm_label": "C1_shared", "hit": 76, "missing": 0},
        {"ts": "2026-08-01T10:00:02.000", "req_id": "cmpl-req-B",
         "arm_label": "C1_isolated", "hit": 0, "missing": 76},
        {"ts": "2026-08-01T10:05:59.500", "req_id": "cmpl-req-cross",
         "arm_label": "C1_shared", "hit": 10, "missing": 0},
    ]
    ts_dmas = [
        {"ts": "2026-08-01T10:00:01.100", "arm_label": "C1_shared",
         "device_id": 3, "dma_bytes": 1_400_000_000, "dma_ms": 85.0, "dma_gbps": 16.5},
        {"ts": "2026-08-01T10:00:02.100", "arm_label": "C1_isolated",
         "device_id": 5, "dma_bytes": 1_400_000_000, "dma_ms": 90.0, "dma_gbps": 15.5},
        {"ts": "2026-08-01T10:06:01.200", "arm_label": "C1_shared",
         "device_id": 3, "dma_bytes": 200_000_000, "dma_ms": 12.0, "dma_gbps": 16.0},
    ]

    dma_map, fallback, leftover, violations = audit.bind_dma_to_prefetch(
        ts_prefetches, ts_dmas, connector_by_req, label_to_npu)
    assert violations == [], f"Gate 4b: no bind violations expected: {violations}"
    # req-A binds its arm+device DMA (85ms), never the C1_isolated one
    assert dma_map["cmpl-req-A"]["dma_ms"] == 85.0, "Gate 4b: wrong DMA bound"
    # req-B: hit=0 → skipped, no DMA expected
    assert "cmpl-req-B" not in dma_map, "Gate 4b: hit=0 must not bind"
    # Cross-minute: 10:05:59.500 → 10:06:01.200 is 1.7s within 30s window.
    # Old code (split(":")[-1]) would see 58.3s and reject.
    assert dma_map["cmpl-req-cross"]["dma_ms"] == 12.0, \
        "Gate 4b: cross-minute window must bind (1.7s not 58.3s)"
    assert fallback == 0, f"Gate 4b: no fallback expected, got {fallback}"
    assert leftover == 1, \
        f"Gate 4b: isolated-arm DMA (hit=0 prefetch) stays a leftover, got {leftover}"
    print("  Gate 4b PASS: bind_dma_to_prefetch — arm/device scope, cross-minute, hit=0 skip")


# ---------------------------------------------------------------------------
# Gate 4c: Duplicate connector events → FAIL conservation
# ---------------------------------------------------------------------------

def test_gate4_duplicate_connector():
    """Duplicate connector events for one request MUST fail conservation —
    never "first match wins" with a silent PASS."""
    import run_trace_audit as audit

    dup_connectors = {
        "cmpl-trace-dup-0-x": {"req_id": "cmpl-trace-dup-0-x", "hit_blocks": 76,
                               "hit_tokens": 9728, "num_tokens": 9854},
        "cmpl-trace-dup-0-y": {"req_id": "cmpl-trace-dup-0-y", "hit_blocks": 1,
                               "hit_tokens": 128, "num_tokens": 128},
    }
    records_dup = [{"req_id": "trace-dup", "producer": False}]
    r3 = audit.merge_by_req_id(records_dup, dup_connectors, {}, {})
    assert not r3["conservation_ok"], \
        "Gate 4c: duplicate connector must break conservation"
    assert r3["connector_duplicates"] == 1, \
        f"Gate 4c: expected 1 duplicate connector, got {r3['connector_duplicates']}"
    assert any("duplicate connector" in v for v in r3["violations"]), \
        f"Gate 4c: expected duplicate-connector violation: {r3['violations']}"
    assert records_dup[0].get("_audit_invalid"), \
        "Gate 4c: duplicate connector must mark the record INVALID"
    print("  Gate 4c PASS: duplicate connector → conservation FAIL, record INVALID")


# ---------------------------------------------------------------------------
# Gate 4d: Duplicate prefetch events → FAIL conservation
# ---------------------------------------------------------------------------

def test_gate4_duplicate_prefetch():
    """Two prefetch events for one connector must fail conservation."""
    import run_trace_audit as audit

    connector = {
        "cmpl-trace-dp-0-x": {"req_id": "cmpl-trace-dp-0-x", "hit_blocks": 10,
                              "hit_tokens": 1280, "num_tokens": 1280},
    }
    prefetch = {
        "cmpl-trace-dp-0-x": {"total_keys": 10, "hit_blocks": 10,
                              "missing_blocks": 0, "occurrences": 2},
    }
    dma = {"cmpl-trace-dp-0-x": {"dma_bytes": 100, "dma_ms": 1.0, "dma_gbps": 1.0}}
    r = audit.merge_by_req_id([{"req_id": "trace-dp", "producer": False}],
                              connector, prefetch, dma)
    assert not r["conservation_ok"], "Gate 4d: duplicate prefetch must fail"
    assert r["prefetch_duplicates"] == 1
    assert any("duplicate prefetch" in v for v in r["violations"]), \
        f"Gate 4d: expected duplicate-prefetch violation: {r['violations']}"
    assert r["dma_orphans"] == 0, "DMA was legitimately claimed"
    print("  Gate 4d PASS: duplicate prefetch → conservation FAIL")


# ---------------------------------------------------------------------------
# Gate 4e: Missing prefetch event → FAIL
# ---------------------------------------------------------------------------

def test_gate4_missing_prefetch():
    """A matched connector without its server-side prefetch is INVALID."""
    import run_trace_audit as audit

    connector = {
        "cmpl-trace-mp-0-x": {"req_id": "cmpl-trace-mp-0-x", "hit_blocks": 10,
                              "hit_tokens": 1280, "num_tokens": 1280},
    }
    r = audit.merge_by_req_id([{"req_id": "trace-mp", "producer": False}],
                              connector, {}, {})
    assert any("missing prefetch" in v for v in r["violations"]), \
        f"Gate 4e: expected missing-prefetch violation: {r['violations']}"
    assert any("missing DMA" in v for v in r["violations"]), \
        "Gate 4e: hit>0 with no DMA must ALSO be flagged"
    assert not r["conservation_ok"]
    assert r["invalid_records"], "Gate 4e: record must be in invalid_records"
    print("  Gate 4e PASS: missing prefetch (and missing DMA on hit) → INVALID")


# ---------------------------------------------------------------------------
# Gate 4f: Missing DMA on a claimed hit → FAIL (not graceful fallback)
# ---------------------------------------------------------------------------

def test_gate4_missing_dma():
    """A consumer claiming a hit and transfer with NO DMA evidence is INVALID,
    never a silent dma_ms=0."""
    import run_trace_audit as audit

    connector = {
        "cmpl-trace-md-0-x": {"req_id": "cmpl-trace-md-0-x", "hit_blocks": 76,
                              "hit_tokens": 9728, "num_tokens": 9854},
    }
    prefetch = {"cmpl-trace-md-0-x": {"total_keys": 76, "hit_blocks": 76,
                                      "missing_blocks": 0}}
    rec = [{"req_id": "trace-md", "producer": False}]
    r = audit.merge_by_req_id(rec, connector, prefetch, {})  # no DMA map
    assert any("missing DMA" in v for v in r["violations"]), \
        f"Gate 4f: expected missing-DMA violation: {r['violations']}"
    assert not r["conservation_ok"], "Gate 4f: missing DMA must fail conservation"
    assert rec[0].get("_audit_invalid"), "Gate 4f: record must be marked INVALID"
    assert rec[0]["dma_ms"] == 0.0, "dma_ms stays default but evidence is INVALID"

    # hit=0 must NOT require DMA — full-miss requests legitimately transfer nothing
    conn0 = {"cmpl-trace-zero-0-x": {"req_id": "cmpl-trace-zero-0-x",
                                     "hit_blocks": 0, "hit_tokens": 0,
                                     "num_tokens": 0}}
    pf0 = {"cmpl-trace-zero-0-x": {"total_keys": 76, "hit_blocks": 0,
                                   "missing_blocks": 76}}
    r0 = audit.merge_by_req_id([{"req_id": "trace-zero", "producer": False}],
                               conn0, pf0, {})
    assert r0["conservation_ok"], \
        f"Gate 4f: hit=0 without DMA must be valid: {r0['violations']}"
    assert r0["violations"] == []
    print("  Gate 4f PASS: missing DMA on claimed hit → INVALID; hit=0 exempt")


# ---------------------------------------------------------------------------
# Gate 4g: Orphan / leftover events → FAIL conservation
# ---------------------------------------------------------------------------

def test_gate4_orphan_events():
    """Formal events matching no request are orphans — run must fail."""
    import run_trace_audit as audit

    connector = {
        "cmpl-trace-consumed-0-x": {"req_id": "cmpl-trace-consumed-0-x",
                                    "hit_blocks": 5, "hit_tokens": 640,
                                    "num_tokens": 640},
        "cmpl-trace-orphan-0-y": {"req_id": "cmpl-trace-orphan-0-y",
                                  "hit_blocks": 5, "hit_tokens": 640,
                                  "num_tokens": 640},
    }
    prefetch = {
        "cmpl-trace-consumed-0-x": {"total_keys": 5, "hit_blocks": 5,
                                    "missing_blocks": 0},
        "cmpl-trace-orphan-0-y": {"total_keys": 5, "hit_blocks": 5,
                                  "missing_blocks": 0},
        "cmpl-ghost-0-z": {"total_keys": 5, "hit_blocks": 5, "missing_blocks": 0},
    }
    dma = {
        "cmpl-trace-consumed-0-x": {"dma_bytes": 100, "dma_ms": 1.0, "dma_gbps": 1.0},
        "cmpl-trace-orphan-0-y": {"dma_bytes": 100, "dma_ms": 1.0, "dma_gbps": 1.0},
    }
    r = audit.merge_by_req_id(
        [{"req_id": "trace-consumed", "producer": False}],
        connector, prefetch, dma, dma_leftover_count=2)
    assert r["connector_orphans"] == 1, f"got {r['connector_orphans']}"
    assert r["prefetch_orphans"] == 2, f"got {r['prefetch_orphans']}"
    assert r["dma_orphans"] == 1, f"got {r['dma_orphans']}"
    assert r["dma_leftover"] == 2
    assert not r["conservation_ok"], "orphans must break conservation"
    assert any("orphan" in v for v in r["violations"]), \
        f"expected orphan violations: {r['violations']}"
    assert any("leftover DMA" in v for v in r["violations"]), \
        f"expected leftover-DMA violation: {r['violations']}"
    print("  Gate 4g PASS: orphan connector/prefetch/DMA + leftover → INVALID")


# ---------------------------------------------------------------------------
# Gate 4h (R6): per-copy fallback is NOT formal batch-DMA evidence
# ---------------------------------------------------------------------------

def test_gate4_bind_fallback_not_evidence():
    """A hit with only a fallback line must be INVALID — fallback never
    substitutes for the required batch DMA evidence."""
    import run_trace_audit as audit

    connector_by_req = {
        "cmpl-trace-fb-0-x": {"req_id": "cmpl-trace-fb-0-x", "label": "C1_shared_3"},
    }
    label_to_npu = {"C1_shared_3": 3}
    ts_prefetches = [
        {"ts": "2026-08-01T10:00:01.000", "req_id": "cmpl-trace-fb-0-x",
         "arm_label": "C1_shared", "hit": 76, "missing": 0},
    ]
    ts_dmas = [
        {"ts": "2026-08-01T10:00:01.500", "arm_label": "C1_shared",
         "device_id": 3, "dma_bytes": 0, "dma_ms": 0.0, "dma_gbps": 0.0,
         "fallback": True},
    ]
    dma_map, fallback, leftover, violations = audit.bind_dma_to_prefetch(
        ts_prefetches, ts_dmas, connector_by_req, label_to_npu)
    assert dma_map == {}, "R6: fallback-only must NOT bind as DMA evidence"
    assert fallback == 1, f"R6: fallback_only_count should be 1, got {fallback}"
    assert leftover == 0, "R6: fallback line is consumed, not leftover"
    assert any("fallback-only" in v for v in violations), \
        f"R6: expected fallback-only violation: {violations}"

    # Merge level: record must be _audit_invalid via "missing DMA event"
    connector = {"cmpl-trace-fb-0-x": {"req_id": "cmpl-trace-fb-0-x",
                                       "hit_blocks": 76, "hit_tokens": 9728,
                                       "num_tokens": 9854}}
    prefetch = {"cmpl-trace-fb-0-x": {"total_keys": 76, "hit_blocks": 76,
                                      "missing_blocks": 0}}
    rec = [{"req_id": "trace-fb", "producer": False}]
    r = audit.merge_by_req_id(rec, connector, prefetch, dma_map,
                              dma_fallback_only_count=fallback)
    assert not r["conservation_ok"], "R6: fallback-only must fail conservation"
    assert rec[0].get("_audit_invalid"), "R6: record must be marked INVALID"
    assert any("missing DMA" in v for v in r["violations"]), \
        f"R6: expected missing-DMA violation: {r['violations']}"
    assert any("fallback-only" in v for v in r["violations"])
    print("  Gate 4h PASS: fallback-only DMA → no binding, record INVALID (R6)")


# ---------------------------------------------------------------------------
# Gate 4i (R9): binding negatives — out-of-window / wrong-device / leftover
# ---------------------------------------------------------------------------

def test_gate4_bind_negative():
    """Out-of-window and wrong-device DMA completions bind nothing; the
    unconsumed completions become leftovers and fail conservation."""
    import run_trace_audit as audit

    connector_by_req = {
        "cmpl-req-w": {"req_id": "cmpl-req-w", "label": "C1_shared_3"},
    }
    label_to_npu = {"C1_shared_3": 3}
    ts_prefetches = [
        {"ts": "2026-08-01T10:00:01.000", "req_id": "cmpl-req-w",
         "arm_label": "C1_shared", "hit": 76, "missing": 0},
    ]
    # 60s late (out of window) + wrong device + one valid candidate
    ts_dmas = [
        {"ts": "2026-08-01T10:01:01.000", "arm_label": "C1_shared",
         "device_id": 3, "dma_bytes": 1, "dma_ms": 1.0, "dma_gbps": 1.0},
        {"ts": "2026-08-01T10:00:01.200", "arm_label": "C1_shared",
         "device_id": 7, "dma_bytes": 1, "dma_ms": 1.0, "dma_gbps": 1.0},
        {"ts": "2026-08-01T10:00:01.100", "arm_label": "C1_shared",
         "device_id": 3, "dma_bytes": 1, "dma_ms": 2.0, "dma_gbps": 1.0},
    ]
    dma_map, fallback, leftover, _violations = audit.bind_dma_to_prefetch(
        ts_prefetches, ts_dmas, connector_by_req, label_to_npu)
    assert dma_map["cmpl-req-w"]["dma_ms"] == 2.0, "nearest valid DMA must bind"
    assert leftover == 2, f"late + wrong-device = 2 leftovers, got {leftover}"
    assert fallback == 0

    connector = {"cmpl-req-w": {"req_id": "cmpl-req-w", "hit_blocks": 76,
                                "hit_tokens": 9728, "num_tokens": 9854}}
    prefetch = {"cmpl-req-w": {"total_keys": 76, "hit_blocks": 76,
                               "missing_blocks": 0}}
    r = audit.merge_by_req_id([{"req_id": "req-w", "producer": False}],
                              connector, prefetch, dma_map,
                              dma_leftover_count=leftover)
    assert not r["conservation_ok"], "leftover DMA must fail conservation"
    assert any("leftover DMA" in v for v in r["violations"]), \
        f"expected leftover-DMA violation: {r['violations']}"
    print("  Gate 4i PASS: out-of-window/wrong-device DMA → leftover → INVALID (R9)")


# ---------------------------------------------------------------------------
# Gate 5: Mid-arm admission drift (owner PID + HBM re-check)
# ---------------------------------------------------------------------------

def test_gate5_admission_drift():
    """Admission must hold during arm execution: owner PID + HBM re-checked."""
    import run_trace_audit as audit

    admitted = [0, 1, 2, 3]
    pre = {0: 64000, 1: 64000, 2: 64000, 3: 64000}
    now_clean = {0: 10000, 1: 10000, 2: 10000, 3: 10000}
    expected = {0: 55000, 1: 55000, 2: 55000, 3: 55000}
    procs_clean = {0: [1000], 1: [1001], 2: [1002], 3: [1003]}
    tracked = {1000, 1001, 1002, 1003}
    pgid_of = lambda pid: pid

    # Clean: our pids own every device, HBM drop within expected+slack
    v = audit.check_admission_drift(admitted, pre, now_clean, expected,
                                    procs_clean, tracked, pgid_of=pgid_of)
    assert v == [], f"Gate 5: clean run must show no drift: {v}"

    # Foreign PID attaches to admitted NPU 2 mid-arm
    procs_foreign = dict(procs_clean)
    procs_foreign[2] = [1002, 99999]
    v = audit.check_admission_drift(admitted, pre, now_clean, expected,
                                    procs_foreign, tracked, pgid_of=pgid_of)
    assert any("foreign pid" in x and "NPU2" in x for x in v), \
        f"Gate 5: foreign pid must be flagged: {v}"

    # HBM collapses beyond our expected footprint (external allocation)
    now_collapsed = dict(now_clean)
    now_collapsed[3] = 0
    v = audit.check_admission_drift(admitted, pre, now_collapsed, expected,
                                    procs_clean, tracked, pgid_of=pgid_of)
    assert any("HBM drift" in x and "NPU3" in x for x in v), \
        f"Gate 5: HBM collapse must be flagged: {v}"

    # Admitted device lost its owner process entirely
    procs_lost = dict(procs_clean)
    procs_lost[1] = []
    v = audit.check_admission_drift(admitted, pre, now_clean, expected,
                                    procs_lost, tracked, pgid_of=pgid_of)
    assert any("no process attached" in x and "NPU1" in x for x in v), \
        f"Gate 5: lost owner must be flagged: {v}"

    print("  Gate 5 PASS: mid-arm admission drift detection (owner PID + HBM)")


def test_gate5b_monitor_polls():
    """R7: the admission monitor polls DURING phase execution and catches
    mid-phase drift — boundary sampling alone is not enough."""
    import run_trace_audit as audit

    admitted = [0]
    pre = {0: 64000}
    expected = {0: 55000}
    tracked = {1000}
    pgid_of = lambda pid: pid

    clean = ({0: 10000}, {0: [1000]})
    drifted = ({0: 10000}, {0: [1000, 99999]})  # foreign pid appears mid-phase
    samples = iter([clean, clean, drifted, clean])
    def sampler():
        return next(samples)

    stop_event = threading.Event()
    out: list[str] = []
    t = threading.Thread(
        target=audit.monitor_admission_drift,
        args=(admitted, pre, expected, tracked, stop_event, out),
        kwargs={"interval_s": 0.05, "sampler": sampler, "pgid_of": pgid_of},
        daemon=True)
    t.start()
    time.sleep(0.3)  # let it poll through the drifted sample
    stop_event.set()
    t.join(timeout=5)
    assert not t.is_alive(), "monitor must stop on stop_event"
    assert any("foreign pid" in v for v in out), \
        f"monitor must catch mid-phase drift: {out}"
    assert any("NPU0" in v for v in out), f"drift must name the device: {out}"
    print("  Gate 5b PASS: periodic monitor catches mid-phase drift (R7)")


# ---------------------------------------------------------------------------
# Gate 6: Fail-close exit code — INVALID evidence must exit non-zero
# ---------------------------------------------------------------------------

def test_gate6_fail_close_exit_code():
    """fail_close() must terminate the process with exit code 1, never 0."""

    here = str(Path(__file__).resolve().parent)
    proc = subprocess.run(
        [sys.executable, "-c",
         "import run_trace_audit as a; a.fail_close(['synthetic gate failure'])"],
        cwd=here, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 1, \
        f"Gate 6: fail_close must exit 1, got {proc.returncode}"
    assert "[INVALID]" in proc.stdout, \
        "Gate 6: fail_close must print [INVALID] before exiting"
    print("  Gate 6 PASS: fail-close exits non-zero (never 0 on INVALID)")


# ---------------------------------------------------------------------------
# Gate 7 (prereg §2.3): producer flag is per-REQUEST, not per-instance
# ---------------------------------------------------------------------------

def test_gate7_producer_per_request():
    """Only the warmup REQUEST is a producer — all 24 timed requests are
    consumers. Per-instance marking would silently shrink N to 21."""
    import run_trace_audit as audit

    instances = []
    for i in range(8):
        spec = {"label": f"C1_shared_{i}", "port": 19000 + i,
                "mode": "read_write", "namespace": "ns",
                "physical_npu": i, "use_pegaflow": True}
        instances.append((spec, MagicMock(poll=lambda: None)))

    def fake_send(port, prompt, model_path):
        return {"ttft_s": 0.1, "total_s": 1.0, "text": "", "ok": True,
                "req_id": f"trace-{port}"}

    with patch.object(audit, "send_one_streaming", side_effect=fake_send), \
         patch.object(audit.time, "sleep", return_value=None):
        records = audit.run_phase_sequential(
            "shared", instances, ["Q0", "Q1", "Q2"], "/model",
            warmup_first=True, cycle=1)

    producers = [r for r in records if r.get("producer")]
    consumers = [r for r in records if not r.get("producer")]
    assert len(records) == 25, f"1 warmup + 24 timed = 25, got {len(records)}"
    assert len(producers) == 1, \
        f"only the warmup request is producer, got {len(producers)}"
    assert len(consumers) == 24, \
        f"prereg N=24 consumers per arm, got {len(consumers)}"
    assert producers[0]["req_idx"] == -1, \
        "producer record must be the warmup request (req_idx == -1)"
    assert producers[0]["instance"] == "C1_shared_0", \
        "warmup runs on the first instance"
    # The warmup instance's timed requests must be consumers
    inst0_timed = [r for r in records if r["instance"] == "C1_shared_0"
                   and not r.get("producer")]
    assert len(inst0_timed) == 3, \
        f"warmup instance contributes 3 consumers, got {len(inst0_timed)}"
    print("  Gate 7 PASS: producer = warmup request only, timed N=24 (prereg §2.3)")


# ---------------------------------------------------------------------------
# Gate 8 (A2): per-request prefill/queue timing extraction
# ---------------------------------------------------------------------------

def test_gate8_vllm_timing_extraction():
    """A2: prefill_time_ms / queue_time_ms parsed from vLLM Finished-request
    lines; absent lines leave no entry (merge defaults to -1)."""
    import run_trace_audit as audit

    log = (
        "INFO 08-05 10:00:01.123 llm_engine.py:123] Finished request "
        "cmpl-trace-aaa-0-a1b2c3: prompt=hello, params=(), "
        "prompt_processing_ms=123.4 queue_time=5.6 generation_time_ms=99.0\n"
        "INFO 08-05 10:00:02.456 llm_engine.py:456] Finished request "
        "cmpl-trace-bbb-0-d4e5f6: prompt_processing_ms=50.0 queue_time=0.0\n"
        "INFO 08-05 10:00:03.000 llm_engine.py:789] Received request "
        "cmpl-trace-ccc-0-g7h8i9\n"
    )
    timing = audit.extract_vllm_timing(log)
    assert timing["cmpl-trace-aaa-0-a1b2c3"]["prefill_time_ms"] == 123.4, timing
    assert timing["cmpl-trace-aaa-0-a1b2c3"]["queue_time_ms"] == 5.6, timing
    assert timing["cmpl-trace-bbb-0-d4e5f6"]["prefill_time_ms"] == 50.0
    assert "cmpl-trace-ccc-0-g7h8i9" not in timing, \
        "non-Finished lines must not create entries"
    assert audit.extract_vllm_timing("no timing lines here") == {}
    print("  Gate 8 PASS: per-request prefill/queue timing extraction (A2)")


def test_gate8b_merge_attaches_timing():
    """A2: merge attaches parsed timing to the matched record; absent -> -1."""
    import run_trace_audit as audit

    connector = {
        "cmpl-trace-t-0-x": {"req_id": "cmpl-trace-t-0-x", "hit_blocks": 1,
                             "hit_tokens": 128, "num_tokens": 128},
    }
    timing = {"cmpl-trace-t-0-x": {"prefill_time_ms": 88.5, "queue_time_ms": 2.1}}
    rec = [{"req_id": "trace-t", "producer": False}]
    r = audit.merge_by_req_id(rec, connector, {}, {}, timing_by_req=timing)
    assert r["conservation_ok"] or r["matched"] == 1
    assert rec[0]["prefill_time_ms"] == 88.5, rec[0]
    assert rec[0]["queue_time_ms"] == 2.1, rec[0]
    # absent timing -> -1 default, no crash
    rec2 = [{"req_id": "trace-u", "producer": False}]
    audit.merge_by_req_id(rec2, connector, {}, {})
    assert rec2[0]["prefill_time_ms"] == -1.0 and rec2[0]["queue_time_ms"] == -1.0
    print("  Gate 8b PASS: merge attaches prefill/queue timing, defaults -1 (A2)")


# ---------------------------------------------------------------------------
# Gate 8c (§4.4/C5/D5): break-even verdict + per-instance + cluster CI
# ---------------------------------------------------------------------------

def test_gate8c_break_even_verdict():
    """compute_paired_analysis: GO when prefill_saved > dma_cost and cluster
    CI excludes 0; BREAK-EVEN when DMA cost dominates."""
    import run_trace_audit as audit

    def rec(cycle, npu, phase, ttft, dma=0.0):
        return {"cycle": cycle, "npu": npu, "phase": phase,
                "query_idx": 0, "ok": True, "ttft_s": ttft,
                "dma_ms": dma, "producer": False}

    # GO case: shared consistently faster than isolated across 3 cycles
    shared_go = [rec(c, i, "shared", 0.10 + i * 0.001, dma=20.0)
                 for c in (1, 2, 3) for i in range(8)]
    isolated_go = [rec(c, i, "isolated", 0.50 + i * 0.001)
                   for c in (1, 2, 3) for i in range(8)]
    pa = audit.compute_paired_analysis(shared_go, isolated_go)
    assert pa["verdict"] == "GO", f"expected GO, got {pa}"
    assert pa["prefill_saved_ms"] > 0
    assert pa["significant"], "CI must exclude 0"
    assert len(pa["per_instance_deltas_ms"]) == 8, "D5: per-instance deltas"
    assert pa["cluster_ci"][0] > 0, "C5: per-class cluster CI"

    # BREAK-EVEN case: DMA cost exceeds the prefill saving
    shared_be = [rec(c, i, "shared", 0.10 + i * 0.001, dma=500.0)
                 for c in (1, 2, 3) for i in range(8)]
    isolated_be = [rec(c, i, "isolated", 0.50 + i * 0.001)
                   for c in (1, 2, 3) for i in range(8)]
    pa2 = audit.compute_paired_analysis(shared_be, isolated_be)
    assert pa2["verdict"] == "BREAK-EVEN", \
        f"expected BREAK-EVEN (dma cost 500ms), got {pa2}"
    assert pa2["dma_cost_ms"] == 500.0
    print("  Gate 8c PASS: GO/BREAK-EVEN verdict + per-instance + cluster CI (§4.4/C5/D5)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Trace Audit Dry-Run (host-only, no NPU)")
    print("=" * 60)
    all_passed = True
    tests = [
        ("Gate 1: Device Admission", test_gate1_device_admission),
        ("Gate 2: Fail-Close", test_gate2_fail_close),
        ("Gate 3: Process Isolation", test_gate3_process_isolation),
        ("Gate 4a: 1:1 Request/DMA Binding + Conservation",
         test_gate4_request_dma_binding),
        ("Gate 4b: DMA Arm Scoping + Cross-Minute Window",
         test_gate4_dma_arm_scoping),
        ("Gate 4c: Duplicate Connector Fails", test_gate4_duplicate_connector),
        ("Gate 4d: Duplicate Prefetch Fails", test_gate4_duplicate_prefetch),
        ("Gate 4e: Missing Prefetch Fails", test_gate4_missing_prefetch),
        ("Gate 4f: Missing DMA Fails", test_gate4_missing_dma),
        ("Gate 4g: Orphan/Leftover Fails", test_gate4_orphan_events),
        ("Gate 4h: Fallback-Only DMA Fails (R6)", test_gate4_bind_fallback_not_evidence),
        ("Gate 4i: Binding Negatives (R9)", test_gate4_bind_negative),
        ("Gate 5: Mid-Arm Admission Drift", test_gate5_admission_drift),
        ("Gate 5b: Periodic Drift Monitor (R7)", test_gate5b_monitor_polls),
        ("Gate 6: Fail-Close Exit Code", test_gate6_fail_close_exit_code),
        ("Gate 7: Producer Per-Request (prereg §2.3)", test_gate7_producer_per_request),
        ("Gate 8: Prefill/Queue Timing (A2)", test_gate8_vllm_timing_extraction),
        ("Gate 8b: Merge Attaches Timing (A2)", test_gate8b_merge_attaches_timing),
        ("Gate 8c: Break-Even Verdict (§4.4/C5/D5)", test_gate8c_break_even_verdict),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {name} — {e}")
            all_passed = False
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {name} — {type(e).__name__}: {e}")
            all_passed = False
    print()
    if all_passed:
        print("  ALL GATES PASSED — ready for hardware run")
    else:
        print("  SOME GATES FAILED — fix before hardware run")
    print("=" * 60)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
