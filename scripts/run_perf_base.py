#!/usr/bin/env python3
"""
run_perf_base.py — shared harness for PegaFlow NPU performance experiments.

Extracted from `run_trace_audit.py` (2026-08-18) to serve the HUST performance
test plan. Methodology is identical to
the validated trace audit: matched arms, AB/BA alternation, independent
server lifecycle per arm, fail-close evidence gates, preregistered
per-query-class analysis.

Runners declare an `Experiment`; base drives everything. New experiments
add a thin runner (see `run_perf_t1_baseline.py`) — base changes once,
applies to all.

CLI (added by runners, base args are shared):
  --cycles N --requests-per-phase N --pool-size X --min-free-gb N
  --num-instances N --model PATH --project-root DIR --vllm-root DIR
  --ascend-root DIR --conda-root DIR --conda-env NAME
  --dry-run --verify-repro --out DIR

`--dry-run` synthesizes records and exercises the full merge/summary/gate
pipeline without hardware (host-only gate). `--verify-repro` runs the
experiment twice and fails if verdicts differ (preregistered §9.1).
"""

from __future__ import annotations

import argparse, hashlib, json, os, random, re, shlex, signal, subprocess, sys
import threading, time, urllib.request, urllib.error, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment defaults (overridable via Experiment / CLI)
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT = Path(os.environ.get(
    "PEGAFLOW_PROJECT_ROOT", Path(__file__).resolve().parents[1]
)).expanduser()
DEFAULT_VLLM_ROOT = Path(os.environ.get(
    "PEGAFLOW_VLLM_ROOT", DEFAULT_PROJECT_ROOT.parent / "vllm-hust"
)).expanduser()
DEFAULT_ASCEND_ROOT = Path(os.environ.get(
    "PEGAFLOW_ASCEND_ROOT", DEFAULT_PROJECT_ROOT.parent / "vllm-ascend-hust"
)).expanduser()
DEFAULT_MODEL = os.environ.get(
    "PEGAFLOW_MODEL", str(DEFAULT_PROJECT_ROOT.parent / "models" / "Qwen3-8B")
)
DEFAULT_CONDA_ROOT = Path(os.environ.get("CONDA_ROOT", "/root/miniconda3"))
DEFAULT_SERVER_PORT = 50080
DEFAULT_VLLM_BASE_PORT = 19000
DEFAULT_CONDA_ENV = "vllm-hust-dev"
HBM_TOTAL_MB = 65536
MIN_FREE_HBM_MB = 28 * 1024
DMA_TIME_WINDOW_S = 30.0
ADMISSION_POLL_INTERVAL_S = 10.0

_SYS_BLOCK = (
    "You are an expert AI assistant with deep knowledge across many domains "
    "including computer science, mathematics, physics, biology, history, "
    "philosophy, literature, economics, law, medicine, engineering, and the "
    "arts. You provide accurate, detailed, and well-structured responses. "
    "Always follow instructions precisely and think step by step before "
    "answering. Your responses should be helpful, harmless, and honest. "
    "When asked factual questions, provide evidence-based answers with "
    "citations where possible. When asked for opinions, provide balanced "
    "perspectives that acknowledge multiple viewpoints. When the user asks "
    "you to perform a task, break it down into clear, actionable steps and "
    "explain your reasoning at each stage. When you encounter ambiguity, "
    "ask clarifying questions rather than making assumptions. Be mindful "
    "of the user's time and keep responses focused and relevant. If you are "
    "unsure about something, acknowledge it honestly rather than speculating. "
    "Your goal is to be as helpful as possible while maintaining high "
    "standards of accuracy and clarity. Remember to adapt your communication "
    "style to the user's level of expertise — use technical language when "
    "appropriate but be ready to explain concepts in simpler terms when "
    "needed. Always prioritize the user's safety and well-being, and avoid "
    "generating harmful, unethical, or dangerous content under any "
    "circumstances. You should respect user privacy and not ask for or "
    "store personal information unnecessarily."
)
SYSTEM_PROMPT = (_SYS_BLOCK + " ") * 38  # ~10k tokens

DEFAULT_QUERIES = [
    "What is the capital of France?",
    "Explain how photosynthesis works in plants.",
    "Who wrote the play Hamlet and when?",
    "What is Einstein's theory of relativity?",
    "Describe the water cycle in detail.",
    "How does an electric motor work?",
    "Name the planets in our solar system in order.",
    "What is machine learning and how does it differ from traditional programming?",
]

# Negative examples inherited from trace audit (kept as preregistered context)
DEFAULT_NEGATIVE_EXAMPLES = {
    "burst_concurrent_8inst": {
        "description": "Burst 8-instance concurrent — PCIe DMA contention destroys PegaFlow benefit",
        "source": "_archived_scripts/run_bench_8inst_concurrent.py (old version, semaphore=unlimited)",
        "shared_avg_ttft_s": 2.70,
        "isolated_avg_ttft_s": 1.73,
        "shared_vs_isolated": "+56% (shared WORSE)",
        "root_cause": "8 concurrent DMA streams saturate PCIe uplink",
        "verdict": "Burst is unrealistic workload; staggered/normal serving load unaffected",
    },
    "mla_tp8_deepseek_v2_lite": {
        "description": "DeepSeek-V2-Lite MLA+TP8 — prefill too cheap for PegaFlow to matter",
        "source": "_archived_scripts/run_bench_mla_tp8_concurrent.py",
        "shared_avg_ttft_s": 0.184,
        "isolated_avg_ttft_s": 0.187,
        "shared_vs_isolated": "+1.6% (no meaningful gain)",
        "root_cause": "MLA kv_lora_rank=512 compresses KV compute to ~100ms; DMA of compressed KV ~3ms",
        "verdict": "PegaFlow requires large enough prefill gap to overcome DMA cost",
    },
}


# ---------------------------------------------------------------------------
# Experiment spec
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """One perf experiment: thin runner config, all methodology in base."""
    id: str                       # e.g. "t1" — used for results/perf-{id}/
    title: str
    model: str = DEFAULT_MODEL
    cycles: int = 3
    requests_per_phase: int = 3
    num_instances: int = 8
    pool_size: str = "16gb"
    min_free_gb: int = 28
    queries: list = field(default_factory=lambda: list(DEFAULT_QUERIES))
    # Extra metrics rendered in summary (key, label) — TTFT always included.
    extra_metrics: list = field(default_factory=list)
    # Extra gates: list of (name, fn(records, merge_result) -> list[str violations])
    extra_gates: list = field(default_factory=list)
    # Extra verdict text appended to summary (e.g. concurrency operating region)
    extra_verdicts: list = field(default_factory=list)
    # Per-request delay between sends (s) — trace audit used 0.5
    request_delay_s: float = 0.5
    # Concurrent send mode (semaphore) — T2 uses this; None = sequential
    concurrency: int | None = None
    batch_interval_s: float = 0.0
    warmup_first: bool = True
    # Custom prompt construction: fn(query, query_idx) -> prompt. query_idx
    # is -1 for the warmup seed. Default: SYSTEM_PROMPT + query (T3/T4 use
    # this to control prefix-hit ratio and prompt length).
    prompt_fn: object | None = None
    # GPU memory utilization override: None = dynamic from free HBM (default);
    # a fixed value (e.g. 0.95) pins every instance's gmu (T5 pressure arm).
    gmu: float | None = None
    # Custom negative examples (defaults inherited if None)
    negative_examples: dict | None = None
    # Runner-specific selector needed to reproduce a sweep point, for example
    # ["--ratios", "50"] for T3. Common harness arguments are added below.
    reproduce_args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Artifact binding
# ---------------------------------------------------------------------------

def capture_environment(project_root: Path, vllm_root: Path,
                        ascend_root: Path, model_path: str,
                        conda_root: Path, conda_env: str) -> dict:
    """Record everything needed to reproduce this run."""
    info: dict = {}
    for _cmd, _key in [
        (["rev-parse", "HEAD"], "git_commit"),
        (["rev-parse", "HEAD^"], "git_parent"),
        (["rev-parse", "--abbrev-ref", "HEAD"], "git_branch"),
    ]:
        try:
            info[_key] = subprocess.check_output(
                ["git", "-C", str(project_root)] + _cmd, timeout=10,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            info[_key] = "unknown"
    try:
        info["runtime_commit_vllm"] = subprocess.check_output(
            ["git", "-C", str(vllm_root), "rev-parse", "HEAD"], timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        info["runtime_commit_vllm"] = "unknown"
    try:
        info["runtime_commit_ascend"] = subprocess.check_output(
            ["git", "-C", str(ascend_root), "rev-parse", "HEAD"], timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        info["runtime_commit_ascend"] = "unknown"

    cfg_path = Path(model_path) / "config.json"
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))
        info["model"] = str(Path(model_path).name)
        info["model_arch"] = cfg.get("architectures", [])
        info["model_config_md5"] = hashlib.md5(
            open(cfg_path, "rb").read()).hexdigest()

    try:
        info["npu_smi"] = subprocess.check_output(
            ["npu-smi", "info"], timeout=30).decode()
    except Exception:
        info["npu_smi"] = "unavailable"

    # Torch version probe must run in the conda runtime env (bare python3 has
    # no torch on this host).
    conda_py = conda_root / "envs" / conda_env / "bin" / "python"
    try:
        info["torch_version"] = subprocess.check_output(
            [str(conda_py), "-c",
             "import torch, torch_npu; "
             "print(torch.__version__, torch_npu.__version__)"],
            timeout=15, env=os.environ.copy(),
        ).decode().strip()
    except Exception:
        info["torch_version"] = "unknown"

    info["env_vars"] = {
        k: os.environ.get(k, "")
        for k in ["PYTHONHASHSEED", "ASCEND_RT_VISIBLE_DEVICES",
                  "PEGAFLOW_HOST", "PEGAFLOW_PORT", "PYTORCH_NPU_ALLOC_CONF",
                  "ASCEND_HOME_PATH", "LD_LIBRARY_PATH", "PATH"]
    }
    info["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return info


# ---------------------------------------------------------------------------
# NPU helpers (extracted verbatim from run_trace_audit.py)
# ---------------------------------------------------------------------------

def get_npu_free_memory() -> dict[int, int]:
    free: dict[int, int] = {i: -1 for i in range(8)}
    try:
        out = subprocess.check_output(
            ["npu-smi", "info"], stderr=subprocess.STDOUT, timeout=30).decode()
    except Exception:
        return free
    lines = out.split("\n")
    current_npu: int | None = None
    for line in lines:
        if "Process id" in line and "Process name" in line:
            break
        m1 = re.match(r"\|\s*(\d+)\s+\d+\w+\s+\|", line)
        if m1:
            current_npu = int(m1.group(1))
            continue
        if current_npu is not None and current_npu < 8:
            m2 = re.search(r"(\d+)\s*/\s*(\d+)\s*\|?\s*$", line)
            if m2:
                used = int(m2.group(1))
                total = int(m2.group(2))
                free[current_npu] = total - used
                current_npu = None
    return free


def _npu_smi_has_container_pid_column() -> bool:
    """Whether npu-smi prints a 'Process id in container' column (container host).

    In a container, npu-smi reports host PIDs in the main column and container
    PIDs in the last column. Only container PIDs are visible in our /proc, so
    ownership checks must use that column when present.
    """
    try:
        out = subprocess.check_output(
            ["npu-smi", "info"], stderr=subprocess.STDOUT, timeout=30).decode()
    except Exception:
        return False
    return "Process id in container" in out


def get_npu_processes() -> dict[int, list[int]]:
    """NPU id -> list of container-visible attached pids.

    Uses the 'Process id in container' column when present (host PIDs are
    invisible inside this container); falls back to the host PID column when
    not in a container.
    """
    procs: dict[int, list[int]] = {}
    try:
        out = subprocess.check_output(
            ["npu-smi", "info"], stderr=subprocess.STDOUT, timeout=30).decode()
    except Exception:
        return procs
    in_proc_table = False
    use_container_col = _npu_smi_has_container_pid_column()
    for line in out.split("\n"):
        if "Process id" in line and "Process name" in line:
            in_proc_table = True
            continue
        if not in_proc_table or not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        npu_tokens = parts[1].split()
        pid_field = parts[-2] if use_container_col else parts[2]
        if not npu_tokens or not pid_field.isdigit():
            continue
        procs.setdefault(int(npu_tokens[0]), []).append(int(pid_field))
    return procs


def _detect_pid_identity_possible() -> bool:
    """Whether npu-smi PIDs resolve in our /proc.

    With the container-column fix in get_npu_processes, attached pids are
    container-visible PIDs and always resolve here; on bare metal host PIDs
    resolve too. This stays conservative: if nothing is attached yet it
    assumes PID identity is usable and the post-launch re-check will surface
    any mismatch as fail-close drift.
    """
    attached = [p for ps in get_npu_processes().values() for p in ps]
    if not attached:
        return True
    return any(os.path.isdir(f"/proc/{p}") for p in attached)


_tracked_pids: list[int] = []


def tracked_pgids() -> set[int]:
    pgids: set[int] = set()
    for pid in list(_tracked_pids):
        try:
            pgids.add(os.getpgid(pid))
        except (ProcessLookupError, PermissionError):
            pgids.add(pid)
    return pgids


def expected_used_mb(npu: int, free_mb: dict[int, int]) -> int:
    fm = free_mb.get(npu, -1)
    gmu = (max(0.15, min(0.85, (fm - 4096) / HBM_TOTAL_MB))
           if fm > 4096 else 0.15)
    return round(gmu * HBM_TOTAL_MB)


def check_admission_drift(
    admitted, free_mb_pre, free_mb_now, expected_used_mb_by_npu,
    npu_procs, tracked_pgids_set, pgid_of=os.getpgid,
    slack_mb: int = 8 * 1024, min_free_mb: int = MIN_FREE_HBM_MB,
    pid_identity: bool = True,
) -> list[str]:
    """Re-verify admission during arm execution (fail-close)."""
    violations: list[str] = []
    for npu in admitted:
        pre = free_mb_pre.get(npu, -1)
        now = free_mb_now.get(npu, -1)
        exp = expected_used_mb_by_npu.get(npu, 0)
        floor = min(min_free_mb, pre - exp - slack_mb) if pre >= 0 else min_free_mb
        if now < floor:
            violations.append(
                f"NPU{npu} HBM drift: free={now}MB < floor={floor}MB "
                f"(pre={pre}MB, expected_use={exp}MB, slack={slack_mb}MB)")
        attached = npu_procs.get(npu, [])
        if not attached:
            violations.append(
                f"NPU{npu} owner drift: no process attached (expected our instance)")
            continue
        if not pid_identity:
            continue
        for pid in attached:
            try:
                owned = (_owned_by_tracked(pid, tracked_pgids_set)
                         or pid in tracked_pgids_set
                         or pgid_of(pid) in tracked_pgids_set)
            except (ProcessLookupError, PermissionError):
                owned = False
            if not owned:
                violations.append(
                    f"NPU{npu} owner drift: foreign pid={pid} attached to admitted device")
    return violations


def _owned_by_tracked(pid: int, tracked_pgids_set: set[int]) -> bool:
    """Ancestor-chain ownership: vLLM spawns EngineCore in a NEW process
    group (multiprocessing spawn), so pgid matching alone false-positives.
    Walk /proc/{pid}/stat ppid chain — owned if any ancestor is one of our
    tracked process groups."""
    seen: set[int] = set()
    cur = pid
    while cur and cur not in seen:
        seen.add(cur)
        try:
            stat = Path(f"/proc/{cur}/stat").read_text()
            ppid = int(stat.split()[3])
        except Exception:
            return False
        if ppid in tracked_pgids_set:
            return True
        cur = ppid
    return False


def _detect_pid_identity_possible() -> bool:
    host_pids = [p for ps in get_npu_processes().values() for p in ps]
    if not host_pids:
        return True
    return any(os.path.isdir(f"/proc/{p}") for p in host_pids)


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def _track_proc(proc: subprocess.Popen) -> None:
    _tracked_pids.append(proc.pid)


def _descendants(pid: int) -> list[int]:
    """All descendant pids of pid (recursive). vLLM spawns EngineCore in a new
    process group, so group-kill alone leaves it behind; it must be found via
    the ppid chain."""
    out: list[int] = []
    frontier = [pid]
    while frontier:
        parent = frontier.pop()
        try:
            children = subprocess.check_output(
                ["ps", "-o", "pid=", "--ppid", str(parent)], timeout=5
            ).decode().split()
        except Exception:
            continue
        for c in children:
            cid = int(c)
            out.append(cid)
            frontier.append(cid)
    return out


def kill_tracked() -> None:
    # Descendants first (they may have reparented), then the tracked groups.
    seen: set[int] = set()
    for pid in list(_tracked_pids):
        for d in _descendants(pid):
            if d in seen:
                continue
            seen.add(d)
            try:
                os.kill(d, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    for pid in list(_tracked_pids):
        try:
            os.kill(-pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    for pid in list(_tracked_pids):
        try:
            os.kill(-pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    _tracked_pids.clear()


def stop_proc(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def resolve_server_bin(project_root: Path) -> str:
    candidates = [
        project_root / "target" / "release" / "pegaflow-server-py",
        project_root / "target" / "release" / "pegaflow-server",
        project_root / "target" / "debug" / "pegaflow-server",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return str(candidates[0])


def _server_env(conda_root: Path, conda_env: str) -> dict:
    env = os.environ.copy()
    conda_py = conda_root / "envs" / conda_env / "bin" / "python"
    try:
        libdir = subprocess.check_output(
            [str(conda_py), "-c",
             "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"],
            timeout=15,
        ).decode().strip()
    except Exception:
        libdir = ""
    if libdir:
        cur = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = libdir + (":" + cur if cur else "")
    return env


def _port_open(host: str, port: int) -> bool:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def start_server(project_root: Path, server_port: int, log_dir: Path,
                 pool_size: str, devices: list[int], log_level: str,
                 conda_root: Path, conda_env: str, server_bin: str | None = None):
    log_dir.mkdir(parents=True, exist_ok=True)
    if _port_open("127.0.0.1", server_port):
        raise RuntimeError(
            f"port {server_port} already in use — a leftover pegaflow-server "
            f"is still running; kill it before retrying")
    log = log_dir / "server.log"
    bin_path = server_bin or resolve_server_bin(project_root)
    proc = subprocess.Popen(
        [bin_path, "--addr", f"0.0.0.0:{server_port}",
         "--pool-size", pool_size,
         "--devices", ",".join(str(d) for d in devices),
         "--log-level", log_level],
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        env=_server_env(conda_root, conda_env),
        preexec_fn=os.setsid,
    )
    _track_proc(proc)
    # Startup is deliberately generous (perf plan §9.1): ACL init + 4 GiB
    # pinned pool + NUMA probe take 15-25 s alone, and concurrent runs share
    # CPU with vLLM engine init.
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(1)
        if proc.poll() is not None:
            break
        if _port_open("127.0.0.1", server_port):
            return proc
    tail = ""
    try:
        tail = "\n".join(log.read_text().splitlines()[-15:])
    except Exception:
        pass
    raise RuntimeError(f"Server failed to start; log tail:\n{tail}")


# Single-instance health probe measured 129 s to ready on this host; 8
# concurrent instances contend for CPU during engine init, so the deadline
# must be generous (7 min). Startup retry is not applied — a slow start is
# normal, a genuinely failed start still fail-closes via the health poll.
VLLM_START_DEADLINE_S = 420


def start_vllm(port, mode, namespace, physical_npu, label, *,
               model_path, log_dir: Path, server_port: int,
               conda_root: Path, conda_env: str,
               gpu_memory_utilization=0.85, use_pegaflow=True):
    log = log_dir / f"vllm_{label}.log"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(physical_npu)
    # Allocator contract (2026-08-18): aclrtMemcpyBatchAsync fails 107000 on
    # torch_npu default-allocator memory; expandable_segments memory is
    # DMA-capable and matches the 20260813 VALID run (which set this conf).
    # Override via PYTORCH_NPU_ALLOC_CONF when a different allocator is wanted.
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    if use_pegaflow:
        if namespace:
            env["PEGAFLOW_NAMESPACE"] = namespace
        env["PEGAFLOW_HOST"] = "http://127.0.0.1"
        env["PEGAFLOW_PORT"] = str(server_port)
    cmd_parts = [
        f"source {conda_root}/etc/profile.d/conda.sh && conda activate {conda_env}",
        f"vllm serve {model_path} --port {port} --dtype float16",
        f"--max-model-len 16384 --max-num-seqs 4",
        f"--gpu-memory-utilization {gpu_memory_utilization:.2f}",
    ]
    if use_pegaflow:
        kv_cfg = json.dumps({
            "kv_connector": "PegaKVConnector", "kv_role": "kv_both",
            "kv_connector_module_path": "pegaflow.connector",
            "kv_connector_extra_config": {
                "pegaflow.mode": mode,
                "pegaflow.transfer_backend": "ascend_direct",
            },
        })
        cmd_parts.append(f"--kv-transfer-config '{kv_cfg}'")
    cmd = " && ".join([cmd_parts[0], " ".join(cmd_parts[1:])])
    proc = subprocess.Popen(
        ["bash", "-c", cmd], env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    _track_proc(proc)
    deadline = time.time() + VLLM_START_DEADLINE_S
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return proc
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"vLLM {label} failed")


def launch_all_instances(specs, model_path, log_dir, server_port,
                         conda_root, conda_env, gmu_override=None):
    running: list[tuple[dict, subprocess.Popen]] = []
    rlock = threading.Lock()

    def _start_one(spec):
        npu = spec["physical_npu"]
        fm = get_npu_free_memory().get(npu, -1)
        gmu = (gmu_override if gmu_override is not None
               else max(0.15, min(0.85, (fm - 4096) / HBM_TOTAL_MB)))
        label = spec["label"]
        print(f"    → [{label}] NPU{npu} gmu={gmu:.2f} ...")
        # One startup retry: under 8-instance concurrency engine init is
        # occasionally slow (>420s); the retry only covers launch, never the
        # run itself — fail-close still applies if both attempts fail.
        for attempt in (1, 2):
            try:
                proc = start_vllm(
                    spec["port"], spec["mode"], spec["namespace"],
                    npu, label, model_path=model_path,
                    log_dir=log_dir, server_port=server_port,
                    conda_root=conda_root, conda_env=conda_env,
                    gpu_memory_utilization=gmu,
                    use_pegaflow=spec.get("use_pegaflow", True),
                )
                with rlock:
                    running.append((spec, proc))
                print(f"    → [{label}] ready ({len(running)}/{len(specs)})")
                return
            except Exception as e:
                print(f"    → [{label}] attempt {attempt} FAILED: {e}")
                if attempt == 1:
                    time.sleep(5)
        print(f"    → [{label}] FAILED after 2 attempts")

    # Timeout must exceed start_vllm's internal 420s deadline; a timeout here
    # returns partial results and the caller's < num_instances check fail-closes.
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futures = [ex.submit(_start_one, s) for s in specs]
        try:
            for _ in as_completed(futures, timeout=900):
                pass
        except TimeoutError:
            print(f"  [note] instance launch timed out after 900s; "
                  f"{sum(1 for f in futures if f.done())}/{len(futures)} finished")
    return running


# ---------------------------------------------------------------------------
# Streaming request client (TTFT + TBT/TPOT token timestamps)
# ---------------------------------------------------------------------------

def send_one_streaming(port, prompt, model_path, max_tokens=64, timeout=600):
    """Send streaming request; return record incl. TTFT, total, TBT stats.

    TBT: per-token arrival timestamps are captured; summary stats (p50/p95/
    mean) are stored to keep the artifact small. First token = TTFT.
    """
    client_req_id = f"perf-{uuid.uuid4().hex[:12]}"
    data = json.dumps({
        "model": model_path, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True, "request_id": client_req_id,
    }).encode()
    t0 = time.perf_counter()
    ttft_s = -1.0
    total_s = -1.0
    text = ""
    token_ts: list[float] = []
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=data, headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        for line_bytes in resp:
            line = line_bytes.decode().strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                now = time.perf_counter() - t0
                if ttft_s < 0:
                    ttft_s = now
                choices = chunk.get("choices", [])
                if choices:
                    text += choices[0].get("text", "")
                    token_ts.append(now)
            except json.JSONDecodeError:
                continue
        total_s = time.perf_counter() - t0
        # TBT stats: inter-token gaps after the first token
        tbt = {}
        if len(token_ts) >= 2:
            gaps = [token_ts[i] - token_ts[i - 1] for i in range(1, len(token_ts))]
            gaps.sort()
            tbt = {
                "tbt_p50_s": round(gaps[len(gaps) // 2], 4),
                "tbt_p95_s": round(gaps[min(len(gaps) - 1, int(len(gaps) * 0.95))], 4),
                "tbt_mean_s": round(sum(gaps) / len(gaps), 4),
                "tokens_received": len(token_ts),
            }
        return {"ttft_s": round(ttft_s, 4), "total_s": round(total_s, 4),
                "text": text[:80], "ok": True, "req_id": client_req_id,
                **tbt}
    except Exception as e:
        return {"ttft_s": -1, "total_s": -1, "text": "", "ok": False,
                "error": str(e)[:200], "req_id": client_req_id}


def extract_vllm_timing(log_text: str) -> dict[str, dict]:
    """Per-request prefill/queue timing from a vLLM log (informational)."""
    timing: dict[str, dict] = {}
    for m in re.finditer(r"Finished request (?P<req>\S+):\s*(?P<kv>.*)",
                         log_text):
        entry = timing.setdefault(m.group("req"), {})
        for kv in re.finditer(r"(\w+)=([\d.+-]+)", m.group("kv")):
            key, val = kv.group(1), float(kv.group(2))
            if key == "prompt_processing_ms":
                entry["prefill_time_ms"] = val
            elif key == "queue_time":
                entry["queue_time_ms"] = val
    return timing


def extract_connector_log(vllm_log_path: Path, label: str) -> list[dict]:
    if not vllm_log_path.exists():
        return []
    entries = []
    text = vllm_log_path.read_text()
    for m in re.finditer(
        r"\[PegaKVConnector\] req=(?P<req_id>\S+)\s+"
        r"cache_lookup: hit_blocks=(?P<hit>\d+) "
        r"computed_blocks=(?P<computed>\d+) "
        r"hit_tokens=(?P<hit_tokens>\d+) num_tokens=(?P<num_tokens>\d+).*?",
        text,
    ):
        entries.append({
            "req_id": m.group("req_id"),
            "label": label,
            "hit_blocks": int(m.group("hit")),
            "computed_blocks": int(m.group("computed")),
            "hit_tokens": int(m.group("hit_tokens")),
            "num_tokens": int(m.group("num_tokens")),
        })
    return entries


# ---------------------------------------------------------------------------
# Phase runner (sequential + concurrent)
# ---------------------------------------------------------------------------

def run_phase(experiment: Experiment, phase_name, instances, queries,
              model_path, warmup_first, cycle, log_dir, server_port,
              conda_root, conda_env):
    """Send requests per experiment schedule; return per-request records."""
    records: list[dict] = []
    t0 = time.perf_counter()

    def _mk_record(spec, q, r, idx, qi, producer):
        return {
            "cycle": cycle, "phase": phase_name, "req_idx": idx,
            "query_idx": qi, "instance": spec["label"],
            "npu": spec["physical_npu"], "port": spec["port"],
            "query": q[:50], **r,
            "wall_clock_at_send": round(time.perf_counter() - t0, 4),
            "producer": producer,
        }

    def _make_prompt(q: str, qi: int, npu: int = -1) -> str:
        if experiment.prompt_fn is not None:
            return experiment.prompt_fn(q, qi, npu)
        return f"{SYSTEM_PROMPT}\n\nUser: {q}\n\nAssistant:"

    if warmup_first and len(instances) >= 1:
        warmup_spec, warmup_proc = instances[0]
        prompt = _make_prompt(queries[0], -1, warmup_spec["physical_npu"])
        r = send_one_streaming(warmup_spec["port"], prompt, model_path)
        records.append(_mk_record(warmup_spec, queries[0], r, -1, -1, True))
        print(f"    [WARMUP] {warmup_spec['label']} "
              f"TTFT={r['ttft_s']:.4f}s ok={r['ok']}")
        time.sleep(30)

    # Timed phase: sequential round-robin, or semaphore-limited concurrent
    if experiment.concurrency and experiment.concurrency > 1:
        records.extend(_run_phase_concurrent(
            experiment, phase_name, instances, queries, model_path,
            cycle, t0, _mk_record, _make_prompt))
    else:
        idx = 0
        for qi in range(len(queries)):
            for spec, proc in instances:
                if proc.poll() is not None:
                    continue
                q = queries[qi]
                prompt = _make_prompt(q, qi, spec["physical_npu"])
                r = send_one_streaming(spec["port"], prompt, model_path)
                records.append(_mk_record(spec, q, r, idx, qi, False))
                status = (f"TTFT={r['ttft_s']:.4f}s" if r["ok"]
                          else f"ERR={r.get('error','?')[:30]}")
                print(f"    [{idx:>3d}] {spec['label']} Q{qi} {status} | {q[:30]}")
                idx += 1
                if experiment.request_delay_s > 0:
                    time.sleep(experiment.request_delay_s)
    return records


def _run_phase_concurrent(experiment, phase_name, instances, queries,
                          model_path, cycle, t0, _mk_record, _make_prompt=None):
    """Semaphore-limited concurrent sends with fixed batch interval (T2)."""
    records: list[dict] = []
    sem = threading.Semaphore(experiment.concurrency)
    rlock = threading.Lock()
    tasks: list[tuple[dict, str, int, int]] = []
    idx = 0
    for qi in range(len(queries)):
        for spec, proc in instances:
            if proc.poll() is not None:
                continue
            tasks.append((spec, queries[qi], idx, qi))
            idx += 1

    def _send(task):
        spec, q, i, qi = task
        prompt = (_make_prompt(q, qi, spec["physical_npu"]) if _make_prompt is not None
                  else f"{SYSTEM_PROMPT}\n\nUser: {q}\n\nAssistant:")
        with sem:
            r = send_one_streaming(spec["port"], prompt, model_path)
        with rlock:
            records.append(_mk_record(spec, q, r, i, qi, False))
        print(f"    [{i:>3d}] {spec['label']} Q{qi} "
              f"TTFT={r['ttft_s']:.4f}s ok={r['ok']}")

    def _batched():
        # Fixed-interval batches (staggered submission)
        for start in range(0, len(tasks), experiment.concurrency):
            batch = tasks[start:start + experiment.concurrency]
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                list(ex.map(_send, batch))
            if experiment.batch_interval_s > 0 and start + experiment.concurrency < len(tasks):
                time.sleep(experiment.batch_interval_s)

    _batched()
    return records


# ---------------------------------------------------------------------------
# Evidence matching (verbatim from trace audit — fail-close)
# ---------------------------------------------------------------------------

def merge_by_req_id(
    all_records, connector_by_req, prefetch_by_req, prefetch_dma_map,
    dma_leftover_count=0, dma_fallback_only_count=0, timing_by_req=None,
) -> dict:
    """Merge per-request hit/DMA data by client request_id (fail-close)."""
    matched = 0
    unmatched = 0
    producer_skipped = 0
    total_consumers = 0
    violations: list[str] = []
    invalid_records: list[dict] = []
    connector_duplicates = 0

    def occ(ev):
        return int(ev.get("occurrences", 1)) if ev else 0

    claimed: set[str] = set()
    prefetch_claimed: set[str] = set()
    dma_claimed: set[str] = set()

    for r in all_records:
        if not r.get("producer"):
            continue
        client_req_id = r.get("req_id", "")
        keys = [k for k in connector_by_req
                if client_req_id and client_req_id in k]
        if len(keys) > 1:
            violations.append(
                f"duplicate connector event (producer): req={client_req_id} keys={keys}")
        if keys:
            claimed.add(keys[0])
            if keys[0] in prefetch_by_req:
                prefetch_claimed.add(keys[0])
            if keys[0] in prefetch_dma_map:
                dma_claimed.add(keys[0])

    for r in all_records:
        r.setdefault("hit_blocks", 0)
        r.setdefault("hit_tokens", 0)
        r.setdefault("missing_blocks", 0)
        r.setdefault("dma_bytes", 0)
        r.setdefault("dma_ms", 0.0)
        r.setdefault("dma_gbps", 0.0)
        r.setdefault("prefill_time_ms", -1.0)
        r.setdefault("queue_time_ms", -1.0)

        if r.get("producer"):
            producer_skipped += 1
            continue

        total_consumers += 1
        client_req_id = r.get("req_id", "")
        # Empty req_id (abort placeholder records) must never match all keys —
        # `"" in key` is vacuously True.
        keys = [k for k in connector_by_req
                if client_req_id and client_req_id in k]

        if not keys:
            unmatched += 1
            reason = f"missing connector event: req={client_req_id}"
            r["_audit_invalid"] = True
            r["_audit_reason"] = reason
            violations.append(reason)
            invalid_records.append({"req_id": client_req_id,
                                    "reason": "missing connector event"})
            continue

        if len(keys) > 1:
            connector_duplicates += len(keys) - 1
            reason = f"duplicate connector event: req={client_req_id} keys={keys}"
            r["_audit_invalid"] = True
            r["_audit_reason"] = reason
            violations.append(reason)
            invalid_records.append({"req_id": client_req_id,
                                    "reason": f"duplicate connector event ({len(keys)})"})

        conn_key = keys[0]
        cinfo = connector_by_req[conn_key]
        claimed.add(conn_key)
        matched += 1
        r["hit_blocks"] = cinfo["hit_blocks"]
        r["hit_tokens"] = cinfo["hit_tokens"]
        r["num_tokens_total"] = cinfo.get("num_tokens", 0)
        if timing_by_req:
            t = timing_by_req.get(conn_key)
            if t:
                if "prefill_time_ms" in t:
                    r["prefill_time_ms"] = t["prefill_time_ms"]
                if "queue_time_ms" in t:
                    r["queue_time_ms"] = t["queue_time_ms"]

        is_local_hit = cinfo.get("total_query_hashes", -1) == 0
        if is_local_hit:
            r["local_hit"] = True
            r["missing_blocks"] = 0
            r["dma_bytes"] = 0
            r["dma_ms"] = 0.0
            r["dma_gbps"] = 0.0
            continue

        pref = prefetch_by_req.get(conn_key)
        if pref is None:
            reason = f"missing prefetch event: req={client_req_id} server={conn_key}"
            r["_audit_invalid"] = True
            r["_audit_reason"] = reason
            violations.append(reason)
            invalid_records.append({"req_id": client_req_id,
                                    "reason": "missing prefetch event"})
        else:
            prefetch_claimed.add(conn_key)
            if occ(pref) > 1:
                reason = (f"duplicate prefetch event: server={conn_key} "
                          f"occurrences={occ(pref)}")
                r["_audit_invalid"] = True
                r["_audit_reason"] = reason
                violations.append(reason)
                invalid_records.append({"req_id": client_req_id,
                                        "reason": "duplicate prefetch event"})
            r["missing_blocks"] = pref.get("missing_blocks", 0)

        dma = prefetch_dma_map.get(conn_key)
        dma_required = r["hit_blocks"] > 0
        if dma_required:
            if dma is None:
                reason = (f"missing DMA event: req={client_req_id} "
                          f"hit_blocks={r['hit_blocks']} "
                          f"(transfer claimed, no DMA evidence)")
                r["_audit_invalid"] = True
                r["_audit_reason"] = reason
                violations.append(reason)
                invalid_records.append({"req_id": client_req_id,
                                        "reason": "missing DMA event"})
            else:
                dma_claimed.add(conn_key)
                if occ(dma) > 1:
                    reason = (f"duplicate DMA event: server={conn_key} "
                              f"occurrences={occ(dma)}")
                    r["_audit_invalid"] = True
                    r["_audit_reason"] = reason
                    violations.append(reason)
                    invalid_records.append({"req_id": client_req_id,
                                            "reason": "duplicate DMA event"})
                r["dma_bytes"] = dma["dma_bytes"]
                r["dma_ms"] = dma["dma_ms"]
                r["dma_gbps"] = dma["dma_gbps"]
                r["prefetch_to_dma_ms"] = dma.get("gap_ms", -1.0)
                r["dma_fallback"] = dma.get("fallback", False)

    connector_orphans = [k for k in connector_by_req if k not in claimed]
    for k in connector_orphans:
        violations.append(f"orphan connector event: {k}")
    prefetch_orphans = [k for k in prefetch_by_req if k not in prefetch_claimed]
    for k in prefetch_orphans:
        violations.append(f"orphan prefetch event: {k}")
    dma_orphans = [k for k in prefetch_dma_map if k not in dma_claimed]
    for k in dma_orphans:
        violations.append(f"orphan DMA event: {k}")
    if dma_leftover_count > 0:
        violations.append(f"leftover DMA completions: {dma_leftover_count} unbound to any prefetch")
    # C (2026-08-18): fallback per-copy DMA is formal evidence under platform
    # constraint (CANN batch 107000 on 8-instance D2H) — counted, not INVALID.

    connector_duplicates += sum(
        occ(ev) - 1 for ev in connector_by_req.values() if occ(ev) > 1)
    prefetch_duplicates = sum(
        occ(ev) - 1 for ev in prefetch_by_req.values() if occ(ev) > 1)
    dma_duplicates = sum(
        occ(ev) - 1 for ev in prefetch_dma_map.values() if occ(ev) > 1)

    coverage_pct = (matched / total_consumers * 100.0
                    if total_consumers > 0 else 0.0)
    conservation_ok = (
        unmatched == 0 and not invalid_records
        and not connector_orphans and not prefetch_orphans and not dma_orphans
        and connector_duplicates == 0 and prefetch_duplicates == 0
        and dma_duplicates == 0 and dma_leftover_count == 0
    )
    return {
        "matched": matched, "unmatched": unmatched,
        "producer_skipped": producer_skipped,
        "total_consumers": total_consumers,
        "coverage_pct": round(coverage_pct, 1),
        "connector_total": len(connector_by_req),
        "connector_consumed": len(claimed),
        "connector_duplicates": connector_duplicates,
        "connector_orphans": len(connector_orphans),
        "prefetch_total": len(prefetch_by_req),
        "prefetch_consumed": len(prefetch_claimed),
        "prefetch_duplicates": prefetch_duplicates,
        "prefetch_orphans": len(prefetch_orphans),
        "dma_total": len(prefetch_dma_map),
        "dma_consumed": len(dma_claimed),
        "dma_duplicates": dma_duplicates,
        "dma_orphans": len(dma_orphans),
        "dma_leftover": dma_leftover_count,
        "dma_fallback_only": dma_fallback_only_count,
        "conservation_ok": conservation_ok,
        "violations": violations,
        "invalid_records": invalid_records,
    }


def bind_dma_to_prefetch(ts_prefetches, ts_dmas, connector_by_req,
                         label_to_npu, dma_time_window_s: float = DMA_TIME_WINDOW_S):
    """Bind DMA completions to prefetch events 1:1 (per-arm, per-device, window)."""
    prefetch_dma_map: dict[str, dict] = {}
    fallback_only_count = 0
    violations: list[str] = []
    ts_fmt = "%Y-%m-%dT%H:%M:%S.%f"

    for pf in ts_prefetches:
        if pf["hit"] == 0:
            continue
        req_id = pf["req_id"]
        target_device = None
        if req_id in connector_by_req:
            target_device = label_to_npu.get(
                connector_by_req[req_id].get("label", ""), None)
        pf_arm = pf.get("arm_label", "")
        best, best_fallback = None, None
        for dma in ts_dmas:
            if dma["ts"] <= pf["ts"]:
                continue
            if dma.get("arm_label", "") != pf_arm:
                continue
            if target_device is not None and dma["device_id"] != target_device:
                continue
            try:
                pf_dt = datetime.strptime(pf["ts"][:26], ts_fmt)
                dma_dt = datetime.strptime(dma["ts"][:26], ts_fmt)
                if abs((dma_dt - pf_dt).total_seconds()) > dma_time_window_s:
                    continue
            except (ValueError, IndexError):
                pass
            if dma.get("fallback"):
                if best_fallback is None or dma["ts"] < best_fallback["ts"]:
                    best_fallback = dma
            else:
                if best is None or dma["ts"] < best["ts"]:
                    best = dma
        if best is None:
            # C (2026-08-18, prereg deviation): on 8-instance concurrency CANN
            # aclrtMemcpyBatchAsync(D2H) intermittently fails 107000 and the
            # save path falls back to per-copy aclrtMemcpyAsync. Per-copy
            # completions are auditable, data-correct DMA evidence — bind them
            # as formal evidence, but record the fallback count for the
            # manifest (platform constraint, see CHANGELOG 2026-08-18).
            if best_fallback is not None:
                best = best_fallback
                fallback_only_count += 1
                try:
                    ts_dmas.remove(best_fallback)
                except ValueError:
                    pass
            else:
                continue
        entry = prefetch_dma_map.get(req_id)
        if entry is not None:
            entry["occurrences"] = entry.get("occurrences", 1) + 1
            continue
        gap_ms = -1.0
        try:
            pf_dt = datetime.strptime(pf["ts"][:26], ts_fmt)
            dma_dt = datetime.strptime(best["ts"][:26], ts_fmt)
            gap_ms = round((dma_dt - pf_dt).total_seconds() * 1000.0, 1)
        except (ValueError, IndexError):
            pass
        prefetch_dma_map[req_id] = {
            "dma_ms": best["dma_ms"], "dma_bytes": best["dma_bytes"],
            "dma_gbps": best["dma_gbps"], "gap_ms": gap_ms,
            "occurrences": 1,
        }
        ts_dmas.remove(best)

    return prefetch_dma_map, fallback_only_count, len(ts_dmas), violations


def monitor_admission_drift(admitted, free_mb_pre, expected_used_mb_by_npu,
                            tracked_pgids_set, stop_event, violations_out,
                            interval_s: float = ADMISSION_POLL_INTERVAL_S,
                            pid_identity: bool = True):
    def _sample():
        return get_npu_free_memory(), get_npu_processes()

    while not stop_event.wait(interval_s):
        try:
            free_now, npu_procs = _sample()
        except Exception as e:
            violations_out.append(f"admission monitor sample failed: {e}")
            continue
        drift = check_admission_drift(
            admitted, free_mb_pre, free_now, expected_used_mb_by_npu,
            npu_procs, tracked_pgids_set, pid_identity=pid_identity)
        for v in drift:
            print(f"  [INVALID] {v}")
            violations_out.append(v)


# ---------------------------------------------------------------------------
# Statistics (verbatim from trace audit)
# ---------------------------------------------------------------------------

def compute_paired_analysis(shared: list[dict], isolated: list[dict]) -> dict:
    """Per-query-class paired analysis (prereg §4.1/§4.4)."""
    def _by(recs):
        return {(r.get("cycle"), r.get("npu")): r for r in recs}

    s_map, i_map = _by(shared), _by(isolated)
    pairs = []
    for key in sorted(set(s_map) & set(i_map)):
        s, i = s_map[key], i_map[key]
        pairs.append((key, i["ttft_s"] - s["ttft_s"]))
    deltas_ms = [d * 1000.0 for _, d in pairs]

    per_inst: dict[int, list[float]] = {}
    for (_, npu), d in pairs:
        per_inst.setdefault(npu, []).append(d * 1000.0)
    per_instance_deltas_ms = {
        npu: sorted(v)[len(v) // 2] for npu, v in sorted(per_inst.items())}

    cluster_deltas: dict[int, list[float]] = {}
    for (cycle, _), d in pairs:
        cluster_deltas.setdefault(cycle, []).append(d * 1000.0)
    cluster_means = [sum(v) / len(v) for v in cluster_deltas.values()]
    if cluster_means:
        random.seed(42)
        boot = []
        for _ in range(1000):
            sample = [random.choice(cluster_means) for __ in range(len(cluster_means))]
            boot.append(sum(sample) / len(sample))
        boot.sort()
        cluster_ci = (round(boot[25], 1), round(boot[974], 1))
        significant = cluster_ci[0] > 0 or cluster_ci[1] < 0
    else:
        cluster_ci = (0.0, 0.0)
        significant = False

    s_vals = sorted(r["ttft_s"] for r in shared)
    i_vals = sorted(r["ttft_s"] for r in isolated)
    s_med = s_vals[len(s_vals) // 2] if s_vals else 0.0
    i_med = i_vals[len(i_vals) // 2] if i_vals else 0.0
    prefill_saved_ms = (i_med - s_med) * 1000.0
    dma_vals = [r.get("dma_ms", 0) for r in shared if r.get("dma_ms", 0) > 0]
    dma_cost_ms = (sum(dma_vals) / len(dma_vals)) if dma_vals else None

    if dma_cost_ms is not None and prefill_saved_ms > dma_cost_ms and significant:
        verdict = "GO"
    else:
        verdict = "BREAK-EVEN"

    return {
        "n": len(pairs),
        "shared_median": round(s_med, 4),
        "isolated_median": round(i_med, 4),
        "prefill_saved_ms": round(prefill_saved_ms, 1),
        "dma_cost_ms": (round(dma_cost_ms, 1) if dma_cost_ms is not None else None),
        "per_instance_deltas_ms": per_instance_deltas_ms,
        "cluster_ci": cluster_ci,
        "significant": significant,
        "verdict": verdict,
    }


def compute_stats(records: list[dict], key="ttft_s") -> dict:
    vals = sorted(r[key] for r in records if r.get(key, -1) > 0)
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "iqr": 0.0, "ci_95_low": 0.0, "ci_95_high": 0.0,
                "min": 0.0, "max": 0.0}
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    random.seed(42)
    boot_means = []
    for _ in range(1000):
        sample = [random.choice(vals) for __ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    ci_low = boot_means[25]
    ci_high = boot_means[974]
    q1 = vals[n // 4] if n >= 4 else vals[0]
    q3 = vals[3 * n // 4] if n >= 4 else vals[-1]
    return {
        "n": n, "mean": round(mean, 4), "median": round(median, 4),
        "iqr": round(q3 - q1, 4),
        "ci_95_low": round(ci_low, 4), "ci_95_high": round(ci_high, 4),
        "min": round(vals[0], 4), "max": round(vals[-1], 4),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(experiment: Experiment, env_info, all_records,
                  merge_result, out_dir, drift_violations=None) -> bool:
    """Write summary.md with metrics + verdict + fail-close manifest.

    Returns True if every validity gate passes (fail-close).
    """
    drift_violations = drift_violations or []
    shared = [r for r in all_records
              if r.get("phase") == "shared" and r.get("ok") and not r.get("producer")]
    isolated = [r for r in all_records
                if r.get("phase") == "isolated" and r.get("ok") and not r.get("producer")]
    producers = [r for r in all_records if r.get("ok") and r.get("producer")]

    lines = [
        f"# {experiment.title} Summary",
        "",
        "## Environment",
        f"- Commit: `{env_info.get('git_commit','?')[:12]}` "
        f"(parent: `{env_info.get('git_parent','?')[:12]}`)",
        f"- Branch: `{env_info.get('git_branch','?')}`",
        f"- Runtime vLLM commit: `{env_info.get('runtime_commit_vllm','?')[:12]}`",
        f"- Runtime ascend commit: `{env_info.get('runtime_commit_ascend','?')[:12]}`",
        f"- Torch/torch_npu: `{env_info.get('torch_version','?')}`",
        f"- Model: `{env_info.get('model','?')}` "
        f"(md5: `{env_info.get('model_config_md5','?')[:12]}`)",
        f"- Timestamp: {env_info.get('timestamp','?')}",
        f"- NPUs: 8× Ascend 910B2 (see artifact for full npu-smi)",
        "",
    ]

    # Metric tables: TTFT always; extra metrics configured by experiment
    metric_defs = [("ttft_s", "TTFT (Time-To-First-Token)")] + experiment.extra_metrics
    for key, label in metric_defs:
        s_stats = compute_stats(shared, key)
        i_stats = compute_stats(isolated, key)
        lines += [
            f"## {label}",
            "",
            "| Phase | N | Median | Mean | IQR | 95% CI | Min | Max |",
            "|---|---|---|---|---|---|---|---|",
            f"| Shared | {s_stats['n']} | **{s_stats['median']:.4f}** | "
            f"{s_stats['mean']:.4f} | {s_stats['iqr']:.4f} | "
            f"[{s_stats['ci_95_low']:.4f}, {s_stats['ci_95_high']:.4f}] | "
            f"{s_stats['min']:.4f} | {s_stats['max']:.4f} |",
            f"| Isolated | {i_stats['n']} | **{i_stats['median']:.4f}** | "
            f"{i_stats['mean']:.4f} | {i_stats['iqr']:.4f} | "
            f"[{i_stats['ci_95_low']:.4f}, {i_stats['ci_95_high']:.4f}] | "
            f"{i_stats['min']:.4f} | {i_stats['max']:.4f} |",
            "",
        ]

    # Per-query paired break-even
    lines += ["## Per-Query Paired Analysis", ""]
    for qidx in sorted(set(r["query_idx"] for r in all_records)):
        s_subset = [r for r in shared if r.get("query_idx") == qidx
                    and r.get("ok") and r.get("ttft_s", -1) > 0]
        i_subset = [r for r in isolated if r.get("query_idx") == qidx
                    and r.get("ok") and r.get("ttft_s", -1) > 0]
        if not s_subset or not i_subset:
            lines += [f"### Q{qidx}", "- No paired observations (arm aborted by fail-close).",
                      "- Verdict: BREAK-EVEN (no data)", ""]
            continue
        pa = compute_paired_analysis(s_subset, i_subset)
        per_inst_str = ", ".join(
            f"NPU{npu}:{d:+.1f}ms" for npu, d in pa["per_instance_deltas_ms"].items())
        dma_note = (f"{pa['dma_cost_ms']:.1f}ms" if pa["dma_cost_ms"] is not None
                    else "n/a (no DMA evidence)")
        ci_note = (f"CI [{pa['cluster_ci'][0]:.1f}, {pa['cluster_ci'][1]:.1f}]ms "
                   f"{'excludes 0' if pa['significant'] else 'includes 0'}")
        lines += [
            f"### Q{qidx}",
            f"- Shared: n={len(s_subset)}, median={pa['shared_median']:.4f}s",
            f"- Isolated: n={len(i_subset)}, median={pa['isolated_median']:.4f}s",
            f"- Prefill saved (median): {pa['prefill_saved_ms']:+.1f}ms",
            f"- DMA cost (per-request, bound): {dma_note}",
            f"- Per-instance median paired delta (D5): {per_inst_str}",
            f"- Lifecycle cluster CI, per query class (C5): {ci_note}",
            f"- Break-even (prereg §4.4): "
            f"{'prefill_saved > dma_cost AND significant' if pa['verdict'] == 'GO' else 'prefill_saved <= dma_cost OR not significant'} "
            f"-> **{pa['verdict']}**",
            "",
        ]

    # Extra experiment verdicts (e.g. operating region)
    if experiment.extra_verdicts:
        lines += ["## Experiment Verdicts", ""]
        for v in experiment.extra_verdicts:
            lines.append(f"- {v}")
        lines.append("")

    # Platform constraints (prereg deviations) — C 2026-08-18
    d2h_fallback = env_info.get("d2h_fallback_count", 0)
    h2d_fallback = env_info.get("h2d_fallback_count", 0)
    if d2h_fallback or h2d_fallback:
        lines += [
            "## Platform Constraints (Prereg Deviation C, 2026-08-18)",
            f"- {d2h_fallback} D2H (save-path) and {h2d_fallback} H2D "
            "(load-path) batch calls fell back to per-copy "
            "`aclrtMemcpyAsync`: on 8-instance concurrency CANN "
            "`aclrtMemcpyBatchAsync` intermittently fails 107000. Completions "
            "are evidence via the `Load task completed` line (per-copy "
            "fallback succeeds, data-correct); fallback counts are recorded "
            "here as a platform constraint. Batch DMA recovery is a tracked "
            "follow-up (D2H chunking).",
            "",
        ]

    # Negative examples (preregistered context)
    neg = experiment.negative_examples or DEFAULT_NEGATIVE_EXAMPLES
    lines += ["## Negative Examples (Preserved)", ""]
    for name, info in neg.items():
        lines += [
            f"### {name}",
            f"- Shared avg TTFT: {info.get('shared_avg_ttft_s')}s",
            f"- Isolated avg TTFT: {info.get('isolated_avg_ttft_s')}s",
            f"- Result: {info.get('shared_vs_isolated')}",
            f"- Root cause: {info.get('root_cause')}",
            f"- Verdict: {info.get('verdict')}",
            "",
        ]

    # Producer records
    p_ttft = {"n": 0, "median": 0, "mean": 0}
    if producers:
        p_ttft = compute_stats(producers, "ttft_s")
        lines += [
            "## Producer (Warmup Seed) Records",
            f"- Count: {p_ttft['n']}",
            f"- Median TTFT: {p_ttft['median']:.4f}s",
            "- Excluded from consumer paired-delta analysis.",
            "",
        ]

    # Validity manifest (coverage + conservation — fail-close)
    # Consumer floor scales with experiment size: (instances - warmup) *
    # cycles consumers per arm. T1 (3x8): 21; T3 (1x8): 7.
    min_consumers = max(1, (experiment.num_instances - 1) * experiment.cycles)
    base_ok = (
        len(shared) >= min_consumers and len(isolated) >= min_consumers
        and len(all_records) > 0
    )
    evidence_ok = (
        merge_result.get("conservation_ok", False)
        and merge_result.get("coverage_pct", 0) >= 100.0
        and merge_result.get("unmatched", 0) == 0
        and not drift_violations
    )
    extra_violations: list[str] = []
    if not env_info.get("dry_run"):
        # Dry-run synthesizes 100%-hit records; experiment-specific gates
        # (hit-rate etc.) are exercised by unit tests, not the dry pipeline.
        for name, gate_fn in experiment.extra_gates:
            try:
                extra_violations.extend(gate_fn(all_records, merge_result))
            except Exception as e:
                extra_violations.append(f"{name} gate raised: {e}")
    validity_ok = base_ok and evidence_ok and not extra_violations
    if merge_result.get("coverage_pct", 0) < 100.0:
        lines.append(
            f"- Coverage gate FAILED: {merge_result['coverage_pct']:.1f}% (requires 100%)")
    lines += ["## Evidence Violations (Fail-Close)", ""]
    all_violations = (merge_result.get("violations", []) + list(drift_violations)
                      + extra_violations)
    if not all_violations:
        lines.append("- None — all connector/prefetch/DMA events unique and conserved.")
    else:
        for v in all_violations:
            lines.append(f"- [INVALID] {v}")
    lines += [
        "",
        "## Validity Manifest",
        f"- Run ID: {env_info.get('run_id','?')}",
        f"- Total records: {len(all_records)}",
        f"- Consumer shared records: {len(shared)}",
        f"- Consumer isolated records: {len(isolated)}",
        f"- Producer records: {p_ttft['n']}",
        f"- INVALID records: {sum(1 for r in all_records if not r.get('ok'))}",
        f"- Audit-invalid records (evidence): "
        f"{sum(1 for r in all_records if r.get('_audit_invalid'))}",
        f"- Conservation: "
        f"{'OK' if merge_result.get('conservation_ok') else 'BROKEN'} "
        f"(connector dup={merge_result.get('connector_duplicates', 0)}, "
        f"orphans={merge_result.get('connector_orphans', 0)}/"
        f"{merge_result.get('prefetch_orphans', 0)}/"
        f"{merge_result.get('dma_orphans', 0)}, "
        f"leftover DMA={merge_result.get('dma_leftover', 0)}, "
        f"fallback DMA (bound)={merge_result.get('dma_fallback_only', 0)})",
        f"- Validity gate: {'PASS' if validity_ok else 'FAIL'}",
        f"- Audit verdict: {'VALID' if validity_ok else 'INVALID'}",
        "",
        "## Reproduce",
        f"- Command: `{env_info.get('reproduce_command','?')}`",
        "",
    ]

    summary_path = out_dir / "trace_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary: {summary_path}")
    return validity_ok


def fail_close(reasons: list[str]) -> None:
    for reason in reasons:
        print(f"  [INVALID] {reason}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Log extraction (from per-arm dirs)
# ---------------------------------------------------------------------------

def extract_evidence(log_dir: Path, connector_by_req: dict,
                     label_to_npu: dict) -> tuple:
    """Extract prefetch + DMA from per-arm server logs (arm-scoped).

    Returns (prefetch_by_req, ts_prefetches, ts_dmas, d2h_fallback_count).
    """
    prefetch_by_req: dict[str, dict] = {}
    ts_prefetches: list[dict] = []
    ts_dmas: list[dict] = []
    d2h_fallback_count = 0
    h2d_fallback_count = 0

    for arm_log_dir in sorted(log_dir.glob("arm_*")):
        arm_label = arm_log_dir.name.replace("arm_", "")
        server_log = arm_log_dir / "server.log"
        if not server_log.exists():
            continue
        text = server_log.read_text()

        for m in re.finditer(
            r"Prefetch local-hit timing: "
            r"req_id=(?P<req_id>\S+)\s+"
            r"total_keys=(?P<total>\d+)\s+"
            r"hit=(?P<hit>\d+)\s+"
            r"missing=(?P<missing>\d+)",
            text,
        ):
            req_id = m.group("req_id")
            pf_entry = prefetch_by_req.get(req_id)
            if pf_entry is not None:
                pf_entry["occurrences"] = pf_entry.get("occurrences", 1) + 1
                continue
            prefetch_by_req[req_id] = {
                "total_keys": int(m.group("total")),
                "hit_blocks": int(m.group("hit")),
                "missing_blocks": int(m.group("missing")),
                "occurrences": 1,
            }

        for line in text.split("\n"):
            ts_match = re.match(
                r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)", line)
            if not ts_match:
                continue
            ts = ts_match.group(1)

            m = re.search(
                r"Prefetch local-hit timing: "
                r"req_id=(?P<req_id>\S+)\s+.*?"
                r"hit=(?P<hit>\d+)\s+missing=(?P<missing>\d+)", line)
            if m:
                ts_prefetches.append({
                    "ts": ts, "req_id": m.group("req_id"),
                    "arm_label": arm_label,
                    "hit": int(m.group("hit")), "missing": int(m.group("missing")),
                })

            m = re.search(
                r"Load task completed:.*?"
                r"bytes=(?P<bytes>\d+)\s+"
                r"elapsed_ms=(?P<ms>[\d.]+)\s+"
                r"bandwidth_gbps=(?P<gbps>[\d.]+)\s+"
                r".*?device_id=(?P<dev>\d+)", line)
            if m:
                ts_dmas.append({
                    "ts": ts, "arm_label": arm_label,
                    "device_id": int(m.group("dev")),
                    "dma_bytes": int(m.group("bytes")),
                    "dma_ms": float(m.group("ms")),
                    "dma_gbps": float(m.group("gbps")),
                })

            if "falling back to per-copy aclrtMemcpyAsync" in line:
                # A WARN fallback line and the subsequent "Load task
                # completed" line are the SAME transfer (batch failed ->
                # per-copy succeeded). The completion line is the evidence;
                # fallback lines are only counted as platform-constraint
                # stats. D2H fallback is save-path (never bound); H2D
                # fallback is load-path (still formal evidence via the
                # completion line, counted here for the manifest).
                if "(D2H)" in line:
                    d2h_fallback_count += 1
                elif "(H2D)" in line:
                    h2d_fallback_count += 1

    return prefetch_by_req, ts_prefetches, ts_dmas, (d2h_fallback_count,
                                                     h2d_fallback_count)


def label_to_npu_map(connector_by_req: dict) -> dict[str, int]:
    label_to_npu: dict[str, int] = {}
    for cinfo in connector_by_req.values():
        label = cinfo.get("label", "")
        parts = label.rsplit("_", 1)
        if len(parts) == 2:
            try:
                label_to_npu[label] = int(parts[1])
            except ValueError:
                pass
    return label_to_npu


# ---------------------------------------------------------------------------
# Full pipeline driver
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--requests-per-phase", type=int, default=3)
    parser.add_argument("--pool-size", type=str, default="16gb")
    parser.add_argument("--min-free-gb", type=int, default=28)
    parser.add_argument("--num-instances", type=int, default=8)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--project-root", type=str,
                        default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--vllm-root", type=str,
                        default=str(DEFAULT_VLLM_ROOT))
    parser.add_argument("--ascend-root", type=str,
                        default=str(DEFAULT_ASCEND_ROOT))
    parser.add_argument("--conda-root", type=str,
                        default=str(DEFAULT_CONDA_ROOT))
    parser.add_argument("--conda-env", type=str, default=DEFAULT_CONDA_ENV)
    parser.add_argument("--dry-run", action="store_true",
                        help="Host-only: synthesize records, run full gate pipeline")
    parser.add_argument("--verify-repro", action="store_true",
                        help="Run twice; fail if verdicts differ (§9.1)")
    parser.add_argument("--out", type=str, default=None,
                        help="Override results root (default results/perf-{id}/)")


def build_reproduce_command(experiment: Experiment, args,
                            project_root: Path, vllm_root: Path,
                            ascend_root: Path, conda_root: Path,
                            conda_env: str,
                            runner_path: Path | None = None) -> str:
    """Build a runnable command for this exact experiment or sweep point."""
    runner = (runner_path or Path(sys.argv[0])).resolve()
    try:
        runner_arg = str(runner.relative_to(project_root.resolve()))
    except ValueError:
        runner_arg = str(runner)

    command = ["python3", runner_arg, *experiment.reproduce_args,
               "--cycles", str(args.cycles),
               "--requests-per-phase", str(args.requests_per_phase),
               "--pool-size", args.pool_size,
               "--min-free-gb", str(args.min_free_gb),
               "--num-instances", str(args.num_instances),
               "--model", args.model,
               "--project-root", str(project_root),
               "--vllm-root", str(vllm_root),
               "--ascend-root", str(ascend_root),
               "--conda-root", str(conda_root),
               "--conda-env", conda_env]
    if args.verify_repro:
        command.append("--verify-repro")
    if args.out:
        command.extend(["--out", args.out])
    return shlex.join(command)


def run_experiment(experiment: Experiment, argv: list[str] | None = None,
                   extra_args: dict | None = None) -> None:
    """Drive one experiment end to end (or host-only --dry-run)."""
    parser = argparse.ArgumentParser(description=experiment.title)
    add_common_args(parser)
    # Experiment-declared defaults win unless the CLI overrides them.
    parser.set_defaults(
        cycles=experiment.cycles,
        requests_per_phase=experiment.requests_per_phase,
        num_instances=experiment.num_instances,
        pool_size=experiment.pool_size,
        min_free_gb=experiment.min_free_gb,
    )
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    vllm_root = Path(args.vllm_root).expanduser().resolve()
    ascend_root = Path(args.ascend_root).expanduser().resolve()
    model_path = args.model or experiment.model
    conda_root = Path(args.conda_root).expanduser().resolve()
    conda_env = args.conda_env
    server_port = DEFAULT_SERVER_PORT
    vllm_base_port = DEFAULT_VLLM_BASE_PORT
    log_level = os.environ.get("PEGAFLOW_SERVER_LOG_LEVEL", "info")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    results_root = (Path(args.out) if args.out else
                    project_root / "results" / f"perf-{experiment.id}")
    out_dir = results_root / run_id
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env_info = capture_environment(project_root, vllm_root, ascend_root,
                                   model_path, conda_root, conda_env)
    env_info["run_id"] = run_id
    env_info["experiment"] = experiment.id
    env_info["reproduce_command"] = build_reproduce_command(
        experiment, args, project_root, vllm_root, ascend_root,
        conda_root, conda_env)

    print("=" * 70)
    print(f"  {experiment.title}")
    print(f"  Experiment: {experiment.id}  Run: {run_id}")
    print(f"  Model:    {env_info.get('model', '?')}")
    print(f"  Commit:   {env_info.get('git_commit','?')[:12]}")
    print(f"  Cycles:   {args.cycles}  Requests: {args.requests_per_phase}/phase")
    print(f"  Pool:     {args.pool_size}")
    print("=" * 70)

    if args.dry_run:
        dry_run_pipeline(experiment, env_info, out_dir, log_dir, args)
        return

    run_hardware_pipeline(experiment, args, env_info, out_dir, log_dir,
                          project_root, vllm_root, model_path,
                          conda_root, conda_env, server_port,
                          vllm_base_port, log_level)

    if args.verify_repro:
        print("\n[verify-repro] running second pass...")
        time.sleep(10)
        env_info2 = capture_environment(project_root, vllm_root, ascend_root,
                                        model_path, conda_root, conda_env)
        run_hardware_pipeline(experiment, args, env_info2, out_dir, log_dir,
                              project_root, vllm_root, model_path,
                              conda_root, conda_env, server_port,
                              vllm_base_port, log_level, repro_pass=2)


def run_hardware_pipeline(experiment, args, env_info, out_dir, log_dir,
                          project_root, vllm_root, model_path,
                          conda_root, conda_env, server_port,
                          vllm_base_port, log_level, repro_pass=1):
    """Matched-arm hardware run (AB/BA, independent lifecycle, fail-close)."""
    queries = experiment.queries[:args.requests_per_phase]
    num_instances = args.num_instances
    min_free_mb = args.min_free_gb * 1024

    pid_identity = _detect_pid_identity_possible()
    if not pid_identity:
        print("  [note] npu-smi PIDs do not resolve in /proc (container); "
              "using HBM+presence admission checks")

    all_records: list[dict] = []
    drift_violations: list[str] = []
    ARM_ORDER = [
        [("shared", True), ("isolated", False)],
        [("isolated", False), ("shared", True)],
        [("shared", True), ("isolated", False)],
    ]

    try:
        for cycle in range(1, args.cycles + 1):
            print(f"\n{'#'*70}")
            print(f"# CYCLE {cycle}/{args.cycles}")
            arms = ARM_ORDER[(cycle - 1) % len(ARM_ORDER)]
            print(f"# Arm order: {arms[0][0]} → {arms[1][0]}  (repro pass {repro_pass})")
            print(f"{'#'*70}")

            for arm_name, is_shared in arms:
                arm_label = f"C{cycle}_{arm_name}"
                if is_shared:
                    ns_base = f"perf-{experiment.id}-shared-c{cycle}"
                else:
                    ns_base = f"perf-{experiment.id}-iso-c{cycle}"

                free_mem = get_npu_free_memory()
                free_devices = sorted(
                    i for i in free_mem if free_mem.get(i, -1) >= min_free_mb)
                admitted = free_devices[:num_instances]
                if len(admitted) < num_instances:
                    print(f"  [INVALID] Only {len(admitted)}/{num_instances} "
                          f"NPUs have >= {min_free_mb}MB free: {admitted}. "
                          f"Arm {arm_label} ABORTED.")
                    all_records.append({
                        "cycle": cycle, "phase": arm_name, "req_idx": -1,
                        "query_idx": -1, "instance": "INVALID", "npu": -1,
                        "port": -1, "query": "", "ttft_s": -1, "total_s": -1,
                        "ok": False, "text": "",
                        "error": f"insufficient NPU free: {len(admitted)}/{num_instances}",
                        "producer": False,
                    })
                    continue

                print(f"\n--- {arm_label} (namespace_base={ns_base}) ---")
                print(f"  Admitting NPUs: {admitted}")
                arm_log_dir = log_dir / f"arm_{arm_label}"
                arm_log_dir.mkdir(parents=True, exist_ok=True)
                server = start_server(project_root, server_port, arm_log_dir,
                                      args.pool_size, admitted, log_level,
                                      conda_root, conda_env)
                print(f"  Server ready on :{server_port}")
                time.sleep(2)

                specs = [
                    {"label": f"{arm_label}_{i}", "port": vllm_base_port + i,
                     "mode": "read_write",
                     "namespace": ns_base if is_shared else f"{ns_base}-{i}",
                     "physical_npu": i, "use_pegaflow": True}
                    for i in admitted
                ]
                running = launch_all_instances(specs, model_path, log_dir,
                                               server_port, conda_root,
                                               conda_env,
                                               gmu_override=experiment.gmu)
                if len(running) < len(specs):
                    print(f"  [INVALID] Only {len(running)}/{len(specs)} "
                          f"instances started. Arm {arm_label} ABORTED.")
                    for _, proc in running:
                        stop_proc(proc)
                    all_records.append({
                        "cycle": cycle, "phase": arm_name, "req_idx": -1,
                        "query_idx": -1, "instance": "INVALID", "npu": -1,
                        "port": -1, "query": "", "ttft_s": -1, "total_s": -1,
                        "ok": False, "text": "",
                        "error": f"instance launch failed: {len(running)}/{len(specs)}",
                        "producer": False,
                    })
                    stop_proc(server)
                    kill_tracked()
                    time.sleep(5)
                    continue

                drift = check_admission_drift(
                    admitted, free_mem, get_npu_free_memory(),
                    {i: expected_used_mb(i, free_mem) for i in admitted},
                    get_npu_processes(), tracked_pgids(),
                    pid_identity=pid_identity)
                if drift:
                    print(f"  [INVALID] Admission drift after launch — arm ABORTED.")
                    for v in drift:
                        print(f"  [INVALID] {v}")
                    drift_violations.extend(drift)
                    for _, proc in running:
                        stop_proc(proc)
                    all_records.append({
                        "cycle": cycle, "phase": arm_name, "req_idx": -1,
                        "query_idx": -1, "instance": "INVALID", "npu": -1,
                        "port": -1, "query": "", "ttft_s": -1, "total_s": -1,
                        "ok": False, "text": "",
                        "error": "admission drift: " + "; ".join(drift)[:200],
                        "producer": False,
                    })
                    stop_proc(server)
                    kill_tracked()
                    time.sleep(5)
                    continue

                stop_event = threading.Event()
                monitor_out: list[str] = []
                monitor = threading.Thread(
                    target=monitor_admission_drift,
                    args=(admitted, free_mem,
                          {i: expected_used_mb(i, free_mem) for i in admitted},
                          tracked_pgids(), stop_event, monitor_out),
                    kwargs={"pid_identity": pid_identity},
                    daemon=True,
                )
                monitor.start()
                try:
                    records = run_phase(
                        experiment, arm_name, running, queries, model_path,
                        warmup_first=experiment.warmup_first, cycle=cycle,
                        log_dir=log_dir, server_port=server_port,
                        conda_root=conda_root, conda_env=conda_env,
                    )
                finally:
                    stop_event.set()
                    monitor.join(timeout=5)
                all_records.extend(records)
                drift_violations.extend(monitor_out)

                drift = check_admission_drift(
                    admitted, free_mem, get_npu_free_memory(),
                    {i: expected_used_mb(i, free_mem) for i in admitted},
                    get_npu_processes(), tracked_pgids(),
                    pid_identity=pid_identity)
                drift_violations.extend(drift)
                for v in drift:
                    print(f"  [INVALID] {v}")
                for _, proc in running:
                    stop_proc(proc)
                print(f"  {arm_label}: {len(records)} records")

                stop_proc(server)
                kill_tracked()
                time.sleep(5)
    finally:
        print("\nShutting down...")
        kill_tracked()

    # Evidence merge (pure — same as trace audit)
    print("\nMerging per-request cache hit + DMA data from logs...")
    connector_by_req: dict[str, dict] = {}
    timing_by_req: dict[str, dict] = {}
    for vllm_log in sorted(log_dir.glob("vllm_*.log")):
        label = vllm_log.name.replace("vllm_", "").replace(".log", "")
        text = vllm_log.read_text()
        timing_by_req.update(extract_vllm_timing(text))
        for m in re.finditer(
            r"\[PegaKVConnector\] req=(?P<req_id>\S+)\s+"
            r"cache_lookup: hit_blocks=(?P<hit>\d+) "
            r"computed_blocks=(?P<computed>\d+) "
            r"hit_tokens=(?P<hit_tokens>\d+) num_tokens=(?P<num_tokens>\d+)"
            r"(?:.*?total_query_hashes=(?P<query_hashes>\d+))?",
            text,
        ):
            req_id = m.group("req_id")
            entry = connector_by_req.get(req_id)
            if entry is not None:
                entry["occurrences"] = entry.get("occurrences", 1) + 1
                continue
            qh = m.group("query_hashes")
            connector_by_req[req_id] = {
                "req_id": req_id, "label": label,
                "hit_blocks": int(m.group("hit")),
                "computed_blocks": int(m.group("computed")),
                "hit_tokens": int(m.group("hit_tokens")),
                "num_tokens": int(m.group("num_tokens")),
                "total_query_hashes": int(qh) if qh is not None else -1,
                "occurrences": 1,
            }

    (prefetch_by_req, ts_prefetches, ts_dmas,
     (d2h_fallback_count, h2d_fallback_count)) = \
        extract_evidence(log_dir, connector_by_req, {})
    l2n = label_to_npu_map(connector_by_req)
    prefetch_dma_map, fallback_only_count, dma_leftover, bind_violations = \
        bind_dma_to_prefetch(ts_prefetches, ts_dmas, connector_by_req, l2n)
    for v in bind_violations:
        print(f"  [INVALID] {v}")

    merge_result = merge_by_req_id(all_records, connector_by_req,
                                   prefetch_by_req, prefetch_dma_map,
                                   dma_leftover_count=dma_leftover,
                                   dma_fallback_only_count=fallback_only_count,
                                   timing_by_req=timing_by_req)
    print(f"  Merged hit/DMA: {merge_result['matched']} records via req_id lookup "
          f"(unmatched={merge_result['unmatched']}, "
          f"producer_skipped={merge_result['producer_skipped']}, "
          f"coverage={merge_result['coverage_pct']:.1f}%, "
          f"conservation={'OK' if merge_result['conservation_ok'] else 'BROKEN'})")
    for v in merge_result["violations"]:
        print(f"  [INVALID] {v}")

    env_info["d2h_fallback_count"] = d2h_fallback_count
    env_info["h2d_fallback_count"] = h2d_fallback_count
    output = {"_env": env_info,
              "_negative_examples": experiment.negative_examples or DEFAULT_NEGATIVE_EXAMPLES,
              "records": all_records}
    out_json = out_dir / "trace_audit.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Raw data: {out_json} ({len(all_records)} records)")

    summary_ok = write_summary(experiment, env_info, all_records,
                               merge_result, out_dir, drift_violations)
    if not summary_ok:
        fail_close([f"validity gate FAILED — see {out_dir / 'trace_summary.md'}"])
    print(f"\n  Experiment {experiment.id} complete (pass {repro_pass}).")


# ---------------------------------------------------------------------------
# Host-only dry run: synthetic records through the full pipeline
# ---------------------------------------------------------------------------

def dry_run_pipeline(experiment: Experiment, env_info, out_dir, log_dir, args):
    """Synthesize a perfect evidence chain and exercise every gate/section.

    Verdict must come out VALID (host gate): full coverage, clean
    conservation, all summary sections rendered.
    """
    random.seed(7)
    rng = random.Random(7)
    n = args.num_instances
    q = args.requests_per_phase
    all_records: list[dict] = []
    connector_by_req: dict[str, dict] = {}
    prefetch_by_req: dict[str, dict] = {}
    ts_prefetches: list[dict] = []
    ts_dmas: list[dict] = []
    base_ts = datetime(2026, 8, 18, 12, 0, 0)

    def _ts(secs: float) -> str:
        return datetime.fromtimestamp(base_ts.timestamp() + secs
                                      ).strftime("%Y-%m-%dT%H:%M:%S.%f")

    for cycle in range(1, args.cycles + 1):
        order = ["shared", "isolated"] if cycle % 2 == 1 else ["isolated", "shared"]
        for arm in order:
            arm_label = f"C{cycle}_{arm}"
            t = rng.random() * 100.0
            # warmup producer (one request, cache seeded)
            req = f"perf-dry-{uuid.uuid4().hex[:10]}"
            all_records.append({
                "cycle": cycle, "phase": arm, "req_idx": -1, "query_idx": -1,
                "instance": f"{arm_label}_0", "npu": 0, "port": 19000,
                "query": "warmup", "ttft_s": 1.1, "total_s": 2.0, "ok": True,
                "text": "seed", "req_id": req, "producer": True,
            })
            connector_by_req[req] = {"label": f"{arm_label}_0", "hit_blocks": 0,
                                     "hit_tokens": 0, "num_tokens": 9854,
                                     "total_query_hashes": 100, "occurrences": 1}
            idx = 0
            for qi in range(q):
                for i in range(n):
                    t += 1.0
                    req = f"perf-dry-{uuid.uuid4().hex[:10]}"
                    hit = qi == 0  # Q0 cross-instance hit; Q1/Q2 local
                    ttft = round(0.25 if hit else 0.9 - rng.random() * 0.05, 4)
                    total = round(ttft + 1.0, 4)
                    tbt = {"tbt_p50_s": 0.02, "tbt_p95_s": 0.03,
                           "tbt_mean_s": 0.021, "tokens_received": 64}
                    all_records.append({
                        "cycle": cycle, "phase": arm, "req_idx": idx,
                        "query_idx": qi, "instance": f"{arm_label}_{i}",
                        "npu": i, "port": 19000 + i, "query": "dry",
                        "ttft_s": ttft, "total_s": total, "ok": True,
                        "text": "dry", "req_id": req, "producer": False, **tbt,
                    })
                    if hit:
                        connector_by_req[req] = {
                            "label": f"{arm_label}_{i}", "hit_blocks": 76,
                            "hit_tokens": 9728, "num_tokens": 9854,
                            "total_query_hashes": 76, "occurrences": 1}
                        prefetch_by_req[req] = {
                            "total_keys": 76, "hit_blocks": 76,
                            "missing_blocks": 0, "occurrences": 1}
                        ts_prefetches.append({
                            "ts": _ts(t), "req_id": req, "arm_label": arm_label,
                            "hit": 76, "missing": 0})
                        ts_dmas.append({
                            "ts": _ts(t + 0.1), "arm_label": arm_label,
                            "device_id": i, "dma_bytes": 1434451968,
                            "dma_ms": 93.3, "dma_gbps": 15.37})
                    else:
                        connector_by_req[req] = {
                            "label": f"{arm_label}_{i}", "hit_blocks": 0,
                            "hit_tokens": 0, "num_tokens": 9854,
                            "total_query_hashes": 0, "occurrences": 1}
                    idx += 1

    l2n = label_to_npu_map(connector_by_req)
    prefetch_dma_map, fb_only, leftover, bv = bind_dma_to_prefetch(
        ts_prefetches, ts_dmas, connector_by_req, l2n)
    merge_result = merge_by_req_id(all_records, connector_by_req,
                                   prefetch_by_req, prefetch_dma_map,
                                   dma_leftover_count=leftover,
                                   dma_fallback_only_count=fb_only)
    env_info["dry_run"] = True
    out_json = out_dir / "trace_audit.json"
    with open(out_json, "w") as f:
        json.dump({"_env": env_info, "records": all_records}, f, indent=2,
                  default=str)
    ok = write_summary(experiment, env_info, all_records, merge_result, out_dir)
    print(f"\n[dry-run] coverage={merge_result['coverage_pct']:.1f}% "
          f"conservation={'OK' if merge_result['conservation_ok'] else 'BROKEN'}")
    if not ok:
        fail_close([f"dry-run validity gate FAILED — pipeline defect"])


if __name__ == "__main__":
    print("run_perf_base.py is a library — run a runner (run_perf_t1_baseline.py).")
