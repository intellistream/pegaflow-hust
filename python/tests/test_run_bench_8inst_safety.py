import importlib.util
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "run_bench_8inst.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_bench_8inst", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_has_no_global_process_kill() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "pkill" not in source
    assert "kill_all" not in source


def test_stop_proc_terminates_only_tracked_process_group() -> None:
    runner = load_runner()
    proc = subprocess.Popen(["sleep", "60"], preexec_fn=os.setsid)
    runner._track_proc(proc)

    runner.stop_proc(proc)

    assert proc.poll() is not None
    assert proc not in runner._OWNED_PROCESSES


def test_port_probe_does_not_leave_a_listener() -> None:
    runner = load_runner()
    assert runner._port_is_free(18799)
    assert runner._port_is_free(18799)
