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


def test_legacy_and_typed_configs_are_mutually_exclusive() -> None:
    runner = load_runner()
    legacy = runner.build_kv_transfer_config("read_write", "legacy")
    typed = runner.build_kv_transfer_config("read_write", "typed")

    assert legacy["kv_connector"] == "PegaKVConnector"
    assert legacy["kv_connector_module_path"] == "pegaflow.connector"
    assert "kv_connector_selection" not in legacy
    assert "kv_connector" not in typed
    assert "kv_connector_module_path" not in typed
    selection = typed["kv_connector_selection"]
    connector = selection["connectors"][0]
    assert connector["scheduler_component"] == "vllm-hust.pegaflow/scheduler"
    assert connector["worker_component"] == "vllm-hust.pegaflow/worker"
    assert connector["telemetry_component"] == "vllm-hust.pegaflow/telemetry"


def test_typed_mode_declares_explicit_permission_allowlist() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "VLLM_EXTENSION_ALLOWED_PERMISSIONS" in source
    assert '"device_access,ipc,network_egress"' in source
