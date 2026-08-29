# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "run_bench_8inst.py"


def _module():
    spec = importlib.util.spec_from_file_location("pegaflow_real_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_gate_rejects_another_checkout(tmp_path: Path) -> None:
    module = _module()
    selected = tmp_path / "selected"
    source = selected / "python" / "pegaflow" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    outside = tmp_path / "installed" / "pegaflow" / "__init__.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("", encoding="utf-8")
    assert module._source_is_under(str(source), selected)
    assert not module._source_is_under(str(outside), selected)
    assert not module._source_is_under(None, selected)


def test_probe_parser_ignores_logs_before_json() -> None:
    module = _module()
    assert module._last_json_object('INFO runtime\n{"ok": true}\n') == {"ok": True}
    assert module._last_json_object("INFO only\n") is None


def test_runner_has_no_hard_coded_root_conda_activation() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "/root/miniconda3" not in source
    assert "conda activate vllm-hust-dev" not in source
    assert "str(PYTHON_BIN)" in source
    assert "cwd=CORE_ROOT" in source
    assert '"vllm.engine.arg_utils"' in source
    assert '"pegaflow.connector"' in source
    assert '"platform_npu_runtime"' in source
    assert '"server_python_abi"' in source


def test_typed_and_legacy_configs_remain_distinct() -> None:
    module = _module()
    legacy = module.build_kv_transfer_config("read_write", "legacy")
    typed = module.build_kv_transfer_config("read_write", "typed")
    assert legacy["kv_connector_module_path"] == "pegaflow.connector"
    assert "kv_connector_selection" not in legacy
    assert "kv_connector_module_path" not in typed
    assert typed["kv_connector_selection"]["connectors"][0]["connector_id"] == (
        "pegaflow"
    )


def test_runtime_environment_includes_controlled_python_site_packages(
    tmp_path: Path,
) -> None:
    module = _module()
    module.PYTHON_BIN = Path(sys.executable)
    module.CORE_ROOT = tmp_path / "core"
    module.PROJECT_ROOT = tmp_path / "provider"

    metadata = module._controlled_python_metadata(str(module.PYTHON_BIN))
    environment = module._runtime_environment()
    python_paths = environment["PYTHONPATH"].split(os.pathsep)

    assert "error" not in metadata
    assert str(module.CORE_ROOT) == python_paths[0]
    assert str(module.PROJECT_ROOT / "python") == python_paths[1]
    assert set(metadata["site_packages"]).issubset(python_paths)
    if metadata["libdir"]:
        assert environment["LD_LIBRARY_PATH"].split(os.pathsep)[0] == metadata["libdir"]
    if metadata["prefix"] != metadata["base_prefix"]:
        assert environment["VIRTUAL_ENV"] == metadata["prefix"]
        assert environment["PATH"].split(os.pathsep)[0] == str(
            Path(metadata["prefix"]) / "bin"
        )
