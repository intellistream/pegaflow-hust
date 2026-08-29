import importlib
import json
import sys
import tomllib
import types
from pathlib import Path

import pytest
from vllm.plugins.contracts import ComponentPermission
from vllm.plugins.manifest import load_extension_bundle_manifest
from vllm.plugins.startup import resolve_extension_startup

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "packaging" / "extension-bundle-v1.json"
WHEEL_MANIFEST = (
    ROOT / "python" / "pegaflow" / "manifests" / "extension-bundle-v1.json"
)


def test_manifest_declares_split_roles_and_honest_permissions() -> None:
    bundle = load_extension_bundle_manifest(MANIFEST)
    by_id = {component.component_id: component for component in bundle.components}

    assert set(by_id) == {"scheduler", "worker", "telemetry"}
    assert by_id["scheduler"].permissions == (
        ComponentPermission.IPC,
        ComponentPermission.NETWORK_EGRESS,
    )
    assert by_id["worker"].permissions == (
        ComponentPermission.DEVICE_ACCESS,
        ComponentPermission.IPC,
        ComponentPermission.NETWORK_EGRESS,
    )
    assert by_id["telemetry"].permissions == ()


def test_manifest_requires_explicit_host_permission_allowlist() -> None:
    permissions = (
        ComponentPermission.DEVICE_ACCESS,
        ComponentPermission.IPC,
        ComponentPermission.NETWORK_EGRESS,
    )
    resolution = resolve_extension_startup(
        (MANIFEST,), allowed_permissions=permissions
    )
    assert len(resolution.snapshot.components) == 3


def test_telemetry_import_does_not_import_connector_facade() -> None:
    sys.modules.pop("pegaflow.connector.facade", None)
    module = importlib.import_module("pegaflow.connector.telemetry")

    assert "pegaflow.connector.facade" not in sys.modules
    assert callable(module.PegaKVTelemetryProvider.build_kv_connector_stats)
    assert callable(module.PegaKVTelemetryProvider.build_prom_metrics)


def test_legacy_module_export_resolves_to_typed_facade_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = types.ModuleType("pegaflow.pegaflow")
    for name in ("EngineRpcClient", "PyLoadState", "QueryLoading", "QueryReady"):
        setattr(native, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, "pegaflow.pegaflow", native)

    from pegaflow.connector import PegaKVConnector as legacy_class
    from pegaflow.connector.facade import PegaKVConnector as typed_class

    assert legacy_class is typed_class


def test_manifest_is_closed_json_document() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "vllm-hust.pegaflow"
    assert payload["host_api_range"] == ">=1,<2"


def test_wheel_registers_the_same_static_manifest_without_import_hook() -> None:
    project = tomllib.loads(
        (ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8")
    )
    entry_points = project["project"]["entry-points"]

    assert entry_points["vllm.extension_bundles"] == {
        "vllm-hust.pegaflow": "pegaflow.manifests"
    }
    assert WHEEL_MANIFEST.read_bytes() == MANIFEST.read_bytes()
    assert "pegaflow/manifests/extension-bundle-v1.json" in (
        project["tool"]["maturin"]["include"]
    )
