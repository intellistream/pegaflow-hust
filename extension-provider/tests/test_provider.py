from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm_hust_ext.manifest import load_manifest

from vllm_hust_pegaflow_provider import provider as provider_module
from vllm_hust_pegaflow_provider.provider import PegaFlowProvider

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src/vllm_hust_pegaflow_provider/manifests/vllm-hust-extension-v0.2.json"


def test_manifest_preserves_external_service_boundary() -> None:
    manifest = load_manifest(MANIFEST)

    assert manifest.bundle_id == "org.vllm-hust.pegaflow"
    assert manifest.kind == "kv_service_adapter"
    assert manifest.lifecycle_owner == "external_operator"
    assert manifest.runtime.type == "composite"
    assert manifest.requires_services[0].endpoint_config == "health_url"


def test_plan_renders_primary_connector_without_mutating_service() -> None:
    manifest = load_manifest(MANIFEST)
    plan = PegaFlowProvider().plan(
        manifest,
        {
            "grpc_endpoint": "http://pega.example:50055",
            "health_url": "http://pega.example:9091/health",
            "kv_connector_extra_config": {"pegaflow.mode": "read_write"},
        },
        enabled=True,
    )

    assert all(not action.mutating for action in plan.actions)
    assert [action.operation for action in plan.actions] == [
        "render_connector_config",
        "check_service",
    ]
    assert plan.generated_config["kv_transfer_config"] == {
        "kv_connector": "PegaKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "pegaflow.connector",
        "kv_connector_extra_config": {
            "pegaflow.host": "http://pega.example",
            "pegaflow.mode": "read_write",
            "pegaflow.port": 50055,
        },
    }


def test_plan_rejects_connector_profile_conflation() -> None:
    manifest = load_manifest(MANIFEST)

    with pytest.raises(ValueError, match="separate lifecycle profiles"):
        PegaFlowProvider().plan(
            manifest,
            {"connector": "NixlConnector", "grpc_endpoint": "http://host:50055"},
            enabled=True,
        )


def test_check_reports_reachable_healthy_service(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_manifest(MANIFEST)
    monkeypatch.setattr(provider_module, "_installed_version", lambda: "0.23.3")

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(provider_module, "urlopen", lambda *_args, **_kwargs: Response())
    check = PegaFlowProvider().check(
        manifest,
        {
            "grpc_endpoint": "http://127.0.0.1:50055",
            "health_url": "http://127.0.0.1:9091/health",
        },
    )

    assert check.compatible is True
    assert check.configured is True
    assert check.reachable is True
    assert check.healthy is True


def test_render_is_pure_json() -> None:
    artifact = PegaFlowProvider().render(
        SimpleNamespace(generated_config={"kv_transfer_config": {}})
    )[0]
    assert json.loads(artifact.content) == {"kv_transfer_config": {}}
