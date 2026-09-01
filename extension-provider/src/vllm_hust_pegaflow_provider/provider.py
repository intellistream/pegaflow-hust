"""Read-only PegaFlow host Provider and vLLM connector renderer."""

from __future__ import annotations

import json
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from vllm_hust_ext.manifest import BundleManifest
from vllm_hust_ext.providers.base import (
    PlanAction,
    ProviderCheck,
    ProviderPlan,
    RenderArtifact,
    assess_compatibility,
)

_DISTRIBUTION = "pegaflow-llm-npu"
_CONNECTOR = "PegaKVConnector"
_ROLES = {"kv_producer", "kv_consumer", "kv_both"}


def _installed_version() -> str | None:
    with suppress(PackageNotFoundError):
        return version(_DISTRIBUTION)
    return None


def _endpoint_parts(endpoint: object) -> tuple[str, int]:
    parsed = urlparse(str(endpoint))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
        raise ValueError("grpc_endpoint must be an http(s) URL with an explicit port")
    return f"{parsed.scheme}://{parsed.hostname}", parsed.port


class PegaFlowProvider:
    """Delegate PegaFlow lifecycle while rendering its primary vLLM connector."""

    name = "pegaflow"

    def supports(self, manifest: BundleManifest) -> bool:
        return manifest.host.provider == self.name

    def _connector_config(self, configuration: dict[str, Any]) -> dict[str, Any]:
        connector = configuration.get("connector", _CONNECTOR)
        if connector != _CONNECTOR:
            raise ValueError(
                "this profile supports PegaKVConnector only; PD and NIXL connectors "
                "require separate lifecycle profiles"
            )
        role = configuration.get("kv_role", "kv_both")
        if role not in _ROLES:
            raise ValueError(f"unsupported PegaFlow KV role: {role}")
        endpoint = configuration.get("grpc_endpoint")
        if endpoint is None:
            raise ValueError("grpc_endpoint is required")
        host, port = _endpoint_parts(endpoint)
        extra = configuration.get("kv_connector_extra_config", {})
        if not isinstance(extra, dict):
            raise ValueError("kv_connector_extra_config must be an object")
        extra = dict(extra)
        for key, value in {"pegaflow.host": host, "pegaflow.port": port}.items():
            if key in extra and extra[key] != value:
                raise ValueError(f"{key} conflicts with grpc_endpoint")
            extra[key] = value
        return {
            "kv_connector": connector,
            "kv_role": role,
            "kv_connector_module_path": "pegaflow.connector",
            "kv_connector_extra_config": extra,
        }

    def plan(
        self,
        manifest: BundleManifest,
        configuration: dict[str, Any],
        *,
        enabled: bool,
    ) -> ProviderPlan:
        connector = self._connector_config(configuration)
        actions = [
            PlanAction(
                "render_connector_config",
                "vllm",
                "vllm",
                details={"enabled": enabled},
            )
        ]
        actions.extend(
            PlanAction(
                "check_service",
                service.service_id,
                manifest.lifecycle_owner,
                details={"protocol": service.protocol},
            )
            for service in manifest.requires_services
        )
        return ProviderPlan(
            manifest.bundle_id,
            self.name,
            tuple(actions),
            {"kv_transfer_config": connector},
            (
                "PegaFlow lifecycle and stored KV data remain owned by the external operator; "
                "the Manager will not start, stop, clear, upgrade, or delete them.",
                "PD and NIXL connectors are outside this first profile.",
            ),
        )

    def render(self, plan: ProviderPlan) -> tuple[RenderArtifact, ...]:
        return (
            RenderArtifact(
                "pegaflow-vllm-connector.json",
                "application/json",
                json.dumps(plan.generated_config, indent=2, sort_keys=True),
            ),
        )

    def check(self, manifest: BundleManifest, configuration: dict[str, Any]) -> ProviderCheck:
        detected = _installed_version()
        defaults = {} if detected is None else {"pegaflow-engine": detected}
        compatible, evidence = assess_compatibility(
            manifest,
            configuration,
            detected_host_version=detected,
            default_protocol_versions=defaults,
        )
        if compatible is False:
            return ProviderCheck(False, False, evidence=evidence)
        try:
            self._connector_config(configuration)
        except ValueError as error:
            return ProviderCheck(
                compatible,
                False,
                degraded=True,
                evidence=evidence + (str(error),),
            )
        health_url = configuration.get("health_url")
        if not health_url:
            return ProviderCheck(
                compatible,
                False,
                degraded=True,
                evidence=evidence + ("health_url is required",),
            )
        try:
            request = Request(str(health_url), method="GET")
            with urlopen(request, timeout=2) as response:  # noqa: S310
                healthy = 200 <= response.status < 300
                return ProviderCheck(
                    compatible,
                    True,
                    reachable=True,
                    healthy=healthy,
                    degraded=not healthy,
                    evidence=evidence + (f"PegaFlow health endpoint returned {response.status}",),
                )
        except HTTPError as error:
            return ProviderCheck(
                compatible,
                True,
                reachable=True,
                healthy=False,
                degraded=True,
                evidence=evidence + (f"PegaFlow health endpoint returned {error.code}",),
            )
        except (OSError, URLError) as error:
            return ProviderCheck(
                compatible,
                True,
                reachable=False,
                healthy=False,
                degraded=True,
                evidence=evidence + (f"PegaFlow service is unreachable: {error}",),
            )
