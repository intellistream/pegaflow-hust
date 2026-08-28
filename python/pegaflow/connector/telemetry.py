"""API-plane telemetry provider without scheduler or worker imports."""

from __future__ import annotations

from typing import Any

from pegaflow.connector.connector_metrics import (
    PegaKVConnectorStats,
    PegaPromMetrics,
)


class PegaKVTelemetryProvider:
    """Construct PegaFlow stats and Prometheus codecs in the API process."""

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> PegaKVConnectorStats | None:
        if data is None:
            return None
        return PegaKVConnectorStats(data=data)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: Any,
        metric_types: dict[type[Any], type[Any]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> PegaPromMetrics:
        return PegaPromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )


__all__ = ["PegaKVTelemetryProvider"]
