"""Lazy compatibility facade for PegaFlow vLLM connector implementations.

Keeping this package initializer lightweight lets the API-plane telemetry
provider load without importing Torch or worker implementation modules.
"""

from __future__ import annotations

from typing import Any

_FACADE_EXPORTS = {
    "KVConnectorRole",
    "NoopKVConnector",
    "PegaKVConnector",
    "_map_device",
    "_resolve_device_id",
}


def __getattr__(name: str) -> Any:
    if name not in _FACADE_EXPORTS:
        raise AttributeError(name)
    from pegaflow.connector import facade

    return getattr(facade, name)


__all__ = sorted(_FACADE_EXPORTS)
