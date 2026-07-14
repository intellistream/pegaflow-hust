"""
Debug module for PegaFlow Ascend E2E Save/Load diagnostics.

Usage:
    PEGAFLOW_DEBUG_SAVE_PATH=1 python -m vllm.entrypoints.openai.api_server ...

Search for `[PegaKVConnector.DEBUG]` in both vLLM and pegaflow-server logs.
"""

import os


def debug_save_enabled() -> bool:
    """Check if save/load debug tracing is enabled via environment variable."""
    return os.environ.get("PEGAFLOW_DEBUG_SAVE_PATH", "0") == "1"