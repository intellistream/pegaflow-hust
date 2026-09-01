# PegaFlow Provider for vLLM-HUST Extension Manager

This package describes PegaFlow correctly as an externally operated KV and
transfer system plus a vLLM connector. It does not turn the PegaFlow server
into an in-process vLLM plugin and does not grant the Extension Manager
authority to start, stop, upgrade, clear, or delete PegaFlow services or data.

```bash
pip install vllm-hust-ext vllm-hust-pegaflow-provider
pip install /path/to/pegaflow-llm-npu-0.23.3.whl

vllm-hust-ext extension check org.vllm-hust.pegaflow
vllm-hust-ext extension enable org.vllm-hust.pegaflow
vllm-hust-ext run -- vllm serve MODEL
```

The Manager configuration must provide:

```json
{
  "health_url": "http://127.0.0.1:9091/health",
  "grpc_endpoint": "http://127.0.0.1:50055",
  "connector": "PegaKVConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "pegaflow.mode": "read_write"
  }
}
```

The Provider performs read-only health and compatibility checks and renders
`--kv-transfer-config`. The external operator remains responsible for service
lifecycle and data retention. Set a common `PYTHONHASHSEED` explicitly on all
vLLM replicas that are expected to share prefix hashes.
