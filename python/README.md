# PegaFlow Python Package

High-performance key-value storage engine with Python bindings, built with Rust and PyO3.

## Features

- **PegaEngine**: Fast Rust-based key-value storage with Python bindings
- **PegaKVConnector**: vLLM KV connector for distributed inference with KV cache transfer

## Installation

### From Source

```bash
# CUDA 12 (upstream-compatible default)
cd python
uv run maturin develop --release

# HUST Ascend
uv run maturin develop --release --no-default-features --features ascend
```

The package metadata in this branch uses the name `pegaflow-llm-npu`. It is
not currently published to PyPI, so `pip install pegaflow` and
`pip install pegaflow-llm-npu` are not supported installation paths.

The separately maintained upstream CUDA distributions are
[`pegaflow-llm`](https://pypi.org/project/pegaflow-llm/) and
`pegaflow-llm-cu13`.

## Usage

### Basic KV Storage

```python
from pegaflow import PegaEngine

# Create a new engine
engine = PegaEngine()

# Store key-value pairs
engine.put("name", "PegaFlow")
engine.put("version", "0.1.0")

# Retrieve values
name = engine.get("name")  # Returns "PegaFlow"
missing = engine.get("nonexistent")  # Returns None

# Remove keys
removed = engine.remove("name")  # Returns "PegaFlow"
```

### vLLM KV Connector

```python
from vllm import LLM
from vllm.distributed.kv_transfer.kv_transfer_agent import KVTransferConfig

# Configure vLLM to use PegaKVConnector
kv_transfer_config = KVTransferConfig(
    kv_connector="PegaKVConnector",
    kv_role="kv_both",
    kv_connector_module_path="pegaflow.connector",
)

# Create LLM with KV transfer enabled
llm = LLM(
    model="gpt2",
    kv_transfer_config=kv_transfer_config,
)
```

#### Connector Modes

`PegaKVConnector` defaults to `read_write`: it queries PegaFlow for reusable KV
blocks, loads matched blocks into vLLM, and saves newly computed full blocks
back to PegaFlow.

Set `pegaflow.mode` to `save_only` when another vLLM connector is responsible
for reads and PegaFlow should only persist KV blocks for later reuse. This is
intended for `MultiConnector` decode-side setups where an upstream connector
owns the external hit/load path, while PegaFlow records the resulting KV cache.
In `save_only` mode, PegaFlow does not query or load KV blocks.

```bash
vllm serve Qwen/Qwen3-0.6B \
  --kv-transfer-config '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "connectors": [
        {
          "kv_connector": "<external-read-connector>",
          "kv_role": "kv_both"
        },
        {
          "kv_connector": "PegaKVConnector",
          "kv_role": "kv_both",
          "kv_connector_module_path": "pegaflow.connector",
          "kv_connector_extra_config": {
            "pegaflow.mode": "save_only"
          }
        }
      ]
    }
  }'
```

Valid values are `read_write` and `save_only`.

## Development

See the [examples](../examples/) directory for more usage examples.

## Testing

### Running Unit Tests

The test suite includes integration tests that verify the `EngineRpcClient` can correctly communicate with a running `pegaflow-server` instance.

#### Prerequisites

1. **Build the Rust extension**:

   ```bash
   cd python
   uv run maturin develop --release
   ```

2. **Build the server binary**:

   ```bash
   cd ..
   cargo build --release -p pegaflow-server --bin pegaflow-server
   ```

3. **Ensure the selected accelerator is available** (integration tests require
   hardware). For the default CUDA build:
   ```bash
   python -c "import torch; assert torch.cuda.is_available()"
   ```

#### Running Tests

```bash
cd python

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_engine_client.py -v

# Run with coverage
pytest tests/ --cov=pegaflow --cov-report=html
```

#### Test Structure

- **`tests/conftest.py`**: Contains pytest fixtures for:

  - `pega_server`: Automatically starts/stops `pegaflow-server` for integration tests
  - `engine_client`: Creates an `EngineRpcClient` connected to the test server
  - `client_context`: Provides a `ClientContext` representing a vLLM instance with GPU KV cache tensors
  - `registered_instance`: Provides a registered instance ID for query tests

- **`tests/test_engine_client.py`**: Integration tests for:
  - Server connectivity
  - Query operations with various inputs

#### Test Fixtures

The `ClientContext` class abstracts a vLLM instance and provides:

- `register_kv_caches()`: Register GPU KV cache tensors with the server
- `query(block_hashes)`: Query available blocks
- `unregister_context()`: Unregister context from server

Example test usage:

```python
def test_query(client_context):
    """Test query operation."""
    result = client_context.query([])
    assert result is not None
```

## License

MIT
