"""Build script for the _npu_ipc C extension.

This setup.py is used both for standalone builds (e.g., `pip install -e .` when
developing locally) and as a reference for packaging integrations.  When the
library is distributed via `pegaflow-llm`, the build is orchestrated by
`pyproject.toml` / `setup.py` at the project root.

Standalone build:
    cd python/pegaflow/npu_ipc_bindings && python setup.py build_ext --inplace
"""

from setuptools import Extension, setup

_npu_ipc = Extension(
    "npu_ipc_bindings._npu_ipc",
    sources=["_npu_ipc.c"],
    extra_compile_args=["-std=c11", "-O2"],
    extra_link_args=["-ldl"],
)

setup(
    name="npu_ipc_bindings",
    version="0.1.0",
    description="Low-level CANN IPC bindings for Ascend NPU memory sharing",
    ext_modules=[_npu_ipc],
)