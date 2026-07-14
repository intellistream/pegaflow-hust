"""CANN IPC C extension bindings.

Provides fast CPython-native wrappers for Ascend CANN IPC APIs.
When the C extension is not importable, npu_ipc_wrapper falls back
to ctypes transparently.
"""
