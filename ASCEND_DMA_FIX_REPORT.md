# Ascend DMA 问题修复报告

## 已修复

### 1. stream.synchronize() 507001 中毒修复

**文件**: `pegaflow-core/src/device/ascend.rs`

`aclrtMemcpyAsync` 失败后 sync fallback 用 `aclrtMemcpy` 完成拷贝，但后续的 `stream.synchronize()` 对中毒 stream 再次触发 507001。去掉 fallback 后的 sync 调用——同步拷贝返回时数据已完成。

### 2. stream sync 从 error 降级为 warning

**文件**: `pegaflow-core/src/gpu_worker.rs`

`process_load_task` / `process_save_task` 中的 `stream.synchronize()` 失败不再返回 error，改为 warning。数据已通过 sync fallback 完成拷贝。

### 3. Server crash 修复 (torch destructor)

**文件**: `pegaflow-server/src/registry.rs`

`Py<PyAny>` 用 `ManuallyDrop` 包装，防止 torch tensor `tp_dealloc` → `npuSynchronizeDevice` → `c10::Error` C++ 异常 → `std::terminate`。同时去掉 `gc.collect()` 和 `torch.*.empty_cache()` 避免触发相同 crash。

### 4. Save worker 跨 instance 存活

**文件**: `pegaflow-core/src/lib.rs`, `pegaflow-core/src/instance.rs`

`GpuWorkerPool` 从 per-instance 提升为 per-device 全局（存储在 `PegaEngine::gpu_pools`），不随 vLLM instance unregister 而销毁。vLLM 断连后 save worker 存活，继续处理 cooldown 期间的 pending saves。

### 5. Save RPC fire-and-forget

**文件**: `pegaflow-server/src/service.rs`

Save handler 改为 `tokio::spawn` 独立任务，不随 gRPC 连接断开（vLLM SIGKILL）而取消。确保 save 操作完成并插入 cache。

### 6. Warmup pre-kill 延迟

**文件**: `scripts/bench_multi_pegaflow.sh`

Warmup 后加 30s 延迟再 kill vLLM，给 connector 时间提交 pending save RPC。

### 7. 测试工具

**文件**: `pegaflow-core/src/bin/dma_stress.rs`, `pegaflow-core/src/device/ascend_batch.c`

- `dma_stress`: Ascend DMA 压力测试 binary，支持逐个和 batch API 对比。
- `ascend_batch.c`: `aclrtMemcpyBatchAsync` C wrapper，用 CANN header 编译，避免 Rust FFI struct 布局问题。

## 测试结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Save 可靠性 | 0-1 batch | **39 blocks / 337 MB** |
| Cache hit rate | 0% | **100%** (640 hits, 0 misses) |
| H2D load | 0 B | **4.7 GB** |
| 跨实例 token 共享 | 0 | **8192 tokens/instance** |
| Save RPC 成功率 | 部分 | **47/47 OK** |
| Server crash | 有 | **无** |

## 遗留问题

### Load worker 的 stream sync 阻塞

**现象**: benchmark 阶段部分请求超时（28/60 完成），p99 68s，wall-clock 573s。

**原因**: Load worker 调用 `backend.h2d()` → `aclrtMemcpyBatchAsync` 在 CANN 8.5.1 上返回 207000 (FEATURE_NOT_SUPPORTED)，fallback 到逐个 `aclrtMemcpyAsync` + sync `aclrtMemcpy`。sync 拷贝在 NPU 空闲时很快，但在 vLLM compute 负载下阻塞 18-32s。Load worker 串行处理任务，后续请求的 load 被前面阻塞的任务堵住。

**解决方向**:
1. 升级 CANN 到支持 `aclrtMemcpyBatchAsync` 的版本（9.x+）
2. Load worker 用独立 stream 或并行处理多个 load 任务
3. Connector 端降低 load 超时，让 vLLM 更快 fallback 到 recompute

### Batch API 在 CANN 8.5.1 不可用

**现象**: 纯 C 程序调用 `aclrtMemcpyBatchAsync` 返回 207000。CANN 9.1.0 库与当前驱动(25.3.rc1)不兼容，无法验证新版本是否支持。

**解决方向**: 升级 CANN 驱动到匹配 9.x 的版本。
