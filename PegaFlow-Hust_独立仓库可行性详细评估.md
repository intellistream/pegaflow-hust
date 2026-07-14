# PegaFlow-Hust 独立仓库可行性详细评估

## 1. 仓库结构与架构总览

### 1.1 目标仓库结构

```
pegaflow-hust/                              # Fork from novitalabs/pegaflow
│
├── python/                                 # Python package → PyPI: pegaflow-llm-npu
│   ├── pegaflow/
│   │   ├── connector/                      # vLLM connector (KVConnectorBase_V1)
│   │   │   ├── __init__.py                 # 修改: NPU 设备检测 (CudaIPC→NpuIPC)
│   │   │   ├── worker.py                  # 修改: CudaIPCWrapper → NpuIPCWrapper
│   │   │   ├── scheduler.py               # 几乎不变 (设备无关)
│   │   │   ├── common.py                  # 修改: 新增 Ascend transfer 选项
│   │   │   ├── connector_metrics.py        # 不变
│   │   │   └── state_manager.py            # 不变
│   │   ├── npu_ipc_wrapper.py             # 新增: CANN IPC wrapper
│   │   ├── npu_ipc_bindings/               # 新增: CANN IPC C Extension
│   │   │   ├── _npu_ipc.c                  # Python C 扩展源码
│   │   │   └── setup.py                   # C 扩展构建
│   │   └── pegaflow.pyi                   # 类型 stub (更新)
│   ├── pyproject.toml                      # 包名: pegaflow-llm-npu
│   └── Cargo.toml                          # features: ascend
│
├── pegaflow-core/                          # Rust 核心引擎
│   ├── src/
│   │   ├── device/                         # 新增: 设备抽象层 ← 最关键
│   │   │   ├── mod.rs                      # DeviceContext, DeviceStream enum
│   │   │   ├── cuda.rs                     # CUDA 实现 (原 gpu_worker 逻辑迁入)
│   │   │   └── ascend.rs                  # Ascend 实现 (CANN FFI)
│   │   ├── transfer/
│   │   │   ├── mod.rs                      # 修改: TransferBackend trait
│   │   │   ├── memcpy.rs                   # 修改: 适配 DeviceStream
│   │   │   ├── ascend_memcpy.rs           # 新增: Ascend aclrtMemcpy 后端
│   │   │   └── kernel.rs                   # 不变 (CUDA)
│   │   ├── gpu_worker.rs                   # 重改: Cuda→Device
│   │   ├── instance.rs                     # 重改: Arc<CudaContext>→DeviceContext
│   │   ├── pinned_mem.rs                   # 修改: + Ascend 分配路径
│   │   └── lib.rs                          # 修改: 错误类型泛化
│   └── Cargo.toml                          # features: ascend
│
├── pegaflow-server/                        # gRPC + HTTP 服务器
│   ├── src/
│   │   ├── lib.rs                          # 重改: init_cuda → init_device
│   │   ├── registry.rs                     # 少量修改: torch.cuda → torch.npu
│   │   ├── service.rs                      # 注释修改 (CUDA→NPU)
│   │   ├── http_server.rs                  # 注释修改
│   │   └── check_ascend_version.rs         # 替换 check_cuda_version.rs
│   └── Cargo.toml
│
├── pegaflow-transfer/                      # RDMA 传输 (可选, P2P)
│   ├── src/
│   │   ├── lib.rs                          # feature-gate cuda 模块
│   │   ├── cuda_lib/device.rs             # 新增 Ascend 变体
│   │   ├── v2/mr.rs                        # 新增 Ascend MR 注册
│   │   └── rdma_topo.rs                   # nvidia-smi → npu-smi
│   └── Cargo.toml
│
├── pegaflow-common/                        # 公共工具库
│   ├── src/
│   │   └── numa.rs                         # nvidia-smi → npu-smi
│   └── Cargo.toml
│
├── pegaflow-proto/                         # protobuf 定义 (不变)
├── pegaflow-metaserver/                    # 跨节点注册 (不变)
├── pegaflow-pd-wire/                       # P/D 协议 (不变)
├── Cargo.toml                              # workspace features
└── docs/
    └── ascend-feasibility-analysis.md      # 本文档
```

### 1.2 核心数据流 (Ascend 版)

```
┌────────────────────────────────────────────────────────┐
│ vLLM Ascend Worker (camem_allocator)                   │
│                                                        │
│  KV Cache Tensor (aclrtMallocPhysical)                 │
│       ↓                                                │
│  NpuIPCWrapper.__init__(tensor)                       │
│       ↓ aclrtIpcMemGetExportKey(ptr, size) → key       │
│  pickle.dumps(wrapper) → 序列化的 IPC key + metadata   │
│       ↓ [gRPC: register_context_batch]                 │
│                                                        │
├────────────────────────────────────────────────────────┤
│ SchedulerConnector (设备无关)                           │
│   query_prefetch() → 缓存命中检测 + 租约               │
│   build_connector_meta() → LoadIntent / SaveIntent     │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ PegaFlow-Server (Rust, 独立进程)                        │
│                                                        │
│ grpc register_context_batch:                          │
│   pickle.loads(wrapper_bytes) → NpuIPCWrapper          │
│   wrapper.to_tensor() → aclrtIpcMemImportByKey(key)    │
│   tensor.data_ptr() → NPU 虚拟地址 → 传入 Rust 引擎     │
│                                                        │
│ PegaEngine:                                            │
│   Instance { device_ctx: DeviceContext::Ascend }       │
│   GpuWorkerPool { stream: DeviceStream::Ascend }       │
│   TransferBackend: AscendMemcpyBackend                 │
│       → aclrtMemcpyAsync(H2D / D2H)                   │
│   PinnedMemory: aclrtMallocHost → aclrtFreeHost       │
└────────────────────────────────────────────────────────┘
```

### 1.3 架构边界: 设备信息如何在 Python ↔ Rust 间流动

```
[Python: vLLM Worker]              [Rust: PegaFlow Server (via PyO3)]
                                    
① Worker 分配 NPU tensor  ──→ ② NpuIPCWrapper(tensor)
   (camem_allocator)              pickle.dumps(wrapper) = bytes
                                    
③ gRPC: register_context_batch ──→ ④ registry.rs: pickle.loads(bytes)
   (wrapper_bytes)                     → NpuIPCWrapper.to_tensor()
                                       → tensor.data_ptr() = u64
                                       → tensor.untyped_storage().nbytes() = usize
                                       → tensor.device.index = i32
                                   ⑤ 传入 pegaflow-core:
                                      register_context_layer_batch(
                                          device_id: i32,
                                          data_ptrs: &[u64],    ← NPU 虚拟地址
                                          size_bytes: &[usize],
                                          ...
                                      )
                                   ⑥ GpuContext::new(DeviceContext::Ascend { device_id })
                                      → 存储 device_id, kv_caches, worker_pool
                                   
⑦ gRPC: save(instance_id, ...) ──→ ⑧ PegaEngine.save()
                                      → 根据 slot 查找 data_ptr
                                      → build CopyDesc { device: data_ptr, host: ..., size }
                                      → AscendMemcpyBackend::d2h(copies, stream)
                                      → aclrtMemcpyAsync(D2H) 到 pinned memory
```

**关键洞察**: 不同于 CUDA 版本需要 `CudaContext` 保持存活来维持 IPC 映射，**Ascend 版本不需要** — `aclrtIpcMemImportByKey` 返回的指针在整个进程生命周期内有效，只需 `aclrtIpcMemClose(key)` 在实例注销时调用。这意味着 `GpuContext._cuda_ctx: Arc<CudaContext>` 的 Ascend 对应物可以是一个轻量的 `DeviceHandle`，甚至可以是 `()`。

---

## 2. 完整变更文件清单 (35+ 项)

### 2.1 新增文件 (7 个)

| # | 文件 | 行数 | 说明 |
|---|------|------|------|
| N1 | `python/pegaflow/npu_ipc_wrapper.py` | ~150 | CANN IPC wrapper，对应 CudaIPCWrapper |
| N2 | `python/pegaflow/npu_ipc_bindings/_npu_ipc.c` | ~200 | CANN IPC C 扩展 |
| N3 | `python/pegaflow/npu_ipc_bindings/setup.py` | ~30 | C 扩展构建 |
| N4 | `pegaflow-core/src/device/mod.rs` | ~50 | DeviceContext/DeviceStream enum |
| N5 | `pegaflow-core/src/device/cuda.rs` | ~100 | CUDA 实现 (从 gpu_worker 迁出) |
| N6 | `pegaflow-core/src/device/ascend.rs` | ~120 | Ascend 实现 (CANN FFI) |
| N7 | `pegaflow-core/src/transfer/ascend_memcpy.rs` | ~80 | Ascend memcpy 后端 |

### 2.2 重大修改文件 (10 个)

| # | 文件 | 改动行 | 说明 |
|---|------|--------|------|
| M1 | `python/pegaflow/connector/__init__.py` | ~30 | L88 `torch.cuda`→`torch.npu`, L390-412 `_resolve_device_id()`, 注释 |
| M2 | `python/pegaflow/connector/worker.py` | ~25 | L22 `CudaIPCWrapper`→`NpuIPCWrapper`, L738 `synchronize()`, L291 |
| M3 | `pegaflow-core/src/transfer/mod.rs` | ~20 | `TransferBackend` trait 参数 `Arc<CudaStream>`→`&DeviceStream` |
| M4 | `pegaflow-core/src/transfer/memcpy.rs` | ~10 | 适配 trait 签名变更 |
| M5 | `pegaflow-core/src/gpu_worker.rs` | ~60 | `CudaContext`→`DeviceContext`, `CudaStream`→`DeviceStream` |
| M6 | `pegaflow-core/src/instance.rs` | ~30 | `Arc<CudaContext>`→`DeviceContext`, error message |
| M7 | `pegaflow-core/src/pinned_mem.rs` | ~80 | 新增 `allocate_ascend_host()`, `aclrtMallocHost` 路径 |
| M8 | `pegaflow-server/src/lib.rs` | ~80 | `init_cuda_driver`→`init_device`, `detect_cuda_devices`→`detect_npu_devices`, `init_python_cuda`→`init_python_npu` |
| M9 | `pegaflow-server/src/registry.rs` | ~10 | `torch.cuda.init()`→`torch.npu.init()`, `cuda-registry`→`ascend-registry`, `torch.cuda.empty_cache()`→`torch.npu.empty_cache()` |
| M10 | `pegaflow-common/src/numa.rs` | ~40 | `nvidia-smi`→`npu-smi` |

### 2.3 小幅修改文件 (12 个)

| # | 文件 | 改动行 | 说明 |
|---|------|--------|------|
| S1 | `python/pegaflow/connector/common.py` | ~10 | L252 transfer_backend 选项，注释 |
| S2 | `python/src/lib.rs` | ~10 | 文档注释 CUDA→NPU |
| S3 | `python/pegaflow/pegaflow.pyi` | ~20 | 类型 stub 更新 |
| S4 | `python/Cargo.toml` | ~10 | ascend feature |
| S5 | `python/pyproject.toml` | ~5 | 包名 pegaflow-llm-npu |
| S6 | `pegaflow-core/src/lib.rs` | ~10 | `CudaInit`→`DeviceInit` error |
| S7 | `pegaflow-core/Cargo.toml` | ~10 | ascend feature |
| S8 | `pegaflow-server/Cargo.toml` | ~10 | ascend feature |
| S9 | `pegaflow-server/src/check_cuda_version.rs` | ~30 | 替换为 check_ascend_version.rs |
| S10 | `pegaflow-transfer/src/lib.rs` | ~5 | feature-gate cuda module |
| S11 | `pegaflow-transfer/src/cuda_lib/device.rs` | ~10 | + Ascend device 变体 |
| S12 | `pegaflow-transfer/src/v2/mr.rs` | ~20 | Ascend MR 注册 arm |

### 2.4 仅注释修改文件 (4 个)

| # | 文件 | 说明 |
|---|------|------|
| C1 | `pegaflow-server/src/service.rs` | 日志/注释 CUDA→NPU |
| C2 | `pegaflow-server/src/http_server.rs` | 日志/注释 CUDA tensor → NPU tensor |
| C3 | `pegaflow-transfer/src/rdma_topo.rs` | 注释 |
| C4 | `pegaflow-transfer/Cargo.toml` | feature description |

---

## 3. 逐模块工作量分解

### 模块 A: CANN IPC POC + C 扩展 (关键路径, 3 天)

| 任务 | 文件 | 说明 | 小时 |
|------|------|------|------|
| A1 | POC 脚本 | 两进程验证 aclrtIpcMemGetExportKey/ImportByKey 零拷贝 | 4 |
| A2 | POC 脚本 | 验证 camem_allocator 分配的 tensor 可 IPC 导出 | 2 |
| A3 | `npu_ipc_bindings/_npu_ipc.c` | 编写 C 扩展: export_key, import_key, close | 8 |
| A4 | `npu_ipc_bindings/setup.py` | 构建配置, 编译验证 | 2 |
| A5 | `npu_ipc_wrapper.py` | NpuIPCWrapper 类实现 (pickle 兼容) | 6 |
| A6 | 测试 | 单元测试: serialize → deserialize → to_tensor → data_ptr | 2 |
| | | **合计** | **24h (3d)** |

### 模块 B: Python Connector 适配 (1.5 天)

| 任务 | 文件 | 说明 | 小时 |
|------|------|------|------|
| B1 | `connector/__init__.py` | L88 npu.is_available, L390-412 ASCEND_VISIBLE_DEVICES, import NpuIPCWrapper | 3 |
| B2 | `connector/worker.py` | L22 import, L291 NpuIPCWrapper, L738 torch.npu.synchronize | 3 |
| B3 | `connector/common.py` | L252 新增 ascend_direct 选项 | 1 |
| B4 | `python/src/lib.rs` | doc comments | 1 |
| B5 | `pegaflow.pyi` | type stubs 更新 | 1 |
| B6 | `pyproject.toml` + `Cargo.toml` | 包名 pegaflow-llm-npu, ascend feature | 2 |
| B7 | 冒烟测试 | import pegaflow.connector, PegaKVConnector 构造 | 1 |
| | | **合计** | **12h (1.5d)** |

### 模块 C: Rust 引擎设备抽象 (关键路径, 7 天)

| 任务 | 文件 | 说明 | 小时 |
|------|------|------|------|
| C1 | `device/mod.rs` | 定义 DeviceContext, DeviceStream enum + trait 方法 | 4 |
| C2 | `device/cuda.rs` | 从 gpu_worker 迁出 CUDA 初始化逻辑到 device::Cuda | 6 |
| C3 | `device/ascend.rs` | AscendContext, AscendStream: aclrtSetDevice/create_stream/synchronize, 用 `extern "C"` FFI 调用 libascendcl.so | 12 |
| C4 | `transfer/mod.rs` | TransferBackend trait: CudaStream→DeviceStream, CopyDesc 不变 | 2 |
| C5 | `transfer/memcpy.rs` | 适配 trait 变更; CUDA 后端逻辑不变 | 2 |
| C6 | `transfer/ascend_memcpy.rs` | 新增: aclrtMemcpyAsync H2D/D2H, 复用 merge() | 6 |
| C7 | `gpu_worker.rs` | WorkerRuntime{stream: DeviceStream}, init_worker: 匹配 DeviceContext 创建 DeviceStream + backend | 6 |
| C8 | `instance.rs` | _cuda_ctx→_device_ctx, GpuContext::new 增加 Ascend arm | 4 |
| C9 | `lib.rs` | CudaInit→DeviceInit, nvidia-smi message 通用化 | 2 |
| C10 | Cargo.toml (core) | ascend feature, cudarc optional | 2 |
| C11 | 编译 + 单元测试 | `cargo build --no-default-features --features ascend`, 在 Ascend 环境运行 transfer 单元测试 | 10 |
| | | **合计** | **56h (7d)** |

### 模块 D: Pinned Memory Ascend 适配 (1.5 天)

| 任务 | 文件 | 说明 | 小时 |
|------|------|------|------|
| D1 | `pinned_mem.rs` | 新增 allocate_ascend_host(): aclrtMallocHost; error variants | 6 |
| D2 | `pinned_mem.rs` | 新增 allocate_ascend_hugepages(): aclrtMemRegister (如需要) | 4 |
| D3 | 验证 | 与已有 CUDA 钉内存路径对齐测试 | 2 |
| | | **合计** | **12h (1.5d)** |

### 模块 E: 服务端适配 (2 天)

| 任务 | 文件 | 说明 | 小时 |
|------|------|------|------|
| E1 | `registry.rs` | torch.cuda→torch.npu; thread name; empty_cache | 2 |
| E2 | `check_cuda_version.rs`→`check_ascend_version.rs` | 调用 aclrtGetVersion 检查 CANN 版本 | 4 |
| E3 | `lib.rs` | init_cuda_driver→init_device: 调用 aclrtSetDevice 替代 cuInit | 4 |
| E4 | `lib.rs` | detect_cuda_devices→detect_npu_devices: torch.npu.device_count | 2 |
| E5 | `lib.rs` | init_python_cuda→init_python_npu: torch.npu.init + set_device | 2 |
| E6 | `service.rs`, `http_server.rs` | 日志注释 CUDA→NPU | 1 |
| E7 | Cargo.toml (server) | ascend feature | 1 |
| | | **合计** | **16h (2d)** |

### 模块 F: RDMA/拓扑/公共库适配 (2 天)

| 任务 | 文件 | 说明 | 小时 |
|------|------|------|------|
| F1 | `pegaflow-transfer/src/lib.rs` | feature-gate cuda_lib/cuda_sys/cudart_sys | 2 |
| F2 | `pegaflow-transfer/src/cuda_lib/device.rs` | + Ascend(AscendDeviceId) | 1 |
| F3 | `pegaflow-transfer/src/v2/mr.rs` | Ascend MR 注册 arm | 4 |
| F4 | `pegaflow-transfer/src/rdma_topo.rs` | nvidia-smi→npu-smi | 4 |
| F5 | `pegaflow-common/src/numa.rs` | nvidia-smi→npu-smi | 4 |
| F6 | Cargo.toml (transfer, common) | ascend feature | 1 |
| | | **合计** | **16h (2d)** |

### 模块 G: 集成测试 + 修复 (5 天)

| 任务 | 说明 | 小时 |
|------|------|------|
| G1 | 端到端连接测试 | pegaflow-server 启动 + vllm-hust register_context | 6 |
| G2 | Save/Load 功能测试 | 单 request KV save → load → 数据一致性验证 | 8 |
| G3 | 缓存命中测试 | 多 request 共享 prefix → 验证 hit rate + TTFT | 8 |
| G4 | 稳定性测试 | 长时间运行 + 多次注册/注销 | 4 |
| G5 | Bug 修复 | 根据测试结果修复各模块 | 14 |
| | | **合计** | **40h (5d)** |

### 总工作量汇总

| 模块 | 人天 | 关键路径 |
|------|------|----------|
| A: CANN IPC POC + C 扩展 | 3 | ✅ 关键路径 |
| B: Python Connector 适配 | 1.5 | 可并行 |
| C: Rust 引擎设备抽象 | 7 | ✅ 关键路径 |
| D: Pinned Memory | 1.5 | 可并行 |
| E: 服务端适配 | 2 | ✅ 关键路径 |
| F: RDMA/拓扑 (可选) | 2 | 可并行 |
| G: 集成测试 + 修复 | 5 | ✅ 关键路径 |
| **总计 (关键路径)** | **17** | A→C→E→G |
| **总计 (全部)** | **22** | |

---

## 4. 四周冲刺计划 (6/29 → 7/28, 8 月前完成)

### 关键路径: A (3d) → C (7d) → E (2d) → G (5d) = 17 天

### 并行路径: B (1.5d) 可在 A 完成后立即开始; D (1.5d) 可在 C 开始后并行; F (2d) 可在 C 期间并行。

```
第 1 周 (6/29-7/3):  模块 A (CANN IPC POC + C 扩展)   ── 3 天
                     模块 B (Python Connector 适配)     ── 并行, 1.5 天
                     ═══ 第 1 周结束: IPC 可用, connector 可构造 ═══

第 2 周 (7/6-7/10):  模块 C (Rust 引擎设备抽象)        ── 前 4 天
                     模块 D (Pinned Memory)             ── 并行
                     ═══ 第 2 周结束: device/mod.rs 完成 ═══

第 3 周 (7/13-7/17): 模块 C 继续                        ── 后 3 天
                     模块 E (服务端适配)                ── 并行, 2 天
                     模块 F (RDMA/拓扑)                 ── 并行, 2 天
                     ═══ 第 3 周结束: pegaflow-server 可启动 ═══

第 4 周 (7/20-7/24): 模块 G (集成测试 + 修复)           ── 5 天
                     (缓冲: 7/27-7/28 收尾)
                     ═══ 7/28 前完成: E2E 通道打通 ═══
```

### 里程碑

| 日期 | 里程碑 | 验证标准 |
|------|--------|----------|
| 7/1 | POC 通过 | 两进程 CANN IPC 零拷贝验证 |
| 7/3 | IPC 绑定就绪 | NpuIPCWrapper 可 pickle + to_tensor |
| 7/7 | device 抽象完成 | DeviceContext::Ascend 编译通过 |
| 7/10 | transfer 后端完成 | AscendMemcpyBackend::d2h/h2d 编译通过 |
| 7/14 | 引擎编译通过 | `cargo build --features ascend` 成功 |
| 7/17 | pegaflow-server 启动 | 无 CUDA 依赖的二进制文件在 Ascend 环境启动 |
| 7/21 | register_context 通过 | vLLM worker → pegaflow-server 注册 KV cache |
| 7/24 | save + load 通过 | 单 request KV D2H → H2D 数据一致性 |
| 7/28 | 缓存命中通过 | 多 request prefix 共享 → hit + TTFT 下降 |

---

## 5. 完整性检查: 可行性分析漏洞覆盖

以下逐条验证每个已识别的 CUDA 引用在任务规划中有对应的修改项。

| # | 文件:位置 | CUDA 引用 | 覆盖任务 |
|---|----------|-----------|----------|
| 1 | `worker.py:22` | `from pegaflow.ipc_wrapper import CudaIPCWrapper` | B2 |
| 2 | `worker.py:239` | "CUDA device id is unknown" 错误消息 | B2 |
| 3 | `worker.py:291` | `wrapper = CudaIPCWrapper(kv_cache)` | B2 |
| 4 | `worker.py:643` | "CUDA graph replay" 注释 | C1 (comment) |
| 5 | `worker.py:738` | `torch.cuda.synchronize(self._torch_device)` | B2 |
| 6 | `__init__.py:88` | `if torch.cuda.is_available()` | B1 |
| 7 | `__init__.py:155` | "CUDA IPC mappings" 注释 | C1 |
| 8 | `__init__.py:209` | "CUDA graph replay" 注释 | C1 |
| 9 | `__init__.py:390-412` | `_resolve_device_id()` 解析 CUDA_VISIBLE_DEVICES | B1 |
| 10 | `npu_ipc_wrapper.py` (新) | 替代 `ipc_wrapper.py` 的全部 CUDA IPC 调用 | A5 |
| 11 | `npu_ipc_bindings/` (新) | 替代 PyTorch 内置的 `_share_cuda_/_new_shared_cuda_` | A3 |
| 12 | `pegaflow-core/src/transfer/mod.rs` | `use cudarc::driver::CudaStream` | C4 |
| 13 | `transfer/mod.rs:51-57` | `TransferBackend` trait 参数 `Arc<CudaStream>` | C4 |
| 14 | `transfer/memcpy.rs:10` | `use cudarc::driver::{CudaStream, sys}` | C5 |
| 15 | `transfer/memcpy.rs:65` | `sys::cuMemcpyHtoDAsync_v2()` | C5 (不变, CUDA arm) |
| 16 | `transfer/memcpy.rs:83` | `sys::cuMemcpyDtoHAsync_v2()` | C5 (不变, CUDA arm) |
| 17 | `transfer/kernel.rs:14-16` | `CudaContext, CudaFunction, CudaSlice, CudaStream` | 不变 (CUDA-only) |
| 18 | `gpu_worker.rs:3` | `use cudarc::driver::{CudaContext, CudaStream}` | C7 |
| 19 | `gpu_worker.rs:245` | `stream: Arc<CudaStream>` | C7 |
| 20 | `gpu_worker.rs:262-276` | `build_backend(mode, ctx: &Arc<CudaContext>)` | C7 |
| 21 | `gpu_worker.rs:278-298` | `CudaContext::new(device_id)`, `ctx.new_stream()` | C7 |
| 22 | `instance.rs:32` | `use cudarc::driver::CudaContext` | C8 |
| 23 | `instance.rs:230` | `_cuda_ctx: Arc<CudaContext>` | C8 |
| 24 | `instance.rs:263` | `Returns EngineError::CudaInit` doc | C8 |
| 25 | `instance.rs:574-575` | `CudaContext::new(device_id)` | C8 |
| 26 | `pinned_mem.rs:34` | `use cudarc::runtime::sys as rt` | D1 |
| 27 | `pinned_mem.rs:68` | `CudaAllocFailed` error variant | D1 |
| 28 | `pinned_mem.rs:156-182` | `rt::cudaHostAlloc()` | D1 |
| 29 | `pinned_mem.rs:245` | `rt::cudaHostRegister()` | D2 |
| 30 | `pinned_mem.rs:291-301` | `rt::cudaHostGetDevicePointer()` | D1 |
| 31 | `core/lib.rs:78` | `CudaInit(String)` 错误变体 | C9 |
| 32 | `core/lib.rs:95` | `"failed to initialize CUDA"` 消息 | C9 |
| 33 | `core/lib.rs:396` | `"nvidia-smi is available"` 消息 | C9 |
| 34 | `core/Cargo.toml:11-12` | `cudarc/cuda-12080`, `cudarc/cuda-13000` | C10 |
| 35 | `server/lib.rs:1` | `mod check_cuda_version` | E2 |
| 36 | `server/lib.rs:21` | `use cudarc::driver::result as cuda_driver` | E3 |
| 37 | `server/lib.rs:49` | `"CUDA IPC registry"` 注释 | C1 |
| 38 | `server/lib.rs:56` | `"CUDA devices to initialize"` 注释 | C1 |
| 39 | `server/lib.rs:292-294` | `fn init_cuda_driver()` | E3 |
| 40 | `server/lib.rs:297-320` | `fn detect_cuda_devices()` | E4 |
| 41 | `server/lib.rs:322-368` | `fn init_python_cuda()` | E5 |
| 42 | `server/registry.rs:50-58` | `CudaTensorRegistry::new()` → `torch.cuda.init()` | E1 |
| 43 | `server/registry.rs:166` | `torch.cuda.empty_cache()` | E1 |
| 44 | `server/registry.rs:255` | `"cuda-registry"` 线程名 | E1 |
| 45 | `server/check_cuda_version.rs:1` | `use cudarc::runtime::*` | E2 (替换) |
| 46 | `server/http_server.rs:71-118` | "CUDA IPC tensors" "torch.cuda.empty_cache()" 注释 | C1 |
| 47 | `transfer/src/lib.rs:6-8` | `mod cuda_lib; mod cuda_sys; mod cudart_sys` | F1 |
| 48 | `transfer/cuda_lib/device.rs:6-10` | `Device::Cuda(CudaDeviceId)` | F2 |
| 49 | `transfer/v2/mr.rs:4-5` | `cudaPointerGetAttributes`, `CudaDeviceId` | F3 |
| 50 | `transfer/rdma_topo.rs:212-215` | `nvidia-smi` command | F4 |
| 51 | `common/numa.rs:263-308` | `nvidia-smi` → NUMA affinity | F5 |
| 52 | `python/src/lib.rs:262,340,387,543` | doc comments: "CUDA device", "CUDA tensors" | B4 |

**总计 52 个 CUDA 引用点，全部已在任务规划中覆盖。零遗漏。**

---

## 6. 风险登记表

| # | 风险 | 概率 | 影响 | 缓解措施 |
|---|------|------|------|----------|
| R1 | CANN IPC 跨进程内存访问不稳定 | 中 | 高 | 第 1 周 POC 即验证；准备 D2H→CPU→H2D 回退方案 |
| R2 | `camem_allocator` tensor 不可 IPC 导出 | 低 | 高 | 第 1 周 POC 即验证；备选: 新增专用 allocator |
| R3 | `aclrtMemcpyAsync` 性能不达预期 (vs CUDA DMA) | 中 | 中 | 与 `swap_blocks_batch` 做对比 benchmark |
| R4 | `torch.npu.Stream/Event` 语义与 CUDA 不一致 | 中 | 中 | 第 1 周包含流同步测试 |
| R5 | CANN < 8.5 版本缺少 IPC API | 低 | 中 | 文档标记最低 CANN 8.5.0 |
| R6 | `npu-smi` 拓扑输出格式不可解析 | 高 | 低 | 独立 parser，不假设格式兼容；可选 fallback `/sys/class/davinci*` |
| R7 | 没有 GPU 机器可调试 (无 Ascend 环境) | **高** | **极高** | **必须**在阶段 0 前确定 Ascend 硬件 + CANN 8.5 环境可用 |
| R8 | PegaFlow 上游 breaking change (Fork 范围) | 中 | 中 | Fork 基准版本锁定；bugfix 选择性 cherry-pick |

---

## 7. 前置条件 (启动前必须确认)

1. **Ascend 硬件环境就绪** — 至少 1 块 NPU (Ascend 910B/910C) + CANN 8.5.0+
2. **容器内 Python 栈可用** — torch_npu 2.9.0+, vllm-hust 可运行
3. **camem_allocator 启用** — vllm-ascend-hust 的 `COMPILE_CUSTOM_KERNELS` 和 camem_allocator 正常工作
4. **实例间可达** — 两个进程可以同时访问同一个 NPU 设备
5. **编译工具链** — aarch64 GCC, protobuf-compiler, Rust 工具链

---

## 8. 附录: 关键 API 对照

### IPC 内存共享

| 操作 | CUDA API | CANN API |
|------|----------|----------|
| 导出 | `cuIpcGetMemHandle` | `aclrtIpcMemGetExportKey` |
| 导入 | `cuIpcOpenMemHandle` | `aclrtIpcMemImportByKey` |
| 关闭 | `cudaIpcCloseMemHandle` | `aclrtIpcMemClose` |
| IPC 事件 | `cuIpcGetEventHandle` | `aclrtIpcGetEventHandle` |

### 内存传输

| 操作 | CUDA API | CANN API |
|------|----------|----------|
| H2D | `cuMemcpyHtoDAsync_v2` | `aclrtMemcpyAsync(HOST_TO_DEVICE)` |
| D2H | `cuMemcpyDtoHAsync_v2` | `aclrtMemcpyAsync(DEVICE_TO_HOST)` |

### 钉内存

| 操作 | CUDA API | CANN API |
|------|----------|----------|
| 分配 | `cudaHostAlloc` | `aclrtMallocHost` |
| 释放 | `cudaFreeHost` | `aclrtFreeHost` |
| 设备指针 | `cudaHostGetDevicePointer` | `aclrtMallocHost` 返回值显式包含 dev ptr |

### 设备管理

| 操作 | CUDA API | CANN API |
|------|----------|----------|
| 驱动初始化 | `cuInit` | `aclInit` |
| 设置设备 | `cudaSetDevice` | `aclrtSetDevice` |
| 创建流 | `cudaStreamCreate` | `aclrtCreateStream` |
| 同步 | `cudaStreamSynchronize` | `aclrtSynchronizeStream` |
| 记录事件 | `cudaEventRecord` | `aclrtRecordEvent` |
| 同步事件 | `cudaEventSynchronize` | `aclrtSynchronizeEvent` |
| 版本 | `cudaRuntimeGetVersion` | `aclrtGetVersion` |
