# PegaFlow Ascend Save/Load 链路诊断报告

> **日期**: 2026-07-12  
> **目标**: 定位 Save/Load 链路断点，使 KV Cache 命中生效  
> **当前状态**: ❌ Save/Load 链路未打通 | External prefix cache hit rate: 0.0%

---

## 一、完整链路架构

```
┌─────────────────────────────────────────────────────────────┐
│                     SAVE PATH                                │
│                                                              │
│  vLLM Scheduler         vLLM Worker          PegaFlow Server │
│  ─────────────         ───────────          ────────────── │
│                                                              │
│  1. _consume_save_                                         │
│     intent() ──────►  3. wait_for_save() ──►  5. save()    │
│        │                    │                  gRPC handler │
│        ▼                    ▼                      │       │
│  2. build_connector_   4. _process_save_          ▼       │
│     meta() ────────►    batch() ────────►  6. batch_save_  │
│                         engine_client          kv_blocks_   │
│                         .save()               from_ipc()    │
│                                                     │       │
│                                                     ▼       │
│                                              7. GpuWorker   │
│                                                 D2H copy    │
│                                              aclrtMemcpy    │
│                                              Async(D2H)     │
│                                                     │       │
│                                                     ▼       │
│                                              8. Host pinned  │
│                                                 memory pool │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     LOAD PATH                                │
│                                                              │
│  vLLM Scheduler         vLLM Worker          PegaFlow Server │
│  ─────────────         ───────────          ────────────── │
│                                                              │
│  1. get_num_new_                                            │
│     matched_tokens() ─►  5. start_load_kv() ─►  7. load()  │
│        │                     │                  gRPC handler │
│        ▼                     ▼                      │       │
│  2. query_prefetch()   6. engine_client           ▼       │
│     gRPC ◄────────         .load() ──────►  8. batch_load_  │
│        │                                    kv_blocks_      │
│        ▼                                    multi_layer()   │
│  3. QueryReady?                                       │       │
│     ├─ Loading → retry                                ▼       │
│     └─ Ready → hit_blocks                     9. GpuWorker   │
│        │                                       H2D copy      │
│        ▼                                    aclrtMemcpy      │
│  4. update_state_                           Async(H2D)       │
│     after_alloc()                                  │       │
│     → LoadIntent                                    ▼       │
│                                               10. NPU device │
│                                                   memory    │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、断点分析（共 6 个）

### 🔴 BP1: Save Intent 无法为短 prompt 生成（高置信度）

**位置**: `scheduler.py:422-429` `_consume_save_intent()`

**根因代码**:
```python
# scheduler.py:424-425
local_saveable = min(
    len(allocated),
    scheduled // self._ctx.virtual_block_size,  # ← 关键条件
)
```

**分析**:
- `virtual_block_size = block_size × dcp_world_size × pcp_world_size = 128 × 1 × 1 = 128`
- 对于 Qwen2.5-14B（`block_size=128`），一个逻辑块 = 128 tokens
- `scheduled // 128 >= 1` 要求请求累积 ≥ 128 个 scheduled tokens
- 实测 prompt:
  - 短文本 4 tokens → `4 // 128 = 0` → **无 save intent**
  - 长文本 116 tokens → `116 // 128 = 0` → **无 save intent**
- **即使 116 tokens 的"长文本"也不到 1 个 block 的阈值**
- 需要 ≥ 128 tokens 的 prompt 或者多次 decode step 累积到 128 tokens 才会触发

**日志验证方法**:
```bash
PEGAFLOW_DEBUG_SAVE_PATH=1 python -m vllm.entrypoints.openai.api_server ...
# 观察日志:
# [PegaKVConnector.DEBUG] wait_for_save skipped: metadata=True, save_intents={}
```
如果看到 `save_intents={}` 或 `metadata=None`，说明 scheduler 端未生成 save intent。

**证据**: 技术报告中 `External prefix cache hit rate: 0.0%` 且延迟无改善（8.38s vs 8.34s），与无 save intent 一致。

**修复方向**: 
- 使用 ≥ 128 tokens 的 prompt 测试（如 200 token prompt）
- 或在 decode 阶段累积多个 step 后 save intent 应触发

---

### 🔴 BP2: `wait_for_save()` 可能未被 vLLM Ascend attention backend 调用（中置信度）

**位置**: `worker.py:663-691` `wait_for_save()`

**分析**:
- `wait_for_save()` 依赖 vLLM 框架在每个 forward pass 后调用 `connector.wait_for_save()`
- 在 vLLM 标准实现中，`KVConnector_V1` 基类定义了此接口
- 但 Ascend attention backend（`vllm-ascend` 插件）**可能没有正确调用此方法**
- 如果 `wait_for_save()` 从未被调用，即使 scheduler 生成了 save intent，也会被丢弃

**日志验证方法**:
```bash
PEGAFLOW_DEBUG_SAVE_PATH=1 python -m vllm.entrypoints.openai.api_server ...
# 如果完全没有出现 [PegaKVConnector.DEBUG] wait_for_save called 日志
# 说明 vLLM Ascend backend 未调用此方法
```

**间接证据**: `save_kv_layer()` 方法体为 `pass`（`worker.py:658-661`），说明 connector 完全依赖 `wait_for_save()` 触发保存。如果 Ascend backend 不知道调用它，save 链完全断开。

**修复方向**:
- 检查 vllm-ascend 插件代码中是否在 `model_runner` 或 `worker` 中调用了 `connector.wait_for_save()`
- 如果缺失，需要 patch vllm-ascend 或在 `save_kv_layer()` 回调中实现保存逻辑作为 fallback

---

### 🔴 BP3: `aclrtMemcpyAsync(D2H) failed: error code 507899`（高置信度）

**位置**: `pegaflow-core/src/transfer/ascend_memcpy.rs:96` `AscendMemcpyBackend`

**根因**:
- 技术报告明确记录了此错误：`aclrtMemcpyAsync(D2H) failed: error code 507899`
- 错误码 `507899` = `ACL_ERROR_STORAGE_OVER_LIMIT` 或内存分配器不支持 DMA
- **根因**：`torch_npu` 的 `expandable_segments` 分配器分配的内存不支持 DMA 传输
- 即使 BP1/BP2 都通过，到了实际 D2H 拷贝阶段也会失败

**影响范围**:
- **Save (D2H)**: NPU → CPU 拷贝全部失败 → 无 KV Cache 存入主机内存
- **Load (H2D)**: CPU → NPU 拷贝同样失败 → 即使 server 有缓存也无法恢复

**证据**: 旧日志中确认有此错误（详见技术报告 4.0 节）

**修复方向**:
- 使用 `aclrtMallocPhysical` 或 `camem_allocator` 分配的"物理连续"内存
- 短期 workaround: 验证 NPU 原生分配 (`torch.npu.empty()`) 的 buffer 是否支持 DMA
- 或者绕过 `expandable_segments`，使用 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False` 环境变量

---

### 🟡 BP4: Load 路径 - query_prefetch 返回 0 hits（连锁反应）

**位置**: `scheduler.py:495-541` `_count_available_block_prefix()`

**根因**: 由于 BP3，server 端无任何已保存的 block → `query_prefetch` 必然返回 0 hits
- 从 `service.rs:545-559` 看，`QueryReady` 的 `num_hit_blocks` 直接来自 server 缓存命中数
- 无保存 → 无缓存 → `hit_blocks = 0` → `External prefix cache hit rate: 0.0%`

**日志表现**:
```
[PegaKVConnector] req=xxx cache_lookup: hit_blocks=0 computed_blocks=0 ...
```
或：
```
[PegaKVConnector] req=xxx cache_lookup_skipped: mode=read_write
```
（`cache_lookup_skipped` 出现在 `scheduler.py:117-122`，当 `read_enabled=False` 时）

---

### 🟡 BP5: `PEGAFLOW_MODE` 可能未设为 `read_write`（低置信度）

**位置**: `common.py:24-40` `PegaConnectorMode`

**分析**:
- 默认值: `PegaConnectorMode.READ_WRITE`（`common.py:63`）
- 技术报告中提到"配置中未设置 PEGAFLOW_MODE=read_write"
- 但默认值就是 `READ_WRITE`，除非显式覆盖
- `read_enabled` 控制 Load 路径是否启用（`scheduler.py:116-122`）
- `mode` 不影响 Save 路径 — save 始终在 write 模式下启用

**验证方法**:
- 检查启动日志中是否有 `mode=read_write`
- 或 `grep cache_lookup_skipped` 看是否出现 — 出现说明 read disabled

---

### 🟡 BP6: `kv_transfer_config` 中 connector module_path 影响加载行为（低置信度）

**位置**: 启动命令中的 `--kv-transfer-config`

**分析**:
- 当前配置: `"kv_connector_module_path":"pegaflow.connector"`
- 在 vLLM-Ascend 环境下，`VLLM_PLUGINS=ascend` 可能影响 connector 加载路径
- 如果 vLLM 使用 Ascend 的 `general_plugins` 机制，可能未正确加载 PegaKVConnector

**验证方法**:
- vLLM 启动日志中应有: `Creating v1 connector with name: PegaKVConnector`
- 技术报告确认此日志存在 → BP6 当前不成立

---

## 三、断点优先级与修复路径

```
优先级     断点        影响面          阻塞下游      修复难度
─────────────────────────────────────────────────────────
🔴 P0     BP3         Save+Load 全断    全部        中等 (需要 DMA 兼容内存)
🔴 P0     BP1         Save 不触发       BP3         低 (需要更长 prompt)
🔴 P1     BP2         Save 不执行       BP3         低 (需确认 vLLM 调用)
🟡 P2     BP4         Load 无命中       BP3+BP1     无 (BP3 修复后自动恢复)
🟡 P3     BP5         配置问题          无           低 (确认默认值)
🟡 P4     BP6         connector 加载    无           已验证无问题
```

**修复依赖链**:
```
BP3 (aclrtMemcpyAsync D2H)  ← 必须先修复
    │
    ├─► BP1 (prompt 长度) → 使用 ≥128 token prompt
    │
    └─► BP2 (wait_for_save 调用) → 确认 vLLM 调用路径
                │
                ▼
           BP4 自动恢复 (query 有缓存可命中)
```

---

## 四、诊断验证步骤（在 Ascend 硬件上执行）

### 步骤 1: 启用诊断日志

```bash
export PEGAFLOW_DEBUG_SAVE_PATH=1
export RUST_LOG=info,pegaflow_core=debug,pegaflow_server=debug
```

### 步骤 2: 使用长 prompt 测试 Save Intent 生成

```bash
# 使用 ≥ 200 tokens 的 prompt
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/shared_models/Qwen2.5-14B-Instruct",
    "prompt": "'"$(python3 -c 'print("Hello " * 100)')"'",
    "max_tokens": 10
  }'
```

**观察 vLLM 日志中的关键信息**:
```
# Save intent 生成成功:
[PegaKVConnector] req=xxx save_intent: start=... new_blocks=1 ...

# wait_for_save 被调用:
[PegaKVConnector.DEBUG] wait_for_save called: metadata=...
[PegaKVConnector.DEBUG] wait_for_save save_intent: req=xxx block_ids=[...] hashes=1

# Save worker 处理:
[PegaKVConnector.DEBUG] _process_save_batch: layers=96 total_blocks=...
```

**观察 PegaFlow Server 日志**:
```
# Save RPC 到达:
RPC [save]: instance_id=... layers=96 blocks=1 hashes=1

# D2H 拷贝成功或失败:
# 成功: RPC [save] completed: ok layers=96 blocks=1 elapsed_ms=...
# 失败: aclrtMemcpyAsync(D2H) failed: error code 507899
```

### 步骤 3: 重复请求验证缓存命中

```bash
# 发送相同 prompt 第二次
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/shared_models/Qwen2.5-14B-Instruct",
    "prompt": "'"$(python3 -c 'print("Hello " * 100)')"'",
    "max_tokens": 10
  }'
```

**观察**:
```
# 缓存命中:
[PegaKVConnector] req=xxx cache_lookup: hit_blocks=1 computed_blocks=0 hit_tokens=128 ...

# vLLM metrics:
External prefix cache hit rate: 100.0%
```

---

## 五、BP3 修复方案建议

### 方案 A: 禁用 expandable_segments（快速验证）

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
```
- 优点: 立即验证 BP3 是否为 DMA 根因
- 缺点: 可能导致显存碎片化

### 方案 B: 使用 `aclrtMalloc` 中间缓冲

在 `pegaflow-core/src/transfer/ascend_memcpy.rs` 中修改 copy 逻辑:
- 分配一个 `aclrtMalloc` 管理的 DMA 兼容 buffer
- 先 `aclrtMemcpy` device→buffer，再 `aclrtMemcpyAsync` buffer→host
- 两步拷贝，但绕过 expandable_segments 限制

### 方案 C: camem_allocator 集成（技术报告建议）

- 编译 `vllm_ascend_C` 扩展模块
- 使用 `aclrtMallocPhysical` 分配 KV Cache
- CANN IPC + DMA 均可用

---

## 六、总结

| 断点 | 状态 | 修复优先级 | 预计工作量 |
|------|------|-----------|-----------|
| BP1: Save intent 不触发 (短 prompt) | 🔴 需验证 | P0 | 使用 ≥128 token prompt 测试 |
| BP2: wait_for_save 未被调用 | 🔴 需验证 | P1 | 检查 Ascend backend 调用链 |
| BP3: aclrtMemcpyAsync 507899 | 🔴 确认 | P0 | 1-2 天 (方案 A 快速验证) |
| BP4: query 无缓存命中 | 🟡 连锁 | — | BP3 修复后自动恢复 |
| BP5: PEGAFLOW_MODE | 🟢 默认 OK | — | 无需修改 |
| BP6: connector 加载 | 🟢 已验证 | — | 无需修改 |

**核心结论**: Save/Load 链路有**三层屏障**：
1. **BP1** 阻止短 prompt 生成 save intent（使用 ≥128 token prompt 可绕过）
2. **BP2** 可能阻止 save 执行（需确认 vLLM Ascend backend 调用 `wait_for_save()`）
3. **BP3** 阻止 D2H/H2D DMA 拷贝（`507899` 错误，需要分配器层面修复）

三个断点中，**BP3 是最根本的阻塞点** — 即使 BP1 和 BP2 全部通过，D2H 拷贝也会失败，导致 server 端无缓存数据，进而 Load 路径永远返回 0 hits。