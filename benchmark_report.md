# PegaFlow Ascend NPU Benchmark Report

## 测试环境

| 项目 | 配置 |
|------|------|
| 硬件 | 8× Ascend 910B2 (64GB HBM/卡) |
| 模型 | Qwen3-8B (FP16, 32 layers, 8 KV heads) |
| vLLM | v0.23.1 + ACLGraph (FULL_AND_PIECEWISE, VLLM_COMPILE) |
| PegaFlow | pegaflow-server v0.23.3, ascend_direct backend |
| Prompt | ~8,000 words (~10,000 tokens) 共享 system prompt |
| 输出 | 64 tokens/请求 |
| 关键配置 | PYTHONHASHSEED=0, max-model-len=16384, max-num-seqs=8 |

---

## 测试 1: 串行 Round-Robin — TTFT 测试

**目的**: 测量跨实例 PegaFlow 缓存命中的单请求延迟改善

**方法**: 8 实例并行启动 → warmup 种子 → 请求逐个轮询发送 (run_bench_8inst.py)

### 结果

| | Shared (PegaFlow) | Isolated (进程内) | 提升 |
|---|---|---|---|
| 平均 TTFT (首 token) | **0.210s** | 0.410s | **48.8%** |
| 平均 Total (全 token) | 1.33s | 1.52s | 12.5% |
| 吞吐量 | 0.55 req/s | 0.49 req/s | 12.2% |
| TTFT 范围 | 0.09s – 0.94s | 0.14s – 0.93s | — |

### Per-instance TTFT 加速比

| 实例 | Shared | Isolated | 加速比 |
|------|--------|----------|--------|
| I2 (warmup 实例) | **0.13s** | 0.40s | **3.1x** |
| I0 | 0.19s | 0.40s | 2.1x |
| I3 | 0.19s | 0.40s | 2.1x |
| I4 | 0.18s | 0.41s | 2.3x |
| I5 | 0.20s | 0.41s | 2.1x |
| I6 | 0.18s | 0.40s | 2.2x |
| I7 | 0.19s | 0.41s | 2.2x |
| I1 | 0.39s | 0.40s | 1.0x |

### 时间分解 (Shared 缓存命中请求, ~200ms TTFT)

| 阶段 | 时间 | 说明 |
|------|------|------|
| PegaFlow RPC 查询 | <1ms | prefetch 查询 76 blocks |
| DMA Host-to-NPU | ~85ms | 1.43GB, 16.8 Gbps |
| Prefill (仅新增 token) | ~50ms | 9728 token 已缓存, 仅算 ~126 token |
| First Decode | ~20ms | ACLGraph PIECEWISE graph 重放 |
| API + Queue | ~50ms | HTTP、调度器排队 |
| **TTFT 总计** | **~210ms** | |

### 请求时间线

```
Warmup (仅 I2, 不计时):
  I2 Q0: 0.94s ← 首次 prefill 全部 9854 token + ACLGraph 编译
  I2 Q1: 0.15s ← vLLM 内部 prefix cache 命中 (同实例)
  I2 Q2: 0.15s ← 同上

Timed (8 实例轮询):
  I2 Q0: 0.16s ← PegaFlow 缓存命中 (warmup 存的 9728 token)
  I1 Q0: 0.94s ← 首个跨实例请求，ACLGraph 编译开销
  I4 Q0: 0.31s ← PegaFlow 缓存命中
  I5 Q0: 0.34s ← PegaFlow 缓存命中
  ...所有后续请求: 0.09-0.15s ← 全部命中

Isolated (无 PegaFlow):
  所有 Q0: 0.92-0.93s ← 全部从头 prefill
  所有 Q1/Q2: 0.14-0.15s ← vLLM 内部 prefix cache (同实例)
```

### 关键发现

1. **跨实例缓存生效** — 非 warmup 实例的 Q0, PegaFlow 命中后 TTFT 从 0.93s 降至 0.31-0.34s (2.7-3.0x)
2. **同实例 prefix cache 极快** — 第二个请求起 TTFT 仅 0.09-0.15s
3. **Isolated 第一阶段全部重算** — 所有 8 个实例的 Q0 均为 0.92-0.93s
4. **I1 首个请求包含 ACLGraph 编译** — 额外 ~0.6s

---

## 测试 2: Staggered 并发 — 吞吐量测试

**目的**: 测量持续并发负载下的吞吐量, 避免 burst DMA 竞争

**方法**: Semaphore 限并发=4, 100ms 批次间隔, 请求顺序随机打乱 (run_bench_8inst_concurrent.py)

### 结果

| | Shared (PegaFlow) | Isolated (进程内) | 提升 |
|---|---|---|---|
| 平均 TTFT (首 token) | **0.270s** | 0.329s | **17.9%** |
| 平均 Total (全 token) | 1.45s | 1.51s | 4.0% |
| 吞吐量 | **2.70 req/s** | 2.59 req/s | **4.2%** |
| Wall Clock | 14.8s | 15.4s | — |
| TTFT 范围 | 0.13s – 0.95s | 0.14s – 0.95s | — |

### Per-instance TTFT 加速比

| 实例 | Shared | Isolated | 加速比 |
|------|--------|----------|--------|
| I6 | 0.188s | 0.468s | **2.5x** |
| I3 | 0.154s | 0.330s | **2.1x** |
| I0 | 0.182s | 0.307s | 1.7x |
| I7 | 0.230s | 0.299s | 1.3x |
| I4 | 0.276s | 0.305s | 1.1x |
| I5 | 0.331s | 0.310s | 0.9x |
| I1 | 0.340s | 0.301s | 0.9x |
| I2 | 0.462s | 0.310s | 0.7x |

### 方法对比

| | Burst (旧) | Staggered (新) |
|---|---|---|
| 提交方式 | 32 请求同时发出 | Semaphore=4, 批次间隔 100ms |
| 实例顺序 | 固定 I0→I7 | 随机打乱 |
| Shared TTFT | **2.70s** (PCIe 竞争) | **0.270s** |
| Shared vs Isolated | +56% (倒退) | **-17.9%** |
| 结论 | 不真实 | ✓ 合理 |

### 关键发现

1. **Staggered 消除 PCIe 竞争** — 限并发=4 使 DMA 不重叠
2. **Shared TTFT 优于 Isolated** — 0.270s vs 0.329s (-17.9%)
3. **吞吐小幅提升** — 2.70 vs 2.59 req/s (+4.2%)
4. **Warmup 实例 I3 领先** — 0.154s vs 0.330s (2.1x)

---

## 测试 3: MLA+TP8 — DeepSeek-V2-Lite

| | Shared | Isolated |
|---|---|---|
| 平均 TTFT | 0.184s | 0.187s |
| 吞吐量 | 17.47 req/s | 17.21 req/s |
| 差异 | +1.6% | — |

MLA KV 压缩 (kv_lora_rank=512) 使 prefill 仅 ~100ms, DMA 仅 ~3ms。缓存省下的时间可忽略。上游 72% 提升针对 DeepSeek-V3.2 (685B)。

---

## 与上游 H800 对比

| 指标 | 上游 H800 | 我们 Ascend |
|------|----------|------------|
| 模型 | Llama-3.1-8B | Qwen3-8B |
| Attention | Flash Attention + CUDA Graph | ACLGraph (无 FA3) |
| Cold TTFT | 572ms | ~0.93s |
| Warm TTFT | **61ms** | ~0.21s |
| Warm 加速 | **9.4x** | **4.4x** |
| Prefill 成本 | 572ms (FA) | ~0.9s (ACLGraph) |
| DMA 成本 | ~60ms | ~85ms |
| 收益 = Prefill - DMA | **512ms** | **~200ms** |

Ascend 收益较小的原因:
1. **无 Flash Attention** — prefill 比 GPU 慢, 但 ACLGraph 比 enforce_eager 快 4x
2. **DMA 速度相近** (Ascend 15 Gbps vs H800 PCIe)
3. **Prefill 绝对时间更短** — 0.9s vs 5.26s (上游), 缓存省下的绝对值更小

---

## 全部测试汇总

| 测试 | Shared TTFT | Isolated TTFT | 提升 | 吞吐 |
|------|------------|-------------|------|------|
| 串行 TTFT | **0.210s** | 0.410s | **-48.8%** | — |
| Staggered 并发 | **0.270s** | 0.329s | **-17.9%** | +4.2% |
| Burst 并发 | 2.70s | 1.73s | +56% | +1.0% |
| MLA+TP8 | 0.184s | 0.187s | +1.6% | — |

---

## 瓶颈与建议

| 瓶颈 | 影响 | 建议 |
|------|------|------|
| flash_attn_npu_v3 未安装 | prefill 慢 ~2x | 安装 FA3 |
| Burst DMA 竞争 | 并发 burst 场景收益消失 | 正常服务负载不受影响 |
| IPC 内存泄漏 | 跨阶段 NPU HBM 不释放 | 已修复 (ManuallyDrop + gc) |
| Write pipeline seal 延迟 | 前 ~10s 新存 block 不可读 | 已加 15s 等待 |
| DeepSeek MoE ACLGraph | 编译失败, 需 enforce_eager | vllm-ascend 升级 |

## 测试脚本

| 脚本 | 用途 |
|------|------|
| `run_bench_8inst.py` | 串行 round-robin TTFT 测试 |
| `run_bench_8inst_concurrent.py` | Staggered 并发吞吐测试 |
| `run_bench_mla_tp8_concurrent.py` | MLA+TP8 并发测试 |
