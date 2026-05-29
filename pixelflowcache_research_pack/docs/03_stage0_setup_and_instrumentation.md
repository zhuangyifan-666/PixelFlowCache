# 03. Stage 0：复现、统一接口与数据记录

Stage 0 的目标不是发明新方法，而是建立一个能支撑顶会实验的可复现平台。后续所有 profiling、baseline、方法都依赖这个阶段的统一接口和日志。

## 1. 模型范围

纳入四个有官方/作者代码的工作：

| 模型 | 是否纳入 | 原因 |
|---|---:|---|
| JiT | 是 | 有 PyTorch/GPU re-implementation，可读 sampling 代码 |
| PixelGen | 是 | 有官方代码，x-pred + perceptual supervision |
| DeCo | 是 | 有官方代码，frequency-decoupled pixel diffusion |
| PixelDiT | 是 | 有官方代码，dual-level pixel DiT |
| DiP | 否 | 官方代码未公开或不完整时跳过 |

建议 pin 每个 repo 的 commit hash。所有实验日志必须记录：repo、commit、checkpoint、sampling steps、solver、CFG、seed、resolution、GPU 型号。

## 2. 统一 sampler API

每个 repo 的 sampler 命名和返回值不同，先写一个薄 wrapper：

```python
class UnifiedPixelFlowSampler:
    def __init__(self, model, model_type, solver, cfg_scale, cfg_interval, time_grid):
        self.model = model
        self.model_type = model_type  # "xpred" or "vpred"
        self.solver = solver          # "euler", "heun", "ab2"

    def predict_velocity(self, x_t, t, cond, uncond=None, cache_policy=None):
        # returns v_cond, v_uncond, v_cfg, optional diagnostics
        pass

    def step(self, x_t, t_i, t_next, cond, cache_policy):
        # one solver step with cache hooks
        pass

    def sample(self, noise, cond, cache_policy=None):
        pass
```

关键是把 **网络输出** 和 **solver 实际积分的 velocity** 分开记录：

```python
raw_out = model(x_t, t, cond)
if model_type == "xpred":
    x0_pred = raw_out
    v = (x0_pred - x_t) / (1 - t).clamp_min(eps)
else:
    v = raw_out
```

## 3. Hook 系统

需要记录中间层特征。不要一开始就改大段源码，优先使用 PyTorch forward hook 或在 block forward 里插入轻量 recorder。

### 必须记录的张量

| 记录项 | 用途 |
|---|---|
| block input/output | layer temporal smoothness、cache target |
| attention output | ToCa-like token sensitivity |
| MLP output | block cache / MLP cache |
| patch pathway output | PixelDiT branch profiling |
| pixel pathway output | PixelDiT detail sensitivity |
| DeCo DiT features | low-frequency branch profiling |
| DeCo pixel decoder features | high-frequency branch profiling |
| raw model output | x0 或 v 的直接误差 |
| converted velocity | 统一 error criterion |
| CFG conditional/unconditional output | CFG residual profiling |

### 日志组织建议

```
logs/
  model=jit/
    run_id=...
      config.yaml
      sample_meta.jsonl
      latency.jsonl
      features/
        seed_000001_step_000_layer_00.pt
      summaries/
        temporal_smoothness.csv
        velocity_error.csv
        frequency_error.csv
```

不要把所有 feature 都保存为巨型文件。建议两级策略：

1. 小规模 profiling：保存完整 feature；
2. 大规模评估：只保存聚合统计，如 norm、cosine、frequency band error、latency。

## 4. 数据集与采样子集

### Debug 子集

- 16 seeds；
- 4 classes 或 4 prompts；
- 10 / 20 steps；
- 只跑 256 分辨率；
- 目的：确认 hook、cache、metric 不崩。

### Profiling 子集

- 256 到 1024 samples；
- 20 / 50 / 100 steps；
- 每个模型至少 3 个 CFG scale；
- 保存中间统计。

### 主实验子集

- ImageNet class-conditional：尽量 10k / 50k samples；
- text-to-image：按模型可用性选择 GenEval / DPG-Bench 或自建 prompt set；
- 每个 cache 策略至少 3 个 seeds 组。

## 5. 统一时间轴

很多扩散论文用 \(t=1\) 表示噪声，\(t=0\) 表示图像；这几个代码里的 flow sampler 往往是 \(0\rightarrow1\)。为了避免混乱，所有日志统一记录：

```yaml
time_direction: noise_to_image
noise_time: 0.0
image_time: 1.0
```

并记录实际 `t_i`、`t_next`、`dt`。

## 6. Full-compute reference

所有 cache 实验必须有同 seed 的 full-compute reference：

```text
same initial noise
same condition
same solver
same CFG
same steps
only cache_policy differs
```

记录：

- final image；
- all-step velocity summary；
- final perceptual features；
- latency。

这样才能计算 full-vs-cache 的 LPIPS、DINO similarity、frequency error。

## 7. Latency 测量协议

真实 acceleration 论文最容易被质疑 latency。建议：

1. 每个配置先 warm-up 5-10 次；
2. 使用 CUDA events 计时；
3. 同时记录 end-to-end wall-clock；
4. 报 batch size = 1、4、8、16；
5. 报 resolution = 256、512、可行时 1024；
6. 单独测 hook/logging 关闭时的速度；
7. 记录 peak memory。

示例字段：

```json
{
  "model": "PixelDiT",
  "steps": 100,
  "solver": "ab2",
  "cfg": 3.25,
  "cache_policy": "none",
  "batch_size": 8,
  "resolution": 256,
  "latency_ms_mean": 1234.5,
  "latency_ms_std": 12.3,
  "peak_memory_gb": 21.7
}
```

## 8. Sanity checks

在进入 profiling 前必须完成：

- no-cache wrapper 生成结果与原 repo sampler 基本一致；
- 同 seed 重复采样 deterministic；
- x-pred 模型中的 raw output 和 converted velocity 均能正确保存；
- v-pred 模型没有被错误转换；
- CFG interval 开关与原代码一致；
- hooks 不改变输出；
- 计时不包含大规模文件写入；
- cache disabled 时输出完全等价或在浮点容差内等价。

## 9. Stage 0 产出

1. `unified_sampler.py`；
2. `cache_policy.py` 空接口；
3. `feature_recorder.py`；
4. `configs/*.yaml`；
5. no-cache reproductions；
6. latency baseline 表；
7. 每个模型的 sampling trace 文档。

## 10. 何时进入 Stage 1

满足以下条件即可进入：

- 四个模型中至少两个已跑通 full sampling；
- x-pred 和 v-pred 至少各一个；
- hook 能记录 3 个以上层的 feature norm；
- 同 seed full-vs-wrapper 差异接近 0；
- latency 测量可重复。

如果 PixelDiT 或 DeCo 因权重/依赖卡住，不要让项目停滞。先用 JiT + PixelGen 建立 x-pred 实验，再补 v-pred 模型。
