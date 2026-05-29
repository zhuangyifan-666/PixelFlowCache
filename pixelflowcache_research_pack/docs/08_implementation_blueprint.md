# 08. 工程实现蓝图

## 1. 代码结构建议

```
pixelflowcache/
  adapters/
    jit_adapter.py
    pixelgen_adapter.py
    deco_adapter.py
    pixeldit_adapter.py
  sampler/
    unified_sampler.py
    solvers.py
    cfg.py
  cache/
    base_policy.py
    fixed_policy.py
    velocity_policy.py
    branch_policy.py
    solver_policy.py
    cfg_residual_policy.py
    calibration.py
  profiling/
    feature_recorder.py
    smoothness.py
    frequency.py
    cfg_residual.py
    hardware.py
  evaluation/
    metrics.py
    latency.py
    image_quality.py
  configs/
  scripts/
```

## 2. Adapter 接口

每个模型只负责把原 repo 的模型调用转换为统一形式：

```python
class ModelAdapter:
    model_type: Literal["xpred", "vpred"]

    def forward_raw(self, x, t, cond):
        """Return raw model output: x0 for x-pred, v for v-pred."""
        pass

    def raw_to_velocity(self, raw, x, t):
        if self.model_type == "xpred":
            return (raw - x) / torch.clamp(1 - t, min=1e-4)
        return raw

    def get_cache_units(self):
        """Return layer/branch units that can be cached."""
        pass
```

## 3. CacheUnit

```python
@dataclass
class CacheUnit:
    name: str
    unit_type: str  # block, attn, mlp, branch, token
    branch_type: str  # semantic, detail, low_freq, high_freq, patch, pixel
    layer_idx: int
    frequency_weight: float
    threshold: float
    interval_max: int
    last_full_step: int = -1
    cached_tensor: Optional[torch.Tensor] = None
    last_probe: Optional[torch.Tensor] = None
```

## 4. CachePolicy

```python
class CachePolicy:
    def begin_step(self, step, t, x, cond, solver_state):
        pass

    def should_compute(self, unit: CacheUnit, probe_stats):
        pass

    def update(self, unit: CacheUnit, tensor, stats):
        unit.cached_tensor = tensor.detach()
        unit.last_full_step = stats.step

    def reuse(self, unit: CacheUnit):
        return unit.cached_tensor
```

## 5. Solver state

```python
@dataclass
class SolverState:
    solver: str
    step_idx: int
    t: float
    t_next: float
    dt: float
    mode: str  # euler, heun_predictor, heun_corrector, ab2
    v_prev_source: str = "none"  # full/cache/calibrated
    consecutive_cache_steps: int = 0
```

## 6. 统一采样伪代码

```python
def sample(adapter, noise, cond, uncond, policy, solver):
    x = noise
    solver_state = SolverState(...)

    for i in range(num_steps):
        t, t_next, dt = grid[i], grid[i+1], grid[i+1]-grid[i]
        solver_state.update(i, t, t_next, dt)

        if solver.name == "euler":
            v = predict_cfg_velocity(adapter, x, t, cond, uncond, policy, solver_state)
            x = x + dt * v

        elif solver.name == "heun":
            solver_state.mode = "heun_predictor"
            v1 = predict_cfg_velocity(adapter, x, t, cond, uncond, policy, solver_state)
            x_hat = x + dt * v1
            solver_state.mode = "heun_corrector"
            v2 = predict_cfg_velocity(adapter, x_hat, t_next, cond, uncond, policy, solver_state)
            x = x + 0.5 * dt * (v1 + v2)

        elif solver.name == "ab2":
            v = predict_cfg_velocity(adapter, x, t, cond, uncond, policy, solver_state)
            if solver_state.v_prev is None or policy.reset_history:
                x_next = x + dt * v
            else:
                x_next = x + dt * (1.5 * v - 0.5 * solver_state.v_prev)
            solver_state.v_prev = v.detach()
            x = x_next

    return x
```

## 7. Hook 实现建议

### Forward hook

```python
def make_hook(name, recorder):
    def hook(module, inputs, output):
        recorder.record(name, output)
    return hook
```

优点：不改原模型。缺点：如果要在 forward 中替换 cached tensor，hook 不够，需要 wrapper block。

### Wrapper block

```python
class CacheableBlock(nn.Module):
    def __init__(self, block, unit, policy):
        super().__init__()
        self.block = block
        self.unit = unit
        self.policy = policy

    def forward(self, x, *args, **kwargs):
        if self.policy.should_compute(self.unit):
            y = self.block(x, *args, **kwargs)
            self.policy.update(self.unit, y)
            return y
        else:
            return self.policy.reuse(self.unit)
```

优点：能真正跳过计算。缺点：需要小心 residual connection、dropout、conditioning。

## 8. 特征保存策略

不要把所有 tensor 都落盘。推荐在线聚合：

```python
stats = {
    "norm": tensor.norm().item(),
    "delta_norm": (tensor - prev).norm().item(),
    "cosine": cosine(tensor, prev).item(),
    "mean": tensor.mean().item(),
    "std": tensor.std().item(),
}
```

只有小规模 profiling 保存完整 tensor。

## 9. Frequency analyzer

```python
def frequency_bands(x, bands=(0.15, 0.45)):
    X = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    # build radial masks: low/mid/high
    return low_energy, mid_energy, high_energy
```

对 RGB 图像、velocity、x0_pred 都可用。对 patch tokens 需要先 unpatchify。

## 10. Config schema

```yaml
model:
  name: PixelDiT
  type: vpred
  repo_commit: null
sampling:
  steps: 100
  solver: ab2
  cfg_scale: 3.25
  cfg_interval: [0.1, 1.0]
cache:
  policy: pixelflowcache
  probe_layers: [0, 1, 2]
  velocity_normalization: true
  branch_aware: true
  solver_aware: true
  cfg_aware: true
  calibration: false
profiling:
  save_full_features: false
  frequency_analysis: true
  log_interval: 1
```

## 11. Unit tests

### 基础测试

- `cache_policy=none` 与原 sampler 输出一致；
- `cache_policy=fixed, K=1` 与 full 输出一致；
- x-pred raw_to_velocity 公式正确；
- CFG split/chunk 正确；
- Heun predictor/corrector 时间正确；
- AB2 first step fallback Euler。

### 数值测试

- no-cache latency 接近原 repo；
- cache 后 FLOPs/latency 确实下降；
- memory 不随 steps 线性爆炸；
- deterministic seed。

## 12. 常见 bug

1. 把 \(t\) 方向搞反；
2. x-pred 模型忘记除以 \(1-t\)；
3. CFG conditional/unconditional 顺序反了；
4. cached tensor 没 detach 导致显存累积；
5. token cache gather/scatter 破坏 shape；
6. hook 写盘进入 latency 计时；
7. cache K=1 与 full 不一致；
8. Heun corrector 使用了错误的 \(t\)；
9. AB2 使用 cached stale \(v_{prev}\) 却没有 reset；
10. 对 pixel branch 使用 patch branch 的 cache threshold。

## 13. 实验脚本建议

```bash
python scripts/run_sample.py \
  --model jit \
  --config configs/jit_256.yaml \
  --cache none \
  --num-samples 1024

python scripts/run_profile.py \
  --model pixeldit \
  --profile branch_smoothness \
  --num-samples 256

python scripts/run_eval.py \
  --model deco \
  --cache pixelflowcache \
  --budget medium
```

## 14. 工程优先级

1. 先做 block/branch coarse cache；
2. 再做 online probe；
3. 再做 solver-aware reset；
4. 最后做 token-wise 和 calibration。

这样可以尽早得到真实 latency improvement。
