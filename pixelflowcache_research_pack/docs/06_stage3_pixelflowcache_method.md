# 06. Stage 3：PixelFlowCache 方法设计

本阶段把前面的 profiling 发现转化为一个统一 cache 框架。建议把方法设计成 training-free 或 calibration-light，这样更容易强调即插即用与实际部署价值。

## 1. 方法总览

PixelFlowCache 包含五个模块：

1. **Velocity-domain cache criterion**：所有模型都用 solver 实际积分的 velocity error 衡量 cache risk。
2. **Branch / frequency-aware allocation**：语义/低频/patch 分支更激进，细节/高频/pixel 分支更保守。
3. **Solver-aware reset**：根据 Euler / Heun / AB2 的误差传播决定何时 full compute。
4. **CFG-aware residual policy**：考虑 guidance scale 对 conditional/unconditional cache error 的放大。
5. **Lightweight calibration**：可选，对 cached feature 或 velocity 做时间/频率相关校正。

## 2. 统一 risk function

对每个可缓存单元 \(u\)（layer、block、token、branch），定义：

\[
R_{u,t}=w^{param}_t\cdot w^{freq}_{u,t}\cdot w^{solver}_t\cdot w^{cfg}_t\cdot E_{u,t}.
\]

当 \(R_{u,t}<\tau_u\) 时允许 cache，否则 full compute。

### 2.1 Parameterization weight

x-pred 模型：

\[
w^{param}_t=\frac{1}{1-t+\epsilon}.
\]

v-pred 模型：

\[
w^{param}_t=1.
\]

### 2.2 Frequency / branch weight

语义/低频 branch：

\[
w^{freq}_{u,t}<1.
\]

高频/细节 branch：

\[
w^{freq}_{u,t}>1.
\]

也可以从 profiling 中学习：

\[
w^{freq}_{u}=\frac{corr(E^{band}_u,\Delta Q)}{\sum_u corr(E^{band}_u,\Delta Q)}.
\]

### 2.3 Solver weight

Euler：

\[
w^{solver}_t=|\Delta t_t|.
\]

Heun：

- predictor cache：\(w^{solver}=|\Delta t|/2\)，因为 corrector 可能修正；
- corrector cache：\(w^{solver}=|\Delta t|\)，更谨慎；
- both cache：\(w^{solver}=2|\Delta t|\)。

AB2：

\[
w^{solver}_t=|\Delta t_t|(1.5+0.5\cdot I[v_{t-1}\text{ cached}]).
\]

### 2.4 CFG weight

\[
w^{cfg}_t=s \quad \text{if conditional branch cached},
\]

\[
w^{cfg}_t=|1-s| \quad \text{if unconditional branch cached}.
\]

如果缓存 residual \(r=v_c-v_u\)：

\[
w^{cfg}_t=s.
\]

### 2.5 Error proxy

如果有 full output reference，只在 profiling 中使用：

\[
E_{u,t}=\|v^{cache}_{u,t}-v^{full}_{t}\|.
\]

实际推理中不能 full compute，所以用 online probe：

\[
\hat{E}_{u,t}=\phi(\Delta h^{probe}_t, \Delta emb_t, \Delta x_t, \Delta r^{cfg}_t, E^{freq}_{low,t}).
\]

training-free 版本可以用线性组合：

\[
\hat{E}_{u,t}=a\|h^{probe}_t-h^{probe}_{t-1}\|+b\|emb_t-emb_{t-1}\|+c\|x_t-x_{t-1}\|.
\]

## 3. Cache decision

```python
def should_cache(unit, t, probe, model_type, solver_state, cfg_state):
    param_w = 1.0 / max(1 - t, eps) if model_type == "xpred" else 1.0
    freq_w = unit.frequency_weight
    solver_w = solver_weight(solver_state)
    cfg_w = cfg_weight(cfg_state, unit)
    err_hat = estimate_error(probe, unit)
    risk = param_w * freq_w * solver_w * cfg_w * err_hat
    return risk < unit.threshold
```

## 4. Cache unit 选择

### Coarse units

- full block output；
- attention output；
- MLP output；
- branch output；
- shallow/deep split。

优点：真实 latency 好，工程简单。

### Fine units

- patch token；
- pixel token；
- high/low frequency component；
- conditional residual。

优点：质量更好；缺点：实现复杂，可能影响 kernel efficiency。

建议论文主方法以 **branch/layer coarse cache** 为主体，token/frequency 作为增强模块。这样更容易获得真实 wall-clock speedup。

## 5. Algorithm v0：Velocity-normalized fixed cache

最小可行版本：

1. 每 K 步 full compute；
2. x-pred 模型 late-stage 强制 reset；
3. v-pred 模型按固定 interval cache；
4. 输出统一评估 velocity error。

伪代码：

```python
for i, (t, t_next) in enumerate(time_grid):
    if model_type == "xpred" and t > t_late:
        use_cache = False
    else:
        use_cache = (i % K != 0)
    v = model_forward_with_optional_cache(x, t, cond, use_cache)
    x = solver_step(x, v, t, t_next)
```

用途：作为过渡版，快速验证 x-pred late reset 是否有效。

## 6. Algorithm v1：Branch/frequency-aware cache

对每个 unit 设置不同 cache interval：

| unit | interval |
|---|---:|
| semantic / patch / low-frequency | long |
| middle | medium |
| pixel / detail / high-frequency | short |

伪代码：

```python
for block in model.blocks:
    unit = cache_units[block]
    if step - unit.last_full_step < unit.interval:
        h = unit.cached_feature
    else:
        h = block(h)
        unit.update(h, step)
```

DeCo：

```text
DiT branch: K=3 or 4
pixel decoder: K=1 or 2
```

PixelDiT：

```text
patch pathway: K=3 or 4
pixel pathway: K=1 or 2
```

JiT / PixelGen：

```text
middle semantic blocks: K=3
shallow/deep detail-sensitive blocks: K=1 or 2
late timestep: reset
```

## 7. Algorithm v2：Online probe adaptive cache

先计算浅层 probe，再决定后续深层是否 cache：

```python
h = run_probe_layers(x, t, cond)
probe_delta = norm(h - cache.prev_probe)

for unit in deep_units:
    risk = compute_risk(unit, t, probe_delta, solver_state, cfg_state)
    if risk < threshold:
        h = unit.reuse(h)
    else:
        h = unit.compute(h)
```

注意：probe layers 的计算必须计入 latency。选择 probe 层时要权衡：

- 太浅：预测不准；
- 太深：省不了多少。

建议先选前 2-4 个 block。

## 8. Algorithm v3：Solver-aware cache

### Euler

直接按 risk 判断。

### Heun

推荐默认：predictor 可 cache，corrector 更保守。

```python
v_pred = forward_with_cache(x, t, mode="predictor")
x_hat = x + dt * v_pred

if step % corrector_full_interval == 0 or risk_high:
    v_corr = forward_full(x_hat, t_next)
else:
    v_corr = forward_with_cache(x_hat, t_next, mode="corrector")

x_next = x + 0.5 * dt * (v_pred + v_corr)
```

### AB2

```python
if v_prev_source == "cache" and current_risk > tau:
    reset_history()
    x_next = x + dt * v_current
else:
    x_next = x + dt * (1.5 * v_current - 0.5 * v_prev)
```

## 9. Algorithm v4：CFG residual cache

可选增强。

### 方案 A：分支独立 cache

conditional 和 unconditional 分开判断：

```python
v_u = forward(uncond, cache_policy_u)
v_c = forward(cond, cache_policy_c)
v = v_u + cfg * (v_c - v_u)
```

### 方案 B：residual cache

全算 uncond，缓存 residual：

```python
v_u = forward_full(uncond)
if residual_safe:
    r = cache.residual
else:
    v_c = forward_full(cond)
    r = v_c - v_u
v = v_u + cfg * r
```

### 方案 C：uncond cache + cond full

适合 conditional 细节更重要的情况。

## 10. Calibration-light

### Feature affine calibration

\[
h'_t=a_{u,t}h^{cache}_t+b_{u,t}.
\]

\(a,b\) 可以通过小 calibration set 拟合，不训练主模型。

### Velocity residual calibration

\[
v'_t=v^{cache}_t+\gamma_t(v^{cache}_t-v^{cache}_{t-1}).
\]

### Frequency calibration

只修正高频或中频：

\[
v'_t=v^{cache}_t+\mathcal{F}^{-1}(m_H\cdot\delta_H).
\]

其中 \(\delta_H\) 可以是从最近 full step 估计的 high-frequency residual。

## 11. Per-model adapter

### JiT adapter

- model_type = xpred；
- cache target：Transformer block output；
- criterion：velocity-normalized；
- solver：Euler/Heun；
- policy：late reset + corrector full。

### PixelGen adapter

- model_type = xpred；
- cache target：JiT backbone block output；
- late low-noise 阶段更保守；
- 可用 LPIPS/DINO proxy 评估 local/global perceptual preservation。

### DeCo adapter

- model_type = vpred；
- cache target：DiT branch aggressively；
- pixel decoder conservatively；
- frequency-weighted threshold。

### PixelDiT adapter

- model_type = vpred；
- cache target：patch pathway aggressively；
- pixel pathway conservatively；
- AB2 history reset；
- compact tokens 可做 token-level extension。

## 12. 复杂度分析

总 latency：

\[
T=T_{probe}+\sum_u I[compute_u]T_u+T_{cache\_op}+T_{calibration}.
\]

speedup：

\[
S=\frac{T_{full}}{T_{cache}}.
\]

如果 token-wise cache 的 \(T_{cache\_op}\) 过大，实际 speedup 会下降。因此 method 里要优先保证 cache unit 的硬件友好性。

## 13. 方法消融顺序

1. Fixed cache；
2. + velocity-normalized threshold；
3. + branch/frequency allocation；
4. + solver-aware reset；
5. + CFG-aware policy；
6. + calibration。

每加一个模块都要在 quality-latency curve 上显示收益。

## 14. 最小可发表版本

如果时间有限，最小版本可以是：

- velocity-normalized criterion；
- branch/frequency-aware cache；
- solver-aware reset；
- 四模型验证；
- 不做复杂 calibration。

这已经能形成清晰区别：**pixel-space flow specific cache**。
