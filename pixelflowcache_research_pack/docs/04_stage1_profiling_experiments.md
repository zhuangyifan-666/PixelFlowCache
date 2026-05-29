# 04. Stage 1：Profiling 实验设计

Stage 1 是整篇论文的动机来源。目标是证明：pixel-space flow diffusion 的 cache 敏感性不是简单的“相邻 timestep feature 相似”，而是由 parameterization、frequency、branch、solver、CFG 共同决定。

## P1. Layer temporal smoothness heatmap

### 目的

找出哪些层在相邻 timestep 中最稳定，作为 layer-wise cache 的基础。

### 做法

对每个模型、每个 step、每个 block 记录 full-compute feature：

\[
S_{l,t}=\frac{\|h_{l,t+1}-h_{l,t}\|_2}{\|h_{l,t}\|_2+\epsilon}.
\]

也记录 cosine distance：

\[
C_{l,t}=1-\frac{h_{l,t+1}\cdot h_{l,t}}{\|h_{l,t+1}\|\|h_{l,t}\|}.
\]

### 实验变量

- 模型：JiT / PixelGen / DeCo / PixelDiT；
- steps：20 / 50 / 100；
- CFG：1.0 / default / high；
- layer type：attention output、MLP output、block output。

### 输出图

- x-axis：timestep；
- y-axis：layer index；
- color：normalized feature change。

### 预期

- 中间层或语义层可能更稳定；
- late-stage 可能更敏感；
- PixelDiT patch pathway 比 pixel pathway 更稳定；
- DeCo DiT branch 比 pixel decoder 更稳定。

### 解释方式

如果某些层在多数样本上稳定，就适合固定或 adaptive cache。如果 smoothness 与最终质量无关，则需要更高级的 velocity/error criterion。

## P2. Branch temporal smoothness

### 目的

验证 branch-aware cache 的必要性。

### 适用模型

- DeCo：DiT branch vs pixel decoder；
- PixelDiT：patch-level pathway vs pixel-level pathway；
- JiT / PixelGen：可用 shallow/middle/deep blocks 或 token frequency proxy 替代显式 branch。

### 指标

\[
S^{branch}_{b,t}=\frac{\|H_{b,t+1}-H_{b,t}\|}{\|H_{b,t}\|+\epsilon}.
\]

### 额外分析

计算 branch feature change 与 final LPIPS / FID degradation 的相关性：

\[
\rho_b=Spearman(S^{branch}_{b,t},\Delta Q).
\]

### 关键图

- branch smoothness curve；
- branch cache ablation bar chart。

### 可能结论

如果 low-frequency/patch branch 比 high-frequency/pixel branch 平滑，可以支持：

> semantic branch long-cache, detail branch short-cache.

## P3. Feature error 到 velocity error 的映射

### 目的

证明 feature error 不是最终 cache risk 的充分指标，velocity-space error 更贴近采样。

### 做法

对某层强制使用前一步 feature：

\[
h^{cache}_{l,t}=h^{full}_{l,t-1}.
\]

运行后续层得到 output，与 full output 比较：

- feature error：\(\|h^{cache}_{l,t}-h^{full}_{l,t}\|\)；
- raw output error：\(\|\hat{x}^{cache}_0-\hat{x}^{full}_0\|\) 或 \(\|v^{cache}-v^{full}\|\)；
- velocity error：统一为 \(\|v^{cache}-v^{full}\|\)；
- final image LPIPS / DINO / DCT error。

### 分析

做相关性比较：

| predictor | target |
|---|---|
| feature L2 | final LPIPS |
| feature cosine | final LPIPS |
| raw output error | final LPIPS |
| velocity error | final LPIPS |
| frequency-weighted velocity error | final LPIPS |

如果 velocity error 相关性最高，就能支撑 PixelFlowCache 的核心 criterion。

## P4. x-pred late-stage velocity amplification

### 目的

专门分析 JiT / PixelGen 的 x-pred cache 风险。

### 做法

对每个 timestep，注入相同相对幅度的 \(\Delta \hat{x}_0\) 或使用真实 cache 误差，计算：

\[
A_t=\frac{\|\Delta v_t\|}{\|\Delta \hat{x}_{0,t}\|}=\frac{1}{1-t+\epsilon}.
\]

但只画理论曲线不够，还要画真实误差：

\[
E^{real}_t=\left\|\frac{\hat{x}^{cache}_{0,t}-\hat{x}^{full}_{0,t}}{1-t+\epsilon}\right\|.
\]

### 关键图

- timestep vs \(\|\Delta \hat{x}_0\|\)；
- timestep vs \(\|\Delta v\|\)；
- timestep vs final quality degradation when caching one stage。

### 可能结论

即使 \(\hat{x}_0\) error 看起来不大，late-stage velocity error 也会显著增大。因此 x-pred 模型需要 late-stage reset 或 velocity-normalized threshold。

## P5. Frequency error decomposition

### 目的

分析 cache 主要破坏低频结构还是高频细节。

### 做法

对 full 和 cache 的 velocity / x0 output 做 DCT 或 FFT：

\[
\Delta V=\mathcal{F}(v^{cache})-\mathcal{F}(v^{full}).
\]

将频率分成 L/M/H 三个 band：

```text
low:    radius <= r1
middle: r1 < radius <= r2
high:   radius > r2
```

记录：

\[
E_L=\|\Delta V_L\|,\quad E_M=\|\Delta V_M\|,\quad E_H=\|\Delta V_H\|.
\]

### 关键实验

- cache semantic branch vs detail branch；
- early/mid/late timestep；
- smooth background vs edge-heavy image；
- low CFG vs high CFG。

### 输出图

- frequency band error stacked bar；
- 2D FFT/DCT error heatmap；
- final image difference map。

### 可能结论

- DeCo 的 pixel decoder cache 更容易导致 high-frequency error；
- PixelDiT pixel pathway cache 更容易破坏 texture；
- semantic branch cache 可能主要影响 low-frequency layout。

## P6. CFG residual stability

### 目的

研究 conditional/unconditional/residual 三者哪个更适合缓存。

### 做法

记录：

\[
v_u(t),\quad v_c(t),\quad r(t)=v_c(t)-v_u(t).
\]

比较相邻 timestep 的变化：

\[
S_u=\|v_u(t+1)-v_u(t)\|,
\]

\[
S_c=\|v_c(t+1)-v_c(t)\|,
\]

\[
S_r=\|r(t+1)-r(t)\|.
\]

### 策略测试

1. cache both cond/uncond；
2. cache only uncond；
3. cache only cond；
4. cache residual \(r\)；
5. full uncond + cached residual；
6. cached uncond + full residual。

### 判断标准

看同等 latency 下 final LPIPS / FID。若 residual cache 表现好，可以成为方法亮点。

## P7. Solver sensitivity

### 目的

证明 cache policy 需要 solver-aware。

### 做法

同一模型同一 seed 下比较：

- full Euler；
- fixed-cache Euler；
- full Heun；
- predictor-cache Heun；
- corrector-cache Heun；
- both-cache Heun；
- AB2 full；
- AB2 cache with/without history reset。

### 指标

- per-step velocity error；
- final LPIPS to full；
- FID / IS；
- latency；
- error accumulation curve。

### 关键图

- solver variant vs quality-speed Pareto；
- error accumulation over steps。

### 可能结论

Heun 的 corrector full compute 可以显著降低 cache risk；AB2 需要历史 reset；Euler 对当前 velocity threshold 更敏感。

## P8. Token / spatial sensitivity

### 目的

在 JiT / PixelGen 无显式 branch 的情况下，建立 token-level cache 依据。

### 做法

对每个 patch token 计算：

- feature temporal change；
- attention entropy；
- image-space edge magnitude；
- local frequency energy；
- object boundary / saliency proxy；
- final quality sensitivity。

### cache test

分别 cache：

1. smooth background tokens；
2. edge tokens；
3. random tokens；
4. high-attention tokens；
5. low-attention tokens。

### 预期

背景 token 更可缓存，边缘 / 小物体 / 文字 token 更敏感。

## P9. Hardware microbenchmark

### 目的

防止算法看似节省 FLOPs，但实际 latency 没有提升。

### 做法

分别测：

- no hook no cache；
- hook only；
- fixed block cache；
- token gather/scatter cache；
- branch cache；
- calibration；
- FlashAttention on/off。

### 输出

| 策略 | FLOPs saving | latency saving | memory overhead | note |
|---|---:|---:|---:|---|

### 结论用途

如果 token-wise cache 由于 scatter/gather 变慢，可以把主方法转向 layer/branch cache 或 block-level coarse cache。

## 10. Profiling 后的决策

| 观察 | 方法选择 |
|---|---|
| x-pred late error 强 | velocity-normalized threshold + late reset |
| branch smoothness 差异强 | branch-aware cache 是主方法 |
| frequency error 与 LPIPS 强相关 | frequency-weighted criterion |
| CFG residual 稳定 | guidance-residual cache |
| solver 差异强 | solver-aware reset/corrector policy |
| token sensitivity 强 | token-aware cache |
| hardware token cache 慢 | coarse branch/layer cache 优先 |

## 11. Stage 1 最小成功标准

至少拿到三类可发表级动机图：

1. x-pred vs v-pred cache error propagation；
2. branch/frequency cache sensitivity；
3. solver/CFG 对 cache risk 的影响。

只要这三点成立，后续方法就有明确新意。
