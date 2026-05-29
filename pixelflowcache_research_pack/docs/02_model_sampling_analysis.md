# 02. 四个模型的采样过程与 cache 含义

## 1. 统一采样视角

四个目标模型的采样都可以写成 flow ODE：

\[
\frac{dx_t}{dt}=v_\theta(x_t,t,c),
\]

从 image-shaped Gaussian noise 开始：

\[
x_{t=0}\sim\mathcal{N}(0,I),
\]

沿 \(t:0\rightarrow1\) 积分到 clean image。

区别在于网络到底直接输出什么：

| 模型 | 输出参数化 | 采样中实际积分 | 典型 solver | cache 关键问题 |
|---|---|---|---|---|
| JiT | clean image \(\hat{x}_0\) | \((\hat{x}_0-x_t)/(1-t)\) | Euler / Heun | x-pred error 会被 velocity conversion 放大 |
| PixelGen | clean image \(\hat{x}_0\) | \((\hat{x}_0-x_t)/(1-t)\) | Heun | low-noise 阶段感知细节敏感 |
| DeCo | velocity \(\hat{v}\) | \(\hat{v}\) | Euler | frequency branch 差异明显 |
| PixelDiT | velocity \(\hat{v}\) | \(\hat{v}\) | FlowDPMSolver / AB2-like | patch pathway 与 pixel pathway 敏感性不同 |

## 2. x-pred flow 的 cache error

JiT / PixelGen 的网络输出：

\[
\hat{x}_0=f_\theta(x_t,t,c).
\]

采样器转换成：

\[
\hat{v}=\frac{\hat{x}_0-x_t}{1-t+\epsilon}.
\]

如果某个 cache 策略导致：

\[
\hat{x}^{cache}_0=\hat{x}^{full}_0+\Delta \hat{x}_0,
\]

则：

\[
\hat{v}^{cache}=\hat{v}^{full}+\frac{\Delta \hat{x}_0}{1-t+\epsilon}.
\]

结论：同样的 \(\hat{x}_0\) 误差，在 late stage 可能造成更大的 velocity 误差。cache criterion 不应只看 \(\|\Delta \hat{x}_0\|\)，而应看：

\[
E^{xpred}_t=\left\|\frac{\Delta \hat{x}_0}{1-t+\epsilon}\right\|.
\]

这也是 PixelFlowCache 的第一条设计原则。

## 3. v-pred flow 的 cache error

DeCo / PixelDiT 的网络直接输出：

\[
\hat{v}=f_\theta(x_t,t,c).
\]

cache error 直接进入 ODE：

\[
E^{vpred}_t=\|\hat{v}^{cache}-\hat{v}^{full}\|.
\]

这类模型没有 x-pred conversion 的分母放大，但高频 velocity 或 pixel branch 的误差可能直接影响纹理和边界。

## 4. Euler solver 下的误差传播

Euler 更新：

\[
x_{i+1}=x_i+\Delta t_i v_i.
\]

如果 cache 造成 \(\Delta v_i\)：

\[
\Delta x_{i+1}\approx \Delta x_i+\Delta t_i\Delta v_i.
\]

含义：

- 大 step size 阶段更需要谨慎；
- 如果 step size 非均匀，应将 \(\Delta t_i\) 放进 cache risk；
- velocity error 和 solver step 是乘法关系。

可定义风险：

\[
R^{Euler}_{i}=|\Delta t_i|\cdot E_i.
\]

## 5. Heun solver 下的误差传播

Heun：

1. predictor：\(\tilde{x}_{i+1}=x_i+\Delta t_i v_i\)
2. corrector：\(v_{i+1}=f_\theta(\tilde{x}_{i+1},t_{i+1})\)
3. update：\(x_{i+1}=x_i+\frac{\Delta t_i}{2}(v_i+v_{i+1})\)

cache 可以发生在 predictor 前向、corrector 前向，或二者都发生。策略空间：

| 策略 | 速度 | 风险 | 适合场景 |
|---|---:|---:|---|
| predictor full + corrector cache | 中 | 中 | corrector smoother 时 |
| predictor cache + corrector full | 中 | 低 | 希望用 corrector 修正误差 |
| predictor/cache + corrector/cache | 高 | 高 | smooth timestep 区间 |
| alternating full corrector | 中高 | 低中 | 保守默认策略 |

建议先从 **predictor cache + corrector full** 做，因为 corrector 可以纠正 predictor cache error；若速度不够，再逐步放开 corrector cache。

## 6. AB2 / multistep solver 下的误差传播

PixelDiT 代码中的 FlowDPMSolver 风格可抽象为：

\[
x_{i+1}=x_i+\Delta t_i(1.5v_i-0.5v_{i-1}).
\]

当前和上一轮 velocity 都影响更新。如果 \(v_{i-1}\) 来自 cache，误差会进入下一步。建议策略：

- 维护 `v_prev_source` 标记：full / cache / calibrated；
- 连续 cache 超过 K 步后强制 full reset；
- 如果 \(\|v_i-v_{i-1}\|\) 或 shallow-probe residual 突变，丢弃 multistep cache history；
- 对 AB2 里的 \(v_{i-1}\) 做更严格校验。

## 7. CFG 对 cache 的放大

CFG：

\[
v_{cfg}=v_u+s(v_c-v_u).
\]

cache error：

\[
\Delta v_{cfg}=s\Delta v_c+(1-s)\Delta v_u.
\]

当 \(s=3\) 时，conditional error 权重为 3，unconditional error 权重为 -2。两者都会被放大。

实验问题：

1. \(v_u\)、\(v_c\)、\(r=v_c-v_u\) 哪个在相邻 timestep 更平滑？
2. cache unconditional branch 比 cache conditional branch 是否更安全？
3. CFG interval 外是否可以更激进缓存？
4. 高 CFG scale 下是否需要降低 cache ratio？

可定义：

\[
R^{CFG}_t=s\|\Delta v_c\|+|1-s|\|\Delta v_u\|.
\]

## 8. Frequency-space 的误差

pixel diffusion 的误差不能只看整体 L2，因为视觉质量对频率敏感。将 velocity 或 \(\hat{x}_0\) 做 DCT / FFT：

\[
\mathcal{F}(v)=v_L+v_M+v_H.
\]

cache risk：

\[
R^{freq}_t=\lambda_L\|\Delta v_L\|+\lambda_M\|\Delta v_M\|+\lambda_H\|\Delta v_H\|.
\]

权重可有三种选择：

1. 均匀权重，用于纯分析；
2. 视觉敏感权重，强调中高频边缘；
3. 从 full-vs-cache final LPIPS / DINO 相关性回归得到。

## 9. 对四个模型的具体 cache 含义

### JiT

- 没有显式高频 branch；
- 大 patch Transformer 承担全部生成；
- 建议先 cache block output / MLP output，而不是最终 \(\hat{x}_0\)；
- late timestep 需要用 \(1/(1-t)\) 做风险惩罚；
- Heun 下优先保留 corrector full compute。

### PixelGen

- 与 JiT 一样是 x-pred flow；
- 额外有 perceptual supervision，低噪声阶段可能对纹理更敏感；
- cache 策略应在 low-noise / late stage 更保守；
- 可用 LPIPS / DINO feature distance 作为 cache 后质量 proxy。

### DeCo

- 显式 frequency-decoupled：DiT 偏低频语义，pixel decoder 偏高频细节；
- 适合 branch-aware cache：DiT branch 可更激进，pixel decoder 保守；
- frequency-aware profiling 是关键证据。

### PixelDiT

- dual-level architecture：patch-level for global semantics, pixel-level for texture details；
- patch pathway 可长 interval cache，pixel pathway 短 interval 或校正；
- Pixel Token Compaction 后的 compact tokens 可能更适合 adaptive token cache。

## 10. 设计原则总结

1. cache criterion 应该以 solver 实际积分的 velocity error 为核心；
2. x-pred 模型不能只看 \(\hat{x}_0\) error；
3. pixel-space 模型必须看 frequency / perceptual error；
4. 有显式 branch 的模型应 branch-aware；
5. solver 不是无关实现细节，而是 cache error propagation 的一部分；
6. CFG scale 越大，cache policy 应越保守；
7. 最好用 online probe 做 sample-specific 判断，而不是固定 schedule。
