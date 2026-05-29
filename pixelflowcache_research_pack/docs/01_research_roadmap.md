# 01. 总研究路线：从“能加速”到“顶会级贡献”

## 1. 论文定位

目标不是简单把 ToCa、TeaCache、DiCache 或 FoCa 移植到 pixel diffusion 上，而是提出一个针对 **pixel-space flow diffusion** 的 cache 框架。四个目标模型有共同点：

1. 采样直接发生在 RGB pixel space 或 patchified pixel space，而不是 VAE latent space；
2. 采样是 flow / ODE 式多步积分；
3. 每一步通常并行更新整张图像；
4. 模型结构中存在“全局语义”和“局部/高频细节”的分工；
5. CFG 和 solver 的选择会直接影响 cache 误差传播。

顶会级工作应该回答的不只是“cache 以后速度快多少”，而是：

> **pixel-space flow models 的中间特征、输出向量场、频率误差和最终视觉质量之间是什么关系？怎样利用这些关系设计比通用 DiT cache 更稳的加速方案？**

## 2. 推荐核心 claim

> Existing feature caching methods for diffusion transformers mainly exploit timestep-wise feature similarity, but they ignore the parameterization-, frequency-, and solver-dependent error propagation of pixel-space flow diffusion. We first characterize this behavior across x-pred and v-pred pixel diffusion models, then propose PixelFlowCache, a velocity-space controlled, branch/frequency-aware, solver-aware cache framework.

中文表述：

> 现有 DiT cache 默认 feature 相似就能复用，但 pixel-space flow diffusion 里，cache 误差是否安全取决于网络输出参数化、像素频率结构、模型分支、ODE solver 和 CFG residual。我们系统分析这些因素，并设计一个统一的 PixelFlowCache。

## 3. 研究问题 RQ

### RQ1：x-pred 与 v-pred 的 cache 误差是否不同？

JiT / PixelGen 输出 \(\hat{x}_0\)，采样时转换为：

\[
\hat{v}=\frac{\hat{x}_0-x_t}{1-t}.
\]

如果 cache 引入 \(\Delta \hat{x}_0\)，velocity error 为：

\[
\Delta \hat{v}=\frac{\Delta \hat{x}_0}{1-t}.
\]

因此 x-pred 模型的 late-stage cache 误差可能被放大。DeCo / PixelDiT 直接输出 \(v\)，误差传播形式不同。这个差异可以作为论文第一个新发现。

### RQ2：pixel-space cache 误差主要破坏低频语义还是高频细节？

DeCo 明确把 DiT 和 pixel decoder 分为低频语义与高频细节。PixelDiT 也用 patch-level pathway 和 pixel-level pathway 分工。PixelGen 用 LPIPS 与 DINO 分别约束局部纹理和全局语义。这说明 pixel diffusion 的质量不只由 feature L2 决定，还由 frequency / perceptual error 决定。

### RQ3：哪些 branch / layer / token 最适合缓存？

可能规律：

- patch-level global tokens 更平滑，适合更长 cache interval；
- pixel-level texture tokens 更敏感，应更频繁重算；
- 背景 / 平滑区域 token 可缓存；
- 边缘、文字、小物体、纹理复杂区域需保守缓存；
- 低噪声阶段的细节 refinement 不适合大范围缓存。

### RQ4：solver 如何改变 cache 风险？

Euler、Heun、AB2 / multistep 对 velocity error 的容忍度不同。Heun 有 predictor 和 corrector 两次前向，PixelDiT 的 multistep 更新可能把上一轮 velocity error 带入下一步。cache policy 应该与 solver 绑定，而不是只按 timestep interval 固定复用。

### RQ5：CFG 是否放大 cache error？

CFG 输出：

\[
 v_{cfg}=v_u+s(v_c-v_u).
\]

cache 误差传播为：

\[
\Delta v_{cfg}=s\Delta v_c+(1-s)\Delta v_u.
\]

当 \(s>1\) 时，conditional / unconditional 分支误差可能被放大。需要研究是否可以缓存 CFG residual \(r=v_c-v_u\)，或者分别对 conditional / unconditional 设置不同 cache 策略。

### RQ6：能否构造 training-free 的 online cache criterion？

理想策略不重新训练模型，只用浅层 probe、低频 residual、CFG residual、time embedding 等低成本信号预测当前 step 是否安全缓存。

### RQ7：cache + calibration 是否比单纯复用稳定？

高加速比下纯 cache 可能崩。可以尝试轻量 correction：

\[
 h' = a_{l,t}h_{cache}+b_{l,t}
\]

或者直接做 velocity / frequency-space correction。

### RQ8：真实 latency 是否能超过已有方法？

顶会 acceleration 论文必须报真实 wall-clock。需要区分 FLOPs saving 和 actual latency saving，关注 FlashAttention、token gather/scatter、cache memory、batch size、resolution scaling。

## 4. 六阶段路线

| 阶段 | 名称 | 目标 | 主要产出 |
|---|---|---|---|
| Stage 0 | 复现与统一接口 | 跑通四个官方模型，统一 sampler / hook / logging | baseline samples、latency、统一 API |
| Stage 1 | Profiling | 证明 pixel flow cache 的误差规律 | 5-9 张 motivation figures |
| Stage 2 | Baseline | 建立公平比较体系 | fixed cache、ToCa-like、TeaCache-like、DiCache-like、FoCa-like |
| Stage 3 | 方法 v1 | 提出 PixelFlowCache 核心策略 | velocity-aware + branch/frequency-aware + solver-aware cache |
| Stage 4 | 主实验与消融 | 证明质量-速度 Pareto 优势 | ImageNet / T2I 指标、latency、ablation |
| Stage 5 | 论文与补充材料 | 构造完整顶会叙事 | paper draft、figures、appendix |

## 5. 论文贡献的最小闭环

一篇强论文至少需要下面四类证据：

### 贡献 A：系统性发现

展示 x-pred/v-pred、frequency、branch、solver、CFG 对 cache error 的不同影响。最好每个发现都有一张 heatmap 或曲线。

### 贡献 B：统一方法

PixelFlowCache 不只是经验规则，而是从 solver 实际积分的 velocity error 出发，统一处理 x-pred 和 v-pred 模型。

### 贡献 C：跨模型验证

在 JiT、PixelGen、DeCo、PixelDiT 四个官方实现上验证，覆盖两类输出参数化和两类架构分工。

### 贡献 D：真实加速

不只报 theoretical FLOPs，也报真实 wall-clock、显存、batch size、resolution sensitivity。

## 6. 建议优先级

最优先做：

1. 统一 sampler 和 hook；
2. x-pred late-stage velocity amplification profiling；
3. branch/frequency temporal smoothness profiling；
4. fixed cache + ToCa-like + TeaCache-like baseline；
5. PixelFlowCache v1。

次优先做：

1. CFG residual cache；
2. Heun predictor/corrector cache 分析；
3. calibration-light correction。

不建议一开始做：

1. 训练一个大型 cache controller；
2. 太复杂的 token routing；
3. 只追 FLOPs 不管真实 latency；
4. 在所有模型上强行使用同一种层编号策略。

## 7. 预期最强卖点

如果 profiling 支持以下结论，论文会比较有冲击力：

- JiT / PixelGen 的 \(\hat{x}_0\) 误差在低 \(1-t\) 阶段转换成 velocity 后显著放大；
- DeCo 的 low-frequency DiT branch 比 high-frequency pixel decoder 更可缓存；
- PixelDiT 的 patch pathway 比 pixel pathway temporal smoothness 更强；
- CFG residual 的变化比 full conditional velocity 更可预测，或至少 CFG scale 会显著放大 cache 误差；
- solver-aware reset 比固定 cache interval 在同等 latency 下质量更稳。

这些发现与 pixel-space flow 的结构强相关，能够和已有 latent DiT cache 区分开。
