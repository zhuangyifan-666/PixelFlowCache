# 10. 风险登记表与决策树

## 1. 主要风险

| 风险 | 概率 | 影响 | 应对 |
|---|---:|---:|---|
| 官方代码/权重难跑 | 中 | 高 | 先跑两个模型建立闭环，后续补齐 |
| hook 后显存爆炸 | 高 | 中 | 只保存统计，完整 feature 仅小样本 |
| token cache 无真实 speedup | 高 | 高 | 主方法以 coarse branch/layer cache 为主 |
| x-pred amplification 实验不明显 | 中 | 中 | 转向 frequency/solver/CFG 作为主贡献 |
| branch smoothness 差异不明显 | 中 | 中 | 使用 output frequency error 而非 feature smoothness |
| baseline 过强 | 中 | 高 | 强调跨模型、真实 latency、质量稳定性 |
| calibration 吃掉速度 | 中 | 中 | 作为 optional，不放主方法核心 |
| FID 改善不稳定 | 中 | 高 | 增加 full-to-cache LPIPS/DINO/frequency 指标 |
| 审稿人认为 incremental | 中 | 高 | 强化系统性 analysis 和 unified velocity criterion |

## 2. 决策树

### Step 1：x-pred profiling 是否强？

如果 JiT / PixelGen 中 late-stage velocity error 明显放大：

- 把 velocity-normalized criterion 放为主贡献；
- x-pred late reset 是核心模块；
- 主图放 amplification curve。

如果不明显：

- 不强推该点；
- 仅作为统一公式；
- 主贡献转向 branch/frequency/solver。

### Step 2：branch/frequency profiling 是否强？

如果 DeCo / PixelDiT 的 semantic vs detail branch 差异明显：

- 主方法采用 branch-aware cache；
- title 可强调 frequency-aware；
- 主实验重点放 DeCo / PixelDiT。

如果不明显：

- 用 final output frequency error，而不是 branch feature smoothness；
- 做 token/spatial sensitivity；
- 转向 solver-aware 和 CFG-aware。

### Step 3：CFG residual 是否稳定？

如果 residual 更平滑：

- 加 guidance-residual cache；
- 作为一个新模块。

如果不稳定：

- 只用 CFG-aware risk weight；
- 高 CFG 下更保守，不做 residual cache。

### Step 4：solver 差异是否明显？

如果 Heun/AB2 对 cache 敏感：

- 加 solver-aware reset；
- 这是强贡献。

如果不明显：

- solver-aware 作为 safety mechanism；
- 不要过度展开。

### Step 5：真实 latency 是否达标？

如果 coarse cache latency 好：

- 主方法用 coarse cache；
- token cache 只做质量增强。

如果 coarse cache 质量差：

- 用 branch + calibration；
- 或降低 speedup target，强调 near-lossless。

如果 token cache latency 差：

- 不把 token cache 作为核心；
- 附录中解释硬件限制。

## 3. 项目里程碑

### Milestone 1：两模型跑通

- JiT + DeCo 或 JiT + PixelDiT；
- x-pred/v-pred 各一个；
- full vs wrapper 一致。

### Milestone 2：三张动机图

- layer smoothness；
- velocity error；
- frequency/branch error。

### Milestone 3：baseline curve

- fewer steps；
- fixed cache；
- online-probe baseline。

### Milestone 4：PixelFlowCache v1

- velocity + branch + solver；
- 至少两个模型优于 baseline。

### Milestone 5：完整主实验

- 四模型；
- speed-quality Pareto；
- ablation；
- visual cases。

## 4. 成功/失败判据

### 强成功

- 四模型中三到四个显著优于 baseline；
- 真实 speedup ≥ 1.8x；
- FID/LPIPS 接近 full；
- profiling 发现清晰。

### 中等成功

- 两个模型强，另外两个中等；
- speedup 1.5x-2.0x；
- 作为 workshop 或主会边缘有希望。

### 需要转向

- 真实 latency 无提升；
- profiling 无规律；
- baseline 几乎不可超越。

转向选项：

1. 从通用 cache 转向 DeCo/PixelDiT 的 branch-specific acceleration；
2. 从 feature cache 转向 solver-aware step allocation；
3. 从 image generation 转向 high-res 或 video pixel diffusion；
4. 从 training-free 转向 small learned cache controller。

## 5. 不要做的事情

- 不要只报 FLOPs；
- 不要只在一个模型上验证；
- 不要忽略 fewer-step baseline；
- 不要在 x-pred 模型上直接复用 v-pred cache criterion；
- 不要把所有模块一次性堆上而没有消融；
- 不要过早训练复杂 controller。
