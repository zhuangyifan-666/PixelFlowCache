# 09. 论文写作计划

## 1. 题目候选

1. PixelFlowCache: Velocity- and Frequency-aware Caching for Pixel-space Flow Diffusion
2. Caching Pixel Flow Diffusion Models with Velocity-space Error Control
3. Beyond Feature Similarity: Solver-aware Caching for Pixel-space Diffusion Transformers
4. Parameterization-aware Feature Caching for Pixel Diffusion Transformers

最推荐第 1 个，因为它覆盖 velocity、frequency、pixel-space flow 三个关键词。

## 2. Abstract 草案

> Pixel-space diffusion models have recently emerged as a promising alternative to latent diffusion by eliminating the VAE bottleneck and generating images directly in RGB space. However, their high-dimensional sampling process makes inference expensive. Existing feature caching methods for diffusion transformers mainly exploit timestep-wise feature similarity, but they overlook how cache errors propagate through pixel-space flow samplers. In this work, we first systematically analyze caching behavior across x-prediction and velocity-prediction pixel diffusion models. We find that cache errors are parameterization-dependent, frequency-dependent, branch-dependent, and solver-dependent. Based on these observations, we propose PixelFlowCache, a training-free/cache-calibration-light framework that controls cache decisions in velocity space, allocates cache budgets by semantic/detail branches and frequency sensitivity, and resets cache according to the ODE solver. Experiments on JiT, PixelGen, DeCo, and PixelDiT show that PixelFlowCache achieves a better quality-latency trade-off than fixed caching, timestep-aware caching, token-wise caching, and online-probe caching baselines.

## 3. Introduction 结构

### 第一段：pixel diffusion 兴起

讲 VAE latent diffusion 有重建瓶颈，pixel diffusion 回到 RGB 空间，代表 JiT、PixelGen、DeCo、PixelDiT。

### 第二段：问题是推理贵

pixel space 维度高，DiT/Transformer 计算大，多步 flow sampling 开销高。

### 第三段：已有 cache 不够

已有 feature caching 利用相邻 timestep feature 相似，但多数基于 latent DiT 或 video DiT，没有考虑 pixel-space flow 的误差传播。

### 第四段：关键观察

列出四个观察：

1. x-pred cache error 会经过 \(1/(1-t)\) 进入 velocity；
2. pixel-space cache error 在频率上不均匀；
3. semantic branch 与 detail branch 可缓存性不同；
4. solver / CFG 会放大或延续 cache error。

### 第五段：方法

PixelFlowCache：velocity criterion + frequency/branch allocation + solver-aware reset + optional CFG residual/cache calibration。

### 第六段：贡献

- 首个系统研究 pixel-space flow diffusion cache；
- 提出 velocity-space cache risk；
- 提出 branch/frequency/solver-aware cache；
- 四模型验证。

## 4. Related Work 分组

1. Diffusion / flow matching sampling acceleration；
2. Feature caching for diffusion transformers；
3. Pixel-space diffusion models；
4. Efficient Transformer inference；
5. Frequency/perceptual analysis of image generation。

不要把 related work 写成流水账，要围绕差异：现有 cache 缺少 pixel-space flow-specific error control。

## 5. Method 章节安排

### 3.1 Preliminaries

- flow sampling；
- x-pred vs v-pred；
- CFG；
- solver。

### 3.2 Cache error in pixel flow

给出：

\[
\Delta v=\Delta \hat{x}_0/(1-t)
\]

和 CFG error 公式。

### 3.3 Velocity-space cache criterion

定义统一 risk。

### 3.4 Branch/frequency-aware cache allocation

讲 semantic/detail 分工。

### 3.5 Solver-aware reset and CFG-aware policy

讲 Euler/Heun/AB2 和 CFG。

### 3.6 Implementation

讲 cache units、online probe、calibration 可选。

## 6. Experiments 章节安排

### 4.1 Setup

模型、数据、metrics、baselines。

### 4.2 Profiling findings

展示动机图。

### 4.3 Main results

质量-速度 Pareto 和主表。

### 4.4 Ablation

每个模块的贡献。

### 4.5 Analysis

frequency maps、cache decision、failure cases。

## 7. 建议主图

### Figure 1：问题示意

显示 pixel-space flow sampling，从 noise image 到 clean image，中间有 x-pred/v-pred 两类。标注：cache error 进入 velocity、frequency、solver。

### Figure 2：Profiling heatmaps

四个子图：

- x-pred velocity amplification；
- DeCo branch smoothness；
- PixelDiT patch vs pixel pathway；
- CFG residual stability。

### Figure 3：方法框架

输入：current x_t, t, condition。输出：cache decisions。中间模块：probe -> velocity risk -> branch/frequency allocation -> solver-aware reset -> cache / compute。

### Figure 4：Quality-latency Pareto

每个模型一个 subplot。

### Figure 5：Ablation

去掉每个模块后的结果。

### Figure 6：Visual comparison

Full、fewer steps、baseline cache、PixelFlowCache。

## 8. 表格

### Table 1：模型与采样差异

x-pred / v-pred、solver、branch structure、CFG。

### Table 2：Main results

四模型主指标。

### Table 3：Ablation

模块消融。

### Table 4：Latency breakdown

forward、cache op、probe、calibration、memory。

## 9. Rebuttal 预案

### 质疑 1：只是组合已有 cache？

回答：核心不是组合，而是 pixel-space flow-specific error analysis。展示 x-pred/v-pred、frequency、solver 证据。

### 质疑 2：baseline 不够强？

回答：包含 fewer steps、fixed cache、token-wise、timestep-aware、online-probe、forecast-calibration，并按 latency budget 调参。

### 质疑 3：只在一个模型有效？

回答：覆盖四个架构：x-pred, v-pred, frequency-decoupled, dual-level。

### 质疑 4：真实速度不明显？

回答：报 wall-clock 和 FLOPs，优先 coarse branch/layer cache，分析 token cache overhead。

### 质疑 5：需要训练 controller？

回答：主方法 training-free；calibration 是 optional/lightweight。

## 10. Supplementary 建议

- 各模型具体 hook 点；
- 更多 frequency maps；
- 更多 classes/prompts；
- cache thresholds；
- failure cases；
- exact latency protocol；
- pseudo-code；
- ethical / environmental impact：加速降低计算成本。

## 11. 最终投稿前 checklist

- 四模型至少两个强结果；
- x-pred/v-pred 分析成立；
- 真实 latency 可信；
- baseline 足够强；
- 消融清楚；
- 图像样例无 cherry-pick 争议；
- 代码可整理开源；
- 附录完整。
