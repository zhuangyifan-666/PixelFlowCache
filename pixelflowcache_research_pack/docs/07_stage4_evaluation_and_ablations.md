# 07. Stage 4：主实验、消融和评估协议

## 1. 实验目标

证明 PixelFlowCache 在相同或更低质量损失下，比 fewer-step、fixed cache、ToCa-like、TeaCache-like、DiCache-like、FoCa-like baseline 有更高真实 speedup。

## 2. 主实验矩阵

| 维度 | 取值 |
|---|---|
| 模型 | JiT, PixelGen, DeCo, PixelDiT |
| 分辨率 | 256, 512, 可行时 1024 |
| steps | 20, 50, 100 |
| CFG | default, low, high |
| solver | Euler, Heun, AB2/FlowDPMSolver |
| cache budget | mild, medium, aggressive |
| 数据 | ImageNet class-conditional, optional T2I benchmarks |

不要一开始铺满所有组合。建议分三层：

1. Debug matrix：小样本快速验证；
2. Profiling matrix：中等样本拿动机图；
3. Final matrix：只保留最重要组合做大规模指标。

## 3. 图像质量指标

### Class-to-image

- FID；
- IS；
- sFID；
- precision / recall；
- gFID 或 clean-FID；
- LPIPS-to-full，同 seed 对比 full compute；
- DINO similarity-to-full；
- frequency-band error。

### Text-to-image

如果官方模型支持 T2I，可报：

- GenEval；
- DPG-Bench；
- CLIPScore；
- prompt-object alignment；
- text rendering / small object / multi-object subset。

### 为什么要 full-to-cache 指标

FID 是分布指标，可能无法揭示同 seed 轨迹变化。cache 加速需要证明“同一初始噪声下，cache 没有把图像轨迹带偏太多”。因此要报：

\[
LPIPS(I^{cache}, I^{full}),
\]

\[
1-cos(DINO(I^{cache}),DINO(I^{full})),
\]

以及 frequency error。

## 4. 加速指标

必须报：

- end-to-end latency；
- CUDA event latency；
- FLOPs reduction；
- peak memory；
- cache memory；
- batch size sensitivity；
- resolution sensitivity；
- 是否兼容 FlashAttention；
- 是否包含 CFG 双分支前向。

## 5. Quality-latency Pareto

主图建议使用：

- x-axis：speedup 或 latency；
- y-axis：FID / LPIPS-to-full / DINO distance；
- 每条线：一个方法；
- 每个点：一个 cache budget。

这个图比单点表格更有说服力。

## 6. 消融实验

### A1. 去掉 velocity-normalized criterion

x-pred 模型中，将 criterion 从：

\[
\|\Delta \hat{x}_0\|/(1-t)
\]

改成：

\[
\|\Delta \hat{x}_0\|.
\]

预期：late-stage 质量下降。

### A2. 去掉 branch/frequency allocation

所有 branch 使用同一 interval。预期：DeCo / PixelDiT 质量下降，尤其高频细节。

### A3. 去掉 solver-aware reset

Heun / AB2 下使用固定 cache。预期：error accumulation 更明显。

### A4. 去掉 CFG-aware policy

conditional/unconditional 统一 cache。预期：高 CFG scale 下质量下降。

### A5. 去掉 online probe

使用固定 schedule。预期：outlier classes/prompts 上失败更多。

### A6. 去掉 calibration

如果使用 calibration，则必须展示它的边际收益和 latency cost。

## 7. Stress tests

### High-frequency stress

- 草地、头发、纹理布料、城市建筑、文字、细线条。

### Small object stress

- 多个小物体；
- 图像边缘小物体；
- 高密度场景。

### Semantic stress

- 多对象组合；
- prompt 中方位关系；
- 类别细粒度区分。

### Sampling stress

- 高 CFG；
- 少步数；
- 高分辨率；
- aggressive cache budget。

## 8. 可视化

建议每个模型展示：

1. full vs cache 同 seed 图像；
2. difference map；
3. high-frequency error map；
4. cache decision timeline；
5. layer/branch cache heatmap；
6. failure examples。

## 9. 统计显著性

- 每个配置至少 3 个 seed groups；
- latency 报 mean ± std；
- LPIPS/DINO 报 confidence interval；
- FID 最好使用相同 sample count；
- 不要在不同 sample count 的 FID 间做强结论。

## 10. 主表模板

| Model | Method | Steps | Speedup | FID ↓ | IS ↑ | LPIPS-full ↓ | DINO-full ↑ | Freq-H ↓ | Memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

## 11. 论文中最重要的 6 张图

1. Motivation heatmap：x-pred/v-pred error propagation；
2. Branch/frequency sensitivity；
3. PixelFlowCache algorithm diagram；
4. Quality-latency Pareto；
5. Ablation bar chart；
6. Visual comparison + failure cases。

## 12. 评估成功标准

建议设定目标：

- mild budget：几乎无质量损失，1.3x-1.5x；
- medium budget：质量损失小于 fewer-step，1.8x-2.3x；
- aggressive budget：质量仍优于强 baseline，2.5x+；
- 至少两个模型上明显优于所有 baseline；
- 四个模型上整体趋势成立。

如果某个模型上没有超过强 baseline，也不要隐藏。可以分析该模型结构/solver 不适合某类 cache，并作为 limitation。
