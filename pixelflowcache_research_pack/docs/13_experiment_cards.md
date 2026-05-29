# 13. 实验卡片：可以直接照着执行

每张卡片都对应一个具体实验。建议把每个卡片实现为一个 script 或 config group。

## Card P1：Layer temporal smoothness

**问题**：哪些层的 feature 随 timestep 最平滑？

**输入**：full sampler，无 cache，N=256 samples。

**记录**：每层 block output 的 norm、delta norm、cosine distance。

**步骤**：

1. 固定 seed、class/prompt、solver、CFG；
2. 每一步采样时记录 layer output summary；
3. 计算相邻 timestep 变化；
4. 对 sample 求均值和分位数；
5. 画 layer × timestep heatmap。

**成功信号**：某些层/阶段明显更平滑，可作为 cache 候选。

**失败解释**：如果所有层变化都大，转向 online probe 或 output/solver-aware cache。

## Card P2：Branch cache sensitivity

**问题**：语义分支是否比细节分支更适合缓存？

**模型**：DeCo、PixelDiT。

**步骤**：

1. 只 cache semantic/patch branch；
2. 只 cache detail/pixel branch；
3. 两者都 cache；
4. 都不 cache；
5. 比较 final LPIPS、frequency error、latency。

**成功信号**：semantic/patch branch cache 的质量损失明显小于 detail/pixel branch。

## Card P3：x-pred conversion error

**问题**：JiT / PixelGen 中 \(\hat{x}_0\) cache error 是否被 \(1/(1-t)\) 放大？

**步骤**：

1. full forward 得到 \(\hat{x}^{full}_0\)；
2. cache forward 得到 \(\hat{x}^{cache}_0\)；
3. 计算 \(\Delta x_0\)；
4. 计算 \(\Delta v=\Delta x_0/(1-t)\)；
5. 画 timestep curves。

**成功信号**：late-stage \(\Delta v\) 明显大于 \(\Delta x_0\)，且与 final LPIPS 更相关。

## Card P4：Frequency band error

**问题**：cache 破坏哪一段频率？

**步骤**：

1. 对 full/cache velocity 做 FFT/DCT；
2. 分 low/mid/high band；
3. 计算 band-wise error；
4. 按 cache unit、timestep、模型分组；
5. 与 final LPIPS/DINO 相关。

**成功信号**：high-frequency branch cache 导致 high-band error，且视觉上对应纹理/边缘损坏。

## Card P5：CFG residual stability

**问题**：\(v_u\)、\(v_c\)、\(r=v_c-v_u\) 谁更稳定？

**步骤**：

1. 每步分别记录 cond/uncond velocity；
2. 计算 residual；
3. 计算 temporal delta；
4. 不同 CFG scale 下重复；
5. 尝试 residual cache。

**成功信号**：residual 或 uncond branch 更可缓存；高 CFG 下普通 cache 质量下降。

## Card P6：Heun predictor/corrector cache

**问题**：Heun 中 cache predictor 还是 corrector 更安全？

**步骤**：

1. predictor full + corrector cache；
2. predictor cache + corrector full；
3. both cache；
4. alternating full corrector；
5. 比较质量/速度。

**成功信号**：predictor cache + corrector full 是较稳的折中。

## Card P7：AB2 history reset

**问题**：PixelDiT multistep 采样是否需要 cache history reset？

**步骤**：

1. AB2 fixed cache without reset；
2. AB2 cache with reset every K；
3. AB2 cache with risk-triggered reset；
4. 比较 error accumulation curve。

**成功信号**：risk-triggered reset 减少 trajectory drift。

## Card B1：Fewer-step baseline

**问题**：cache 是否优于直接减步数？

**步骤**：

1. full steps = 100；
2. fewer steps = 75/50/33/25；
3. 同样 solver/CFG；
4. 计算质量和 latency。

**成功信号**：PixelFlowCache 在同 latency 下质量更好。

## Card B2：Fixed cache baseline

**问题**：简单 cache 能到什么程度？

**步骤**：

1. K=2/3/4/5；
2. layers=all/middle/deep；
3. block/attn/mlp output；
4. 画 Pareto。

**成功信号**：PixelFlowCache 比 fixed cache Pareto 更优。

## Card M1：PixelFlowCache v1 主实验

**方法**：velocity criterion + branch/frequency allocation + solver reset。

**步骤**：

1. 用 Stage 1 结果设置 branch weights；
2. 运行 mild/medium/aggressive budgets；
3. 与 fixed/DiCache-like/ToCa-like 对比；
4. 报 latency/FID/LPIPS/DINO/frequency。

**成功信号**：至少两个模型明显优于 strongest baseline。

## Card A1：消融

**步骤**：

1. full method；
2. no velocity normalization；
3. no branch/frequency；
4. no solver reset；
5. no CFG-aware；
6. no online probe；
7. no calibration。

**成功信号**：每个模块都有合理贡献，且不是单一 trick。

## Card V1：视觉案例

**步骤**：

1. 选 8 个 seed，不只 cherry-pick；
2. full、fewer-step、baseline cache、PixelFlowCache 并排；
3. 放 difference map 和 FFT error；
4. 标注边缘、文字、小物体、纹理。

**成功信号**：PixelFlowCache 在纹理和语义上更接近 full。
