# 12. 12 周执行计划

这个计划按“先拿证据，再做方法，再扩实验”的节奏设计。真实进度会受模型权重、GPU、依赖环境影响，可以按里程碑而不是日历硬推。

## Week 1：环境与最小复现

### 目标

跑通至少两个模型：一个 x-pred，一个 v-pred。建议优先 JiT + DeCo 或 JiT + PixelDiT。

### 任务

1. clone 官方 repo，记录 commit；
2. 安装环境，下载 checkpoint；
3. 跑官方 sample script；
4. 保存 no-cache baseline samples；
5. 写 `ModelAdapter.forward_raw()` 和 `raw_to_velocity()`；
6. 确认 wrapper no-cache 输出与官方 sampler 一致。

### 产出

- `repro_log.md`；
- 每个模型 16 张 baseline 图；
- latency baseline；
- 初版 unified sampler。

### 风险点

如果 PixelDiT 环境复杂，先不要卡住；先用 JiT + PixelGen 做 x-pred 流程，用 DeCo 做 v-pred。

## Week 2：Hook 与日志系统

### 目标

能稳定记录 block/branch output 的 summary stats，不影响采样结果。

### 任务

1. 实现 `FeatureRecorder`；
2. 记录 layer norm、delta norm、cosine；
3. 记录 raw output 与 velocity；
4. 记录 CFG cond/uncond；
5. 对小 batch 做 feature 保存测试；
6. 排除 logging I/O 对 latency 的影响。

### 产出

- `feature_summary.csv`；
- no-cache 与 hook-only 输出一致性报告；
- P1/P3 所需日志。

## Week 3：基础 profiling P1/P3/P4

### 目标

拿到第一批动机图：layer smoothness、feature-to-velocity、x-pred amplification。

### 任务

1. JiT / PixelGen 跑 P4；
2. DeCo / PixelDiT 跑 P1；
3. 对某些层强制 cache one-step，测 velocity error；
4. 计算 feature error 与 final LPIPS 的相关性。

### 产出

- layer-timestep heatmap；
- x-pred amplification curve；
- feature/velocity/final quality scatter；
- 初步结论：velocity criterion 是否成立。

## Week 4：branch/frequency/CFG profiling P2/P5/P6

### 目标

确认 pixel-space 特异性：branch/frequency/CFG。

### 任务

1. DeCo：DiT branch vs pixel decoder；
2. PixelDiT：patch pathway vs pixel pathway；
3. 做 FFT/DCT band error；
4. 测 \(v_u\)、\(v_c\)、\(r=v_c-v_u\) 的 temporal stability；
5. 高 CFG stress test。

### 产出

- branch smoothness curve；
- frequency error heatmap；
- CFG residual stability curve；
- 对是否加入 guidance-residual cache 做决策。

## Week 5：baseline v0

### 目标

建立最低限度公平 baseline。

### 任务

1. fewer steps；
2. fixed block cache；
3. layer-wise cache；
4. output cache；
5. 统一 latency budget；
6. 每个模型至少 3 个 cache budget。

### 产出

- baseline quality-latency curve；
- fixed cache failure cases；
- 主方法需要超过的门槛。

## Week 6：strong baseline v1

### 目标

加入更强 cache baseline，防止论文被质疑。

### 任务

1. ToCa-like token cache；
2. TeaCache-like timestep/input proxy；
3. DiCache-like shallow online probe；
4. FoCa-like forecast/correction 简化版；
5. 记录真实 latency 和 overhead。

### 产出

- strong baseline table；
- token cache 是否硬件友好的结论；
- baseline 排名。

## Week 7：PixelFlowCache v1

### 目标

实现最小主方法：velocity + branch/frequency + solver reset。

### 任务

1. x-pred velocity-normalized threshold；
2. DeCo/PixelDiT branch interval；
3. Heun corrector full / AB2 history reset；
4. 与 fixed baseline 对比；
5. 画 cache decision timeline。

### 产出

- PixelFlowCache v1 results；
- 第一次完整 method ablation；
- 对是否加入 CFG/cache calibration 做决策。

## Week 8：PixelFlowCache v2

### 目标

加入 online probe 和 CFG-aware 策略。

### 任务

1. shallow probe error estimator；
2. CFG risk weight；
3. residual cache optional；
4. 高 CFG stress test；
5. outlier sample test。

### 产出

- adaptive vs fixed schedule 对比；
- high-CFG ablation；
- outlier failure reduction。

## Week 9：Calibration-light 与 aggressive speedup

### 目标

提高 aggressive cache 质量。

### 任务

1. feature affine calibration；
2. frequency residual calibration；
3. velocity residual extrapolation；
4. 分析 calibration latency cost；
5. 只保留收益明显的模块。

### 产出

- aggressive budget Pareto；
- calibration on/off table；
- latency breakdown。

## Week 10：主实验扩展

### 目标

扩大 sample count，做可投稿主表。

### 任务

1. 四模型主配置；
2. 256/512 resolution；
3. default/high CFG；
4. 计算 FID/IS/LPIPS/DINO/frequency；
5. 画 Pareto curves。

### 产出

- main result table；
- main Pareto figure；
- visual comparison。

## Week 11：消融、可视化和失败案例

### 目标

补齐论文可信度。

### 任务

1. 去掉每个模块；
2. 负控：random layer/token/frequency；
3. failure cases；
4. cache decision timeline；
5. frequency maps；
6. latency breakdown。

### 产出

- ablation table；
- negative control；
- supplemental figures。

## Week 12：论文初稿

### 目标

形成完整 paper draft。

### 任务

1. 写 introduction；
2. 写 method；
3. 写 experiments；
4. 整理 related work；
5. 写 appendix；
6. 内部 rebuttal checklist。

### 产出

- 8 页主文初稿；
- appendix；
- figures；
- reproducibility checklist。

## 可压缩版本：6 周路线

如果时间有限：

1. Week 1：JiT + DeCo 跑通；
2. Week 2：profiling P1/P3/P4/P5；
3. Week 3：baseline；
4. Week 4：PixelFlowCache v1；
5. Week 5：主实验；
6. Week 6：写作。

最小论文只做 JiT / DeCo / PixelDiT 三个也可以，但最好保留 PixelGen 作为 x-pred perceptual variant。
