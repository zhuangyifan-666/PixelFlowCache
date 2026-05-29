# PixelFlowCache 研究路线包

这个压缩包是围绕“为 JiT、DeCo、PixelDiT、PixelGen 四类 pixel-space flow diffusion 模型设计 cache 推理加速方案”的研究路线整理。它假设目标是冲击 CVPR / ICCV / NeurIPS / ICLR 这类顶会，因此文档不只列想法，而是把每个阶段的实验目的、实验协议、方法设计、可视化、消融和论文叙事都拆开。

## 研究核心

四个有官方/作者代码的 pixel diffusion 工作可以分成两类：

- **x-pred flow**：JiT、PixelGen。网络输出 clean image \(\hat{x}_0\)，采样时转换成 velocity 再积分。
- **v-pred flow**：DeCo、PixelDiT。网络直接输出 velocity / vector field。

主张：现有 DiT cache 大多把“相邻 timestep feature 相似”作为出发点，但 pixel-space flow diffusion 的 cache 误差应该从 **velocity-space、frequency-space、branch structure、solver behavior、CFG residual** 五个维度重新理解。

建议论文暂定名：

> **PixelFlowCache: Velocity-, Frequency-, and Solver-aware Caching for Pixel-space Flow Diffusion**

## 推荐阅读顺序

1. `docs/01_research_roadmap.md`：总路线和研究问题。
2. `docs/02_model_sampling_analysis.md`：四个模型采样过程统一解读。
3. `docs/03_stage0_setup_and_instrumentation.md`：如何搭环境、统一 sampler、挂 hook、记录 profiling 数据。
4. `docs/04_stage1_profiling_experiments.md`：每个 profiling 实验怎么做、画什么图、怎么看结果。
5. `docs/05_stage2_baselines.md`：baseline 体系和公平比较协议。
6. `docs/06_stage3_pixelflowcache_method.md`：PixelFlowCache 方法细节、公式和伪代码。
7. `docs/07_stage4_evaluation_and_ablations.md`：主实验、消融、stress test 和指标。
8. `docs/08_implementation_blueprint.md`：工程实现蓝图。
9. `docs/09_paper_writing_plan.md`：论文写法、图表安排、贡献组织。
10. `docs/10_risk_register_and_decision_tree.md`：风险表和决策树。
11. `docs/11_references.md`：官方代码、论文和 cache baseline 参考链接。

## 附带文件

- `prior_notes/`：前面已经读过的 diffusion cache 和 pixel diffusion 中文阅读笔记。
- `templates/*.csv`：实验矩阵、指标表、消融表模板。
- `configs/pixelflowcache_minimal.yaml`：建议的统一配置草案。
- `manifest.json`：文件清单。

## 一句话路线

先复现四个官方模型的无 cache 采样；然后做 layer / branch / frequency / solver / CFG 的系统 profiling；用这些 profiling 证明 pixel-space flow cache 与 latent DiT cache 的误差传播不同；最后提出一个 training-free 或 calibration-light 的 cache 框架，把 x-pred/v-pred、frequency/branch、solver、CFG residual 纳入同一个决策函数。
