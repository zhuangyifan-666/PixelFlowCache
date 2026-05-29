# diffusion_cache_reading_pack 阅读笔记

## 处理概况

- 压缩包内共有 21 篇 PDF，5 个主题目录。
- PDF 总页数：351 页。
- 文本抽取约 20 万词；所有 PDF 均可抽取文本并用 PDFium 渲染页面，无需 OCR。
- 主题集中在扩散模型/扩散 Transformer 的推理加速，核心技术是 feature caching、layer/token/block caching、feature forecasting、cache error calibration，以及与 quantization/pruning/video generation 的组合。

## 总体脉络

1. **2024 基础阶段**：从 U-Net 或 denoising network 内部冗余出发，发现相邻 timestep 的中间特征/模块输出高度相似，于是缓存 block 或高层特征。
2. **DiT 粒度细化阶段**：从整层/整块缓存进一步细分到 layer、token、channel、dimension，并开始考虑不同层、不同 token、不同 timestep 的异质性。
3. **误差控制阶段**：简单复用 cache 会累计误差，后续方法开始做 scale-shift、SVD calibration、gradient propagation、noise filtering、linear modulation、Taylor/ODE forecasting、input-output relational prediction。
4. **自适应调度阶段**：固定 cache interval 逐渐被 non-uniform、sample-specific、runtime adaptive、DP-optimized schedule 替代。
5. **系统组合阶段**：缓存与量化、剪枝、蒸馏、视频生成框架结合，目标从单纯 FLOPs 降低转向真实 latency、显存、视频一致性和部署质量。

## 逐篇速记

| 目录 | 论文 | 页数 | 核心方法 | 关键结论/指标 |
|---|---:|---:|---|---|
| 00_foundations | 2024_CVPR_Cache_Me_if_You_Can_Block_Caching | 10 | 分析 denoising network 中 block 输出随 timestep 平滑变化，缓存 layer/block 输出；加入轻量 scale-shift alignment，并根据 block change 自动推导 schedule。 | 在 LDM/EMU 和 DDIM/DPM 设置下约 1.5×-1.8× 加速，固定计算预算下比直接减少步数更能保持细节。 |
| 00_foundations | 2024_CVPR_DeepCache_Accelerating_Diffusion_Models_for_Free | 11 | 利用 U-Net 架构中高层特征跨相邻步骤高度冗余的性质，复用 high-level features，同时以低成本更新 low-level features。 | Stable Diffusion v1.5 约 2.3× 加速且 CLIP 仅下降 0.05；LDM-4-G 约 4.1× 加速且 FID 仅小幅下降。 |
| 01_dit_token_layer_cache | 2024_NeurIPS_Learning_to_Cache_Layer_Caching | 23 | L2C 将 Transformer layer 作为 cache 单元，训练 timestep-variant、input-invariant router，经连续优化后离散化成静态计算图。 | U-ViT-H/2 的 cache steps 可移除 93.68% 计算，总体 46.84% 计算，FID 几乎不降；同速下优于 DDIM/DPM-Solver 和已有 cache baseline。 |
| 01_dit_token_layer_cache | 2025_ICLR_ToCa_Token_wise_Feature_Caching | 22 | ToCa 做 token-wise feature caching，根据 temporal redundancy 和 error propagation 设计 token selection scores，并允许不同层/类型/深度采用不同 cache ratio。 | 在 OpenSora 和 PixArt-α 上分别达到约 2.36×、1.93× 加速，几乎不损失生成质量；指出不同 token 的 cache 破坏性可相差约 10×。 |
| 01_dit_token_layer_cache | 2025_ICML_HarmoniCa_Feature_Caching_DiT | 24 | 学习式 cache 框架；用 Step-wise Denoising Training 对齐训练/推理轨迹，用 Image Error Proxy-Guided Objective 对齐最终图像质量目标。 | 覆盖 8 个模型、4 个 sampler、256 到 2K 分辨率；在 PixArt-α 上可带来约 40% latency reduction/2.07× theoretical speedup，训练时间比前法少约 25%。 |
| 02_forecast_calibration_error | 2025_CVPR_Increment_Calibrated_Caching_Channel_Aware_SVD | 10 | 对 cached activations 做 increment calibration；用预训练权重的低秩 SVD 生成校准参数，并用 channel-aware SVD 缓解 outlier activations。 | 相比 35-step DDIM 可减少超过 45% 计算，IS 提升约 12，FID 增加小于 0.06；主要贡献是 training-free cache calibration。 |
| 02_forecast_calibration_error | 2025_ICCV_Gradient_Optimized_Cache_GOC | 10 | GOC 使用 cached gradient propagation：用 gradient queue 估计 cached/recomputed feature 的差异并传播补偿；再用 inflection-aware optimization 避免相反梯度更新。 | 在 50% cached blocks 下，报告 IS 216.28、FID 3.907，相比 baseline DiT 有明显质量提升且计算成本基本不变。 |
| 02_forecast_calibration_error | 2025_ICCV_OmniCache_Trajectory_Oriented_Global_Cache | 11 | 从全局 sampling trajectory 出发，而非只看局部相似度；依据 trajectory curvature 分配 cache reuse，并估计/过滤 cache-induced noise。 | 在 OpenSora、Latte 等上约 2×-2.5× 加速且质量接近无损；强调早期 cache reuse 更容易被后续步骤纠正，晚期错误更难恢复。 |
| 02_forecast_calibration_error | 2025_ICCV_TaylorSeer_From_Reusing_to_Forecasting | 11 | 将“cache-then-reuse”改为“cache-then-forecast”，用 feature 的时间连续性和 Taylor expansion 预测未来 timestep 特征。 | 在 FLUX、HunyuanVideo 上约 5× near-lossless acceleration；高加速比下比直接复用 cache 更稳。 |
| 02_forecast_calibration_error | 2025_NeurIPS_Diffusion_on_Demand_Selective_Caching_Modulation | 34 | 不只在“复用 cache”和“完整推理”之间二选一，而是用 learned modulation gate 选择性地对 cached features 做轻量 linear modulation。 | DiT-XL/2 上 FLOPs 和 latency 分别约降低 2.93×、2.15×；PixArt-α 上约 2.83× FLOPs、1.50× latency；仅调制少量层即可有效。 |
| 02_forecast_calibration_error | 2026_AAAI_FoCa_Feature_Caching_as_ODE | 9 | 将 hidden-feature sequence 视作 feature-ODE；FoCa 使用 BDF2 predictor + Heun corrector 做 Forecast-then-Calibrate。 | 无额外训练下，FLUX 5.50×、HunyuanVideo 6.45×、Inf-DiT 3.17×，DiT 上 4.53× 仍保持高质量；重点是大 skip interval 下稳定性。 |
| 02_forecast_calibration_error | 2026_ICLR_CEM_Cumulative_Error_Minimization | 29 | CEM 是 plug-in scheduler：先 offline error modeling 建模 timestep 与 cache interval 对误差的影响，再用 dynamic programming 最小化 cumulative error。 | 可接入 TaylorSeer、ToCa、DuCa、FasterSD、量化模型等，不增加 online compute；目标是保留原加速比同时提升 fidelity。 |
| 03_adaptive_cache_2026 | 2026_AAAI_ProCache_Constraint_Aware_Feature_Caching | 16 | ProCache 做 constraint-aware non-uniform schedule search，并在 cached segments 中对 deep blocks 和 high-importance tokens 做 selective computation。 | 在 DiT-XL/2 ImageNet 上约 2.90× speedup；报告可减少 56.2% quality degradation，高加速区间比固定 interval 稳。 |
| 03_adaptive_cache_2026 | 2026_AAAI_SortBlock_Similarity_Aware_Feature_Reuse | 9 | SortBlock 根据相邻 timestep 的 block-wise residual similarity 排序，动态决定 recomputation ratio，并用轻量 linear prediction 缓解 skipped block 误差。 | 在 Flux.1-dev、Wan2.1、HunyuanVideo 等上超过 2× 加速，强调 training-free 和跨模型泛化。 |
| 03_adaptive_cache_2026 | 2026_ICLR_DiCache_Let_Diffusion_Model_Determine_Its_Own_Cache | 26 | 用 shallow-layer online probe 实时估计 cache error，做 sample-specific schedule；Dynamic Cache Trajectory Alignment 用浅层轨迹辅助深层 multi-step cache 近似。 | 在 Wan2.1、HunyuanVideo、FLUX 上优于固定策略；示例结果包含 HunyuanVideo 约 2.34×、Flux 约 3.22× 等。 |
| 03_adaptive_cache_2026 | 2026_ICLR_HyCa_Hybrid_Feature_Caching | 24 | 将 feature evolution 建模为 dimension-wise mixture of ODEs；对不同维度 cluster 并分配不同 solver，采用 One-Time Choosing + All-Time Solving 降低开销。 | FLUX 约 5.55×、HunyuanVideo 约 5.56×、Qwen-Image/Edit 约 6.24× near-lossless；还能与 distillation 叠加到更高加速。 |
| 03_adaptive_cache_2026 | 2026_ICLR_Relational_Feature_Caching_RFC | 21 | RFC 利用 module input-output relationship；RFE 从 input feature variation 估计 output feature change，RCS 用 input prediction error 代理调度 full computation。 | 解决纯 temporal extrapolation 对 output feature irregular magnitude 预测不足的问题；在 ImageNet、DrawBench、VBench 等设置下主打 SOTA trade-off。 |
| 03_adaptive_cache_2026 | 2026_ICLR_ScalingCache_Dynamic_Interval_Caching | 18 | 用 first-order differential scaling coefficients 做更轻量 feature prediction，并在 runtime 根据 cumulative error 动态调整 cache interval。 | Wan2.1/HunyuanVideo 约 2.5× 加速且 VBench 仅约 0.5% 下降；FLUX 约 3.1× near-lossless；offline 分析样本少，online 开销低。 |
| 04_cache_quant_sparse_video | 2025_CVPR_CacheQuant_Comprehensively_Accelerated_Diffusion_Models | 12 | 共同优化 temporal caching 与 structural quantization；用 dynamic programming schedule 处理二者非正交误差，用 decoupled error correction 逐步纠正耦合误差。 | Stable Diffusion/MS-COCO 上约 5.18× speedup 和 4× compression，CLIP 仅下降约 0.02；强调 cache+quant 不能简单叠加。 |
| 04_cache_quant_sparse_video | 2025_CVPR_TeaCache_Timestep_Embedding_Aware_Cache | 11 | TeaCache 用 timestep embedding modulated noisy inputs 来估计 model output difference，再据此决定 output caching；避免直接计算昂贵 model outputs。 | Open-Sora-Plan 上最高约 4.41× 加速，VBench 仅约 -0.07%；视频场景中 latency-quality trade-off 明显优于 uniform timestep caching。 |
| 04_cache_quant_sparse_video | 2025_ICCV_QuantCache_Adaptive_Quantization_Hierarchical_Caching | 10 | 面向视频 DiT，将 hierarchical latent caching、adaptive importance-guided quantization、structural redundancy-aware pruning 联合起来。 | Open-Sora 上 end-to-end latency speedup 约 6.72×，生成质量损失较小；重点是端到端系统加速而非单一 cache 策略。 |

## 横向结论

- **从复用到预测**：早期方法主要“直接复用”上一时刻特征；后期方法普遍把 cache 特征当作时间序列或 ODE 轨迹来预测下一步，从而支撑更大的 cache interval。
- **从全局固定到局部自适应**：固定 interval 简单但容易错过不同 timestep/layer/token 的异质性；高质量方法通常要按层、token、通道、维度或样本动态调节。
- **误差是核心瓶颈**：只减少 FLOPs 不够，真正制约 high-ratio acceleration 的是 cache-induced error 的累积和传播；因此 calibration、gradient/noise correction、DP schedule、relational prediction 成为主线。
- **training-free 是主流，但 learned 方法有空间**：大量方法强调无训练、即插即用；HarmoniCa、DoD 等学习式方法则在 router/gate/modulation 上取得更强适配性，但需要额外训练或校准成本。
- **视频生成更依赖系统组合**：TeaCache、QuantCache、OmniCache、ScalingCache、DiCache、HyCa 等都强调视频任务；视频不仅看单帧质量，还要看 motion、consistency、latency、显存和长序列稳定性。

## 读后建议

若要快速建立论文脉络，可按以下顺序读：

1. Block Caching + DeepCache：理解最基本的 cache/reuse 直觉。
2. L2C + ToCa：理解 DiT 里 layer/token 粒度为何重要。
3. TaylorSeer + FoCa + RFC + ScalingCache：理解 feature forecasting/ODE/relational prediction 主线。
4. CEM + DiCache + ProCache + HyCa：理解 adaptive scheduling 和 runtime/sample-specific cache。
5. CacheQuant + TeaCache + QuantCache：理解 cache 与量化、视频、系统部署的结合。
