# Pixel-space Diffusion 论文阅读笔记

本笔记基于上传压缩包 `Pixel.zip` 中的 5 篇 PDF：

| 文件 | 论文 | 页数 | 主关键词 |
|---|---:|---:|---|
| `2511.13720v2.pdf` | **Back to Basics: Let Denoising Generative Models Denoise** / JiT | 18 | x-prediction, raw pixel ViT, large patches, no tokenizer |
| `2511.18822v3.pdf` | **DiP: Taming Diffusion Models in Pixel Space** | 25 | global-local decoupling, Patch Detailer Head, large patch DiT |
| `2511.19365v2.pdf` | **DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation** | 18 | frequency decoupling, pixel decoder, JPEG/DCT frequency-aware FM loss |
| `2511.20645v2.pdf` | **PixelDiT: Pixel Diffusion Transformers for Image Generation** | 25 | dual-level DiT, pixel-wise AdaLN, pixel token compaction, T2I 1K |
| `2602.02493v2.pdf` | **PixelGen: Improving Pixel Diffusion with Perceptual Supervision** | 25 | x-prediction, LPIPS, P-DINO, noise gating, perceptual supervision |

> 说明：下面的数值均为论文 PDF 中报告的结果。不同论文的训练轮数、采样器、NFE、CFG 设置、模型规模、数据和评测协议不完全一致，因此横向比较只能看趋势，不能简单视作严格同设置排行榜。

---

## 1. 总体技术脉络

这 5 篇论文都在回答同一个问题：**能不能绕开 VAE，直接在 pixel space 里训练扩散/flow 模型，同时又保持 latent diffusion 的质量和效率？**

它们的共同判断是：pixel diffusion 的困难不只是“维度太高”，而是至少有四个耦合瓶颈：

1. **预测目标不合适**：传统 `epsilon`/`v` prediction 要网络在高维像素空间里预测含噪量，容易让有限容量网络把能力浪费在高维噪声上。
2. **全频率混在一个 DiT 里建模**：全局语义、低频结构、高频纹理、像素级细节都塞进同一个 patch-level Transformer，会导致训练慢、细节差或算力爆炸。
3. **标准逐像素 loss 不符合感知质量**：均匀 MSE/flow matching 对所有像素一视同仁，容易优化到模糊样本或无意义的细节/噪声。
4. **像素级 token 交互昂贵**：要保留细节需要小 patch 或像素级建模，但直接对 H×W pixel tokens 做 attention 不可承受。

5 篇论文分别从不同层面解决这些问题：

| 路线 | 代表论文 | 核心思路 |
|---|---|---|
| **改预测目标** | JiT, PixelGen | 让网络直接预测 clean image，即 x-prediction；采样时再转换为 velocity。 |
| **做全局-局部/高低频解耦** | DiP, DeCo, PixelDiT | DiT 负责全局语义或低频，另一个轻量模块负责局部/高频/像素细节。 |
| **改训练监督** | PixelGen, DeCo, PixelDiT | 加感知损失、DINO/REPA 对齐、频域加权，让优化目标更贴近视觉质量。 |
| **让像素级建模可计算** | PixelDiT | 用 pixel token compaction 降低 attention 序列长度，同时用 pixel-wise AdaLN 保留细粒度调制。 |

一句话概括这组论文的共同结论：**pixel diffusion 正在从“直接把 latent DiT 搬到像素空间”转向“目标、结构、频率、感知监督的联合重设计”。**

---

## 2. 逐篇阅读

### 2.1 JiT / Back to Basics: Let Denoising Generative Models Denoise

**核心问题**：为什么现有扩散模型明明叫 denoising，却通常不直接预测 clean image，而是预测 noise 或 velocity？在高维像素空间里，这个选择是否根本影响可学习性？

**核心观点**：

自然图像位于低维流形上，而 noise、velocity 等含噪量分布在高维空间。对有限容量网络来说，预测 clean image 是“回到低维流形”，预测 noise/velocity 则要保留高维噪声信息。低维 latent 里这个差异被隐藏了；到了 raw pixel，差异会变得非常明显。

**方法**：

JiT 使用极简的 raw-pixel ViT：

- 将图像切成大 patch，例如 ImageNet-256 用 patch size 16，单 patch 维度为 16×16×3=768。
- 不用 VAE tokenizer，不用 perceptual/adversarial loss，不用 self-supervised pretraining。
- 网络输出 clean image `x_pred`。
- 训练时仍使用 velocity-space loss：`v_pred = (x_pred - z_t)/(1-t)`，再与真实 `v = (x - z_t)/(1-t)` 做 L2。
- 采样时也用 `x_pred` 转成 velocity，套 Euler/Heun 等 ODE solver。

**关键实验结论**：

- 在 ImageNet-256、JiT-B/16 的九宫格实验里，只要网络直接输出 `x`，无论 loss 在 x、epsilon、v 空间都能工作；但网络直接输出 epsilon 或 v 会灾难性失败。论文报告的 FID 量级是：x-pred + v-loss 约 8.62，而 epsilon/v-pred 多数达到数十到数百。
- 在 ImageNet-64、patch size 4 的低维输入下，epsilon/v/x 三种输出都能工作，说明问题主要来自高维像素 patch。
- 大 patch 不一定是坏事：JiT/16 在 256，JiT/32 在 512，甚至 JiT/64 在 1024 都保持 16×16 token grid，计算量随分辨率增长不明显。
- bottleneck patch embedding 反而有帮助。把 768-d raw patch 先压到 32/64/128 等低维再投影到 Transformer hidden dim，能改善 FID，符合“自然图像内在低维”的假设。
- 参考结果：JiT-G/16 在 ImageNet-256 报告 FID 1.82；JiT-G/32 在 ImageNet-512 报告 FID 1.78。它没有用 tokenizer、perceptual loss 或 self-supervised pretraining。

**我认为这篇的意义**：

JiT 给后续 pixel diffusion 提供了一个非常强的基础假设：**别让网络在像素空间里预测高维噪声，先让它预测干净图像。** 这为 PixelGen 的感知监督提供了接口，也和 DeCo 的“把高频噪声从 DiT 里拿掉”形成呼应。

**局限**：

JiT 的优势是极简和概念清楚，但它仍依赖较大模型和较长训练；在强 CFG 或 text-to-image 场景下，单靠 x-prediction 还不足以完全解决感知质量和语义对齐问题。

---

### 2.2 DiP: Taming Diffusion Models in Pixel Space

**核心问题**：大 patch DiT 在像素空间中效率高，但会丢失 patch 内高频纹理；小 patch 能保细节但计算太贵。如何同时获得 large-patch 的效率和 small/local 模块的细节能力？

**核心方法**：

DiP 把生成拆成两个阶段/职责：

1. **DiT backbone**：使用大 patch，例如 P=16，负责全局结构和长程依赖。
2. **Patch Detailer Head**：轻量局部模块，接收每个 patch 的 DiT context feature `s_i` 和原始 noisy patch `p_i`，并行预测该 patch 的 noise/velocity 细节。

最终选择的是 **post-hoc refinement**：Patch Detailer Head 放在 DiT 最后，不插入中间层。这样实现简单，也能把标准 DiT 当作 black-box backbone。

**Patch Detailer Head 设计对比**：

论文比较了 Standard MLP、Coordinate-based MLP、Intra-Patch Attention 和 Convolutional U-Net。结果显示：

- MLP 缺少空间归纳偏置，效果很差。
- Intra-Patch Attention 有提升但内存/计算较高。
- Coordinate-based MLP 接近 PixNerd/NeRF 风格，效果较好但训练成本更高。
- 轻量 **Convolutional U-Net** 最终最稳：只增加很少参数，却显著补足局部纹理。

**关键结果**：

- ImageNet-256，DiP-XL/16 报告 FID 1.79，参数约 631M。
- 论文强调相对 PixelFlow 有显著速度优势：例如 PixelFlow-XL/4 约 7.50s/image，DiP 约 0.70–0.92s/image，达到接近 8–10× 的推理加速说法。
- ImageNet-512，DiP-XL/32 报告 FID 2.31，说明 large-patch + local head 在高分辨率也能保持效率。
- 消融中，单纯加深/加宽 DiT 不如加局部 inductive bias 划算：DiT-only 增大模型可以降 FID，但参数、训练小时和延迟显著上升；Patch Detailer Head 更高效。

**我认为这篇的意义**：

DiP 的核心不是“再加一个后处理器”，而是提出了一个很实用的分工：**global DiT 不该被迫学习 patch 内所有纹理，局部模块才是高频细节的合适载体。** 它和 DeCo、PixelDiT 是同一方向：把全局语义和局部像素建模解耦。

**局限**：

DiP 的局部 head 使用卷积 U-Net，虽然效果强，但不再是纯 Transformer；此外它主要在 class-to-image ImageNet 上验证，论文也把 text-to-image / video 作为未来方向。

---

### 2.3 DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation

**核心问题**：像素空间里，低频语义和高频细节/噪声混在一个 DiT 中，会让 DiT 同时承担语义建模和高频重建，训练慢且易受高频噪声干扰。

**核心方法**：

DeCo 明确提出 **frequency-decoupled pixel diffusion**：

1. **DiT 只看 downsampled / patchified low-resolution input**，专注低频语义。
2. **pixel decoder 看 full-resolution noisy image**，在 DiT 的语义条件下生成高频细节和最终 pixel velocity。
3. **frequency-aware flow-matching loss** 用 JPEG/DCT 感知先验对频率成分加权。

pixel decoder 是 attention-free 的轻量线性/MLP 模块：

- 输入 full-resolution `x_t` 加位置编码，构成 dense query。
- DiT 输出 `x_low` 上采样后生成 AdaLN 的调制参数。
- pixel decoder 经过多层 decoder block 输出 pixel velocity。

训练目标为：标准 FM loss + frequency-aware FM loss + REPA alignment loss。

**频域损失细节**：

- 将预测 velocity 和真实 velocity 转到 YCbCr。
- 对 8×8 block 做 DCT。
- 用 JPEG quantization table 的倒数做自适应权重；论文默认 quality=85。
- 直觉：人眼更敏感的低频/亮度成分权重更大，不重要的高频噪声被弱化。

**关键结果**：

- 在同样 200K training steps、no-CFG 的比较中，baseline FID 61.10，JiT+REPA FID 39.06，DeCo w/o frequency loss FID 34.12，完整 DeCo FID 31.35。
- ImageNet-256 with CFG：DeCo-XL/16 报告 FID 1.62；使用 Heun 50-step、600 epoch 的设置下报告 FID 1.69，优于 JiT-H/16 的 1.86，且参数更少。
- ImageNet-512：DeCo-XL/16 报告 FID 2.22，优于 PixNerd-XL/16 的 2.84。
- text-to-image：DeCo-XXL/16 报告 GenEval overall 0.86，在表中高于 PixNerd-XXL/16 的 0.73、OmniGen2 的 0.80、SD3/FLUX/DALL-E 3 的 0.67–0.68；DPG-Bench average 81.4。
- 消融显示：decoder hidden size 32、depth 3、decoder patch size 1、AdaLN interaction、FreqFM weight 1、JPEG quality 85 是默认较优配置。

**我认为这篇的意义**：

DeCo 把 pixel diffusion 的瓶颈解释成“频率混叠”：DiT 更适合低频语义，像素级 decoder 更适合高频重建。这个视角比单纯 global/local 更明确，因为它把问题和 DCT/JPEG 感知先验联系起来。

**局限**：

它使用 REPA 和外部感知/表征先验，已不再像 JiT 那样完全极简。频域 loss 的 JPEG 权重是很实用的手工先验，但是否对非自然图像、医学/科学图像等领域通用还需要验证。

---

### 2.4 PixelDiT: Pixel Diffusion Transformers for Image Generation

**核心问题**：能否做一个 **fully transformer-based** 的 pixel diffusion，不依赖 VAE，也不依赖卷积/NeRF 风格 decoder，同时还能高效建模像素细节？

**核心方法**：

PixelDiT 提出 **dual-level DiT architecture**：

1. **Patch-level DiT**：负责全局语义结构。
2. **Pixel-level DiT / PiT blocks**：负责像素级细节 refinement。

关键机制有两个：

#### 2.4.1 Pixel-wise AdaLN

传统 AdaLN 把一个全局条件向量广播到所有 token。PixelDiT 认为这对像素级更新太粗。于是它用 patch-level semantic token 生成每个 patch 内 `p²` 个像素各自的 AdaLN 参数，让每个像素都有独立的 scale/shift/gate。

这让 pixel-level pathway 能够做“context-aligned per-pixel update”。

#### 2.4.2 Pixel Token Compaction

直接对 H×W pixel tokens 做全局 attention 太贵。PixelDiT 先把每个 patch 内的 `p²` pixel tokens 压缩成 compact patch token，再做全局 attention，之后再展开回 pixel tokens。

对于 p=16，attention sequence 从 H×W 降到 H/16 × W/16，序列长度缩小 256 倍；论文还强调这不是 VAE 式永久压缩，而是 attention 操作前后的临时压缩/展开，高频信息仍通过残差和展开层保留。

**训练与 T2I 扩展**：

- 使用 Rectified Flow velocity matching。
- 加 REPA/DINOv2 feature alignment。
- class-conditioned ImageNet 默认 PixelDiT-XL：patch-level depth 26，pixel-level depth 4，hidden 1152，pixel hidden 16，参数约 797M。
- text-to-image 使用 MM-DiT blocks 只扩展 patch-level pathway；pixel-level pathway 不直接接收 text tokens，而是通过 patch semantic tokens 间接获得文本意图。
- T2I 采用 Gemma-2 text encoder，训练 26M image-text pairs，先 512 后 1024 微调。

**关键结果**：

- ImageNet-256：PixelDiT-XL 320 epoch 报告 gFID 1.61；80 epoch 已有 gFID 2.36。
- ImageNet-512：PixelDiT 报告 gFID 1.81，recall 0.67。
- T2I 512：PixelDiT-T2I 报告 GenEval 0.78、DPG 83.7、throughput 1.07 samples/s。
- T2I 1024：报告 GenEval 0.74、DPG 83.5、throughput 0.33 samples/s。
- train-free image editing：论文用 FlowEdit 展示，PixelDiT 避免 VAE reconstruction artifact，对背景小文字等细节保存更好。

**消融重点**：

- Vanilla DiT/16 80 epoch gFID 9.84。
- 加 RoPE/RMSNorm 到 8.53。
- dual-level 但不做 token compaction 会 OOM。
- 加 pixel token compaction 到 3.50。
- 再加 pixel-wise AdaLN 到 2.36；320 epoch 到 1.61。
- 无 pixel-pathway attention 比完整模型差，说明 compact global attention 对局部更新与全局语义对齐是必要的。
- patch size 更小会加速收敛/提升质量，但随模型变大收益变小；XL 下 p=16 已接近 p=8，因此默认 p=16 是质量/计算折中。

**我认为这篇的意义**：

PixelDiT 是这组论文中最系统的“纯 Transformer pixel diffusion”方案。DiP/DeCo 用局部 decoder 解决细节，PixelDiT 则直接在 Transformer 内部设计像素级 pathway 与压缩机制。它对未来大规模 pixel-space T2I 很重要，因为它证明：**像素级建模不是一定要卷积或神经场，只要 token 组织设计得好，Transformer 也能做。**

**局限**：

方法复杂度较高，对训练 recipe、REPA、solver、CFG、patch size、depth allocation 等都敏感；论文 appendix 中也提到去掉 representation alignment 后训练会不稳定甚至发散。

---

### 2.5 PixelGen: Improving Pixel Diffusion with Perceptual Supervision

**核心问题**：JiT 的 x-prediction 已经让 pixel diffusion 更可训练，但标准逐像素 flow/pixel loss 仍会把容量浪费在感知不重要的像素上，导致样本发糊、频谱和真实图像存在差距。

**核心方法**：

PixelGen 在 JiT 的 x-prediction 基础上增加三件事：

1. **LPIPS loss**：用 VGG/LPIPS 特征监督局部纹理和细节。
2. **P-DINO loss**：用 DINOv2-B 的最后层 patch features 做 cosine alignment，监督全局语义。
3. **Noise gating**：只在低噪声阶段打开感知损失；论文设置 `t >= 0.3` 时启用，即前 30% 高噪声 timesteps 不做 perceptual supervision。

训练目标可以概括为：

`L = flow matching loss + λ1 * gate(t) * LPIPS + λ2 * gate(t) * P-DINO + REPA`

其中网络仍直接输出 clean image `x_theta`，再转成 velocity 计算 FM loss。

**为什么要 noise gating**：

高噪声 timesteps 下预测图还很模糊，此时强行让 LPIPS/DINO 特征靠近 clean image 会过约束早期去噪，降低 sample coverage。消融显示全 timestep 感知损失虽然 FID/precision 强，但 recall 下降；`τ=0.3` 能取得更好平衡。

**关键结果**：

- 200K steps no-CFG、ImageNet-256：JiT-L/16 baseline FID 23.67；加 LPIPS 后 10.00；再加 P-DINO 后 7.46；加 noise-gate 后 FID 7.53、recall 从 0.58 回升到 0.60。
- PixelGen-L/16 100K steps 即 FID 10.50；200K steps FID 7.53。
- ImageNet-256 no-CFG：PixelGen-XL/16 80 epoch 报告 FID 5.11，强于多项 latent/pixel baseline 的 no-CFG 设置。
- ImageNet-256 with CFG：PixelGen-XL/16 160 epoch 报告 FID 1.83。
- T2I：PixelGen-XXL/16 报告 GenEval overall 0.79；论文强调从零预训练仅 6 天、8×H800。

**消融重点**：

- LPIPS 权重默认 0.1：太小提升弱，太大 recall 下滑。
- P-DINO 权重默认 0.01：更大权重 FID 可能继续改善，但 recall 下降。
- DINO 最后一层最好；浅层或多层组合不如最后层，说明 P-DINO 更像语义监督而非低级纹理监督。
- REPA 与 perceptual supervision 互补：无 REPA 的 PixelGen FID 11.81 已优于有 REPA 的 JiT baseline 23.67；加 REPA 后到 7.53。

**我认为这篇的意义**：

PixelGen 最直接地指出：**x-prediction 让感知损失变得自然可用。** 如果网络输出的是 noise/velocity，LPIPS/DINO 这类 clean-image encoder 很难直接用；而 x-prediction 输出干净图像估计，刚好可以接感知监督。这把 JiT 的目标选择和 latent diffusion 里常见的 perceptual/GAN/tokenizer supervision 联系起来。

**局限**：

PixelGen 已依赖 pretrained VGG/LPIPS、DINOv2、REPA 等外部监督，不再是完全 self-contained。论文也承认 CFG 下仍落后最强 latent baseline，且 perceptual losses 与 coverage/recall 存在张力。

---

## 3. 横向对比

### 3.1 方法差异表

| 方法 | 是否用 VAE | 网络直接输出 | 核心结构 | 额外监督 | 最强/代表结果 |
|---|---|---|---|---|---|
| JiT | 否 | clean image `x` | plain ViT on large raw-pixel patches | 无额外 loss / 无 tokenizer / 无预训练 | ImageNet-256 JiT-G/16 FID 1.82；512 JiT-G/32 FID 1.78 |
| DiP | 否 | 主要是 noise/velocity 预测框架 | DiT + Patch Detailer Head，最终用 local Conv U-Net | 使用 DDT/representation alignment setup | ImageNet-256 FID 1.79；较 PixelFlow 报告高得多的速度优势 |
| DeCo | 否 | velocity | low-res DiT + full-res pixel decoder | REPA + JPEG/DCT frequency-aware FM loss | ImageNet-256 FID 1.62；512 FID 2.22；T2I GenEval 0.86 |
| PixelDiT | 否 | velocity | patch-level DiT + pixel-level PiT；pixel-wise AdaLN + token compaction | REPA/DINOv2 alignment | ImageNet-256 gFID 1.61；512 gFID 1.81；T2I 1024 GenEval 0.74 |
| PixelGen | 否 | clean image `x` | JiT-style x-prediction backbone | LPIPS + P-DINO + noise gate + REPA | no-CFG FID 5.11/80ep；CFG FID 1.83；T2I GenEval 0.79 |

### 3.2 共同趋势

#### 趋势 1：x-prediction 成为 pixel diffusion 的关键起点

JiT 证明在高维像素空间中，网络直接预测 clean image 比预测 noise/velocity 更合理。PixelGen 进一步说明 x-prediction 不只是数值稳定，还让感知损失可以直接作用在网络输出上。

不过 PixelDiT 和 DeCo 也表明，不是所有强方法都必须 x-prediction。只要结构上把低频/高频、patch/pixel 分工做好，velocity prediction 仍可以很强。

#### 趋势 2：大 patch 是效率基础，但必须补局部细节

JiT 用大 patch 获得低 token 数；DiP、DeCo、PixelDiT 都接受这个效率出发点，但认为只用 patch-level DiT 会缺局部纹理。因此三者分别补了：

- DiP：Patch Detailer Head / local Conv U-Net。
- DeCo：full-resolution attention-free pixel decoder。
- PixelDiT：pixel-level PiT + pixel-wise AdaLN。

这说明 pixel diffusion 的关键不是“要不要大 patch”，而是：**大 patch 之后如何把 patch 内细节还回来。**

#### 趋势 3：global semantics 与 local/frequency detail 需要解耦

DiP 是 global-local 解耦；DeCo 是 low-frequency/high-frequency 解耦；PixelDiT 是 patch-level/pixel-level 解耦。三者的语言不同，但结构思想相近：

- DiT/Transformer 擅长全局语义和低频结构。
- 局部模块/像素 pathway 擅长纹理、高频、边缘、patch 内细节。

#### 趋势 4：外部视觉先验仍然很重要

虽然这些论文都强调摆脱 VAE，但很多结果仍用到外部先验：

- PixelGen 用 LPIPS/VGG、DINOv2、REPA。
- PixelDiT 用 REPA/DINOv2 alignment。
- DeCo 用 REPA 和 JPEG perceptual frequency prior。
- DiP 用 DDT/representation alignment 的训练设置。
- JiT 是其中最 self-contained 的，但它的性能上限和训练效率也更依赖 scaling。

所以更准确地说，它们摆脱的是 **VAE tokenizer bottleneck**，而不是完全摆脱视觉先验。

### 3.3 和 latent diffusion 的关系

这组论文不是简单宣称 latent diffusion 过时，而是指出 latent diffusion 的两阶段 pipeline 有结构性问题：

- VAE reconstruction upper bound 限制生成质量。
- VAE artifact 在编辑、局部文字、细节保真上会被后续 diffusion 放大。
- VAE training 本身常需要 perceptual/GAN losses，pipeline 不完全 end-to-end。

pixel diffusion 的优势是：

- 原生无 VAE reconstruction artifact。
- 训练和采样都在同一空间，目标更直接。
- 对科学图像、医学图像、特殊数据等没有现成 tokenizer 的场景更有吸引力。

但代价是：

- 像素空间维度高，训练 recipe 更敏感。
- 如果不引入结构解耦或感知监督，模型容易慢、糊、细节差。
- 很多强结果仍需要外部 DINO/LPIPS/REPA/JPEG 先验。

---

## 4. 对后续研究/复现的启发

### 4.1 最值得优先复现的 baseline

如果目标是快速验证 pixel diffusion：

1. **先复现 JiT-L/16 或 JiT-B/16 的 x-prediction + v-loss**，因为概念最清楚，变量最少。
2. 再加 **PixelGen 的 LPIPS + P-DINO + noise gate**，因为是在 JiT 基础上最直接的 loss 改造。
3. 如果关注高分辨率效率，再看 **DeCo 的 pixel decoder** 或 **DiP 的 Patch Detailer Head**。
4. 如果目标是纯 Transformer 大模型路线，则看 **PixelDiT**，但实现复杂度和训练 recipe 成本最高。

### 4.2 对 diffusion cache / 推理加速的启发

这些 pixel-space 方法对 caching 有一些新机会：

- **DiP/DeCo/PixelDiT 都显式拆出全局语义路径和局部细节路径**。在采样后期，全局语义变化可能比局部细节变化更慢，因此 patch-level semantic tokens 或 global DiT blocks 可能比 pixel decoder / PiT 细节层更适合 cache。
- **DeCo 的 DiT 输出低频语义**，理论上更平滑、更稳定，适合做 timestep cache 或 feature forecast；而 pixel decoder 负责高频，cache 误差更容易表现成纹理 artifact。
- **PixelDiT 的 pixel token compaction 产生 compact patch tokens**，这可能是 cache 的自然对象：它比 raw pixel tokens 更短，又携带全局 attention 信息。
- **PixelGen 的 perceptual supervision 可能改变特征时间平滑性**。低噪声阶段感知 loss 打开后，特征可能更贴近 clean manifold，但也可能使局部纹理更新更敏感。cache 策略需要区分高噪声/低噪声阶段。
- **JiT 的 x-prediction 输出 clean image**，可考虑直接缓存/预测 `x_pred` 或中间 denoised representation，而不是缓存 noise/velocity，但需要控制 `(1-t)` 分母带来的低噪声误差放大。

一个可研究方向：在 DeCo/PixelDiT 这类分层结构中，对 global semantic branch 使用较激进 cache，对 local/frequency branch 使用保守 cache，形成 **frequency-aware / branch-aware caching**。

### 4.3 可能的综述主线

可以把这几篇归纳成一个综述标题：

> From Latent Bottlenecks to Pixel Manifolds: Prediction Targets, Frequency Decoupling, and Perceptual Supervision for Pixel-Space Diffusion

推荐组织结构：

1. VAE bottleneck 与 pixel diffusion 的复兴。
2. 预测目标：epsilon/v 到 x-prediction。
3. 结构解耦：global-local、low-high frequency、patch-pixel。
4. 训练监督：REPA、LPIPS、DINO、frequency-aware loss。
5. 计算效率：large patch、token compaction、lightweight decoder。
6. 开放问题：外部先验依赖、CFG/recall trade-off、T2I scaling、cache/quant/pruning 部署。

---

## 5. 个人总结

这 5 篇论文合起来说明，pixel diffusion 已经不再只是“把扩散模型搬回像素空间”的朴素尝试。它们正在形成一套较完整的设计原则：

- **目标上**：尽量让网络预测更接近图像流形的量，JiT/PixelGen 强调 x-prediction。
- **结构上**：让 Transformer 做它擅长的低频/全局语义，让局部或像素路径处理高频细节。
- **监督上**：逐像素 loss 不够，需要 perceptual、semantic、frequency-aware 或 representation alignment。
- **计算上**：保持大 patch 的 token 效率，同时通过 local head、pixel decoder 或 token compaction 补足 patch 内建模。

我对这组论文的排序理解是：

- **最基础的理论/范式转折**：JiT。
- **最工程简洁的结构增强**：DiP。
- **最明确的频率视角**：DeCo。
- **最系统的纯 Transformer 架构**：PixelDiT。
- **最直接的 loss/监督改进**：PixelGen。

综合看，未来 pixel diffusion 的核心竞争点可能不在“是否无 VAE”本身，而在于：**无 VAE 之后，如何用更好的预测目标、结构解耦和感知监督把 raw pixel 的高维复杂性重新组织成模型可学习、可采样、可部署的形式。**
