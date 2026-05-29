# 11. 参考链接与基线清单

以下链接用于定位官方代码、论文和主要 cache baseline。建议在正式论文中使用 BibTeX，并记录访问日期与 commit hash。

- **JiT GitHub**: https://github.com/LTH14/JiT
- **JiT paper**: https://arxiv.org/abs/2511.13720
- **DeCo GitHub**: https://github.com/Zehong-Ma/DeCo
- **DeCo paper**: https://arxiv.org/html/2511.19365v1
- **DeCo project**: https://zehong-ma.github.io/DeCo/
- **PixelDiT GitHub**: https://github.com/NVlabs/PixelDiT
- **PixelDiT project**: https://pixeldit.github.io/
- **PixelDiT paper**: https://arxiv.org/html/2511.20645v1
- **PixelGen GitHub**: https://github.com/Zehong-Ma/PixelGen
- **PixelGen paper**: https://arxiv.org/html/2602.02493v2
- **PixelGen project**: https://zehong-ma.github.io/PixelGen/
- **ToCa paper**: https://arxiv.org/html/2410.05317v3
- **ToCa GitHub**: https://github.com/Shenyi-Z/ToCa
- **TeaCache paper**: https://arxiv.org/html/2411.19108v2
- **TeaCache GitHub**: https://github.com/ali-vilab/TeaCache
- **DiCache paper**: https://arxiv.org/abs/2508.17356
- **DiCache GitHub**: https://github.com/Bujiazi/DiCache
- **FoCa paper**: https://arxiv.org/abs/2508.16211
- **Frequency-aware caching paper**: https://arxiv.org/html/2510.08669v1


## 需要重点复查的 baseline

### ToCa / token-wise cache

关注点：token sensitivity、layer type/depth 分配、training-free、真实 latency 与 FlashAttention 兼容性。

### TeaCache / timestep embedding aware cache

关注点：不用昂贵 model output，而用低成本输入/时间信号估计输出变化。

### DiCache / online probe

关注点：shallow-layer online probe、runtime adaptive、multi-step trajectory alignment。

### FoCa / forecast then calibrate

关注点：feature caching as ODE、历史特征预测、calibration 解决大 skip interval 误差。

### Frequency-aware caching

关注点：如果已有工作也使用 frequency 视角，需要在 related work 中区分：你的对象是 pixel-space flow models，并把 frequency 与 output parameterization / solver / branch 绑定。

## 建议补充阅读

- Flow Matching / Rectified Flow 采样和 solver；
- DiT / PixArt / FLUX cache acceleration；
- DCT/FFT perceptual image error；
- LPIPS / DINO perceptual metrics；
- efficient Transformer inference and FlashAttention。
