# 05. Stage 2：Baseline 体系与公平比较

一个 cache 加速工作最容易被审稿人质疑 baseline 不公平。Stage 2 的目标是建立从简单到强的 baseline 梯度，并规定调参、速度预算和质量预算。

## 1. Baseline 总表

| Baseline | 类型 | 实现成本 | 用途 |
|---|---|---:|---|
| Fewer steps | 非 cache | 低 | 证明 cache 比直接减少步数好 |
| Fixed interval block cache | cache | 低 | 最基本对照 |
| Layer-wise cache | cache | 中 | 对比 layer selection |
| Token-wise cache / ToCa-like | cache | 中高 | 对比 fine-grained token cache |
| TeaCache-like | output/timestep cache | 中 | 对比 timestep embedding / output diff proxy |
| DiCache-like | online probe | 中 | 对比 adaptive runtime cache |
| FoCa-like | forecast + calibrate | 高 | 对比 feature trajectory / ODE cache |
| PixelFlowCache | proposed | 中高 | 主方法 |

## 2. Fewer-step baseline

### 目的

如果你的方法 2x 加速，但 50 steps 直接降到 25 steps 也差不多质量，那么 cache 的意义不大。必须比较。

### 设定

- Full: 100 steps；
- Fewer-step: 75 / 50 / 33 / 25 steps；
- 保持同 solver、CFG、time shift；
- 不改模型。

### 报告方式

画 quality-latency curve，而不是只报一个点。

## 3. Fixed interval block cache

### 策略

每 K 步 full compute 一次，其余步复用某些 block output：

```text
if step % K == 0:
    compute block normally and store
else:
    reuse cached block output
```

### 变量

- K = 2 / 3 / 4 / 5；
- cache layers：all / shallow / middle / deep / selected；
- cache tensor：block output / attn output / MLP output。

### 意义

这是最低门槛。PixelFlowCache 必须显著优于它。

## 4. Layer-wise cache

### 策略

根据 Stage 1 的 layer smoothness 排序选择可缓存层，但不使用 pixel-specific criterion。

### 对照目的

如果 layer-wise cache 已经很强，说明你的新意必须来自 velocity/frequency/solver，而不是简单层选择。

## 5. ToCa-like token-wise cache

### 核心思想

不同 token 的 cache sensitivity 不同，按 token 决定复用/重算。

### 移植策略

- patch tokens：按 temporal change、attention score、edge proxy 排序；
- 每层选 top-p sensitive tokens 重算，其余缓存；
- 对 PixelDiT pixel pathway 要谨慎，因为 token 数和 memory pattern 可能导致 latency 下降不明显。

### 注意

ToCa-like baseline 不一定完全复现原论文所有细节，但要实现同类思想，并在文档中说明差异。

## 6. TeaCache-like baseline

### 核心思想

利用 timestep embedding 或 low-cost input-side signal 预测相邻输出差异，从而决定是否 cache。

### 移植到 pixel flow

可以用：

\[
D_t=\|emb(t)-emb(t-1)\|.
\]

或：

\[
D_t=\|x_t-x_{t-1}\|/\|x_{t-1}\|.
\]

当 \(D_t<\tau\) 时 cache。

### 对照目的

证明只用 timestep/input signal 不够，需要 velocity/frequency/branch-aware criterion。

## 7. DiCache-like online probe

### 核心思想

用浅层 feature 的在线变化预测深层/输出变化。

### 移植实现

每步先计算前 L_probe 层：

\[
P_t=\|h^{shallow}_t-h^{shallow}_{t-1}\|.
\]

若 \(P_t<\tau\)，则复用后续深层 cache；否则 full compute。

### 公平性

probe 层的计算必须计入 latency。

### 对照目的

证明你的 online probe 不只是浅层变化，而是 velocity/frequency/solver normalized。

## 8. FoCa-like forecast + calibrate

### 核心思想

不简单复用上一帧 feature，而是用历史 feature 预测当前 feature，再做校正。

### 简化实现

一阶：

\[
\hat{h}_t=h_{t-1}+(h_{t-1}-h_{t-2}).
\]

二阶 / BDF-like：

\[
\hat{h}_t=a h_{t-1}+b h_{t-2}+c h_{t-3}.
\]

校正：

\[
h'_t=\alpha_t \hat{h}_t+\beta_t.
\]

### 对照目的

如果 FoCa-like 很强，你的方法需要证明 pixel-specific error criterion 仍有优势。

## 9. Output cache baseline

### 策略

直接复用上一 timestep 的 model output：

- x-pred：复用 \(\hat{x}_{0,t-1}\)，再转换为当前 velocity；
- v-pred：复用 \(v_{t-1}\)。

### 价值

这是低成本但通常质量差的 baseline，可以展示 feature cache 的必要性。

## 10. 公平调参协议

每个 baseline 都按 latency budget 调参，而不是随意挑最好结果。

### 推荐预算

| 预算 | 目标 speedup |
|---|---:|
| mild | 1.2x - 1.5x |
| medium | 1.5x - 2.0x |
| aggressive | 2.0x - 3.0x |
| extreme | >3.0x |

每个预算下比较质量。

## 11. 统一报告格式

每个模型每个 baseline 报：

- latency speedup；
- FLOPs reduction；
- memory overhead；
- FID / IS / sFID；
- LPIPS-to-full；
- DINO similarity-to-full；
- frequency error；
- failure examples。

## 12. 负控实验

为了证明不是偶然，需要加入负控：

1. random layer cache；
2. random token cache；
3. reverse frequency weighting；
4. disabling CFG interval；
5. x-pred 模型不做 velocity normalization。

如果 PixelFlowCache 明显优于这些负控，可信度更高。

## 13. Stage 2 产出

1. baseline implementation；
2. baseline tuning table；
3. quality-latency Pareto curves；
4. 每个模型的 baseline ranking；
5. 主方法需要超越的 strong baseline 列表。
