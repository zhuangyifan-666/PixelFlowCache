from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from pfc.eval.jit_runtime import JiTRuntimeConfig


class FakeXEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x.flatten(2).transpose(1, 2)


class FakeTimestepEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.stack([t, 2.0 * t], dim=-1)


class FakeLabelEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(4, 2)
        with torch.no_grad():
            self.embedding_table.weight.copy_(
                torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [-0.2, -0.1]])
            )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class FakeRope:
    def __init__(self, value: float) -> None:
        self.value = value


class FakeBlock(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.calls = 0
        self.sequence_lengths: list[int] = []
        self.conditions: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope: FakeRope) -> torch.Tensor:
        self.calls += 1
        self.sequence_lengths.append(int(x.shape[1]))
        self.conditions.append(c.detach().clone())
        return x + (self.index + 1) * 0.05 * c.unsqueeze(1) + rope.value * 0.01


class FakeFinalLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + 0.1 * c.unsqueeze(1)


class FakeJiT(nn.Module):
    def __init__(self, *, depth: int = 6, in_context_start: int = 2, in_context_len: int = 2) -> None:
        super().__init__()
        self.t_embedder = FakeTimestepEmbedder()
        self.y_embedder = FakeLabelEmbedder()
        self.x_embedder = FakeXEmbedder()
        self.pos_embed = nn.Parameter(torch.full((1, 4, 2), 0.01), requires_grad=False)
        self.blocks = nn.ModuleList(FakeBlock(index) for index in range(depth))
        self.final_layer = FakeFinalLayer()
        self.patch_size = 1
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.in_context_posemb = nn.Parameter(torch.full((1, in_context_len, 2), 0.02))
        self.feat_rope = FakeRope(1.0)
        self.feat_rope_incontext = FakeRope(2.0)
        self.num_classes = 3

    def unpatchify(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        assert patch_size == 1
        batch, tokens, channels = x.shape
        side = int(tokens**0.5)
        assert side * side == tokens
        return x.transpose(1, 2).reshape(batch, channels, side, side)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y).to(device=t_emb.device, dtype=t_emb.dtype)
        c = t_emb + y_emb
        hidden = self.x_embedder(x)
        hidden = hidden + self.pos_embed.to(device=hidden.device, dtype=hidden.dtype)
        for index, block in enumerate(self.blocks):
            if self.in_context_len > 0 and index == self.in_context_start:
                context = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                context = context + self.in_context_posemb.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                hidden = torch.cat([context, hidden], dim=1)
            rope = self.feat_rope if index < self.in_context_start else self.feat_rope_incontext
            hidden = block(hidden, c, rope)
        hidden = hidden[:, self.in_context_len :]
        return self.unpatchify(self.final_layer(hidden, c), self.patch_size)


class FakeDenoiser:
    def __init__(self, net: FakeJiT | None = None) -> None:
        self.net = net or FakeJiT()
        self.num_classes = self.net.num_classes
        self.t_eps = 0.05


def runtime_config(*, steps: int, num_samples: int = 2, batch_size: int = 2) -> JiTRuntimeConfig:
    return JiTRuntimeConfig(
        jit_dir=Path("unused"),
        ckpt_dir=Path("unused"),
        run_id="fake",
        run_dir=Path("unused"),
        preview_dir=Path("unused"),
        img_size=2,
        num_samples=num_samples,
        batch_size=batch_size,
        steps=steps,
        cfg=3.0,
        interval_min=0.1,
        interval_max=1.0,
    )
