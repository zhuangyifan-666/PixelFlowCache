from __future__ import annotations

import time
from typing import Any

import torch

from pfc.cache.dicache_policy import DCTAResult, DiCachePolicy
from pfc.diagnostics.tensor_stats import l2_norm
from pfc.eval.jit_runtime import (
    JiTRuntimeConfig,
    cfg_enabled,
    combine_cfg_velocity,
    xpred_to_velocity,
)


class JiTDiCacheExecutor:
    """Clean-room split-forward executor for the released JiT model interface."""

    _REQUIRED_ATTRIBUTES = (
        "t_embedder",
        "y_embedder",
        "x_embedder",
        "pos_embed",
        "blocks",
        "final_layer",
        "unpatchify",
        "patch_size",
        "in_context_len",
        "in_context_start",
        "in_context_posemb",
        "feat_rope",
        "feat_rope_incontext",
        "num_classes",
    )

    def __init__(self, net: Any) -> None:
        missing = [name for name in self._REQUIRED_ATTRIBUTES if not hasattr(net, name)]
        if missing:
            raise AttributeError(f"JiT model is missing required DiCache attributes: {missing}")
        self.net = net
        self.total_blocks = len(net.blocks)
        if self.total_blocks <= 1:
            raise ValueError("JiT DiCache requires at least two Transformer blocks")

    def prepare_common_input(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        t_emb = self.net.t_embedder(t)
        h0 = self.net.x_embedder(x)
        pos_embed = self.net.pos_embed.to(device=h0.device, dtype=h0.dtype)
        h0 = h0 + pos_embed
        if h0.ndim != 3:
            raise ValueError(f"JiT x_embedder must produce [B,N,C], got {tuple(h0.shape)}")
        return h0, t_emb, int(h0.shape[1])

    def prepare_branch_condition(
        self,
        t_emb: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_emb = self.net.y_embedder(labels).to(device=t_emb.device, dtype=t_emb.dtype)
        return y_emb, t_emb + y_emb

    def run_blocks_range(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        label_embedding: torch.Tensor,
        *,
        start: int,
        end: int,
        num_image_tokens: int,
    ) -> torch.Tensor:
        if not 0 <= start <= end <= self.total_blocks:
            raise ValueError(
                f"invalid JiT block range [{start}, {end}) for {self.total_blocks} blocks"
            )
        context_len = int(self.net.in_context_len)
        context_start = int(self.net.in_context_start)
        image_only_len = num_image_tokens
        with_context_len = num_image_tokens + context_len
        sequence_len = int(hidden.shape[1])
        if sequence_len not in {image_only_len, with_context_len}:
            raise ValueError(
                "JiT hidden sequence must contain image tokens with at most one context prefix: "
                f"got {sequence_len}, expected {image_only_len} or {with_context_len}"
            )
        context_present = context_len > 0 and sequence_len == with_context_len

        for index in range(start, end):
            if context_len > 0 and index == context_start:
                if context_present:
                    raise ValueError("JiT in-context tokens would be inserted more than once")
                context_tokens = label_embedding.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                ).unsqueeze(1).repeat(1, context_len, 1)
                context_posemb = self.net.in_context_posemb.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                context_tokens = context_tokens + context_posemb
                hidden = torch.cat([context_tokens, hidden], dim=1)
                context_present = True
            elif context_len > 0 and index > context_start and not context_present:
                raise ValueError(
                    "resuming JiT after in_context_start requires the existing context prefix"
                )
            rope = self.net.feat_rope if index < context_start else self.net.feat_rope_incontext
            hidden = self.net.blocks[index](hidden, condition, rope)
        return hidden

    def extract_image_tokens(
        self,
        hidden: torch.Tensor,
        num_image_tokens: int,
    ) -> torch.Tensor:
        sequence_len = int(hidden.shape[1])
        if sequence_len == num_image_tokens:
            return hidden
        expected_with_context = num_image_tokens + int(self.net.in_context_len)
        if int(self.net.in_context_len) > 0 and sequence_len == expected_with_context:
            return hidden[:, -num_image_tokens:]
        raise ValueError(
            "cannot extract JiT image-token tail: "
            f"got sequence length {sequence_len}, expected {num_image_tokens} "
            f"or {expected_with_context}"
        )

    def finalize_output(
        self,
        image_tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        patches = self.net.final_layer(image_tokens, condition)
        return self.net.unpatchify(patches, self.net.patch_size)

    def forward_split_full(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        *,
        probe_depth: int,
    ) -> torch.Tensor:
        if not 1 <= probe_depth < self.total_blocks:
            raise ValueError("probe_depth must split the JiT block stack")
        h0, t_emb, num_image_tokens = self.prepare_common_input(x, t)
        y_emb, condition = self.prepare_branch_condition(t_emb, labels)
        probe_hidden = self.run_blocks_range(
            h0,
            condition,
            y_emb,
            start=0,
            end=probe_depth,
            num_image_tokens=num_image_tokens,
        )
        full_hidden = self.run_blocks_range(
            probe_hidden,
            condition,
            y_emb,
            start=probe_depth,
            end=self.total_blocks,
            num_image_tokens=num_image_tokens,
        )
        return self.finalize_output(
            self.extract_image_tokens(full_hidden, num_image_tokens),
            condition,
        )


def sample_jit_dicache(
    model: Any,
    labels: torch.Tensor,
    noise: torch.Tensor,
    config: JiTRuntimeConfig,
    *,
    executor: JiTDiCacheExecutor,
    policy: DiCachePolicy,
    mode: str = "dicache_style",
    solver_stage: str = "euler",
    collect_step_records: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Euler JiT sampling with one shared cond/uncond DiCache decision per step."""

    if solver_stage != "euler":
        raise NotImplementedError("adapted JiT DiCache currently supports solver_stage='euler' only")
    if config.steps != policy.total_steps:
        raise ValueError("JiT runtime steps must match DiCachePolicy.total_steps")
    if executor.total_blocks != policy.total_blocks:
        raise ValueError("JiT executor blocks must match DiCachePolicy.total_blocks")

    outputs: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    device = noise.device
    timesteps = torch.linspace(0.0, 1.0, config.steps + 1, device=device, dtype=noise.dtype)

    for batch_start in range(0, config.num_samples, config.batch_size):
        batch_end = min(batch_start + config.batch_size, config.num_samples)
        z = noise[batch_start:batch_end].clone()
        batch_labels = labels[batch_start:batch_end]
        null_labels = torch.full_like(batch_labels, model.num_classes)
        policy.clear_batch()

        for step_idx in range(config.steps):
            t_scalar = timesteps[step_idx]
            t_next_scalar = timesteps[step_idx + 1]
            dt = t_next_scalar - t_scalar
            t_value = step_idx / config.steps
            t_next_value = (step_idx + 1) / config.steps
            dt_value = 1.0 / config.steps
            t = t_scalar.expand(z.shape[0], 1, 1, 1)
            flat_t = t.flatten()
            cfg_active = cfg_enabled(t_value, config.interval_min, config.interval_max)
            cfg_scale_interval = config.cfg if cfg_active else 1.0

            branch_labels = {"cond": batch_labels, "uncond": null_labels}
            branch_inputs: dict[str, torch.Tensor] = {}
            branch_conditions: dict[str, torch.Tensor] = {}
            branch_label_embeddings: dict[str, torch.Tensor] = {}
            branch_probe_hidden: dict[str, torch.Tensor] = {}
            branch_probe_images: dict[str, torch.Tensor] = {}

            started = time.perf_counter()
            shared_common: tuple[torch.Tensor, torch.Tensor, int] | None = None
            if policy.share_cfg_prefix:
                shared_common = executor.prepare_common_input(z, flat_t)
            num_image_tokens: int | None = None
            for branch, current_labels in branch_labels.items():
                if shared_common is None:
                    h0, t_emb, branch_num_image_tokens = executor.prepare_common_input(
                        z, flat_t
                    )
                else:
                    h0, t_emb, branch_num_image_tokens = shared_common
                if num_image_tokens is None:
                    num_image_tokens = branch_num_image_tokens
                elif branch_num_image_tokens != num_image_tokens:
                    raise ValueError("cond/uncond JiT prefix token counts differ")
                y_emb, condition = executor.prepare_branch_condition(t_emb, current_labels)
                branch_inputs[branch] = h0
                branch_label_embeddings[branch] = y_emb
                branch_conditions[branch] = condition
                probe_hidden = executor.run_blocks_range(
                    h0,
                    condition,
                    y_emb,
                    start=0,
                    end=policy.probe_depth,
                    num_image_tokens=branch_num_image_tokens,
                )
                branch_probe_hidden[branch] = probe_hidden
                branch_probe_images[branch] = executor.extract_image_tokens(
                    probe_hidden, branch_num_image_tokens
                )
            assert num_image_tokens is not None
            policy.add_host_dispatch_time(
                "probe_host_dispatch_time_sec", time.perf_counter() - started
            )

            decision = policy.decide(
                step_idx=step_idx,
                input_feature=branch_inputs["cond"],
                probe_features=branch_probe_images,
            )
            branch_full_images: dict[str, torch.Tensor] = {}
            dcta_results: dict[str, DCTAResult] = {}
            if decision.decision == "full":
                started = time.perf_counter()
                for branch in branch_labels:
                    full_hidden = executor.run_blocks_range(
                        branch_probe_hidden[branch],
                        branch_conditions[branch],
                        branch_label_embeddings[branch],
                        start=policy.probe_depth,
                        end=executor.total_blocks,
                        num_image_tokens=num_image_tokens,
                    )
                    full_image = executor.extract_image_tokens(full_hidden, num_image_tokens)
                    branch_full_images[branch] = full_image
                    policy.record_refresh(
                        branch,
                        input_feature=branch_inputs[branch],
                        probe_feature=branch_probe_images[branch],
                        full_feature=full_image,
                        step_idx=step_idx,
                    )
                policy.add_host_dispatch_time(
                    "deep_compute_host_dispatch_time_sec", time.perf_counter() - started
                )
            else:
                started = time.perf_counter()
                for branch in branch_labels:
                    result = policy.approximate_residual(
                        branch,
                        current_probe_residual=(
                            branch_probe_images[branch] - branch_inputs[branch]
                        ),
                        degenerate=decision.dcta_degenerate[branch],
                    )
                    dcta_results[branch] = result
                    branch_full_images[branch] = branch_inputs[branch] + result.residual
                policy.add_host_dispatch_time(
                    "dcta_host_dispatch_time_sec", time.perf_counter() - started
                )

            started = time.perf_counter()
            x_pred = {
                branch: executor.finalize_output(
                    branch_full_images[branch], branch_conditions[branch]
                )
                for branch in branch_labels
            }
            policy.add_host_dispatch_time(
                "final_layer_host_dispatch_time_sec", time.perf_counter() - started
            )
            policy.finish_step(
                decision,
                input_feature=branch_inputs["cond"],
                probe_features=branch_probe_images,
                dcta_results=dcta_results,
                t=t_value,
            )

            v_cond = xpred_to_velocity(x_pred["cond"], z, t, model.t_eps)
            v_uncond = xpred_to_velocity(x_pred["uncond"], z, t, model.t_eps)
            v_cfg = combine_cfg_velocity(v_cond, v_uncond, cfg_scale_interval)
            if collect_step_records:
                records.append(
                    {
                        "record_type": "jit_step",
                        "mode": mode,
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "step_idx": step_idx,
                        "t": t_value,
                        "t_next": t_next_value,
                        "dt": dt_value,
                        "cfg_enabled": cfg_active,
                        "cfg_scale": config.cfg,
                        "dicache_decision": decision.decision,
                        "dicache_reason": decision.reason,
                        "velocity_l2": l2_norm(v_cfg),
                    }
                )
            z = z + dt * v_cfg
        policy.finalize_batch_statistics()
        outputs.append(z.detach())
    return torch.cat(outputs, dim=0), records
