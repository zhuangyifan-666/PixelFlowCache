from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


DEFAULT_JIT_PLAN_NAMES = ("early", "middle", "late", "early_middle", "whole")
DIAGNOSTIC_JIT_PLAN_NAMES = ("middle_late",)


@dataclass(frozen=True)
class JiTBoundaryPlan:
    name: str
    start_block: int
    end_block: int
    description: str

    def validate(self, num_blocks: int) -> None:
        if not 0 <= self.start_block < self.end_block <= num_blocks:
            raise ValueError(
                f"invalid JiT boundary plan {self.name}: "
                f"[{self.start_block}, {self.end_block}) for {num_blocks} blocks"
            )

    @property
    def skipped_block_count(self) -> int:
        return self.end_block - self.start_block

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_jit_boundary_plans(
    num_blocks: int,
    names: Iterable[str] | None = None,
) -> list[JiTBoundaryPlan]:
    if num_blocks < 3:
        raise ValueError("JiT Stage-1 boundary plans require at least three blocks")
    split_a = max(1, num_blocks // 3)
    split_b = max(split_a + 1, 2 * num_blocks // 3)
    split_b = min(split_b, num_blocks - 1)
    available = {
        "early": JiTBoundaryPlan("early", 0, split_a, "First third of JiT blocks."),
        "middle": JiTBoundaryPlan("middle", split_a, split_b, "Middle third of JiT blocks."),
        "late": JiTBoundaryPlan("late", split_b, num_blocks, "Final third of JiT blocks."),
        "early_middle": JiTBoundaryPlan(
            "early_middle", 0, split_b, "First two thirds of JiT blocks."
        ),
        "middle_late": JiTBoundaryPlan(
            "middle_late", split_a, num_blocks, "Diagnostic final two-thirds plan."
        ),
        "whole": JiTBoundaryPlan("whole", 0, num_blocks, "Whole JiT block stack."),
    }
    selected_names = list(names or DEFAULT_JIT_PLAN_NAMES)
    if not selected_names:
        raise ValueError("at least one JiT boundary plan must be selected")
    unknown = sorted(set(selected_names) - set(available))
    if unknown:
        raise ValueError(f"unknown JiT boundary plans: {unknown}")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("JiT boundary plans must be unique")
    resolved = [available[name] for name in selected_names]
    for plan in resolved:
        plan.validate(num_blocks)
    return resolved
