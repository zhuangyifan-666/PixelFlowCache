from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import torch


HistoryKey = tuple[int, str, str]


@dataclass(frozen=True)
class BoundaryHistoryItem:
    step_idx: int
    t: float
    boundary_input: torch.Tensor
    boundary_output: torch.Tensor


class JiTFreshBoundaryHistory:
    def __init__(self, max_history: int = 3) -> None:
        if max_history < 2:
            raise ValueError("max_history must be at least two")
        self.max_history = int(max_history)
        self._items: dict[HistoryKey, deque[BoundaryHistoryItem]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )

    def append(
        self,
        *,
        sample_global_index: int,
        branch: str,
        boundary_plan: str,
        step_idx: int,
        t: float,
        boundary_input: torch.Tensor,
        boundary_output: torch.Tensor,
    ) -> None:
        key = self._key(sample_global_index, branch, boundary_plan)
        items = self._items[key]
        if items and step_idx <= items[-1].step_idx:
            raise ValueError("fresh history step indices must increase strictly")
        items.append(
            BoundaryHistoryItem(
                step_idx=int(step_idx),
                t=float(t),
                boundary_input=boundary_input.detach().clone(),
                boundary_output=boundary_output.detach().clone(),
            )
        )

    def select_age(
        self,
        *,
        sample_global_index: int,
        branch: str,
        boundary_plan: str,
        current_step_idx: int,
        age: int,
    ) -> BoundaryHistoryItem | None:
        if age <= 0:
            raise ValueError("history age must be positive; age 0 uses the current fresh capture")
        target_step = int(current_step_idx) - int(age)
        item = next(
            (
                value
                for value in reversed(self._items.get(self._key(sample_global_index, branch, boundary_plan), ()))
                if value.step_idx == target_step
            ),
            None,
        )
        if item is not None and item.step_idx >= current_step_idx:
            raise AssertionError("future leakage: selected history is not older than the current step")
        return item

    def latest_two(
        self,
        *,
        sample_global_index: int,
        branch: str,
        boundary_plan: str,
        current_step_idx: int,
    ) -> tuple[BoundaryHistoryItem, BoundaryHistoryItem] | None:
        past = [
            item
            for item in self._items.get(self._key(sample_global_index, branch, boundary_plan), ())
            if item.step_idx < current_step_idx
        ]
        if len(past) < 2:
            return None
        first, second = past[-2], past[-1]
        if not first.step_idx < second.step_idx < current_step_idx:
            raise AssertionError("future leakage in Taylor history")
        return first, second

    def taylor_order_1(
        self,
        *,
        sample_global_index: int,
        branch: str,
        boundary_plan: str,
        current_step_idx: int,
    ) -> torch.Tensor | None:
        pair = self.latest_two(
            sample_global_index=sample_global_index,
            branch=branch,
            boundary_plan=boundary_plan,
            current_step_idx=current_step_idx,
        )
        if pair is None:
            return None
        first, second = pair
        denominator = first.step_idx - second.step_idx
        if denominator == 0:
            raise ValueError("Taylor history contains duplicate step indices")
        weight_first = (current_step_idx - second.step_idx) / denominator
        weight_second = (current_step_idx - first.step_idx) / (second.step_idx - first.step_idx)
        return weight_first * first.boundary_output + weight_second * second.boundary_output

    def clear_sample(self, sample_global_index: int) -> None:
        for key in [key for key in self._items if key[0] == int(sample_global_index)]:
            del self._items[key]

    def clear(self) -> None:
        self._items.clear()

    def item_count(self) -> int:
        return sum(len(items) for items in self._items.values())

    @staticmethod
    def _key(sample_global_index: int, branch: str, boundary_plan: str) -> HistoryKey:
        return int(sample_global_index), str(branch), str(boundary_plan)
