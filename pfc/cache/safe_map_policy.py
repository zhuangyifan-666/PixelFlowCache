from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class SafeMapCachePolicy:
    policy_name = "SafeMapCachePolicy"

    def __init__(
        self,
        *,
        safe_map: dict[str, Any] | None = None,
        enabled: bool = True,
        model_name: str | None = None,
        safe_map_path: Path | str | None = None,
        boundary_groups: dict[str, list[str]] | None = None,
        module_to_boundary: dict[str, str] | None = None,
        max_age: int | None = None,
        solver_stages: set[str] | list[str] | tuple[str, ...] | None = None,
        branches: set[str] | list[str] | tuple[str, ...] | None = None,
        fallback_to_global_branch: bool = True,
        default_safe: bool = False,
        debug_jsonl_path: Path | str | None = None,
    ) -> None:
        if safe_map is None and safe_map_path is None:
            raise ValueError("SafeMapCachePolicy requires safe_map or safe_map_path")
        self.safe_map_path = Path(safe_map_path) if safe_map_path is not None else None
        self.safe_map = dict(safe_map or self._load_json(self.safe_map_path))
        self.enabled = bool(enabled)
        self.model_name = str(model_name or self.safe_map.get("model_name") or "unknown")
        self.boundary_groups = {
            str(key): [str(item) for item in value]
            for key, value in (boundary_groups or self.safe_map.get("boundary_groups") or {}).items()
        }
        inferred_module_to_boundary = self._infer_module_to_boundary(self.boundary_groups)
        self.module_to_boundary = {
            str(key): str(value)
            for key, value in (
                module_to_boundary or self.safe_map.get("module_to_boundary") or inferred_module_to_boundary
            ).items()
        }
        self.max_age = int(max_age if max_age is not None else self.safe_map.get("max_age", 1))
        if self.max_age <= 0:
            raise ValueError("max_age must be positive")
        self.solver_stages = set(str(item) for item in (solver_stages or self.safe_map.get("solver_stages") or {"euler"}))
        self.branches = set(str(item) for item in (branches or self.safe_map.get("branches") or {"global"}))
        self.fallback_to_global_branch = bool(fallback_to_global_branch)
        self.default_safe = bool(default_safe)
        self.debug_jsonl_path = Path(debug_jsonl_path) if debug_jsonl_path is not None else None
        self.safe_table = self.safe_map.get("safe") or {}
        self.u_ratio_table = self.safe_map.get("u_ratio") or {}
        self.safe_lambda = self.safe_map.get("lambda", self.safe_map.get("safe_lambda"))
        self.quantile = self.safe_map.get("quantile")
        self.lte_floor = self.safe_map.get("lte_floor")
        self.eps = self.safe_map.get("eps")
        self._stats = self._empty_stats()
        self._by_boundary_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._by_age_counts: defaultdict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._by_step_counts: defaultdict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    def should_cache_module(self, module_name: str) -> bool:
        return str(module_name) in self.module_to_boundary

    def is_active(self, step_idx: int, t: float, module_name: str, cfg_branch: str, solver_stage: str) -> bool:
        del step_idx, t
        if not self.enabled:
            return False
        if str(solver_stage) not in self.solver_stages:
            return False
        if not self.should_cache_module(module_name):
            return False
        if str(cfg_branch) in self.branches:
            return True
        return self.fallback_to_global_branch and "global" in self.branches

    def should_refresh(self, step_idx: int, t: float, module_name: str, cfg_branch: str, solver_stage: str) -> bool:
        del step_idx, t, module_name, cfg_branch, solver_stage
        return True

    def should_reuse(self, step_idx: int, t: float, module_name: str, cfg_branch: str, solver_stage: str) -> bool:
        del step_idx, t, module_name, cfg_branch, solver_stage
        return False

    def should_reuse_entry(
        self,
        *,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
        entry: Any | None,
    ) -> bool:
        boundary = self.module_to_boundary.get(str(module_name))
        branch = self._select_branch(str(cfg_branch))
        reason = "safe_reuse"
        age: int | None = None
        safe = False

        if not self.enabled or str(solver_stage) not in self.solver_stages:
            reason = "inactive"
        elif boundary is None:
            reason = "missing_boundary"
        elif str(cfg_branch) not in self.branches and not (self.fallback_to_global_branch and "global" in self.branches):
            reason = "inactive"
        elif entry is None:
            reason = "missing_entry_refresh"
        else:
            age = int(step_idx) - int(entry.step_idx)
            if age <= 0:
                reason = "nonpositive_age_refresh"
            elif age > self.max_age:
                reason = "over_age_refresh"
            else:
                safe = self._safe_lookup(str(solver_stage), branch, boundary, int(step_idx), age)
                reason = "safe_reuse" if safe else "unsafe_refresh"

        self._mark(
            reason=reason,
            boundary=boundary,
            step_idx=int(step_idx),
            age=age,
            branch=branch,
            solver_stage=str(solver_stage),
            module_name=str(module_name),
            t=float(t),
        )
        return bool(safe)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "enabled": self.enabled,
            "model_name": self.model_name,
            "safe_map_path": str(self.safe_map_path) if self.safe_map_path is not None else None,
            "boundary_groups": {key: list(value) for key, value in sorted(self.boundary_groups.items())},
            "module_to_boundary": dict(sorted(self.module_to_boundary.items())),
            "max_age": self.max_age,
            "solver_stages": sorted(self.solver_stages),
            "branches": sorted(self.branches),
            "fallback_to_global_branch": self.fallback_to_global_branch,
            "default_safe": self.default_safe,
            "safe_lambda": self.safe_lambda,
            "quantile": self.quantile,
            "lte_floor": self.lte_floor,
            "eps": self.eps,
            "debug_jsonl_path": str(self.debug_jsonl_path) if self.debug_jsonl_path is not None else None,
        }

    def summary(self) -> dict[str, Any]:
        stats = dict(self._stats)
        ages = stats.pop("_ages")
        mean_age = sum(ages) / len(ages) if ages else 0.0
        max_observed_age = max(ages) if ages else 0
        return {
            "policy": self.policy_name,
            "config": self.to_dict(),
            "stats": {
                **stats,
                "mean_age": mean_age,
                "max_age": max_observed_age,
                "by_boundary": self._nested_to_dict(self._by_boundary_counts),
                "by_age": {str(key): dict(value) for key, value in sorted(self._by_age_counts.items())},
                "by_step": {str(key): dict(value) for key, value in sorted(self._by_step_counts.items())},
            },
        }

    @classmethod
    def from_path(cls, path: Path | str, **kwargs: Any) -> "SafeMapCachePolicy":
        return cls(safe_map_path=path, **kwargs)

    @staticmethod
    def _load_json(path: Path | None) -> dict[str, Any]:
        if path is None:
            raise ValueError("safe_map_path must not be None")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _infer_module_to_boundary(boundary_groups: dict[str, list[str]]) -> dict[str, str]:
        return {str(module_name): str(boundary) for boundary, modules in boundary_groups.items() for module_name in modules}

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "total_decisions": 0,
            "safe_reuse": 0,
            "safe_refresh": 0,
            "unsafe_refresh": 0,
            "missing_entry_refresh": 0,
            "over_age_refresh": 0,
            "nonpositive_age_refresh": 0,
            "missing_boundary_refresh": 0,
            "inactive_refresh": 0,
            "_ages": [],
        }

    def _select_branch(self, cfg_branch: str) -> str:
        if cfg_branch in self.branches:
            return cfg_branch
        if self.fallback_to_global_branch and "global" in self.branches:
            return "global"
        return cfg_branch

    def _safe_lookup(self, solver_stage: str, branch: str, boundary: str, step_idx: int, age: int) -> bool:
        value = self._lookup(self.safe_table, solver_stage)
        value = self._lookup(value, branch)
        if value is None and self.fallback_to_global_branch and branch != "global":
            value = self._lookup(self._lookup(self.safe_table, solver_stage), "global")
        value = self._lookup(value, boundary)
        value = self._lookup(value, step_idx)
        value = self._lookup(value, age, list_age=True)
        if value is None:
            return self.default_safe
        return bool(value)

    @staticmethod
    def _lookup(node: Any, key: str | int, *, list_age: bool = False) -> Any:
        if node is None:
            return None
        if isinstance(node, dict):
            for candidate in (key, str(key)):
                if candidate in node:
                    return node[candidate]
            return None
        if isinstance(node, list):
            idx = int(key)
            if 0 <= idx < len(node):
                return node[idx]
            if list_age and 1 <= idx <= len(node):
                return node[idx - 1]
        return None

    def _mark(
        self,
        *,
        reason: str,
        boundary: str | None,
        step_idx: int,
        age: int | None,
        branch: str,
        solver_stage: str,
        module_name: str,
        t: float,
    ) -> None:
        if reason == "safe_reuse":
            self._stats["safe_reuse"] += 1
            if age is not None:
                self._stats["_ages"].append(age)
        else:
            self._stats["safe_refresh"] += 1
            if reason == "unsafe_refresh":
                self._stats["unsafe_refresh"] += 1
            elif reason == "missing_entry_refresh":
                self._stats["missing_entry_refresh"] += 1
            elif reason == "over_age_refresh":
                self._stats["over_age_refresh"] += 1
            elif reason == "nonpositive_age_refresh":
                self._stats["nonpositive_age_refresh"] += 1
            elif reason == "missing_boundary":
                self._stats["missing_boundary_refresh"] += 1
            elif reason == "inactive":
                self._stats["inactive_refresh"] += 1
        self._stats["total_decisions"] += 1
        boundary_key = boundary or "<missing>"
        self._by_boundary_counts[boundary_key][reason] += 1
        if age is not None:
            self._by_age_counts[age][reason] += 1
        self._by_step_counts[step_idx][reason] += 1
        self._write_debug(
            {
                "policy": self.policy_name,
                "reason": reason,
                "reuse": reason == "safe_reuse",
                "boundary": boundary,
                "module_name": module_name,
                "step_idx": step_idx,
                "age": age,
                "branch": branch,
                "solver_stage": solver_stage,
                "t": t,
            }
        )

    @staticmethod
    def _nested_to_dict(value: defaultdict[str, defaultdict[str, int]]) -> dict[str, dict[str, int]]:
        return {key: dict(inner) for key, inner in sorted(value.items())}

    def _write_debug(self, payload: dict[str, Any]) -> None:
        if self.debug_jsonl_path is None:
            return
        self.debug_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
