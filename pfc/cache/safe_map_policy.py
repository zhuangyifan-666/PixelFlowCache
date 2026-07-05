from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def canonical_module_name(module_name: str) -> str:
    parts = str(module_name).split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "blocks" and parts[idx + 1].isdigit():
            return f"blocks.{parts[idx + 1]}"
    return str(module_name)


def compute_safe_map_density(safe_map: dict[str, Any]) -> dict[str, Any]:
    safe_root = safe_map.get("safe") or {}
    totals: dict[str, Any] = {
        "safe_total": 0,
        "safe_true": 0,
        "safe_density": 0.0,
        "by_boundary": defaultdict(lambda: {"safe_total": 0, "safe_true": 0, "safe_density": 0.0}),
        "by_age": defaultdict(lambda: {"safe_total": 0, "safe_true": 0, "safe_density": 0.0}),
        "by_branch": defaultdict(lambda: {"safe_total": 0, "safe_true": 0, "safe_density": 0.0}),
        "by_stage": defaultdict(lambda: {"safe_total": 0, "safe_true": 0, "safe_density": 0.0}),
    }

    for stage, stage_node in _dict_items(safe_root):
        for branch, branch_node in _dict_items(stage_node):
            for boundary, boundary_node in _dict_items(branch_node):
                for _step, step_node in _dict_items(boundary_node):
                    for age, value in _iter_age_values(step_node):
                        if not isinstance(value, bool):
                            continue
                        is_safe = bool(value)
                        _density_mark(totals, "safe_total", "safe_true", is_safe)
                        for group, key in (
                            ("by_boundary", str(boundary)),
                            ("by_age", str(age)),
                            ("by_branch", str(branch)),
                            ("by_stage", str(stage)),
                        ):
                            _density_mark(totals[group][key], "safe_total", "safe_true", is_safe)

    _finalize_density(totals)
    for group in ("by_boundary", "by_age", "by_branch", "by_stage"):
        values = dict(sorted(totals[group].items(), key=lambda item: item[0]))
        for payload in values.values():
            _finalize_density(payload)
        totals[group] = values
    return totals


def _dict_items(node: Any):
    if isinstance(node, dict):
        yield from node.items()


def _iter_age_values(node: Any):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
    elif isinstance(node, list):
        for idx, value in enumerate(node, start=1):
            yield idx, value


def _density_mark(payload: dict[str, Any], total_key: str, true_key: str, is_safe: bool) -> None:
    payload[total_key] += 1
    if is_safe:
        payload[true_key] += 1


def _finalize_density(payload: dict[str, Any]) -> None:
    total = int(payload.get("safe_total", 0))
    payload["safe_density"] = float(payload.get("safe_true", 0)) / total if total else 0.0


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
        raw_boundary_groups = boundary_groups or self.safe_map.get("boundary_groups") or {}
        self.boundary_groups = {
            str(boundary): [canonical_module_name(str(item)) for item in modules]
            for boundary, modules in raw_boundary_groups.items()
        }
        inferred_module_to_boundary = self._infer_module_to_boundary(self.boundary_groups)
        raw_module_to_boundary = module_to_boundary or self.safe_map.get("module_to_boundary") or inferred_module_to_boundary
        self.module_to_boundary = {
            canonical_module_name(str(module)): str(boundary)
            for module, boundary in raw_module_to_boundary.items()
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
        self.safe_density = compute_safe_map_density(self.safe_map)
        self._stats = self._empty_stats()
        self._reuse_ages: list[int] = []
        self._by_reason: defaultdict[str, int] = defaultdict(int)
        self._by_boundary: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._by_age: defaultdict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._by_step: defaultdict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._by_branch: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._by_solver_stage: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    def should_cache_module(self, module_name: str) -> bool:
        return canonical_module_name(str(module_name)) in self.module_to_boundary

    def is_active(self, step_idx: int, t: float, module_name: str, cfg_branch: str, solver_stage: str) -> bool:
        del step_idx, t
        if not self.enabled:
            return False
        if canonical_module_name(str(module_name)) not in self.module_to_boundary:
            return False
        if str(solver_stage) not in self.solver_stages:
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
        canonical_module = canonical_module_name(str(module_name))
        boundary = self.module_to_boundary.get(canonical_module)
        entry_step_idx = int(entry.step_idx) if entry is not None else None
        age = int(step_idx) - entry_step_idx if entry_step_idx is not None else None
        branch = str(cfg_branch)
        lookup_found = False
        u_ratio = None
        safe = False
        reason = "unsafe_refresh"

        if not self.enabled:
            reason = "policy_disabled"
        elif boundary is None:
            reason = "module_not_managed"
        elif str(solver_stage) not in self.solver_stages:
            reason = "solver_stage_not_found"
        else:
            branch, branch_ok = self._select_branch(branch)
            if not branch_ok:
                reason = "branch_not_found"
            else:
                self._stats["total_managed_calls"] += 1
                if entry is None:
                    reason = "missing_entry_refresh"
                elif age is None or age <= 0:
                    reason = "non_positive_age_refresh"
                elif age > self.max_age:
                    reason = "over_age_refresh"
                else:
                    lookup = self._safe_lookup(str(solver_stage), branch, boundary, int(step_idx), int(age))
                    safe = lookup["safe"]
                    lookup_found = lookup["found"]
                    u_ratio = lookup["u_ratio"]
                    reason = "safe_reuse" if safe else lookup["reason"]

        self._record_decision(
            reason=reason,
            boundary=boundary,
            step_idx=int(step_idx),
            age=age,
            branch=branch,
            solver_stage=str(solver_stage),
            module_name=canonical_module,
            original_module_name=str(module_name),
            t=float(t),
            entry_exists=entry is not None,
            entry_step_idx=entry_step_idx,
            safe_lookup_found=lookup_found,
            safe=safe,
            u_ratio=u_ratio,
        )
        return bool(safe)

    def mark_reuse_committed(
        self,
        *,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
        entry: Any,
    ) -> None:
        del t, cfg_branch, solver_stage
        age = int(step_idx) - int(entry.step_idx)
        self._stats["safe_reuse_committed"] += 1
        self._reuse_ages.append(age)

    def mark_refresh_committed(
        self,
        *,
        step_idx: int,
        t: float,
        module_name: str,
        cfg_branch: str,
        solver_stage: str,
        entry: Any | None,
        refreshed_entry: Any | None = None,
        entry_matches: bool | None = None,
    ) -> None:
        del t, module_name, cfg_branch, solver_stage, entry, refreshed_entry
        self._stats["refresh_committed"] += 1
        if entry_matches is False:
            self._stats["entry_mismatch_refresh"] += 1
            self._by_reason["entry_mismatch_refresh"] += 1
            self._by_step[int(step_idx)]["entry_mismatch_refresh"] += 1

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
            "safe_density": self.safe_density,
        }

    def summary(self) -> dict[str, Any]:
        mean_age = sum(self._reuse_ages) / len(self._reuse_ages) if self._reuse_ages else 0.0
        max_age = max(self._reuse_ages) if self._reuse_ages else 0
        stats = dict(self._stats)
        return {
            "policy": self.policy_name,
            "config": self.to_dict(),
            "stats": {
                **stats,
                "safe_reuse": stats["safe_reuse_committed"],
                "safe_refresh": stats["refresh_decisions"],
                "mean_age": mean_age,
                "mean_age_of_reuse": mean_age,
                "max_age": max_age,
                "max_age_of_reuse": max_age,
                "by_reason": dict(sorted(self._by_reason.items())),
                "by_boundary": self._nested_to_dict(self._by_boundary),
                "by_age": {str(key): dict(value) for key, value in sorted(self._by_age.items())},
                "by_step": {str(key): dict(value) for key, value in sorted(self._by_step.items())},
                "by_branch": self._nested_to_dict(self._by_branch),
                "by_solver_stage": self._nested_to_dict(self._by_solver_stage),
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
        return {canonical_module_name(module): boundary for boundary, modules in boundary_groups.items() for module in modules}

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "total_managed_calls": 0,
            "safe_reuse_decisions": 0,
            "safe_reuse_committed": 0,
            "refresh_decisions": 0,
            "refresh_committed": 0,
            "missing_entry_refresh": 0,
            "unsafe_refresh": 0,
            "over_age_refresh": 0,
            "non_positive_age_refresh": 0,
            "boundary_not_found": 0,
            "module_not_managed": 0,
            "solver_stage_not_found": 0,
            "branch_not_found": 0,
            "step_not_found": 0,
            "age_not_found": 0,
            "entry_mismatch_refresh": 0,
        }

    def _select_branch(self, cfg_branch: str) -> tuple[str, bool]:
        if cfg_branch in self.branches:
            return cfg_branch, True
        if self.fallback_to_global_branch and "global" in self.branches:
            return "global", True
        return cfg_branch, False

    def _safe_lookup(self, solver_stage: str, branch: str, boundary: str, step_idx: int, age: int) -> dict[str, Any]:
        stage_node = self._lookup(self.safe_table, solver_stage)
        if stage_node is None:
            return self._lookup_result(False, False, "solver_stage_not_found", None)
        branch_node = self._lookup(stage_node, branch)
        if branch_node is None and self.fallback_to_global_branch and branch != "global":
            branch_node = self._lookup(stage_node, "global")
        if branch_node is None:
            return self._lookup_result(False, False, "branch_not_found", None)
        boundary_node = self._lookup(branch_node, boundary)
        if boundary_node is None:
            return self._lookup_result(False, False, "boundary_not_found", None)
        step_node = self._lookup(boundary_node, step_idx)
        if step_node is None:
            return self._lookup_result(False, self.default_safe, "step_not_found", None)
        value = self._lookup(step_node, age, list_age=True)
        if value is None:
            return self._lookup_result(False, self.default_safe, "age_not_found", None)
        u_ratio = self._u_ratio_lookup(solver_stage, branch, boundary, step_idx, age)
        if bool(value):
            return self._lookup_result(True, True, "safe_reuse", u_ratio)
        return self._lookup_result(True, False, "unsafe_refresh", u_ratio)

    @staticmethod
    def _lookup_result(found: bool, safe: bool, reason: str, u_ratio: Any) -> dict[str, Any]:
        return {"found": bool(found), "safe": bool(safe), "reason": reason, "u_ratio": u_ratio}

    def _u_ratio_lookup(self, solver_stage: str, branch: str, boundary: str, step_idx: int, age: int) -> Any:
        value = self._lookup(self.u_ratio_table, solver_stage)
        value = self._lookup(value, branch)
        if value is None and self.fallback_to_global_branch and branch != "global":
            value = self._lookup(self._lookup(self.u_ratio_table, solver_stage), "global")
        value = self._lookup(value, boundary)
        value = self._lookup(value, step_idx)
        return self._lookup(value, age, list_age=True)

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

    def _record_decision(
        self,
        *,
        reason: str,
        boundary: str | None,
        step_idx: int,
        age: int | None,
        branch: str,
        solver_stage: str,
        module_name: str,
        original_module_name: str,
        t: float,
        entry_exists: bool,
        entry_step_idx: int | None,
        safe_lookup_found: bool,
        safe: bool,
        u_ratio: Any,
    ) -> None:
        if reason == "safe_reuse":
            self._stats["safe_reuse_decisions"] += 1
        elif reason in self._stats:
            self._stats[reason] += 1
            if reason.endswith("_refresh") or reason in {
                "unsafe_refresh",
                "over_age_refresh",
                "missing_entry_refresh",
                "non_positive_age_refresh",
                "step_not_found",
                "age_not_found",
                "boundary_not_found",
            }:
                self._stats["refresh_decisions"] += 1
        else:
            self._stats["refresh_decisions"] += 1
        self._by_reason[reason] += 1
        boundary_key = boundary or "<missing>"
        self._by_boundary[boundary_key][reason] += 1
        if age is not None:
            self._by_age[int(age)][reason] += 1
        self._by_step[int(step_idx)][reason] += 1
        self._by_branch[str(branch)][reason] += 1
        self._by_solver_stage[str(solver_stage)][reason] += 1
        self._write_debug(
            {
                "policy": self.policy_name,
                "step_idx": step_idx,
                "t": t,
                "module_name": module_name,
                "original_module_name": original_module_name,
                "boundary": boundary,
                "branch": branch,
                "solver_stage": solver_stage,
                "entry_exists": entry_exists,
                "entry_step_idx": entry_step_idx,
                "candidate_age": age,
                "safe_lookup_found": safe_lookup_found,
                "safe": safe,
                "u_ratio": u_ratio,
                "lambda": self.safe_lambda,
                "reason": reason,
                "reuse": reason == "safe_reuse",
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
