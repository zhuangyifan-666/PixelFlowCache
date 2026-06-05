from __future__ import annotations

from collections import Counter
from pathlib import Path

from pfc.eval.label_schedule import (
    label_schedule_summary,
    load_label_schedule,
    make_imagenet_class_balanced_labels,
    save_label_schedule,
)


def test_label_schedule_under_1000_is_sequential() -> None:
    assert make_imagenet_class_balanced_labels(8) == list(range(8))


def test_label_schedule_over_1000_is_balanced() -> None:
    labels = make_imagenet_class_balanced_labels(2501)
    counts = Counter(labels)
    assert len(counts) == 1000
    assert max(counts.values()) - min(counts.values()) <= 1
    summary = label_schedule_summary(labels)
    assert summary["balanced"] is True


def test_label_schedule_save_load_json_and_csv(tmp_path: Path) -> None:
    labels = make_imagenet_class_balanced_labels(12)
    written = save_label_schedule(labels, tmp_path / "labels")
    assert load_label_schedule(written["json"]) == labels
    assert load_label_schedule(written["csv"]) == labels
    assert load_label_schedule(tmp_path / "labels") == labels

