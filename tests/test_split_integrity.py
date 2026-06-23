"""Train/test split integrity checks (reviewer point 7: data leakage).

These tests document and guard against split leakage in the agricultural datasets:
- CWFID: official split lists image 28 in BOTH train and test (direct leak) -> fixed by dedup.
- WeedsGalore: multi-temporal; the 4 capture dates all span train/val/test (temporal/content
  leakage that we DISCLOSE rather than silently fix). Sample IDs are still disjoint.
- CoFly: single flight with overlapping adjacent frames (spatial leakage, disclosed).

Tests skip automatically when the datasets are not present locally.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cea_plus.dataset import cwfid_split_ids

DATA = Path(__file__).resolve().parents[1] / "data" / "external"
CWFID = DATA / "cwfid"
WEEDS = DATA / "weedsgalore-dataset"


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"dataset not available: {path}")


def test_cwfid_raw_split_has_known_leak():
    """The raw CWFID split file leaks image 28 across train/test (documents the issue)."""
    _require(CWFID / "train_test_split.yaml")
    import yaml

    raw = yaml.safe_load((CWFID / "train_test_split.yaml").read_text(encoding="utf-8"))
    overlap = set(int(i) for i in raw["train"]) & set(int(i) for i in raw["test"])
    assert overlap == {28}, f"unexpected CWFID raw overlap: {overlap}"


def test_cwfid_dedup_makes_train_test_disjoint():
    """After dedup the loader's train and test ids must be disjoint (no leakage)."""
    _require(CWFID / "train_test_split.yaml")
    train = set(cwfid_split_ids(CWFID, "train"))
    test = set(cwfid_split_ids(CWFID, "test"))
    assert train & test == set(), f"CWFID train/test still overlap: {train & test}"
    # test set is preserved intact (we only drop the leak from train)
    assert 28 in test and 28 not in train


def _weeds_split_ids(split: str) -> list[str]:
    path = WEEDS / "splits" / f"{split}.txt"
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_weedsgalore_sample_ids_disjoint():
    """WeedsGalore sample IDs are disjoint across splits (sanity)."""
    _require(WEEDS / "splits" / "train.txt")
    train = set(_weeds_split_ids("train"))
    test = set(_weeds_split_ids("test"))
    assert train & test == set()


def test_weedsgalore_dates_overlap_is_disclosed():
    """Document that WeedsGalore is NOT temporally isolated: dates span train and test.

    This is an intentional record of the leakage we disclose in the paper Limitations,
    not something we fix (the split is the dataset's official one).
    """
    _require(WEEDS / "splits" / "train.txt")
    train_dates = {sid[:10] for sid in _weeds_split_ids("train")}
    test_dates = {sid[:10] for sid in _weeds_split_ids("test")}
    shared = train_dates & test_dates
    # same physical field re-imaged across dates appears in both splits
    assert len(shared) >= 1, "expected shared capture dates (temporal leakage) but found none"
