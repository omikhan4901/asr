"""Reward is SWER delta, independently testable, and strictly monotonic on synthetic pairs."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from bangla_asr.config import GRPOConfig, PipelineConfig, WhisperConfig
from bangla_asr.metrics.swer import compute_swer
from bangla_asr.training.callbacks import EarlyStopping, JsonCsvLogger
from bangla_asr.training.reward import swer_improvement_reward
from bangla_asr.training.train_grpo import (
    compute_group_advantages,
    evaluate_corrections,
    load_correction_records,
)


def test_reward_matches_swer_delta() -> None:
    asr = "নিম্নরক্তচাপ আছে"
    gt = "উচ্চরক্তচাপ আছে"
    correction = "উচ্চরক্তচাপ আছে"
    expected = compute_swer(asr, gt) - compute_swer(correction, gt)
    assert swer_improvement_reward(asr, correction, gt, scale=1.0, clip=10.0) == expected


def test_reward_strictly_monotonic_perfect_vs_garbled() -> None:
    asr = "নিম্নরক্তচাপ আছে"
    gt = "উচ্চরক্তচাপ আছে"
    perfect = gt
    garbled = "ইনসলন আছে"
    r_perfect = swer_improvement_reward(asr, perfect, gt, scale=1.0, clip=10.0)
    r_garbled = swer_improvement_reward(asr, garbled, gt, scale=1.0, clip=10.0)
    assert r_perfect > r_garbled
    assert compute_swer(perfect, gt) < compute_swer(garbled, gt)


def test_reward_scale_and_clip_are_config_not_constants() -> None:
    asr = "নিম্নরক্তচাপ"
    gt = "উচ্চরক্তচাপ"
    raw = swer_improvement_reward(asr, gt, gt, scale=1.0, clip=10.0)
    scaled = swer_improvement_reward(asr, gt, gt, scale=2.0, clip=10.0)
    clipped = swer_improvement_reward(asr, gt, gt, scale=50.0, clip=0.25)
    assert scaled == raw * 2.0
    assert clipped == 0.25


def test_empty_reference_is_excluded_by_default() -> None:
    assert swer_improvement_reward("আমি", "আমি", "") is None
    assert swer_improvement_reward("আমি", "আমি", "", exclude_empty_reference=False) == 0.0
    assert math.isnan(compute_swer("আমি", ""))


def test_evaluate_corrections_logs_excluded_empty_refs() -> None:
    metrics = evaluate_corrections(
        ["আমি", "নিম্নরক্তচাপ আছে"],
        ["আমি", "উচ্চরক্তচাপ আছে"],
        ["", "উচ্চরক্তচাপ আছে"],
        reward_scale=1.0,
        reward_clip=10.0,
        exclude_empty_reference=True,
    )
    assert metrics["excluded_empty_reference"] == 1.0
    assert metrics["n_rewarded"] == 1.0
    assert metrics["mean_reward"] > 0.0


def test_group_advantages_zero_when_flat() -> None:
    adv = compute_group_advantages(np.array([[1.0, 1.0, 1.0]]))
    assert np.allclose(adv, 0.0)


def test_group_advantages_relative() -> None:
    adv = compute_group_advantages(np.array([[0.0, 2.0]]))
    assert adv[0, 1] > adv[0, 0]


def test_load_nbest_and_correction_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "recs.jsonl"
        path.write_text(
            '{"utt_id":"a","asr_hypothesis":"x","ground_truth":"y"}\n'
            '{"utt_id":"b","greedy_text":"g","ground_truth":"y","candidates":'
            '[{"text":"c","rank":0}]}\n',
            encoding="utf-8",
        )
        recs = load_correction_records(str(path))
        assert recs[0]["asr_hypothesis"] == "x"
        assert recs[1]["asr_hypothesis"] == "g"


def test_evaluate_corrections_reports_improvement() -> None:
    metrics = evaluate_corrections(
        ["নিম্নরক্তচাপ আছে"],
        ["উচ্চরক্তচাপ আছে"],
        ["উচ্চরক্তচাপ আছে"],
        reward_scale=1.0,
        reward_clip=10.0,
    )
    assert metrics["mean_reward"] > 0.0
    assert metrics["corr_swer"] < metrics["asr_swer"]


def test_early_stopping_and_logger() -> None:
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.update(1.0) is False
    assert stopper.update(0.5) is False
    assert stopper.update(0.4) is True
    with tempfile.TemporaryDirectory() as tmp:
        logger = JsonCsvLogger(tmp)
        logger.log({"step": 1, "loss": 0.2})
        assert (Path(tmp) / "train_log.jsonl").exists()
        assert (Path(tmp) / "train_log.csv").exists()


def test_grpo_defaults_do_not_change_whisper_baseline() -> None:
    cfg = PipelineConfig()
    assert cfg.whisper == WhisperConfig()
    assert cfg.mbr.temperature == 0.5
    assert cfg.grpo.group_size == 4
    assert GRPOConfig().reward_scale == 1.0
    assert GRPOConfig().exclude_empty_reference_from_batch is True


if __name__ == "__main__":
    test_reward_matches_swer_delta()
    test_reward_strictly_monotonic_perfect_vs_garbled()
    test_reward_scale_and_clip_are_config_not_constants()
    test_empty_reference_is_excluded_by_default()
    test_evaluate_corrections_logs_excluded_empty_refs()
    test_group_advantages_zero_when_flat()
    test_group_advantages_relative()
    test_load_nbest_and_correction_jsonl()
    test_evaluate_corrections_reports_improvement()
    test_early_stopping_and_logger()
    test_grpo_defaults_do_not_change_whisper_baseline()
    print("[PASS] reward / GRPO helper tests")
