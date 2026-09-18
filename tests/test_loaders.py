"""Dataset registry: Bangla manifest works; English clinical hooks stay unimplemented."""

from __future__ import annotations

import tempfile
from pathlib import Path

from bangla_asr.dataset.loaders import BanglaManifestLoader, get_loader


def test_bangla_loader_reads_aligned_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "aligned.csv"
        csv_path.write_text(
            "audio_path,gt,align_score\n"
            "train_001_042.wav,আমি গান গাই,90\n",
            encoding="utf-8",
        )
        records = BanglaManifestLoader().load(str(csv_path))
        assert len(records) == 1
        assert records[0].language == "bn"
        assert records[0].ground_truth == "আমি গান গাই"
        assert records[0].utt_id == "train_001_042"


def test_english_clinical_loader_is_not_implemented() -> None:
    loader = get_loader("meddialog_en_asr")
    try:
        loader.load("unused.csv")
    except NotImplementedError as exc:
        assert "parallel track" in str(exc)
        assert "Bangla baseline" in str(exc)
    else:
        raise AssertionError("English clinical loader must not silently parse data")


def test_unknown_loader_raises() -> None:
    try:
        get_loader("not-a-dataset")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_bangla_loader_reads_aligned_csv()
    test_english_clinical_loader_is_not_implemented()
    test_unknown_loader_raises()
    print("[PASS] dataset loader tests")
