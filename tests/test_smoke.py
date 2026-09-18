"""
Automated smoke test validating all bangla_asr imports, configs, metrics, and rescoring logic.
"""

import sys
import io
import json
import tempfile
from pathlib import Path

# Set UTF-8 encoding for stdout on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

# 1. Config tests
from bangla_asr.config import (
    AudioConfig,
    VADConfig,
    AlignmentConfig,
    WhisperConfig,
    MBRConfig,
    BERTRerankerConfig,
    PipelineConfig,
)

cfg = PipelineConfig()
assert cfg.whisper.num_beams == 12
assert cfg.whisper.sampling_temperature == 0.7
print("[PASS] Config classes verified.")

# 2. Text normalization tests
from bangla_asr.common.text import normalize_bangla_text

sample_bn = "  আমি বাংলায় গান গাই। তুমি কি গান গাও? "
norm_bn = normalize_bangla_text(sample_bn)
assert "।" not in norm_bn
assert "?" not in norm_bn
assert norm_bn == "আমি বাংলায় গান গাই তুমি কি গান গাও"
print("[PASS] Bangla text normalization verified.")

# 3. Metrics tests
from bangla_asr.common.metrics import compute_wer, compute_cer, compute_pairwise_loss

wer = compute_wer(["আমি গান গাই"], ["আমি গান গাই"])
assert wer == 0.0

wer_diff = compute_wer(["আমি গান গাই"], ["আমি বই পড়ি"])
assert wer_diff > 0.0

loss = compute_pairwise_loss("আমি গান গাই", "আমি গান গাই")
assert loss == 0.0
print("[PASS] Metrics calculation verified.")

# 4. MBR Selector test
from bangla_asr.rescoring.mbr import MBRSelector
from bangla_asr.rescoring.mbr import run_mbr_rescoring, softmax_with_temperature

candidates = [
    {"text": "আমি বাংলায় গান গাই", "logprob": -0.15, "rank": 0},
    {"text": "আমি বাংলায় গান খাই", "logprob": -0.35, "rank": 1},
    {"text": "আমি বাংলায় গান গাই", "logprob": -0.18, "rank": 2},
]

selector = MBRSelector(MBRConfig(temperature=0.5))
chosen = selector.select(candidates)
assert chosen["text"] == "আমি বাংলায় গান গাই"
assert "mbr_risk" in chosen
assert np.allclose(softmax_with_temperature(np.array([np.nan, np.inf])), [0.5, 0.5])
rescoring_results = run_mbr_rescoring([
    {"utt_id": "valid", "ground_truth": "reference", "candidates": [{"text": "reference", "rank": 0}]},
    {"utt_id": "empty", "ground_truth": "ignored", "candidates": []},
])
assert rescoring_results["n_utterances"] == 1
print("[PASS] MBR consensus selector verified.")

# 5. Dataset Splitter test
import pandas as pd
from bangla_asr.dataset.splitter import extract_source_id, group_aware_split

assert extract_source_id("train_001_042.wav") == "train_001"
assert extract_source_id("train_001_1042.wav") == "train_001"  # handles 4+ digits

test_df = pd.DataFrame({
    "source_file": ["file_a", "file_a", "file_b", "file_b", "file_c", "file_c"],
    "utt_id": ["a1", "a2", "b1", "b2", "c1", "c2"],
})
train_df, eval_df = group_aware_split(test_df, eval_fraction=0.33, random_seed=42)
train_sources = set(train_df["source_file"])
eval_sources = set(eval_df["source_file"])
assert not train_sources.intersection(eval_sources)
try:
    group_aware_split(test_df.iloc[:2], eval_fraction=0.33, random_seed=42)
    raise AssertionError("Single-group split should fail")
except ValueError:
    pass
print("[PASS] Group-aware dataset splitter verified.")

# 6. VAD packing test: brief pauses merge; long silence creates a new chunk.
from bangla_asr.preprocessing.vad import SileroVADChunker

vad_chunker = object.__new__(SileroVADChunker)
vad_chunker.audio_cfg = AudioConfig(sample_rate=10, max_chunk_duration_s=25.0, min_chunk_duration_s=0.1)
vad_chunker.vad_cfg = VADConfig(max_silence_gap_s=1.0)
vad_chunker.get_speech_timestamps = lambda _: [
    {"start": 0, "end": 5},      # 0.0-0.5 s
    {"start": 12, "end": 17},    # 0.7 s pause: merge
    {"start": 45, "end": 50},    # 2.8 s pause: new chunk
]
vad_chunks = vad_chunker.segment_audio(np.zeros(60, dtype=np.float32))
assert len(vad_chunks) == 2
assert vad_chunks[0]["start_sample"] == 0 and vad_chunks[0]["end_sample"] == 17
assert vad_chunks[1]["start_sample"] == 45 and vad_chunks[1]["end_sample"] == 50

# Continuous spans are split at a quiet point before the hard duration cap,
# rather than always at the exact cap in the middle of speech.
vad_chunker.audio_cfg = AudioConfig(sample_rate=10, max_chunk_duration_s=25.0, min_chunk_duration_s=0.5)
vad_chunker.vad_cfg = VADConfig(max_silence_gap_s=1.0, split_search_window_s=1.0, split_energy_frame_s=0.1)
vad_chunker.get_speech_timestamps = lambda _: [{"start": 0, "end": 260}]
continuous_audio = np.ones(300, dtype=np.float32)
continuous_audio[245] = 0.0
vad_chunks = vad_chunker.segment_audio(continuous_audio)
assert [(chunk["start_sample"], chunk["end_sample"]) for chunk in vad_chunks] == [(0, 245), (245, 260)]
print("[PASS] VAD silence-gap chunk boundaries verified.")

# 7. BERT Reranker Model architecture test
from bangla_asr.models.bert_reranker.model import BanglaBERTReranker
from bangla_asr.models.bert_reranker.ranker import BERTCandidateRanker

bert_cfg = BERTRerankerConfig(use_acoustic_features=True)
with tempfile.TemporaryDirectory() as tmpdir:
    config_path = Path(tmpdir) / "reranker_config.json"
    config_path.write_text(json.dumps({"model_name": "test/model", "use_acoustic_features": False}), encoding="utf-8")
    loaded_cfg = BERTCandidateRanker._load_checkpoint_config(Path(tmpdir) / "best_model.pt")
    assert loaded_cfg.model_name == "test/model"
    assert loaded_cfg.use_acoustic_features is False
print("[PASS] BERT Reranker config verified.")

print("\n>>> ALL SYSTEM UNIT TESTS PASSED SUCCESSFULLY! <<<")
