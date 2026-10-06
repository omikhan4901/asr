"""
Central configuration definitions for the Bangla ASR pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    silence_pad_seconds: float = 0.2
    max_chunk_duration_s: float = 25.0
    min_chunk_duration_s: float = 0.5


@dataclass
class VADConfig:
    repo_or_dir: str = "snakers4/silero-vad"
    model_name: str = "silero_vad"
    sample_rate: int = 16000
    threshold: float = 0.5
    # Silero-internal minimums. Pauses longer than min_silence_duration_ms
    # become candidate chunk boundaries before the packing pass.
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 80
    # Only merge adjacent speech spans when their intervening silence is this
    # short. Longer pauses become separate ASR/subtitle chunks.
    max_silence_gap_s: float = 0.5
    # Soft duration target. Once a packed chunk reaches this length, the next
    # pause starts a new chunk even if the hard max has not been hit.
    # Set close to max_chunk_duration_s (25.0) to match the published
    # pipeline's chunk distribution (Fig 3.3, mean 22.71s) -- i.e. only split
    # early on a genuine long pause, not merely because 15s has elapsed.
    preferred_chunk_duration_s: float = 24.0
    # When continuous speech must be split at max_chunk_duration_s, search
    # backward from that limit for a low-energy boundary rather than cutting
    # at an arbitrary sample in the middle of a word.
    split_search_window_s: float = 1.0
    split_energy_frame_s: float = 0.04
    # A candidate split point only counts as a genuine pause if its local RMS
    # is <= this fraction of the un-split span's average RMS. Without this
    # gate, the backward search always returns *some* point (even the
    # locally-quietest sample of continuous, pause-free speech, e.g. a
    # plosive closure), which can silently cut mid-word. When no candidate
    # clears the bar, the split falls back to a plain hard cut at the
    # duration cap instead of mislabeling a mid-word dip as a safe boundary.
    # Heuristic default, not empirically tuned against real Bangla audio.
    split_quiet_ratio: float = 0.85


@dataclass
class AlignmentConfig:
    min_alignment_score: int = 50
    # Sliding window search bounds relative to sequential corpus pointer
    search_window_lead: int = 5    # Search backwards up to 5 words
    search_window_trail: int = 15   # Search forwards up to 15 words
    # Search recovery if sequential pointer loses track
    max_recovery_lookahead: int = 60
    recovery_score_threshold: float = 65.0
    # On a failed match, advance the corpus pointer by this fraction of the
    # hypothesis word count so later chunks do not keep aligning to the same span.
    failure_advance_frac: float = 0.25


@dataclass
class WhisperConfig:
    model_id: str = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"
    fallback_base_config: str = "openai/whisper-medium"
    adapter_path: Optional[str] = None
    language: str = "bn"
    task: str = "transcribe"
    num_beams: int = 12
    max_new_tokens: int = 320
    length_penalty: float = 1.5
    diverse_beam_search: bool = True
    sampling_temperature: float = 0.7
    # True diverse beam search (num_beam_groups). Off by default so the
    # published sampling-based N-best setup remains the reproducible path.
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    batch_size: int = 2
    attn_implementation: str = "sdpa"


@dataclass
class WhisperTrainConfig:
    model_id: str = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"
    output_dir: str = "checkpoints/whisper_lora"
    language: str = "bn"
    task: str = "transcribe"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    )
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_audio_seconds: float = 30.0
    val_split_frac: float = 0.1
    eval_max_new_tokens: int = 320
    seed: int = 42
    early_stopping_patience: int = 2


@dataclass
class MBRConfig:
    # Softmax scaling temperature for acoustic sequence logprobs
    # Temperature < 1.0 sharpens acoustic confidence; > 1.0 flattens towards uniform consensus
    temperature: float = 0.5
    empty_loss_penalty: float = 1.0


@dataclass
class BERTRerankerConfig:
    model_name: str = "csebuetnlp/banglabert"
    max_seq_length: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    num_epochs: int = 5
    hidden_dropout_prob: float = 0.2
    use_acoustic_features: bool = True
    acoustic_feature_dim: int = 2  # [normalized_rank, z-scored sequence logprob]
    early_stopping_patience: int = 2
    seed: int = 42
    num_workers: int = 0


@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    whisper_train: WhisperTrainConfig = field(default_factory=WhisperTrainConfig)
    mbr: MBRConfig = field(default_factory=MBRConfig)
    bert: BERTRerankerConfig = field(default_factory=BERTRerankerConfig)
