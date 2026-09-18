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
    # pause starts a new chunk even if the hard max has not been hit. This
    # keeps alignment windows short on continuous audiobook speech.
    preferred_chunk_duration_s: float = 15.0
    # When continuous speech must be split at max_chunk_duration_s, search
    # backward from that limit for a low-energy boundary rather than cutting
    # at an arbitrary sample in the middle of a word.
    split_search_window_s: float = 1.0
    split_energy_frame_s: float = 0.04


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
class Wav2Vec2Config:
    """CTC acoustic-model settings. Defaults do not affect the Whisper baseline."""

    model_id: str = "arijitx/wav2vec2-xls-r-300m-bengali"
    kenlm_model_path: Optional[str] = None
    beam_width: int = 12
    lm_alpha: float = 0.5
    lm_beta: float = 1.0
    batch_size: int = 4
    attn_implementation: Optional[str] = None


@dataclass
class SWERConfig:
    """Clinical-severity-weighted error rate. Distinct from Roy 2021 Semantic WER.

    Denominator interpretation (explicit, not hidden):
        Each reference token contributes ``critical_reference_weight`` (5.0) if
        it is marked critical, else ``default_reference_weight`` (1.0). The
        draft specified edit multipliers but not this per-token mass. Review
        before treating SWER as a frozen metric.

    JUDGMENT CALL — do not silently equalize these two numbers:
        critical_substitution_weight = 5.0
        critical_deletion_weight     = 4.0
        critical_reference_weight    = 5.0
        For a single critical reference token this yields
            substitution SWER = 5/5 = 1.0
            deletion     SWER = 4/5 = 0.8
        Implication: dropping a critical term is scored as slightly *less*
        severe than getting it wrong. That was an artifact of the first draft
        numbers, not a supervisor-approved clinical decision. Change the
        weights here if you decide they should be equal (or reversed).
    """

    lexicon_path: Optional[str] = None
    # See class docstring: substitution 5.0 vs deletion 4.0 is a visible
    # unresolved clinical decision, not an implementation bug.
    critical_substitution_weight: float = 5.0
    garbled_span_weight: float = 5.0
    minor_substitution_weight: float = 1.0
    critical_deletion_weight: float = 4.0
    noncritical_deletion_weight: float = 1.0
    hallucinated_insertion_weight: float = 3.0
    filler_insertion_weight: float = 1.0
    critical_reference_weight: float = 5.0
    default_reference_weight: float = 1.0
    # Detector: composite | explicit | character | wordlist_oov | edit_distance
    garbled_detector: str = "composite"
    garbled_wordlist_path: Optional[str] = None
    garbled_edit_threshold: int = 2
    # If True, composite/wordlist_oov treat every OOV token as garbled
    # (high false-positive risk with the starter wordlist).
    treat_oov_as_garbled: bool = False


@dataclass
class MBRConfig:
    # Softmax scaling temperature for acoustic sequence logprobs
    # Temperature < 1.0 sharpens acoustic confidence; > 1.0 flattens towards uniform consensus
    temperature: float = 0.5
    empty_loss_penalty: float = 1.0


@dataclass
class GRPOConfig:
    """Group Relative Policy Optimization for SWER-improvement correction."""

    model_id: str = "WeiboAI/VibeThinker-3B"
    learning_rate: float = 1e-6
    batch_size: int = 1
    group_size: int = 4
    max_new_tokens: int = 128
    sampling_temperature: float = 0.8
    top_p: float = 0.95
    kl_coef: float = 0.05
    reward_scale: float = 1.0
    reward_clip: float = 2.0
    max_steps: int = 100
    eval_every: int = 10
    checkpoint_every: int = 20
    checkpoint_dir: str = "checkpoints/grpo"
    log_dir: str = "logs/grpo"
    seed: int = 42
    max_grad_norm: float = 1.0
    warmup_steps: int = 5
    early_stopping_patience: int = 5
    prompt_style: str = "simple"
    advantage_eps: float = 1e-8
    # Empty-reference SWER is NaN. True (default) drops those pairs from the
    # GRPO batch and logs ``excluded_empty_reference``. False maps them to
    # reward 0.0, which dilutes the batch mean — that was the old behaviour.
    exclude_empty_reference_from_batch: bool = True


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
    wav2vec2: Wav2Vec2Config = field(default_factory=Wav2Vec2Config)
    mbr: MBRConfig = field(default_factory=MBRConfig)
    swer: SWERConfig = field(default_factory=SWERConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    bert: BERTRerankerConfig = field(default_factory=BERTRerankerConfig)
