"""
Batched N-best candidate generation with diverse beam search.
"""

from typing import Dict, List, Optional
import numpy as np
import torch

from bangla_asr.common.audio import pad_silence
from bangla_asr.config import AudioConfig, WhisperConfig
from bangla_asr.models.autoregressive.whisper_model import WhisperASRModel, strip_markers


class BatchedNBestGenerator:
    """Runs batched inference to produce diverse N-best hypotheses per audio chunk."""

    def __init__(
        self,
        model_wrapper: WhisperASRModel,
        audio_config: AudioConfig = AudioConfig(),
        whisper_config: WhisperConfig = WhisperConfig(),
    ):
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.processor = model_wrapper.processor
        self.device = model_wrapper.device
        self.audio_cfg = audio_config
        self.whisper_cfg = whisper_config

    def generate_batch(
        self,
        audio_arrays: List[np.ndarray],
        num_candidates: Optional[int] = None,
    ) -> List[List[Dict]]:
        """
        Transcribes a list of audio arrays.
        Returns a list of candidate lists:
            [[{'text': str, 'logprob': float, 'rank': int}, ...], ...]

        ``num_candidates`` defaults to ``WhisperConfig.num_beams`` so the
        published 12-beam path is unchanged when the argument is omitted.
        """
        if not audio_arrays:
            return []
        n = self.whisper_cfg.num_beams if num_candidates is None else int(num_candidates)
        if n < 1:
            raise ValueError("num_candidates must be >= 1")

        sr = self.audio_cfg.sample_rate
        # Apply configured silence padding
        padded_audios = [
            pad_silence(arr, self.audio_cfg.silence_pad_seconds, sr)
            for arr in audio_arrays
        ]

        inputs = self.processor(
            padded_audios,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        input_features = inputs.input_features.to(self.device, dtype=torch_dtype)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        gen_kwargs = {
            "input_features": input_features,
            "max_new_tokens": self.whisper_cfg.max_new_tokens,
            "num_beams": n,
            "num_return_sequences": n,
            "early_stopping": True,
            "length_penalty": self.whisper_cfg.length_penalty,
            # Beam-search multinomial sampling (published N-best setup): with
            # do_sample=True, sequences_scores are length-penalized beam scores
            # on the temperature-warped distribution, not calibrated log-posteriors.
            "do_sample": bool(self.whisper_cfg.diverse_beam_search),
            "temperature": self.whisper_cfg.sampling_temperature if self.whisper_cfg.diverse_beam_search else 1.0,
            "output_scores": False,  # scores come from score_texts() below
            "return_dict_in_generate": True,
        }
        gen_kwargs.update(self.wrapper.prompt_kwargs)
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        texts = [strip_markers(t) for t in self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)]
        # Candidate scores: exact per-token log-probability under the model (T = 1),
        # by teacher forcing. generate() no longer returns sequences_scores /
        # beam_indices for Whisper, and transition scores without beam indices
        # are taken from the wrong beams.
        avg_logprobs = []
        for i in range(len(audio_arrays)):
            scored = self.wrapper.score_texts(padded_audios[i], texts[i * n : (i + 1) * n], sr)
            avg_logprobs.extend(x["mean_logprob"] for x in scored)

        batch_size = len(audio_arrays)
        results = []

        for i in range(batch_size):
            start = i * n
            end = start + n
            batch_texts = texts[start:end]
            batch_scores = avg_logprobs[start:end]

            candidates = [
                {"text": t, "logprob": float(s), "rank": rk}
                for rk, (t, s) in enumerate(zip(batch_texts, batch_scores))
            ]
            results.append(candidates)

        return results
