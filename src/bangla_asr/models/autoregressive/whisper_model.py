"""
Whisper model loader and configuration manager with robust tokenization & prompt IDs.
"""

import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging as hf_logging

# Per-batch generate() warnings (max_length vs max_new_tokens, duplicate logits
# processors, tokenizer cleanup) are harmless and flood the batch logs.
hf_logging.set_verbosity_error()

from bangla_asr.config import WhisperConfig


class WhisperASRModel:
    """Encapsulates WhisperProcessor and WhisperForConditionalGeneration with verified Bangla prompt IDs."""

    def __init__(self, config: WhisperConfig = WhisperConfig(), device: str = None):
        self.cfg = config
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"Loading Whisper model: {self.cfg.model_id} on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(self.cfg.model_id)

        # Load FP16 on GPU, float32 on CPU
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model_kwargs = {"torch_dtype": torch_dtype}
        if self.device.type == "cuda" and self.cfg.attn_implementation:
            model_kwargs["attn_implementation"] = self.cfg.attn_implementation

        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.cfg.model_id,
            **model_kwargs,
        )
        if self.cfg.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Loading a LoRA adapter requires peft. Install with: pip install peft"
                ) from exc
            print(f"Loading LoRA adapter from {self.cfg.adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, self.cfg.adapter_path)
        self.model = self.model.to(self.device)

        # Prefer the fine-tuned checkpoint's settings. Some regional
        # checkpoints ship an older Whisper generation_config.json, however,
        # which lacks lang_to_id/task_to_id and cannot accept language/task
        # under modern Transformers. Keep a modern base config available to
        # fill only those compatibility fields.
        try:
            gen_config = GenerationConfig.from_pretrained(self.cfg.model_id)
        except Exception:
            gen_config = self.model.generation_config

        try:
            modern_base_config = GenerationConfig.from_pretrained(self.cfg.fallback_base_config)
        except Exception:
            modern_base_config = None

        if modern_base_config is not None:
            for attribute in ("lang_to_id", "task_to_id", "is_multilingual", "no_timestamps_token_id"):
                if not hasattr(gen_config, attribute) and hasattr(modern_base_config, attribute):
                    setattr(gen_config, attribute, getattr(modern_base_config, attribute))

        # Configure the Whisper prompt directly on a generation config which
        # has the modern language/task mappings. Do not pass language/task as
        # generate() kwargs: older fine-tuned configs reject those kwargs
        # before generation begins.
        self.prompt_kwargs = {}
        if hasattr(gen_config, "lang_to_id") and hasattr(gen_config, "task_to_id"):
            gen_config.language = self.cfg.language
            gen_config.task = self.cfg.task
            # Also pass them to generate(): current transformers ignores
            # generation_config.language for multilingual checkpoints such as
            # openai/whisper-large-v3-turbo (it auto-detects and wrote Bangla
            # speech in Devanagari). lang_to_id is guaranteed present here.
            self.prompt_kwargs = {"language": self.cfg.language, "task": self.cfg.task}
            if hasattr(gen_config, "forced_decoder_ids"):
                gen_config.forced_decoder_ids = None
        else:
            # Compatibility fallback for an offline setup where the base
            # generation config could not be downloaded.
            try:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=self.cfg.language,
                    task=self.cfg.task,
                )
            except (AttributeError, ValueError):
                forced_decoder_ids = None
            if hasattr(gen_config, "forced_decoder_ids"):
                gen_config.forced_decoder_ids = forced_decoder_ids
        gen_config.num_beams = self.cfg.num_beams
        gen_config.num_return_sequences = self.cfg.num_beams
        gen_config.early_stopping = True
        gen_config.length_penalty = self.cfg.length_penalty
        # Do NOT default output_scores/return_dict_in_generate to True here.
        # BatchedNBestGenerator needs sequences_scores and sets both flags
        # itself in its own gen_kwargs (self-contained). transcribe_batch()
        # below calls generate() and hands the result straight to
        # batch_decode() expecting a plain token-id tensor -- if this shared,
        # mutable generation_config defaulted return_dict_in_generate to True,
        # it would silently get
        # a GenerateOutput object instead and batch_decode() would iterate
        # its fields (sequences, then scores, ...) rather than decode tokens,
        # surfacing as "Boolean value of Tensor with more than one value is
        # ambiguous" deep inside the tokenizer.

        if self.cfg.diverse_beam_search:
            gen_config.do_sample = True
            gen_config.temperature = self.cfg.sampling_temperature
        else:
            gen_config.do_sample = False

        self.model.generation_config = gen_config
        self.model.eval()
        print("Whisper model successfully loaded and configured.")

    def transcribe_batch(
        self,
        audio_arrays,
        sampling_rate: int = 16000,
        max_new_tokens: int = None,
        pad_seconds: float = 0.2,
    ):
        """Greedy 1-best transcription. Default cap 440 new tokens (Whisper allows 448
        including prompt tokens); dense 25 s chunks reach ~250 tokens."""
        from bangla_asr.common.audio import pad_silence

        if not audio_arrays:
            return []

        padded = [
            pad_silence(arr, pad_seconds, sampling_rate) if pad_seconds > 0 else arr
            for arr in audio_arrays
        ]
        inputs = self.processor(
            padded,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        input_features = inputs.input_features.to(self.device, dtype=torch_dtype)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        gen_kwargs = {
            "input_features": input_features,
            "max_new_tokens": max_new_tokens or 440,
            "num_beams": 1,
            "num_return_sequences": 1,
            "do_sample": False,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        gen_kwargs.update(self.prompt_kwargs)

        with torch.no_grad():
            gen_ids = self.model.generate(**gen_kwargs)
        return [t.strip() for t in self.processor.batch_decode(gen_ids, skip_special_tokens=True)]
