"""
Wav2Vec2 CTC acoustic model with greedy 1-best and optional beam-search N-best.

Greedy argmax is used only when ``n == 1``. Requesting ``n > 1`` requires
``pyctcdecode`` and raises ``CTCBeamSearchUnavailableError`` if it is missing.
KenLM is optional and is never required to produce an acoustic N-best list.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from bangla_asr.config import Wav2Vec2Config
from bangla_asr.models.base import ASRModel, CTCBeamSearchUnavailableError, Hypothesis
from bangla_asr.models.ctc.beam_decoder import CTCBeamDecoder, require_pyctcdecode


def vocab_labels_from_processor(processor) -> List[str]:
    """Return CTC labels ordered by token id for ``pyctcdecode``."""
    vocab = processor.tokenizer.get_vocab()
    size = max(vocab.values()) + 1
    labels = [""] * size
    for token, idx in vocab.items():
        labels[int(idx)] = token
    return labels


class Wav2Vec2Model(ASRModel):
    """XLS-R / Wav2Vec2 CTC wrapper implementing ``ASRModel``."""

    model_family = "wav2vec2_ctc"

    def __init__(
        self,
        config: Optional[Wav2Vec2Config] = None,
        device: Optional[str] = None,
        *,
        processor=None,
        model=None,
        beam_decoder: Optional[CTCBeamDecoder] = None,
    ) -> None:
        self.cfg = config or Wav2Vec2Config()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = processor
        self.model = model
        self._beam_decoder = beam_decoder

        if self.processor is None or self.model is None:
            self._load_checkpoint()

        self.model.eval()
        self._labels = vocab_labels_from_processor(self.processor)

    def _load_checkpoint(self) -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        print(f"Loading Wav2Vec2 CTC model: {self.cfg.model_id} on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.cfg.model_id)
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model_kwargs = {"torch_dtype": torch_dtype}
        if self.device.type == "cuda" and self.cfg.attn_implementation:
            model_kwargs["attn_implementation"] = self.cfg.attn_implementation
        self.model = Wav2Vec2ForCTC.from_pretrained(self.cfg.model_id, **model_kwargs)
        self.model = self.model.to(self.device)

    def _ensure_beam_decoder(self) -> CTCBeamDecoder:
        """Build the CTC beam decoder on first N-best call; fail loud if unavailable."""
        if self._beam_decoder is not None:
            return self._beam_decoder
        require_pyctcdecode()
        self._beam_decoder = CTCBeamDecoder(
            labels=self._labels,
            kenlm_model_path=self.cfg.kenlm_model_path,
            alpha=self.cfg.lm_alpha,
            beta=self.cfg.lm_beta,
        )
        return self._beam_decoder

    def _forward_logits(self, audios: Sequence[np.ndarray], sampling_rate: int) -> List[np.ndarray]:
        """Run the CTC encoder and return per-utterance ``(time, vocab)`` logits."""
        if not audios:
            return []
        inputs = self.processor(
            list(audios),
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = None
        if getattr(inputs, "attention_mask", None) is not None:
            attention_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            if attention_mask is not None:
                logits = self.model(input_values, attention_mask=attention_mask).logits
            else:
                logits = self.model(input_values).logits
        # logits: (batch, time, vocab)
        lengths = None
        if attention_mask is not None:
            # Approximate valid time steps from the attention mask ratio.
            time_steps = logits.shape[1]
            raw_lengths = attention_mask.sum(dim=1).float()
            max_len = raw_lengths.max().clamp(min=1.0)
            lengths = torch.clamp((raw_lengths / max_len * time_steps).long(), min=1, max=time_steps)
        arrays: List[np.ndarray] = []
        for i in range(logits.shape[0]):
            t = int(lengths[i]) if lengths is not None else logits.shape[1]
            arrays.append(logits[i, :t].float().cpu().numpy())
        return arrays

    def greedy_decode_logits(self, logits: np.ndarray) -> str:
        """Collapse-argmax CTC decode using the HuggingFace tokenizer."""
        pred_ids = np.asarray(logits).argmax(axis=-1)
        return str(self.processor.decode(pred_ids)).strip()

    def transcribe_nbest(
        self,
        audio: np.ndarray,
        n: int,
        *,
        sampling_rate: int = 16000,
    ) -> List[Hypothesis]:
        """Greedy 1-best when ``n == 1``; beam-search N-best otherwise."""
        batch = self.transcribe_nbest_batch([audio], n, sampling_rate=sampling_rate)
        return batch[0] if batch else []

    def transcribe_nbest_batch(
        self,
        audios: Sequence[np.ndarray],
        n: int,
        *,
        sampling_rate: int = 16000,
    ) -> List[List[Hypothesis]]:
        """Batched CTC forward + per-utterance greedy or beam decode."""
        if n < 1:
            raise ValueError("n must be >= 1")
        logit_list = self._forward_logits(audios, sampling_rate=sampling_rate)
        if n == 1:
            return [[self._greedy_hypothesis(logits)] for logits in logit_list]

        try:
            decoder = self._ensure_beam_decoder()
        except CTCBeamSearchUnavailableError:
            raise
        except ImportError as exc:
            raise CTCBeamSearchUnavailableError(
                "CTC N-best requires pyctcdecode. Install with: pip install pyctcdecode"
            ) from exc

        batches: List[List[Hypothesis]] = []
        for logits in logit_list:
            beams = decoder.decode_nbest(logits, n=n)
            if not beams:
                raise RuntimeError(
                    "CTC beam search returned no hypotheses. Refusing to fall back to greedy "
                    "1-best, because that would silently invalidate MBR/oracle comparison."
                )
            hyps = [
                Hypothesis(text=text, logprob=score, rank=rank)
                for rank, (text, score) in enumerate(beams)
            ]
            batches.append(hyps)
        return batches

    def _greedy_hypothesis(self, logits: np.ndarray) -> Hypothesis:
        text = self.greedy_decode_logits(logits)
        # Greedy has no calibrated sequence log-prob; use 0.0 so MBR still
        # accepts a singleton list. Callers requesting N-best must use beams.
        return Hypothesis(text=text, logprob=0.0, rank=0)
