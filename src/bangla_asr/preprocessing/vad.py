"""
Voice Activity Detection (VAD) audio segmenter using Silero VAD.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch

from bangla_asr.config import AudioConfig, VADConfig


class SileroVADChunker:
    """
    Detects active speech timestamps and packs them into chunks bounded by min/max durations.
    Long pauses are retained as chunk boundaries, so Whisper does not receive
    unrelated speech separated by extended silence. Guarantees no chunk exceeds
    max_chunk_duration_s even on continuous unpaused speech.
    """

    def __init__(
        self,
        audio_config: AudioConfig = AudioConfig(),
        vad_config: VADConfig = VADConfig(),
    ):
        self.audio_cfg = audio_config
        self.vad_cfg = vad_config

        # Try silero_vad package first, then fallback to torch.hub
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
            self.model = load_silero_vad()
            self._get_timestamps_fn = get_speech_timestamps
            self._use_hub_signature = False
        except ImportError:
            self.model, utils = torch.hub.load(
                repo_or_dir=self.vad_cfg.repo_or_dir,
                model=self.vad_cfg.model_name,
                trust_repo=True,
            )
            self._get_timestamps_fn = utils[0]
            self._use_hub_signature = True

    def get_speech_timestamps(self, audio_array: np.ndarray) -> List[Dict[str, int]]:
        """Returns sample-level [{'start': int, 'end': int}, ...] timestamps."""
        tensor = torch.from_numpy(np.ascontiguousarray(audio_array)).float()
        kwargs = {
            "sampling_rate": self.audio_cfg.sample_rate,
            "threshold": self.vad_cfg.threshold,
            "min_speech_duration_ms": self.vad_cfg.min_speech_duration_ms,
            "min_silence_duration_ms": self.vad_cfg.min_silence_duration_ms,
            "speech_pad_ms": self.vad_cfg.speech_pad_ms,
        }
        try:
            if self._use_hub_signature:
                timestamps = self._get_timestamps_fn(tensor, self.model, **kwargs)
            else:
                timestamps = self._get_timestamps_fn(
                    tensor,
                    self.model,
                    return_seconds=False,
                    **kwargs,
                )
        except TypeError:
            # Older Silero builds only accept threshold + sampling_rate.
            if self._use_hub_signature:
                timestamps = self._get_timestamps_fn(
                    tensor,
                    self.model,
                    sampling_rate=self.audio_cfg.sample_rate,
                    threshold=self.vad_cfg.threshold,
                )
            else:
                timestamps = self._get_timestamps_fn(
                    tensor,
                    self.model,
                    sampling_rate=self.audio_cfg.sample_rate,
                    threshold=self.vad_cfg.threshold,
                    return_seconds=False,
                )
        return timestamps or []

    def _find_low_energy_split(
        self,
        audio_array: np.ndarray,
        chunk_start: int,
        latest_end: int,
        min_samples: int,
    ) -> int:
        """Returns a low-energy split point at or shortly before latest_end.

        Silero can label a long narrated passage as one continuous speech
        region. Splitting exactly at the duration limit then risks cutting a
        syllable. This bounded backward search keeps the chunk within the
        limit while preferring a local energy trough. It deliberately does
        not look beyond latest_end, so it never creates an oversized chunk.
        """
        sr = self.audio_cfg.sample_rate
        search_samples = int(max(0.0, self.vad_cfg.split_search_window_s) * sr)
        frame_samples = max(1, int(max(0.001, self.vad_cfg.split_energy_frame_s) * sr))
        search_start = max(chunk_start + min_samples, latest_end - search_samples)

        if search_start >= latest_end:
            return latest_end

        half_frame = max(1, frame_samples // 2)
        candidates = list(range(search_start, latest_end + 1, frame_samples))
        if candidates[-1] != latest_end:
            candidates.append(latest_end)

        def rms_at(point: int) -> float:
            left = max(chunk_start, point - half_frame)
            right = min(len(audio_array), point + half_frame)
            frame = audio_array[left:right]
            return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if len(frame) else float("inf")

        return min(candidates, key=rms_at)

    def _clip_span(self, start: int, end: int, n_samples: int) -> Tuple[int, int]:
        start = max(0, min(int(start), n_samples))
        end = max(0, min(int(end), n_samples))
        return start, end

    def _split_long_spans(
        self,
        audio_array: np.ndarray,
        raw_timestamps: List[Dict[str, int]],
        max_samples: int,
        min_samples: int,
    ) -> List[Tuple[int, int]]:
        n_samples = len(audio_array)
        normalized_spans = []
        for seg in raw_timestamps:
            s, e = self._clip_span(seg["start"], seg["end"], n_samples)
            if e <= s:
                continue
            curr = s
            while e - curr > max_samples:
                # If a hard max split would leave an undersized tail, move
                # the boundary earlier while remaining within max_samples.
                reserved_tail = e - min_samples
                latest_end = min(curr + max_samples, reserved_tail if reserved_tail > curr else curr + max_samples)
                latest_end = max(curr + 1, latest_end)
                split = self._find_low_energy_split(
                    audio_array, curr, latest_end, min_samples
                )
                if split <= curr:
                    split = min(curr + max_samples, e)
                normalized_spans.append((curr, split))
                curr = split
            if e > curr:
                normalized_spans.append((curr, e))
        return normalized_spans

    def _pack_spans(
        self,
        spans: List[Tuple[int, int]],
        max_samples: int,
        preferred_samples: int,
        max_gap_samples: int,
    ) -> List[Tuple[int, int]]:
        packed_chunks = []
        cur_start, cur_end = spans[0]

        for s, e in spans[1:]:
            silence_gap = max(0, s - cur_end)
            combined = e - cur_start
            current_dur = cur_end - cur_start
            prefer_new_chunk = current_dur >= preferred_samples and silence_gap > 0
            if silence_gap > max_gap_samples or combined > max_samples or prefer_new_chunk:
                packed_chunks.append((cur_start, cur_end))
                cur_start, cur_end = s, e
            else:
                cur_end = e
        packed_chunks.append((cur_start, cur_end))
        return packed_chunks

    def _merge_short_chunks(
        self,
        packed_chunks: List[Tuple[int, int]],
        min_samples: int,
        max_samples: int,
        max_gap_samples: int,
    ) -> List[Tuple[int, int]]:
        """Absorb undersized fragments into a neighbour when the gap is small."""
        if not packed_chunks:
            return []

        merged: List[Tuple[int, int]] = []
        for s, e in packed_chunks:
            if merged and (e - s) < min_samples:
                prev_s, prev_e = merged[-1]
                gap = max(0, s - prev_e)
                if gap <= max_gap_samples and (e - prev_s) <= max_samples:
                    merged[-1] = (prev_s, e)
                    continue
            merged.append((s, e))

        final: List[Tuple[int, int]] = []
        i = 0
        while i < len(merged):
            s, e = merged[i]
            if (e - s) < min_samples and i + 1 < len(merged):
                ns, ne = merged[i + 1]
                gap = max(0, ns - e)
                if gap <= max_gap_samples and (ne - s) <= max_samples:
                    merged[i + 1] = (s, ne)
                    i += 1
                    continue
            if (e - s) >= min_samples:
                final.append((s, e))
            i += 1
        return final

    def segment_audio(self, audio_array: np.ndarray) -> List[Dict]:
        """
        Segments audio into chunks bounded between min_chunk_duration_s and max_chunk_duration_s.
        Each returned dict contains:
            'start_sample', 'end_sample', 'start_sec', 'end_sec', 'duration_sec', 'audio'
        """
        if audio_array is None or len(audio_array) == 0:
            return []

        sr = self.audio_cfg.sample_rate
        max_samples = int(self.audio_cfg.max_chunk_duration_s * sr)
        min_samples = int(self.audio_cfg.min_chunk_duration_s * sr)
        preferred_s = min(
            max(self.vad_cfg.preferred_chunk_duration_s, self.audio_cfg.min_chunk_duration_s),
            self.audio_cfg.max_chunk_duration_s,
        )
        preferred_samples = int(preferred_s * sr)
        max_gap_samples = int(max(0.0, self.vad_cfg.max_silence_gap_s) * sr)

        raw_timestamps = self.get_speech_timestamps(audio_array)
        if not raw_timestamps:
            return []

        normalized_spans = self._split_long_spans(
            audio_array, raw_timestamps, max_samples, min_samples
        )
        if not normalized_spans:
            return []

        packed_chunks = self._pack_spans(
            normalized_spans, max_samples, preferred_samples, max_gap_samples
        )
        packed_chunks = self._merge_short_chunks(
            packed_chunks, min_samples, max_samples, max_gap_samples
        )

        results = []
        for s, e in packed_chunks:
            chunk_audio = audio_array[s:e]
            dur = len(chunk_audio) / sr
            results.append({
                "start_sample": s,
                "end_sample": e,
                "start_sec": round(s / sr, 3),
                "end_sec": round(e / sr, 3),
                "duration_sec": round(dur, 3),
                "audio": chunk_audio,
            })
        return results
