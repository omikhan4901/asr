"""
Web Subtitle Generation Pipeline orchestrating audio extraction, VAD chunking,
Whisper diverse beam candidate generation, and MBR consensus rescoring.
"""

import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np

# Ensure project root and src are on python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
src_dir = PROJECT_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from bangla_asr.common.audio import extract_audio_from_video, load_audio
from bangla_asr.config import AudioConfig, VADConfig, WhisperConfig, MBRConfig
from bangla_asr.models.autoregressive.whisper_model import WhisperASRModel
from bangla_asr.models.autoregressive.generator import BatchedNBestGenerator
from bangla_asr.preprocessing.vad import SileroVADChunker
from bangla_asr.rescoring.mbr import MBRSelector

try:
    from web.backend.srt_utils import chunks_to_srt, get_srt_preview
except ImportError:
    from srt_utils import chunks_to_srt, get_srt_preview


class SubtitleGenerator:
    """Production subtitle generator using the unified bangla_asr library."""

    def __init__(
        self,
        audio_config: AudioConfig = AudioConfig(),
        whisper_config: WhisperConfig = WhisperConfig(num_beams=12, diverse_beam_search=True, sampling_temperature=0.7),
        mbr_config: MBRConfig = MBRConfig(temperature=0.5),
        device: Optional[str] = None,
    ):
        self.audio_cfg = audio_config
        self.whisper_cfg = whisper_config
        self.mbr_cfg = mbr_config

        print("Initializing SubtitleGenerator models...")
        self.vad_chunker = SileroVADChunker(audio_config=self.audio_cfg)
        self.model_wrapper = WhisperASRModel(config=self.whisper_cfg, device=device)
        self.generator = BatchedNBestGenerator(
            model_wrapper=self.model_wrapper,
            audio_config=self.audio_cfg,
            whisper_config=self.whisper_cfg,
        )
        self.mbr_selector = MBRSelector(config=self.mbr_cfg)
        print("SubtitleGenerator initialized.")

    def process_video(
        self,
        video_path: str,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict:
        """
        Executes end-to-end subtitle generation:
          1. Extract audio via ffmpeg
          2. VAD segmenting
          3. Whisper N-best generation (with 200ms padding)
          4. MBR consensus rescoring
          5. SRT generation
        """
        if progress_cb:
            progress_cb("audio", 0, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "extracted.wav"
            extract_audio_from_video(video_path, wav_path, sample_rate=self.audio_cfg.sample_rate)

            if progress_cb:
                progress_cb("audio", 1, 1)
                progress_cb("chunk", 0, 1)

            audio_arr, sr = load_audio(wav_path, target_sr=self.audio_cfg.sample_rate)
            raw_chunks = self.vad_chunker.segment_audio(audio_arr)

            total_chunks = len(raw_chunks)
            if progress_cb:
                progress_cb("chunk", total_chunks, total_chunks)

            if total_chunks == 0:
                return {"srt_content": "", "preview": "", "num_chunks": 0, "total_speech_s": 0.0}

            # Batch generation
            if progress_cb:
                progress_cb("asr", 0, total_chunks)

            chunk_audios = [c["audio"] for c in raw_chunks]
            batch_size = self.whisper_cfg.batch_size
            all_candidates = []

            for i in range(0, total_chunks, batch_size):
                batch = chunk_audios[i:i + batch_size]
                cands_batch = self.generator.generate_batch(batch)
                all_candidates.extend(cands_batch)
                if progress_cb:
                    progress_cb("asr", min(i + len(batch), total_chunks), total_chunks)

            # MBR Rescoring
            if progress_cb:
                progress_cb("rescore", 0, total_chunks)

            subtitled_chunks = []
            for i, (meta, candidates) in enumerate(zip(raw_chunks, all_candidates)):
                chosen = self.mbr_selector.select(candidates)
                subtitled_chunks.append({
                    "start": meta["start_sec"],
                    "end": meta["end_sec"],
                    "text": chosen["text"],
                })
                if progress_cb:
                    progress_cb("rescore", i + 1, total_chunks)

            if progress_cb:
                progress_cb("srt", 0, 1)

            srt_content = chunks_to_srt(subtitled_chunks)
            preview = get_srt_preview(srt_content)

            if progress_cb:
                progress_cb("srt", 1, 1)

            total_speech_sec = sum(c["duration_sec"] for c in raw_chunks)
            return {
                "srt_content": srt_content,
                "preview": preview,
                "num_chunks": total_chunks,
                "total_speech_s": round(total_speech_sec, 2),
            }
