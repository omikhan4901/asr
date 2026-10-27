"""
Web subtitle pipeline: audio extraction, voice-activity chunking, greedy transcription with the released model, SRT.

The same path as the whole-recording evaluation of the paper (pipeline/eval_longform.py: Silero VAD chunks, greedy
decoding, about 14% WER on 22 raw media recordings). Word times come from MMS_FA forced alignment of each chunk's
own transcript, so subtitles can be short without changing the text. The demo no longer samples 12 candidates and
rescores them with MBR: the paper shows that rescoring does not beat greedy decoding, at about 12 times the compute.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Ensure project root and src are on python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
src_dir = PROJECT_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from bangla_asr.common.audio import extract_audio_from_video, load_audio
from bangla_asr.config import AudioConfig, WhisperConfig
from bangla_asr.models.autoregressive.whisper_model import WhisperASRModel
from bangla_asr.preprocessing.forced_align import MMSForcedAligner
from bangla_asr.preprocessing.vad import SileroVADChunker

try:
    from web.backend.srt_utils import chunks_to_srt, get_srt_preview, visible_len, words_to_cues
except ImportError:
    from srt_utils import chunks_to_srt, get_srt_preview, visible_len, words_to_cues

MODEL_ID = os.environ.get("BANGLA_ASR_MODEL", "OmiKhan/bangla-asr-whisper-medium")  # released final model (Apache-2.0)


class VideoTooLong(ValueError):
    """The upload is longer than the demo allows."""


class SubtitleGenerator:
    """Subtitle generator using the unified bangla_asr library."""

    def __init__(
        self,
        audio_config: AudioConfig = AudioConfig(),
        whisper_config: WhisperConfig = WhisperConfig(model_id=MODEL_ID, num_beams=1, diverse_beam_search=False,
                                                      batch_size=8),
        device: Optional[str] = None,
        max_duration_s: Optional[float] = None,
    ):
        self.audio_cfg = audio_config
        self.whisper_cfg = whisper_config
        self.max_duration_s = max_duration_s

        print("Initializing SubtitleGenerator models...")
        self.vad_chunker = SileroVADChunker(audio_config=self.audio_cfg)
        self.model_wrapper = WhisperASRModel(config=self.whisper_cfg, device=device)
        # Word timings for short, readable subtitle lines; the same aligner that built the evaluation split
        self.aligner = MMSForcedAligner(device=device)
        print("SubtitleGenerator initialized.")

    def _timed_words(self, chunk: Dict, text: str, sr: int) -> List[Dict]:
        """Words of one chunk with times in the video, from forced alignment of the chunk's own transcript."""
        words = text.split()
        if not words:
            return []
        offset = chunk["start_sec"]
        try:
            aligned = self.aligner.align(chunk["audio"], words)
        except Exception as exc:  # a failed alignment must not lose the transcript
            print(f"Word alignment failed, spreading the words evenly: {exc}")
            aligned = []
        if any(w.aligned for w in aligned):
            return [{"text": w.text, "start": offset + w.start, "end": offset + w.end} for w in aligned]
        # Fallback: spread the words over the chunk in proportion to their length
        weights = [visible_len(w) + 1 for w in words]
        span = len(chunk["audio"]) / sr / sum(weights)
        out, t = [], offset
        for w, k in zip(words, weights):
            out.append({"text": w, "start": t, "end": t + k * span})
            t += k * span
        return out

    def process_video(
        self,
        video_path: str,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict:
        """
        Executes end-to-end subtitle generation:
          1. Extract audio via ffmpeg
          2. Voice-activity chunking
          3. Greedy transcription of every chunk with the released model, then word times by forced alignment
          4. SRT generation: words grouped into cues of at most two lines and seven seconds
        """
        if progress_cb:
            progress_cb("audio", 0, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "extracted.wav"
            extract_audio_from_video(video_path, wav_path, sample_rate=self.audio_cfg.sample_rate)
            audio_arr, sr = load_audio(wav_path, target_sr=self.audio_cfg.sample_rate)
            seconds = len(audio_arr) / sr
            if self.max_duration_s and seconds > self.max_duration_s:
                raise VideoTooLong(f"The video is {seconds / 60:.0f} minutes long; this demo accepts up to "
                                   f"{self.max_duration_s / 60:.0f} minutes.")

            if progress_cb:
                progress_cb("audio", 1, 1)
                progress_cb("chunk", 0, 1)

            raw_chunks = self.vad_chunker.segment_audio(audio_arr)
            total_chunks = len(raw_chunks)
            if progress_cb:
                progress_cb("chunk", total_chunks, total_chunks)

            if total_chunks == 0:
                return {"srt_content": "", "preview": "", "num_chunks": 0, "total_speech_s": 0.0}

            if progress_cb:
                progress_cb("asr", 0, total_chunks)

            words = []
            batch_size = self.whisper_cfg.batch_size
            for i in range(0, total_chunks, batch_size):
                chunks = raw_chunks[i:i + batch_size]
                texts = self.model_wrapper.transcribe_batch([c["audio"] for c in chunks], sampling_rate=sr,
                                                            num_beams=1)
                for chunk, text in zip(chunks, texts):
                    words.extend(self._timed_words(chunk, text, sr))
                if progress_cb:
                    progress_cb("asr", min(i + len(chunks), total_chunks), total_chunks)

            if progress_cb:
                progress_cb("srt", 0, 1)

            cues = words_to_cues(words, total_s=seconds)
            srt_content = chunks_to_srt(cues)
            preview = get_srt_preview(srt_content)

            if progress_cb:
                progress_cb("srt", 1, 1)

            total_speech_sec = sum(c["duration_sec"] for c in raw_chunks)
            return {
                "srt_content": srt_content,
                "preview": preview,
                "num_chunks": len(cues),
                "total_speech_s": round(total_speech_sec, 2),
            }
