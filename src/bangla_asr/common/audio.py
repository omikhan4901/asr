"""
Robust audio input/output, normalization, resampling, and padding utilities.
"""

import os
import subprocess
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch


def load_audio(
    file_path: Union[str, Path],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file as a mono 1D float32 numpy array resampled to target_sr.
    Attempts torchaudio first, then falls back to soundfile / librosa if available.
    """
    path_str = str(file_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Audio file not found: {path_str}")
    if os.path.getsize(path_str) == 0:
        raise ValueError(f"Audio file is empty: {path_str}")

    # Try torchaudio (standard in PyTorch environments)
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path_str)
        # Average channels to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        audio_array = waveform.squeeze(0).cpu().numpy().astype(np.float32)
        return audio_array, target_sr
    except Exception:
        pass

    # Try soundfile fallback
    try:
        import soundfile as sf
        audio_array, sr = sf.read(path_str, dtype="float32")
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        if sr != target_sr:
            try:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                # Basic linear interpolation fallback if librosa is absent
                orig_len = len(audio_array)
                new_len = int(orig_len * target_sr / sr)
                audio_array = np.interp(
                    np.linspace(0, orig_len, new_len, endpoint=False),
                    np.arange(orig_len),
                    audio_array,
                ).astype(np.float32)
        return audio_array, target_sr
    except Exception:
        pass

    # Try librosa fallback
    try:
        import librosa
        audio_array, sr = librosa.load(path_str, sr=target_sr, mono=True)
        return audio_array.astype(np.float32), target_sr
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio from {path_str} with all available backends: {exc}")


def save_audio(
    file_path: Union[str, Path],
    audio_array: np.ndarray,
    sample_rate: int = 16000,
) -> None:
    """Saves a 1D float32 numpy array to a WAV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
        sf.write(str(path), audio_array, sample_rate)
        return
    except ImportError:
        pass

    try:
        import torchaudio
        tensor = torch.from_numpy(audio_array).unsqueeze(0).float()
        torchaudio.save(str(path), tensor, sample_rate)
        return
    except Exception as exc:
        raise RuntimeError(f"Failed to save audio to {file_path}: {exc}")


def pad_silence(
    audio_array: np.ndarray,
    pad_duration_s: float = 0.2,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Prepends and appends constant zero silence to an audio array."""
    if pad_duration_s <= 0:
        return audio_array
    pad_samples = int(pad_duration_s * sample_rate)
    return np.pad(audio_array, (pad_samples, pad_samples), mode="constant")


def extract_audio_from_video(
    video_path: Union[str, Path],
    out_wav_path: Union[str, Path],
    sample_rate: int = 16000,
) -> None:
    """Extracts a 16kHz mono WAV file from any video or audio container via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", str(sample_rate), "-ac", "1", "-vn",
        str(out_wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-800:]}")
