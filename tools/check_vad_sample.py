import random, sys
from pathlib import Path
sys.path.insert(0, "src")

from bangla_asr.common.audio import load_audio, save_audio
from bangla_asr.config import AudioConfig, VADConfig
from bangla_asr.preprocessing.vad import SileroVADChunker

audio_dir = Path("data/raw/train/audio")
out_dir = Path("scratch/vad_check")
out_dir.mkdir(parents=True, exist_ok=True)

wavs = list(audio_dir.glob("*.wav"))
sample = random.sample(wavs, min(5, len(wavs)))

chunker = SileroVADChunker(AudioConfig(), VADConfig())
for wav in sample:
    audio, sr = load_audio(wav, target_sr=16000)
    chunks = chunker.segment_audio(audio)
    print(f"\n{wav.name}: {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c['start_sec']}s - {c['end_sec']}s ({c['duration_sec']}s)")
        save_audio(out_dir / f"{wav.stem}_{i:03d}.wav", c["audio"], sample_rate=sr)
