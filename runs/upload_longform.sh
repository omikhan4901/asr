#!/usr/bin/env bash
# final model: the 22 raw DL Sprint eval recordings (16 kHz mono FLAC) + their full transcripts -> the private hub
# as zips/longform-eval.zip, for the Kaggle job longform:<run> (pipeline/eval_longform.py). CPU only, ~10-20 min
# (conversion + upload of ~1 GB est.). Resumable: converted files are kept.
#   bash runs/upload_longform.sh
set -eu
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || { [ -x ./Scripts/python.exe ] && echo ./Scripts/python.exe || echo python; } )}
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONIOENCODING=utf-8
"$PY" - <<'EOF'
import shutil, zipfile
from pathlib import Path
import pandas as pd, soundfile as sf
from bangla_asr.common.audio import load_audio
from huggingface_hub import HfApi
recs = sorted(pd.read_csv("data/splits_fa/eval.csv", encoding="utf-8-sig")["recording"].unique())
root = Path("data/longform/longform-eval")
(root / "audio").mkdir(parents=True, exist_ok=True); (root / "annotation").mkdir(parents=True, exist_ok=True)
for r in recs:
    dst = root / "audio" / f"{r}.flac"
    if not dst.exists():
        wav, _ = load_audio(f"data/raw/train/audio/{r}.wav", 16000)
        sf.write(dst.with_suffix(".tmp.flac"), wav, 16000, format="FLAC"); dst.with_suffix(".tmp.flac").replace(dst)
    shutil.copy(f"data/raw/train/annotation/{r}.txt", root / "annotation" / f"{r}.txt")
    print(r, f"{dst.stat().st_size / 1e6:.0f} MB", flush=True)
z = Path("data/longform/longform-eval.zip")
with zipfile.ZipFile(z.with_suffix(".tmp"), "w", zipfile.ZIP_STORED) as f:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            f.write(p, p.relative_to(root.parent).as_posix())
z.with_suffix(".tmp").replace(z)
print(f"{z}: {len(recs)} recordings, {z.stat().st_size / 1e9:.2f} GB; uploading...", flush=True)
HfApi().upload_file(repo_id="OmiKhan/bangla-asr-hub", repo_type="dataset", path_or_fileobj=str(z),
                    path_in_repo="zips/longform-eval.zip", commit_message="22 raw DL Sprint eval recordings (long-form)")
print("done: zips/longform-eval.zip is on the hub")
EOF
