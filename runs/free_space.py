#!/usr/bin/env python3
"""
Free disk space on this PC once every job runs in the cloud: delete audio and archives under data/ and the LoRA
optimizer checkpoints, keep every text file (manifests, N-best lists, predictions, labels, logs, results).

Deleted:
  - audio files under data/ (.flac .wav .mp3 .opus .m4a .ogg): clips and recordings. Every clip set is on the HF hub
    (OmiKhan/bangla-asr-hub, zips/); the raw DL Sprint recordings, Common Voice and OpenSLR 53 are public downloads.
  - .zip / .tar archives under data/ whose name is on the hub's zips/, plus openslr53-train.zip (on both Modal
    volumes and openslr.org). Any other archive is kept and listed.
  - checkpoints/*/checkpoint/ (optimizer state of finished local LoRA runs; the m1_lora adapter is on the hub).
Kept: everything else, including Lib/ and Scripts/ (the Python environment the figures and tests need).

  ./Scripts/python.exe runs/free_space.py          # plan: what would go, size per folder (deletes nothing)
  ./Scripts/python.exe runs/free_space.py --go     # delete (permanent; run the plan first)
"""

import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO = {".flac", ".wav", ".mp3", ".opus", ".m4a", ".ogg"}
ARCHIVE = {".zip", ".tar"}
HUB = "OmiKhan/bangla-asr-hub"
ALSO_SAFE = {"openslr53-train.zip"}  # on both Modal volumes and openslr.org


def hub_zips() -> set[str]:
    from huggingface_hub import HfApi
    return {Path(f).name for f in HfApi().list_repo_files(HUB, repo_type="dataset") if f.startswith("zips/")}


def walk(top: Path):
    stack = [top]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(Path(e.path))
                    elif e.is_file(follow_symlinks=False):
                        yield Path(e.path), e.stat(follow_symlinks=False).st_size
        except OSError as err:
            print("skip", d, err)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="delete (default: only print the plan)")
    args = ap.parse_args()

    on_hub = hub_zips() | ALSO_SAFE
    files, kept_archives = [], []
    by_dir, kept_bytes = defaultdict(int), 0
    for p, size in walk(ROOT / "data"):
        ext = p.suffix.lower()
        if ext in AUDIO or (ext in ARCHIVE and p.name in on_hub):
            files.append((p, size))
            by_dir[p.relative_to(ROOT).parts[1]] += size
        else:
            kept_bytes += size
            if ext in ARCHIVE:
                kept_archives.append(p)
    ckpts = [d for d in (ROOT / "checkpoints").glob("*/checkpoint") if d.is_dir()]
    ck_bytes = sum(s for d in ckpts for _, s in walk(d))

    print("To delete (data/<folder>):")
    for k, v in sorted(by_dir.items(), key=lambda x: -x[1]):
        print(f"  {v / 1e9:7.2f} GB  data/{k}")
    for d in ckpts:
        print(f"  (checkpoint) {d.relative_to(ROOT)}")
    total = sum(s for _, s in files) + ck_bytes
    print(f"Total: {total / 1e9:.2f} GB in {len(files):,} files + {len(ckpts)} checkpoint folders "
          f"({ck_bytes / 1e9:.2f} GB). Kept in data/: {kept_bytes / 1e9:.2f} GB of text and other files.")
    for p in kept_archives:
        print("  kept (not on the hub):", p.relative_to(ROOT))
    if not args.go:
        print("\nNothing deleted. Run again with --go to delete.")
        return
    for i, (p, _) in enumerate(files, 1):
        try:
            p.unlink()
        except OSError as err:
            print("could not delete", p, err)
        if i % 20000 == 0:
            print(f"  {i:,} / {len(files):,} files", flush=True)
    for d in ckpts:
        shutil.rmtree(d, ignore_errors=True)
    for d in sorted({p.parent for p, _ in files}, key=lambda x: -len(x.parts)):  # remove folders left empty
        try:
            d.rmdir()
        except OSError:
            pass
    print(f"Deleted {len(files):,} files and {len(ckpts)} checkpoint folders: {total / 1e9:.2f} GB freed.")


if __name__ == "__main__":
    main()
