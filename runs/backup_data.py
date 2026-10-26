#!/usr/bin/env python3
"""
Copy everything that exists only in the cloud (Hugging Face hub, both Modal volumes) to this PC, into
data/backup/ (git-ignored). Run AFTER the OpenSLR prep has finished. Without --go it only prints the plan.

  ./Scripts/python.exe runs/backup_data.py            # plan: what, from where, sizes, free disk
  ./Scripts/python.exe runs/backup_data.py --go       # copy (resumable: finished items are skipped)
  ./Scripts/python.exe runs/backup_data.py --go --with_openslr   # also the ~15 GB OpenSLR 53 zip (public anyway)

What is copied:
  models/<run>/       every trained model; full models stored as fp16 safetensors (half the size; scoring
                      already runs in fp16 on the GPU, so results are identical); the LoRA adapter as is
  zips/               kathbath-train.zip, shrutilipi-train.zip (+ openslr53-train.zip with --with_openslr)
  modal_<profile>/    every small file of both volumes: run logs, configs, dev predictions, manifests, results
Not copied: optimizer checkpoints (checkpoint/, ~9 GB per run; every run finished), partial/ and best/ copies of
final models, the HF model cache on the volumes, zips that are already on this PC.
Needs: the PC's Modal profiles 'friend' and 'mehboobehsankhan', and `hf auth login` (done).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "backup"
HUB = "OmiKhan/bangla-asr-hub"
VOLUME = "bangla-asr-ft"
# run -> Modal profile whose volume has it (used only when the model is not on the hub)
RUNS = {"m1_full": "mehboobehsankhan", "m1_decoder": "mehboobehsankhan", "m1_encoder": "mehboobehsankhan",
        "m1_lora": "mehboobehsankhan", "m1_full_lr5e-6": "friend", "m1_full_lr2e-5": "friend", "m1_full_4ep": "friend",
        "m2_cv": "friend", "m3_kathbath": "friend", "m4_shrutilipi": "friend", "m5_merged": "friend",
        "m6_harmonized": "friend", "m5_full": "friend", "m5_300": "friend"}  # last two: after the November runs
ZIPS = ["kathbath-train", "shrutilipi-train"]
BIG_DIRS = ("checkpoint", "checkpoint.tmp", "checkpoint.old", "partial", "final", "best")


def free_gb() -> float:
    return shutil.disk_usage(OUT if OUT.exists() else ROOT).free / 1e9


def modal_call(profile: str, code: str) -> str:
    """Run a small Modal SDK snippet under one profile (the active profile is not changed)."""
    env = {**os.environ, "MODAL_PROFILE": profile, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        raise SystemExit(f"Modal ({profile}) failed:\n{r.stderr[-2000:]}")
    return r.stdout


LIST = """
import json, modal
v = modal.Volume.from_name("%s")
print(json.dumps([[e.path, e.size or 0, str(e.type)] for e in v.listdir("/", recursive=True)]))
""" % VOLUME
GET = """
import sys, modal
from pathlib import Path
v = modal.Volume.from_name("%s")
for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
    d = Path(dst); d.parent.mkdir(parents=True, exist_ok=True); t = d.with_name(d.name + ".part")
    with t.open("wb") as f:
        for chunk in v.read_file(src):
            f.write(chunk)
    t.replace(d)
""" % VOLUME


def volume_files(profile):
    files = json.loads(modal_call(profile, LIST).strip().splitlines()[-1])
    return {p.strip("/"): s for p, s, t in files if not t.endswith("DIRECTORY")}


def volume_get(profile, pairs):
    for i in range(0, len(pairs), 50):  # batches keep the argument list short
        chunk = pairs[i:i + 50]
        env = {**os.environ, "MODAL_PROFILE": profile, "PYTHONIOENCODING": "utf-8"}
        args = [x for src, dst in chunk for x in (src, str(dst))]
        r = subprocess.run([sys.executable, "-W", "ignore", "-c", GET, *args], env=env)
        if r.returncode:
            raise SystemExit(f"copy from the {profile} volume failed (run again to continue)")


def hub_files():
    from huggingface_hub import HfApi
    return {s.rfilename: s.size or 0 for s in HfApi().dataset_info(HUB, files_metadata=True).siblings}


def to_fp16(src_dir: Path, dst_dir: Path):
    """Full model folder -> same folder with fp16 weights; checked tensor by tensor (names, shapes)."""
    import torch
    from safetensors.torch import load_file, save_file
    sd = load_file(str(src_dir / "model.safetensors"))
    half = {k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in sd.items()}
    tmp = dst_dir.with_name(dst_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file(half, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    back = load_file(str(tmp / "model.safetensors"))
    assert back.keys() == sd.keys() and all(back[k].shape == sd[k].shape for k in sd), "fp16 copy mismatch"
    for f in src_dir.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy(f, tmp / f.name)
    (tmp / "BACKUP_NOTE.txt").write_text("fp16 copy of the trained fp32 model (scoring runs in fp16 on GPU).\n", encoding="utf-8")
    shutil.rmtree(dst_dir, ignore_errors=True)
    tmp.rename(dst_dir)
    del sd, half, back
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--with_openslr", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("listing the hub and both Modal volumes (read-only)...", flush=True)
    hub = hub_files()
    vols = {p: volume_files(p) for p in ("friend", "mehboobehsankhan")}

    plan = []  # (kind, label, size_bytes, action)
    for run, prof in RUNS.items():
        dst = OUT / "models" / run
        lora = run == "m1_lora"
        on_hub = f"models/{run}/" + ("adapter_model.safetensors" if lora else "model.safetensors") in hub
        size = sum(s for f, s in hub.items() if f.startswith(f"models/{run}/")) if on_hub else \
            sum(s for f, s in vols[prof].items() if f.startswith(f"checkpoints/{run}/final/"))
        done = (dst / ("adapter_model.safetensors" if lora else "model.safetensors")).exists()
        plan.append(("model", run, size, None if done or size == 0 else ("hub" if on_hub else prof)))
    for z in ZIPS + (["openslr53-train"] if a.with_openslr else []):
        dst = OUT / "zips" / f"{z}.zip"
        src = "hub" if f"zips/{z}.zip" in hub else ("friend" if f"zips/{z}.zip" in vols["friend"] else None)
        size = hub.get(f"zips/{z}.zip") or vols["friend"].get(f"zips/{z}.zip", 0)
        plan.append(("zip", z, size, None if dst.exists() and dst.stat().st_size == size else src))
    small = {}
    for prof, files in vols.items():
        for f, s in files.items():
            parts = f.split("/")
            if parts[0] in ("hf", "hf_data", "zips") or any(p in BIG_DIRS for p in parts[1:-1]) or f.endswith(".pt"):
                continue
            dst = OUT / f"modal_{prof}" / f
            if not (dst.exists() and dst.stat().st_size == s):
                small.setdefault(prof, []).append((f, dst, s))
    for prof, items in small.items():
        plan.append(("small", f"{len(items)} small files from the {prof} volume", sum(s for _, _, s in items), prof))

    need = 0.0
    print(f"\n{'item':52s} {'size':>9s}  source")
    for kind, label, size, src in plan:
        eff = size / 2 if kind == "model" and label != "m1_lora" else size
        if src:
            need += eff
        print(f"{kind + ': ' + label:52s} {size / 1e9:8.2f}G  {src or ('already here' if size else 'not trained yet / not found')}"
              + ("  (stored as fp16: half)" if kind == "model" and label != "m1_lora" and src else ""))
    print(f"\nto store: ~{need / 1e9:.1f} GB (+ one 3 GB temporary file at a time); free on this disk: {free_gb():.1f} GB")
    if src_missing := [l for k, l, s, src in plan if src is None and s == 0]:
        print("WARNING, not found anywhere:", src_missing)
    if not a.go:
        print("\nplan only. Run again with --go to copy.")
        return
    if need / 1e9 + 4 > free_gb():
        raise SystemExit("not enough free disk space for this plan (needs ~4 GB margin): free some space first")

    tmp = OUT / "_tmp"
    for kind, label, size, src in plan:
        if not src:
            continue
        print(f"\n== {kind}: {label} from {src}", flush=True)
        if kind == "model":
            shutil.rmtree(tmp, ignore_errors=True)
            if src == "hub":
                from huggingface_hub import snapshot_download
                snapshot_download(HUB, repo_type="dataset", allow_patterns=[f"models/{label}/*"], local_dir=str(tmp))
                got = tmp / "models" / label
            else:
                files = [f for f in vols[src] if f.startswith(f"checkpoints/{label}/final/")]
                volume_get(src, [(f, tmp / Path(f).name) for f in files])
                got = tmp
            if label == "m1_lora":
                dst = OUT / "models" / label
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(got, dst)
            else:
                to_fp16(got, OUT / "models" / label)
            shutil.rmtree(tmp, ignore_errors=True)
        elif kind == "zip":
            dst = OUT / "zips" / f"{label}.zip"
            if src == "hub":
                from huggingface_hub import hf_hub_download
                p = hf_hub_download(HUB, f"zips/{label}.zip", repo_type="dataset", local_dir=str(tmp))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(p, dst)
            else:
                volume_get(src, [(f"zips/{label}.zip", dst)])
            assert dst.stat().st_size == size, f"{label}: size mismatch"
        else:
            volume_get(src, [(f, d) for f, d, _ in small[src]])
        print(f"   done; free disk {free_gb():.1f} GB", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nall copied into data/backup/.")


if __name__ == "__main__":
    main()
