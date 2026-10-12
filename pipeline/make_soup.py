#!/usr/bin/env python3
"""
Weight averaging ("model soup", WiSE-FT) of Whisper checkpoints that share one initialisation (tugstugi).

Every member is a full model folder (model.safetensors + config/tokenizer files) or the word `base` for
tugstugi itself. Weights are uniform per listed member, so repeating a member raises its weight:
  m1_full+m5_merged            -> 0.5 / 0.5
  m1_full+m1_full+m1_full+base -> 0.75 fine-tuned, 0.25 zero-shot (WiSE-FT, alpha 0.75)
The result is one ordinary model folder: same size and speed as any member.

  python pipeline/make_soup.py --members m1_full+m5_merged --models_dir models --out soups/m1_full+m5_merged
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TUGSTUGI = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"


def member_dir(name: str, models_dir: Path) -> Path:
    if name == "base":
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(TUGSTUGI, allow_patterns=["model.safetensors", "*config.json", "tokenizer.json",
                                                                "vocab.json", "merges.txt", "added_tokens.json",
                                                                "special_tokens_map.json", "normalizer.json"]))
    return models_dir / name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", required=True, help="names joined by '+'; 'base' = tugstugi")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "model.safetensors").exists():
        print(f"{out} exists, skipped")
        return
    counts = Counter(args.members.split("+"))
    total = sum(counts.values())
    acc, ref_keys = None, None
    for name, c in counts.items():
        sd = load_file(str(member_dir(name, Path(args.models_dir)) / "model.safetensors"))
        if ref_keys is None:
            ref_keys = set(sd)
            acc = {k: v.float() * (c / total) for k, v in sd.items()}
        else:
            assert set(sd) == ref_keys, f"{name}: different tensor names, not the same architecture"
            for k, v in sd.items():
                acc[k] += v.float() * (c / total)
        del sd
        print(f"added {name} x{c}/{total}", flush=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in acc.items()}, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    # config, tokenizer and generation settings from the first fine-tuned member (base only if alone)
    src = member_dir(next((n for n in counts if n != "base"), "base"), Path(args.models_dir))
    for f in src.iterdir():
        if f.is_file() and f.name != "model.safetensors" and f.suffix in (".json", ".txt"):
            shutil.copy(f, tmp / f.name)
    tmp.rename(out)
    print(f"soup {dict(counts)} -> {out}", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
