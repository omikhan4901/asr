#!/usr/bin/env python3
"""
Turn the resumable checkpoint of a stopped fine-tuning run into a model that can be scored.

11_finetune.py writes final/ only after the last step. When a run stops early (Modal
budget cap, credit exhausted, preemption), the work done so far lives in
<out_dir>/checkpoint/state.pt (trainable weights + optimizer). This rebuilds the model
exactly as 11_finetune.py does (recipe from <out_dir>/config.json), loads the trained
weights and saves them like final/: a LoRA adapter (score with eval_zero_shot.py
--adapter) or a full model (--model). The optimizer state is not needed for scoring.

Output: <out_dir>/partial/ (+ training_state.json with the step it holds). CPU only.
  PYTHONPATH=src ./Scripts/python.exe pipeline/checkpoint_to_model.py --out_dir checkpoints/modal/m1_full
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only: never touches a GPU that is training
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bangla_asr.common.safe_io import atomic_write_text  # noqa: E402

spec = importlib.util.spec_from_file_location("finetune", Path(__file__).with_name("11_finetune.py"))
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True, help="run directory with config.json and checkpoint/")
    a = p.parse_args()
    out = Path(a.out_dir)
    ck = ft.find_checkpoint(out)
    if ck is None:
        sys.exit(f"No checkpoint in {out} (need checkpoint/ or checkpoint.old/ with state.pt and progress.json)")
    recipe = argparse.Namespace(**json.loads((out / "config.json").read_text(encoding="utf-8")))
    state = json.loads((ck / "progress.json").read_text(encoding="utf-8"))

    model, base, processor, _ = ft.build_model(recipe, torch.device("cpu"))
    saved = torch.load(ck / "state.pt", map_location="cpu", weights_only=False)["model"]
    res = model.load_state_dict(saved, strict=False)
    if res.unexpected_keys or set(saved) != set(ft.trainable_state(model)):
        sys.exit(f"{ck} does not match the {recipe.method} model ({len(res.unexpected_keys)} unexpected keys)")

    dst = out / "partial"
    base.config.use_cache = True  # as final/: inference needs the KV cache
    (model if recipe.method == "lora" else base).save_pretrained(dst)
    processor.save_pretrained(dst)
    atomic_write_text(dst / "training_state.json", json.dumps({**state, "from": str(ck)}, indent=2))
    print(f"{recipe.method} model at step {state['step']} (epoch {state['epoch']}) -> {dst}")


if __name__ == "__main__":
    main()
