#!/usr/bin/env python3
"""
Fine-tune tugstugi Whisper-medium on a chunk manifest (Phase 4: M1, E22, matrix runs).

--method
  lora     LoRA on every attention and MLP projection of encoder and decoder
  decoder  full decoder trained, encoder frozen
  encoder  full encoder trained, decoder frozen (E22 error-locus run)
  full     everything

Targets are the manifest text as is; the loss skips the forced prompt tokens
(<|bn|><|transcribe|><|notimestamps|>), so it covers the text and end-of-text.
bf16 autocast over fp32 weights, gradient checkpointing, SpecAugment as in the
checkpoint's config (tugstugi: mask_time_prob 0.05).

Resumable: the latest state (trainable weights, optimizer, scheduler, data
position, RNG) is saved atomically every --save_every optimizer steps, at each
epoch end, when --max_minutes runs out, and on Ctrl+C (finishes the current
step first). Re-running the same command continues from it; a resume with a
different recipe is refused.

Outputs in --out_dir:
  config.json        recipe of the first run
  train_log.csv      step, epoch, loss, lr, grad_norm, s/step, peak GPU memory
  dev_log.csv        step, dev loss, dev WER/CER (greedy, frozen normalizer)
  dev_predictions/step_<N>.jsonl   same format as results/zeroshot predictions
  checkpoint/        latest resumable state
  final/             LoRA adapter (lora) or full model, after the last step
  run_info.json

Example (10-minute smoke test):
  PYTHONPATH=src ./Scripts/python.exe pipeline/11_finetune.py --out_dir checkpoints/smoke_lora \
      --max_minutes 10 --dev_limit 64 --eval_every 40 --memory_probe
"""

import argparse
import csv
import functools
import json
import math
import os
import random
import shutil
import signal
import sys
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.common.audio import load_row_audio, pad_silence
from bangla_asr.common.provenance import RunRecord
from bangla_asr.common.safe_io import atomic_write_text
from bangla_asr.common.stats import corpus_rates, edit_counts
from bangla_asr.config import WhisperConfig
from bangla_asr.models.autoregressive.whisper_model import bangla_generation_config, strip_markers

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
DEFAULT_LR = {"lora": 2e-4, "decoder": 1e-5, "encoder": 1e-5, "full": 1e-5}
# Arguments that define the run; a resume must use the same values.
STOPPED = 3  # exit code when stopped before the last step, so runners do not mark the step done
RECIPE = ["model", "method", "train_manifest", "lr", "batch_size", "grad_accum", "epochs", "max_steps",
          "warmup_steps", "weight_decay", "max_grad_norm", "lora_r", "lora_alpha", "lora_dropout", "seed",
          "train_limit"]


class ChunkDataset(torch.utils.data.Dataset):
    """(log-mel features, token ids) per manifest row; audio sliced on the fly."""

    def __init__(self, rows, token_ids, feature_extractor):
        self.rows, self.ids, self.fe = rows, token_ids, feature_extractor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        audio = load_row_audio(self.rows[i])
        feats = self.fe(audio, sampling_rate=16000, return_tensors="np").input_features[0]
        return feats, self.ids[i]


def collate(batch, pad_id, n_prompt):
    """Decoder input = ids[:-1], labels = ids[1:] with the forced prompt tokens masked."""
    feats = torch.from_numpy(np.stack([b[0] for b in batch]))
    L = max(len(b[1]) for b in batch) - 1
    dec = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    for k, (_, ids) in enumerate(batch):
        t = torch.tensor(ids, dtype=torch.long)
        dec[k, : len(ids) - 1] = t[:-1]
        lab[k, : len(ids) - 1] = t[1:]
        lab[k, : n_prompt] = -100
    return feats, dec, lab


def _ignore_sigint(_):
    """Workers ignore Ctrl+C (Windows sends it to every process); the main process stops cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def append_csv(path, row):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(out, model, opt, sched, state):
    """Write to checkpoint.tmp, then swap, so a power cut never leaves a half-written checkpoint."""
    tmp, cur, old = out / "checkpoint.tmp", out / "checkpoint", out / "checkpoint.old"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    torch.save({"model": trainable_state(model), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "numpy": np.random.get_state(), "python": random.getstate()}}, tmp / "state.pt")
    atomic_write_text(tmp / "progress.json", json.dumps(state, indent=2))
    shutil.rmtree(old, ignore_errors=True)
    if cur.exists():
        cur.rename(old)
    tmp.rename(cur)
    shutil.rmtree(old, ignore_errors=True)


def find_checkpoint(out):
    for name in ("checkpoint", "checkpoint.old"):  # .old survives a cut between the two renames
        if (out / name / "progress.json").exists() and (out / name / "state.pt").exists():
            return out / name
    return None


def build_model(args, device):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(args.model)
    base = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32,
                                                           attn_implementation="sdpa")
    gen_config, prompt_kwargs = bangla_generation_config(WhisperConfig(model_id=args.model), processor, base)
    gen_config.num_beams, gen_config.do_sample = 1, False
    base.generation_config = gen_config
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.method == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                                lora_dropout=args.lora_dropout, target_modules=LORA_TARGETS))
    else:
        model = base
        part = {"decoder": base.model.decoder, "encoder": base.model.encoder, "full": base}[args.method]
        for p in base.parameters():
            p.requires_grad = False
        for p in part.parameters():
            p.requires_grad = True
        # Whisper's encoder positions are a fixed sinusoid table, frozen by the library; keep them fixed.
        base.model.encoder.embed_positions.requires_grad_(False)
    return model.to(device), base, processor, prompt_kwargs


@torch.no_grad()
def evaluate(model, base, processor, prompt_kwargs, rows, args, device, step, out):
    """Greedy dev decoding as in WhisperASRModel.transcribe_batch, plus teacher-forced dev loss."""
    model.eval()
    tok = processor.tokenizer
    preds, loss_sum, n_tok = [], 0.0, 0
    for i in range(0, len(rows), args.eval_batch_size):
        batch = rows[i : i + args.eval_batch_size]
        audios = [load_row_audio(r) for r in batch]
        inp = processor([pad_silence(a, 0.2, 16000) for a in audios], sampling_rate=16000, return_tensors="pt",
                        padding=True, return_attention_mask=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            ids = model.generate(input_features=inp.input_features.to(device), attention_mask=inp.attention_mask.to(device),
                                 max_new_tokens=440, num_beams=1, do_sample=False, **prompt_kwargs)
            feats = torch.from_numpy(np.stack([processor.feature_extractor(a, sampling_rate=16000, return_tensors="np")
                                               .input_features[0] for a in audios]))
            tids = [r["_ids"] for r in batch]
            _, dec, lab = collate(list(zip(feats.numpy(), tids)), tok.eos_token_id, args._n_prompt)
            logits = model(input_features=feats.to(device), decoder_input_ids=dec.to(device)).logits.float()
        mask = lab.to(device) != -100
        lp = torch.log_softmax(logits, -1).gather(-1, lab.to(device).clamp(min=0)[..., None])[..., 0]
        loss_sum += float(-(lp * mask).sum())
        n_tok += int(mask.sum())
        for r, h in zip(batch, processor.batch_decode(ids, skip_special_tokens=True)):
            preds.append({"utt_id": r["utt_id"], "group": str(r["recording"]), "ref": str(r["text"]),
                          "hyp": strip_markers(h)})
    pdir = out / "dev_predictions"
    pdir.mkdir(exist_ok=True)
    atomic_write_text(pdir / f"step_{step}.jsonl", "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds))
    m = corpus_rates([dict(edit_counts(p["hyp"], p["ref"]), group=p["group"]) for p in preds], group_key="group")
    row = {"step": step, "dev_loss": round(loss_sum / max(1, n_tok), 4), "dev_wer": round(100 * m["wer"], 2),
           "dev_cer": round(100 * m["cer"], 2), "n_utt": len(preds), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    append_csv(out / "dev_log.csv", row)
    print(f"[dev] step {step}: loss {row['dev_loss']}  WER {row['dev_wer']}%  CER {row['dev_cer']}%  ({len(preds)} utt)")
    model.train()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=WhisperConfig().model_id)
    p.add_argument("--method", choices=["lora", "decoder", "encoder", "full"], default="lora")
    p.add_argument("--train_manifest", default="data/splits_fa/train.csv")
    p.add_argument("--dev_manifest", default="data/splits_fa/dev.csv")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=None, help="Cap on optimizer steps (also sets the LR schedule length)")
    p.add_argument("--max_minutes", type=float, default=None, help="Stop (and save) after this many minutes of this invocation; re-run to continue")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=None, help="Default: 2e-4 for lora, 1e-5 otherwise")
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_limit", type=int, default=None, help="Use only the first N training rows (tests)")
    p.add_argument("--dev_limit", type=int, default=None, help="Random dev subset (seed 0) for evaluation; default all")
    p.add_argument("--eval_every", type=int, default=0, help="Also evaluate every N steps (always at start and each epoch end)")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--memory_probe", action="store_true",
                   help="Before training, run forward+backward on the longest-target batch and report peak memory")
    args = p.parse_args()
    args.lr = args.lr if args.lr is not None else DEFAULT_LR[args.method]
    sys.stdout.reconfigure(line_buffering=True)  # live log when redirected to a file

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recipe = {k: getattr(args, k) for k in RECIPE}
    if (out / "config.json").exists():
        old = json.loads((out / "config.json").read_text(encoding="utf-8"))
        diff = {k: (old.get(k), v) for k, v in recipe.items() if old.get(k) != v}
        if diff:
            sys.exit(f"{out} was started with a different recipe {diff}. Use a new --out_dir.")
    else:
        atomic_write_text(out / "config.json", json.dumps(recipe, indent=2))

    with RunRecord(out / "run_info.json", inputs=[args.train_manifest, args.dev_manifest]) as rec:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, base, processor, prompt_kwargs = build_model(args, device)
        tok = processor.tokenizer
        tok.set_prefix_tokens(language="bn", task="transcribe", predict_timestamps=False)
        args._n_prompt = len(tok.prefix_tokens) - 1  # <|bn|><|transcribe|><|notimestamps|> as labels
        max_len = base.config.max_target_positions + 1

        def load(manifest, limit=None, sample=False):
            df = pd.read_csv(manifest, encoding="utf-8-sig")
            df = df[df["text"].astype(str).str.strip() != ""]
            if limit and sample:
                df = df.sample(n=min(limit, len(df)), random_state=0)
            elif limit:
                df = df.head(limit)
            rows = df.to_dict("records")
            for r in rows:
                r["_ids"] = tok(str(r["text"])).input_ids
            kept = [r for r in rows if len(r["_ids"]) <= max_len]
            return kept, len(rows) - len(kept)

        train_rows, n_long = load(args.train_manifest, args.train_limit)
        dev_rows, _ = load(args.dev_manifest, args.dev_limit, sample=True)
        n = len(train_rows)
        bs, accum = args.batch_size, args.grad_accum
        steps_per_epoch = (n // bs) // accum
        total = steps_per_epoch * args.epochs if args.max_steps is None else min(args.max_steps, steps_per_epoch * args.epochs)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{args.method}: {n_params / 1e6:.1f}M trainable params | train {n} chunks "
              f"({n_long} dropped: target > {max_len} tokens) | dev {len(dev_rows)} | "
              f"{steps_per_epoch} steps/epoch x {args.epochs} = {total} steps, batch {bs}x{accum}, lr {args.lr}")

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        from transformers import get_linear_schedule_with_warmup
        sched = get_linear_schedule_with_warmup(opt, args.warmup_steps, total)
        state = {"step": 0, "epoch": 0, "micro": 0, "train_seconds": 0.0, "skipped_oom": 0}

        ck = find_checkpoint(out)
        if ck:
            saved = torch.load(ck / "state.pt", map_location="cpu", weights_only=False)
            res = model.load_state_dict(saved["model"], strict=False)
            if res.unexpected_keys or set(saved["model"]) != set(trainable_state(model)):
                sys.exit(f"Checkpoint {ck} does not match the model ({len(res.unexpected_keys)} unexpected keys).")
            opt.load_state_dict(saved["optimizer"])
            sched.load_state_dict(saved["scheduler"])
            torch.set_rng_state(saved["rng"]["torch"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
            np.random.set_state(saved["rng"]["numpy"])
            random.setstate(saved["rng"]["python"])
            state = json.loads((ck / "progress.json").read_text(encoding="utf-8"))
            print(f"Resumed from {ck}: step {state['step']}, epoch {state['epoch']}, micro-batch {state['micro']}")

        stop = {"flag": False}

        def on_sigint(sig, frame):
            if stop["flag"]:
                raise KeyboardInterrupt
            stop["flag"] = True
            print("\nCtrl+C: finishing this step and saving (press again to quit without saving)")

        signal.signal(signal.SIGINT, on_sigint)
        model.train()

        if args.memory_probe:
            longest = sorted(train_rows, key=lambda r: -len(r["_ids"]))[:bs]
            ds = ChunkDataset(longest, [r["_ids"] for r in longest], processor.feature_extractor)
            f, d, l = collate([ds[i] for i in range(len(ds))], tok.eos_token_id, args._n_prompt)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_features=f.to(device), decoder_input_ids=d.to(device), labels=l.to(device)).loss.backward()
            opt.zero_grad(set_to_none=True)
            print(f"[memory probe] batch {bs}, longest targets ({d.shape[1]} tokens): "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
                  f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved of "
                  f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

        if state["step"] == 0 and state["micro"] == 0 and not (out / "dev_log.csv").exists():
            evaluate(model, base, processor, prompt_kwargs, dev_rows, args, device, 0, out)

        ds = ChunkDataset(train_rows, [r["_ids"] for r in train_rows], processor.feature_extractor)
        coll = functools.partial(collate, pad_id=tok.eos_token_id, n_prompt=args._n_prompt)
        done = state["step"] >= total  # already finished: only re-check dev and re-save final/
        win = {"loss": 0.0, "n": 0, "t": time.time()}
        torch.cuda.reset_peak_memory_stats()

        def save(reason):
            state["train_seconds"] = round(state["train_seconds"] + time.time() - t_mark[0], 1)
            t_mark[0] = time.time()
            save_checkpoint(out, model, opt, sched, state)
            print(f"[checkpoint] step {state['step']} saved ({reason})")

        t_mark = [time.time()]
        t_run = time.time()  # --max_minutes counts from here, per invocation
        while state["epoch"] < args.epochs and not done:
            perm = np.random.default_rng(args.seed + state["epoch"]).permutation(n)
            usable = steps_per_epoch * accum * bs
            order = perm[state["micro"] * bs : usable].tolist()
            loader = torch.utils.data.DataLoader(ds, batch_size=bs, sampler=order, collate_fn=coll, drop_last=True,
                                                 num_workers=args.num_workers, pin_memory=True,
                                                 worker_init_fn=_ignore_sigint if args.num_workers else None)
            for f, d, l in loader:
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        loss = model(input_features=f.to(device, non_blocking=True),
                                     decoder_input_ids=d.to(device, non_blocking=True),
                                     labels=l.to(device, non_blocking=True)).loss
                    (loss / accum).backward()
                    win["loss"] += float(loss.detach())
                    win["n"] += 1
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    state["skipped_oom"] += 1
                    print(f"[warn] out of memory on a batch (max target {d.shape[1]} tokens); skipped")
                state["micro"] += 1
                if state["micro"] % accum:
                    continue
                gn = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                state["step"] += 1
                step = state["step"]
                if step % args.log_every == 0 or step == 1:
                    dt = (time.time() - win["t"]) / (args.log_every if step > 1 else 1)
                    row = {"step": step, "epoch": round(state["epoch"] + state["micro"] / (usable // bs), 3),
                           "loss": round(win["loss"] / max(1, win["n"]), 4), "lr": f"{sched.get_last_lr()[0]:.3e}",
                           "grad_norm": round(gn, 3), "s_per_step": round(dt, 2),
                           "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0,
                           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                    append_csv(out / "train_log.csv", row)
                    eta_h = dt * (total - step) / 3600
                    print(f"step {step}/{total} ep {row['epoch']} loss {row['loss']} lr {row['lr']} "
                          f"gn {row['grad_norm']} {row['s_per_step']} s/step peak {row['peak_gib']} GiB eta {eta_h:.1f} h")
                    win = {"loss": 0.0, "n": 0, "t": time.time()}
                if args.eval_every and step % args.eval_every == 0 and step < total:
                    evaluate(model, base, processor, prompt_kwargs, dev_rows, args, device, step, out)
                    win["t"] = time.time()
                over_time = args.max_minutes and (time.time() - t_run) / 60 >= args.max_minutes
                if step >= total:
                    done = True
                    break
                if stop["flag"] or over_time:
                    save("Ctrl+C" if stop["flag"] else f"--max_minutes {args.max_minutes}")
                    print("Stopped. Re-run the same command to continue.")
                    rec.extra = {"progress": state, "recipe": recipe, "stopped_early": True}
                    return STOPPED
                if step % args.save_every == 0:
                    save("periodic")
            else:
                state["epoch"] += 1
                state["micro"] = 0
                evaluate(model, base, processor, prompt_kwargs, dev_rows, args, device, state["step"], out)
                save("epoch end")
                continue
            break

        if done:
            if (out / "dev_log.csv").exists():
                last = list(csv.DictReader((out / "dev_log.csv").open(encoding="utf-8")))[-1]
                if int(last["step"]) != state["step"]:
                    evaluate(model, base, processor, prompt_kwargs, dev_rows, args, device, state["step"], out)
            save("finished")
            final = out / "final"
            base.config.use_cache = True  # turned off for training only; inference needs the KV cache
            (model if args.method == "lora" else base).save_pretrained(final)
            processor.save_pretrained(final)
            atomic_write_text(final / "training_state.json", json.dumps(state, indent=2))
            print(f"Finished {state['step']} steps; weights in {final}")
            rec.outputs = [out / "train_log.csv", out / "dev_log.csv"]
        rec.extra = {"progress": state, "recipe": recipe}


if __name__ == "__main__":
    sys.exit(main())
