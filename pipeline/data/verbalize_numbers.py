#!/usr/bin/env python3
"""
Training text with numbers as spoken words, the reading chosen by the audio (the number fix, v1.1 of the model).

The final model writes numbers well as words and badly as digits (value right in 34-60% of digit outputs) because
its training text mixes both: CV, Kathbath and Shrutilipi write words, OpenSLR 53 (26% of clips) and DL Sprint (3%)
write digits. This rewrites every digit token as words, so the model learns one convention.

  align  each clip with digits: bangla_asr.common.numbers.spoken_forms gives the ways each digit token can be said
         (১৯৯৪ -> উনিশশো চুরানব্বই / এক হাজার নয়শো চুরানব্বই, ৪৫% -> শতাংশ / পার্সেন্ট / শতকরা ...); MMS_FA forced
         alignment scores each reading against the clip's audio (one token at a time, the others fixed) and keeps the
         best. Report per clip: verbalized text, number-word score (lowest mean character posterior over its number
         spans) and the median score of its other words. Nothing is dropped here except clips with a token the
         verbalizer does not handle (১০:৩০, ১/২, ...).
  mix    the training list: clips whose numbers are confirmed, + as many clips that already say numbers in words,
         + as many other clips (seed 0), all from the same manifest. Confirmed = with --judge (the default on Modal),
         the released model in word-only mode wrote one of the token's readings word for word (judge_numbers.py; MMS
         cannot hear long number words, 82% of years); without it, the MMS number score passes --min_score.

Run on Modal next to the audio (pipeline/modal_finetune.py::numbers_align), or locally with the clips unpacked.
  PYTHONPATH=src python pipeline/data/verbalize_numbers.py align --manifest data/splits_fa/matrix/m5-300.csv \
      --out data/splits_fa/matrix/numbers_report.csv [--limit 200]
  PYTHONPATH=src python pipeline/data/verbalize_numbers.py mix --manifest data/splits_fa/matrix/m5-300.csv \
      --report data/splits_fa/matrix/numbers_report.csv --judge data/splits_fa/matrix/numbers_judge.csv --out data/splits_fa/matrix/numbers-v1.csv
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bangla_asr.common.numbers import _UNITS, spoken_forms  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import fold  # noqa: E402

DIGIT = re.compile(r"[0-9০-৯]")
# Number words for the "already in words" group: 2-99 and the multipliers; এক (also "a") and নয় (also "is not") left out.
NUMBER_WORDS = (set(_UNITS[2:]) - {"নয়"}) | {"শো", "হাজার", "লাখ", "লক্ষ", "কোটি", "দশমিক", "শতাংশ"}


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    a, s = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if s != sr:
        import librosa

        a = librosa.resample(a, orig_sr=s, target_sr=sr)
    return a


def plan(text: str):
    """Words with each digit token as a slot of readings: (words, slots) or None if a token is not handled."""
    words = text.split()
    slots = {}
    for i, w in enumerate(words):
        if DIGIT.search(w):
            forms = spoken_forms(w, words[i + 1] if i + 1 < len(words) else "")
            if forms is None:
                return None
            slots[i] = forms
    return words, slots


def render(words, slots, choice):
    """Text with each slot replaced by its chosen reading; also the word indices of each number span."""
    out, spans = [], {}
    for i, w in enumerate(words):
        if i in slots:
            parts = slots[i][choice[i]].split()
            spans[i] = list(range(len(out), len(out) + len(parts)))
            out.extend(parts)
        else:
            out.append(w)
    return out, spans


def word_scores(aligner, em, words, frames=None):
    """Per-word mean character posterior (NaN for words with nothing alignable), or None if alignment fails.
    With frames=[] it is filled with each word's (start, end) frame (None for unaligned words)."""
    toks = aligner.romanize_words(words)
    keep = [i for i, t in enumerate(toks) if t]
    res = aligner._align_window(em, [toks[i] for i in keep])
    if res is None:
        return None
    scores = [float("nan")] * len(words)
    if frames is not None:
        frames[:] = [None] * len(words)
    for i, (a, b, s) in zip(keep, res):
        scores[i] = s
        if frames is not None:
            frames[i] = (a, b)
    return scores


def span_score(scores, idx):
    vals = [scores[j] for j in idx if not np.isnan(scores[j])]
    return min(vals) if vals else float("nan")


def window_score(scores, idx, context=2):
    """Mean over a number span and `context` words each side: a reading that is too short or too long makes its
    neighbours stretch or squeeze, so they must count when readings are compared (span-only favoured short years)."""
    lo, hi = max(0, idx[0] - context), min(len(scores), idx[-1] + context + 1)
    vals = [s for s in scores[lo:hi] if not np.isnan(s)]
    return float(np.mean(vals)) if vals else float("nan")


def heard(aligner, em, frames, idx):
    """What MMS hears from the end of the word before a span to the start of the word after it (romanized)."""
    before = [frames[j] for j in range(idx[0]) if frames[j]]
    after = [frames[j] for j in range(idx[-1] + 1, len(frames)) if frames[j]]
    a = before[-1][1] if before else 0
    b = after[0][0] if after else em.shape[0]
    chars, _ = aligner._greedy_chars(em[a:b])
    inv = {v: k for k, v in aligner.dictionary.items()}
    return "".join(inv[c] for c in chars).replace("|", " ").strip()


def align(args):
    from bangla_asr.preprocessing.forced_align import MMSForcedAligner

    df = pd.read_csv(args.manifest, encoding="utf-8-sig")
    df = df[df["text"].astype(str).str.contains(DIGIT)].reset_index(drop=True)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0).reset_index(drop=True)
    aligner = MMSForcedAligner(device=args.device)
    out, t0 = [], time.time()
    for k, r in df.iterrows():
        row = {"utt_id": r["utt_id"], "source": r["source"], "text": r["text"], "verbalized": "", "choices": "",
               "number_score": float("nan"), "other_median": float("nan"), "status": "", "candidates": "", "heard": ""}
        p = plan(str(r["text"]))
        if p is None:
            row["status"] = "not_handled"
            out.append(row)
            continue
        words, slots = p
        em = aligner.emissions(load_audio(r["audio_path"]))
        choice = {i: 0 for i in slots}
        cand = {}
        for i, forms in slots.items():
            if len(forms) == 1:
                continue
            best, cand[words[i]] = (-1.0, 0), {}
            for c in range(len(forms)):
                trial = {**choice, i: c}
                ws, spans = render(words, slots, trial)
                sc = word_scores(aligner, em, ws)
                s = window_score(sc, spans[i]) if sc is not None else float("nan")
                cand[words[i]][forms[c]] = None if np.isnan(s) else round(s, 4)
                if not np.isnan(s) and s > best[0]:
                    best = (s, c)
            choice[i] = best[1]
        ws, spans = render(words, slots, choice)
        frames = []
        sc = word_scores(aligner, em, ws, frames)
        row.update({"verbalized": " ".join(ws), "candidates": json.dumps(cand, ensure_ascii=False),
                    "choices": json.dumps({words[i]: slots[i][c] for i, c in choice.items()}, ensure_ascii=False)})
        if sc is None:
            row["status"] = "align_failed"
        else:
            num_idx = {j for idx in spans.values() for j in idx}
            nums = [span_score(sc, idx) for idx in spans.values()]
            other = [s for j, s in enumerate(sc) if j not in num_idx and not np.isnan(s)]
            row.update({"number_score": min(nums) if nums else float("nan"),
                        "other_median": statistics.median(other) if other else float("nan"), "status": "ok",
                        "heard": json.dumps({words[i]: heard(aligner, em, frames, idx) for i, idx in spans.items()},
                                            ensure_ascii=False)})
        out.append(row)
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(df)} clips, {time.time() - t0:.0f} s", flush=True)
    rep = pd.DataFrame(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(args.out, index=False, encoding="utf-8-sig")
    ok = rep[rep["status"] == "ok"]
    print(f"done: {len(rep)} clips, {rep['status'].value_counts().to_dict()}", flush=True)
    if len(ok):
        q = ok["number_score"].quantile([0.05, 0.1, 0.25, 0.5, 0.75]).round(3).to_dict()
        print(f"number_score quantiles {q}; other-word median of medians {ok['other_median'].median():.3f}", flush=True)


def _keys(text):
    return [fold(w) for w in normalize_bangla_text(str(text)).split()]


def _anchors(words, i):
    """Keys of the nearest words before and after token i that are not numbers ('' at the edges)."""
    def find(rng):
        for j in rng:
            k = _keys(words[j])
            if k and not DIGIT.search(words[j]):
                return k[-1] if j < i else k[0]
        return ""
    return find(range(i - 1, -1, -1)), find(range(i + 1, len(words)))


def _at(hk, sub, left, right):
    """sub occurs in hk right after `left` or right before `right`: at the number's place, not anywhere (a short year
    such as পাঁচ or নয় is also a common word elsewhere in the sentence). No anchors at all: anywhere."""
    n = len(sub)
    for k in range(len(hk) - n + 1):
        if hk[k:k + n] != sub:
            continue
        if not left and not right:
            return True
        if (left and k > 0 and hk[k - 1] == left) or (right and k + n < len(hk) and hk[k + n] == right):
            return True
    return False


def choose_by_judge(text: str, judge: str):
    """(verbalized text, status): for each digit token the reading the judge (the model in word-only mode) wrote, word
    for word (spelling variants allowed through fold), longest match (উনিশশো চুরানব্বই over চুরানব্বই). No reading
    written -> the clip is dropped. No fallback to 'same value, other words': the training text must be the words said
    (the judge's পঁচিশশো must not become দুই হাজার পাঁচশো)."""
    p = plan(text)
    if p is None:
        return "", "not_handled"
    words, slots = p
    hk = _keys(judge)
    choice = {}
    for i, forms in slots.items():
        left, right = _anchors(words, i)
        exact = [c for c, f in enumerate(forms) if _at(hk, _keys(f), left, right)]
        if not exact:
            return "", "judge_mismatch"
        choice[i] = max(exact, key=lambda c: len(_keys(forms[c])))
    ws, _ = render(words, slots, choice)
    return " ".join(ws), "ok"


def _take_hours(pool: pd.DataFrame, hours: float) -> pd.DataFrame:
    """Random clips of pool (seed 0) until their duration reaches `hours`."""
    if hours <= 0 or pool.empty:
        return pool.iloc[:0]
    s = pool.sample(frac=1, random_state=0)
    return s[s["duration"].cumsum() <= hours * 3600]


def balanced_filler(df, a, has_digit, has_words):
    """Filler clips so every source keeps its share of hours in the manifest (v1.2). The confirmed number clips `a` are
    almost all OpenSLR, which made v1.1's mix 73% OpenSLR (manifest 48%) and drifted DL Sprint dev. The list size is set
    so no source exceeds its share; each source is then filled half with number-word clips, half with other clips."""
    share = df.groupby("source")["duration"].sum() / df["duration"].sum()
    a_h = a.groupby("source")["duration"].sum().reindex(share.index, fill_value=0) / 3600
    total = float((a_h / share).max())
    b_parts, c_parts = [], []
    for src in share.index:
        need = share[src] * total - a_h[src]
        pool_b = df[(df["source"] == src) & ~has_digit & has_words]
        pool_c = df[(df["source"] == src) & ~has_digit & ~has_words]
        b_s = _take_hours(pool_b, need / 2)
        c_s = _take_hours(pool_c, need - b_s["duration"].sum() / 3600)
        b_parts.append(b_s)
        c_parts.append(c_s)
    return pd.concat(b_parts), pd.concat(c_parts)


def mix(args):
    df = pd.read_csv(args.manifest, encoding="utf-8-sig")
    rep = pd.read_csv(args.report, encoding="utf-8-sig")
    if args.judge:  # readings chosen by the model in word-only mode (judge_numbers.py); MMS kept as a cross-check
        jd = pd.read_csv(args.judge, encoding="utf-8-sig")
        rep = rep.merge(jd, on="utt_id", how="left")
        got = [choose_by_judge(t, j) if s == "ok" and isinstance(j, str) else ("", "no_judge")
               for t, j, s in zip(rep["text"], rep["judge"], rep["status"])]
        rep["verbalized"], rep["judge_status"] = [g[0] for g in got], [g[1] for g in got]
        passed = rep[rep["judge_status"] == "ok"]
        mms_ok = rep["number_score"] >= 0.4
        cross = {"judge_ok": int((rep["judge_status"] == "ok").sum()),
                 "judge_status": rep["judge_status"].value_counts().to_dict(),
                 "mms_0.4_ok": int(mms_ok.sum()),
                 "both_ok": int(((rep["judge_status"] == "ok") & mms_ok).sum())}
    else:
        if args.min_score is None:
            sys.exit("give --judge or --min_score")
        passed = rep[(rep["status"] == "ok") & (rep["number_score"] >= args.min_score)]
        cross = {}
    a = df.set_index("utt_id").loc[passed["utt_id"]].reset_index()
    a["text"] = passed["verbalized"].to_numpy()
    text = df["text"].astype(str)
    has_digit = text.str.contains(DIGIT)
    has_words = text.map(lambda t: any(w in NUMBER_WORDS for w in t.split()))
    if args.balance:
        b, c = balanced_filler(df, a, has_digit, has_words)
    else:
        b = df[~has_digit & has_words].sample(n=min(len(a), int((~has_digit & has_words).sum())), random_state=0)
        c = df[~has_digit & ~has_words].sample(n=min(len(a), int((~has_digit & ~has_words).sum())), random_state=0)
    out = pd.concat([a, b, c]).sample(frac=1, random_state=0).reset_index(drop=True)
    out = out[df.columns]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    info = {"report_clips": len(rep), "passed": len(a), "rule": "judge" if args.judge else f"mms>={args.min_score}",
            "number_words": len(b), "other": len(c), "total": len(out), "hours": round(out["duration"].sum() / 3600, 2),
            "status": rep["status"].value_counts().to_dict(), **cross, "balanced": bool(args.balance),
            "hours_share": {s: round(float(v), 1) for s, v in
                            (out.groupby("source")["duration"].sum() / out["duration"].sum() * 100).items()},
            "hours_share_manifest": {s: round(float(v), 1) for s, v in
                                     (df.groupby("source")["duration"].sum() / df["duration"].sum() * 100).items()}}
    if args.judge:
        passed.to_csv(Path(args.out).with_name(Path(args.out).stem + "_choices.csv"), index=False, encoding="utf-8-sig",
                      columns=["utt_id", "source", "text", "judge", "verbalized", "number_score"])
    Path(args.out).with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("align")
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--device", default=None)
    a.set_defaults(func=align)
    m = sub.add_parser("mix")
    m.add_argument("--manifest", required=True)
    m.add_argument("--report", required=True)
    m.add_argument("--min_score", type=float, default=None, help="MMS rule (without --judge)")
    m.add_argument("--judge", default="", help="judge_numbers.py output: readings chosen by the model")
    m.add_argument("--balance", action="store_true", help="filler keeps every source's share of hours (v1.2)")
    m.add_argument("--out", required=True)
    m.set_defaults(func=mix)
    args = ap.parse_args()
    args.func(args)
