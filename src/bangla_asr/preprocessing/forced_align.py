"""
Long-form forced alignment of a full transcript to a full recording, and
word-boundary chunking for ASR training/evaluation.

Pipeline (one recording):
  1. Frame-level character log-posteriors from torchaudio's MMS_FA
     (multilingual wav2vec2 CTC, 20 ms frames), computed in overlapping
     windows so an hour of audio never goes through the model at once.
  2. Transcript words romanized with uroman (MMS_FA's alphabet is Latin).
     Each original word keeps its index, so references stay in Bangla.
  3. Windowed CTC forced alignment with a moving text pointer. Each window
     aligns ``* w_i ... w_j *`` against up to ``window_s`` of audio; the
     ``*`` wildcard absorbs untranscribed audio (music, intros) before and
     after. Only words that end well inside the window are committed.
  4. Words -> chunks: cut at the longest pause so every chunk fits
     [min_chunk_s, max_chunk_s]. The chunk reference is exactly the
     transcript words inside it.

Nothing here depends on an ASR model that we later evaluate, so filtering by
alignment confidence does not bias the evaluation set toward that model.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import unicodedata

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF

from bangla_asr.config import ForcedAlignConfig


@dataclass
class AlignedWord:
    index: int          # position in the transcript word list
    text: str           # original Bangla word (as in the transcript)
    start: float        # seconds
    end: float          # seconds
    score: float        # mean posterior of its characters (0..1); NaN if not aligned
    aligned: bool       # False for words with no alignable characters (digits, symbols)


class MMSForcedAligner:
    """Aligns a transcript to a long recording with MMS_FA + uroman."""

    def __init__(self, cfg: ForcedAlignConfig = ForcedAlignConfig(), device: Optional[str] = None):
        import uroman

        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        bundle = torchaudio.pipelines.MMS_FA
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model(with_star=True).to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.dictionary: Dict[str, int] = bundle.get_dict(star="*")
        self.star_id = self.dictionary["*"]
        self.blank_id = self.dictionary["-"]
        self.uroman = uroman.Uroman()
        # wav2vec2 conv front-end: 320 samples per frame at 16 kHz.
        self.samples_per_frame = 320

    # ------------------------------------------------------------------ text
    def romanize_words(self, words: List[str]) -> List[List[int]]:
        """Token ids per word. Empty list = nothing alignable (digits, symbols)."""
        out = []
        for w in words:
            rom = self.uroman.romanize_string(unicodedata.normalize("NFC", w), lcode="ben").lower()
            out.append([self.dictionary[c] for c in rom if c in self.dictionary and c not in ("-", "*")])
        return out

    # ------------------------------------------------------------- emissions
    @torch.inference_mode()
    def emissions(self, audio: np.ndarray) -> torch.Tensor:
        """Log-posteriors [T, C] for the whole recording (on CPU, float32).

        Overlapping windows: each window is ``emission_window_s`` long and
        only its centre (minus ``emission_context_s`` on each side) is kept,
        so every kept frame saw context on both sides.
        """
        sr = self.sample_rate
        spf = self.samples_per_frame
        win = int(self.cfg.emission_window_s * sr)
        ctx = int(self.cfg.emission_context_s * sr)
        hop = win - 2 * ctx
        n = len(audio)
        n_frames = n // spf
        out = torch.full((n_frames, len(self.dictionary)), float("-inf"))
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))

        start = 0
        while start < n:
            a = max(0, start - ctx)
            b = min(n, start + hop + ctx)
            seg = wav[a:b].unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                seg = seg.half()
            # MMS_FA already returns log-posteriors over its 28 labels; the
            # appended '*' column is a constant 0 (log 1), so no re-normalization:
            # that would halve every score without changing the alignment path.
            em, _ = self.model(seg)
            em = em.float()[0].cpu()
            # frames of this window that fall in the kept region [start, start+hop)
            f_off = a // spf
            keep_a = start // spf
            keep_b = min(n_frames, (start + hop) // spf)
            lo = keep_a - f_off
            hi = min(em.shape[0], keep_b - f_off)
            if hi > lo:
                out[keep_a : keep_a + (hi - lo)] = em[lo:hi]
            start += hop
        # Any frame left at -inf (rounding at the very end) gets the blank.
        bad = torch.isinf(out).all(dim=1)
        if bad.any():
            out[bad] = float("-inf")
            out[bad, self.blank_id] = 0.0
        return out

    # ------------------------------------------------------------- alignment
    def _align_window(
        self,
        em: torch.Tensor,
        word_tokens: List[List[int]],
    ) -> Optional[List[Tuple[int, int, float]]]:
        """Align ``* words *`` to the window. Returns (start_f, end_f, score) per
        word with tokens, or None if infeasible."""
        targets = [self.star_id]
        spans_per_word = []
        for toks in word_tokens:
            spans_per_word.append((len(targets), len(targets) + len(toks)))
            targets.extend(toks)
        targets.append(self.star_id)

        # CTC needs at least one frame per token plus one blank between repeats.
        repeats = sum(1 for x, y in zip(targets, targets[1:]) if x == y)
        if len(targets) + repeats > em.shape[0]:
            return None

        lp = em.unsqueeze(0).to(self.device)
        tg = torch.tensor([targets], dtype=torch.int32, device=self.device)
        ali, scores = AF.forced_align(lp, tg, blank=self.blank_id)
        spans = AF.merge_tokens(ali[0].cpu(), scores[0].exp().cpu(), blank=self.blank_id)
        if len(spans) != len(targets):
            return None

        out = []
        for a, b in spans_per_word:
            ws = spans[a:b]
            score = float(sum(s.score * (s.end - s.start) for s in ws) / max(1, sum(s.end - s.start for s in ws)))
            out.append((ws[0].start, ws[-1].end, score))
        return out

    def align(self, audio: np.ndarray, words: List[str]) -> List[AlignedWord]:
        """Word-level timestamps for the whole transcript."""
        cfg = self.cfg
        em = self.emissions(audio)
        n_frames = em.shape[0]
        fps = self.sample_rate / self.samples_per_frame
        tokens = self.romanize_words(words)
        alignable = [i for i, t in enumerate(tokens) if t]
        if not alignable:
            return []

        # Global speaking-rate estimate: alignable words per second of audio.
        rate = len(alignable) / max(1.0, n_frames / fps)
        win_f = int(cfg.window_s * fps)
        margin_f = int(cfg.commit_margin_s * fps)

        times: Dict[int, Tuple[float, float, float]] = {}
        f0 = 0
        k = 0  # pointer into `alignable`
        while k < len(alignable):
            n_words = max(cfg.min_words_per_window, int(cfg.fill_ratio * cfg.window_s * rate))
            best = None
            # Try a few window offsets; untranscribed stretches (music, ads) can be
            # longer than the leading star can absorb within one window.
            for attempt in range(cfg.max_skip_attempts):
                fa = min(f0 + attempt * (win_f // 2), max(0, n_frames - 1))
                fb = min(n_frames, fa + win_f)
                chunk_ids = alignable[k : k + n_words]
                res = self._align_window(em[fa:fb], [tokens[i] for i in chunk_ids])
                if res is None:
                    continue
                head = [s for _, _, s in res[: cfg.probe_words]]
                head_score = float(np.mean(head)) if head else 0.0
                if best is None or head_score > best[0]:
                    best = (head_score, fa, fb, chunk_ids, res)
                if head_score >= cfg.good_window_score:
                    break
            if best is None:
                # Window too short for the remaining text: give up on the tail.
                break
            _, fa, fb, chunk_ids, res = best
            last = len(chunk_ids) == len(alignable) - k
            # Commit words ending before the window's last margin (all, at the end of text).
            limit = fb if (last or fb >= n_frames) else fb - margin_f
            committed = 0
            for wi, (s, e, sc) in zip(chunk_ids, res):
                if e + fa > limit and committed >= 1:
                    break
                times[wi] = ((fa + s) / fps, (fa + e) / fps, sc)
                committed += 1
            if not last and committed > cfg.holdback_words:
                # Hold back the last few committed words: they are re-aligned in the
                # next window with right-context, which is more reliable.
                for wi in chunk_ids[committed - cfg.holdback_words : committed]:
                    times.pop(wi, None)
                committed -= cfg.holdback_words
            k += committed
            f0 = int(times[alignable[k - 1]][1] * fps) if k > 0 else f0 + win_f // 2

        # Assemble all words; unalignable ones get the gap between their neighbours.
        result: List[AlignedWord] = []
        prev_end = 0.0
        for i, w in enumerate(words):
            if i in times:
                s, e, sc = times[i]
                result.append(AlignedWord(i, w, s, e, sc, True))
                prev_end = e
            else:
                result.append(AlignedWord(i, w, prev_end, prev_end, float("nan"), False))
        # Give unaligned words the start of the next aligned word as their end.
        next_start = n_frames / fps
        for wobj in reversed(result):
            if wobj.aligned:
                next_start = wobj.start
            else:
                wobj.end = max(wobj.start, next_start)
        return result


def chunk_words(
    words: List[AlignedWord],
    cfg: ForcedAlignConfig = ForcedAlignConfig(),
    total_duration: Optional[float] = None,
) -> List[Dict]:
    """Group aligned words into chunks that end at pauses.

    A chunk grows until adding the next word would exceed ``max_chunk_s``,
    or the next pause is longer than ``break_pause_s``. When it is too long,
    it is cut at the longest pause inside it that leaves at least
    ``min_chunk_s`` on the left. Audio boundaries extend into the
    surrounding pauses by at most ``edge_pad_s`` (never past the pause midpoint).
    """
    ws = [w for w in words if w.aligned or w.text.strip()]
    if not ws:
        return []
    total = total_duration if total_duration is not None else ws[-1].end + cfg.edge_pad_s

    groups: List[List[AlignedWord]] = []
    cur: List[AlignedWord] = [ws[0]]
    for w in ws[1:]:
        pause = w.start - cur[-1].end
        if pause >= cfg.break_pause_s and cur[-1].end - cur[0].start >= cfg.min_chunk_s:
            groups.append(cur)
            cur = [w]
            continue
        if w.end - cur[0].start <= cfg.max_chunk_s:
            cur.append(w)
            continue
        # Too long: cut at the longest pause that leaves >= min_chunk_s on the left.
        best_j, best_gap = None, -1.0
        for j in range(1, len(cur)):
            if cur[j - 1].end - cur[0].start < cfg.min_chunk_s:
                continue
            gap = cur[j].start - cur[j - 1].end
            if gap >= best_gap:
                best_j, best_gap = j, gap
        if best_j is None:
            groups.append(cur)
            cur = [w]
        else:
            groups.append(cur[:best_j])
            cur = cur[best_j:] + [w]
    groups.append(cur)

    chunks = []
    for gi, g in enumerate(groups):
        left_lim = groups[gi - 1][-1].end if gi > 0 else 0.0
        right_lim = groups[gi + 1][0].start if gi + 1 < len(groups) else total
        start = max(g[0].start - cfg.edge_pad_s, (left_lim + g[0].start) / 2, 0.0)
        end = min(g[-1].end + cfg.edge_pad_s, (g[-1].end + right_lim) / 2, total)
        scores = [w.score for w in g if w.aligned]
        chunks.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "n_words": len(g),
            "first_word": g[0].index,
            "last_word": g[-1].index,
            "text": " ".join(w.text for w in g),
            "mean_score": round(float(np.mean(scores)), 4) if scores else float("nan"),
            "min_score": round(float(np.min(scores)), 4) if scores else float("nan"),
            "low_score_frac": round(float(np.mean([s < cfg.low_word_score for s in scores])), 4) if scores else 1.0,
            "unaligned_words": sum(1 for w in g if not w.aligned),
        })
    return chunks
