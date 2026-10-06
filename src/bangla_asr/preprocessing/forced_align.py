"""
Long-form forced alignment of a full transcript to a full recording, and
word-boundary chunking for ASR training/evaluation.

Pipeline (one recording):
  1. Frame-level character log-posteriors from torchaudio's MMS_FA
     (multilingual wav2vec2 CTC, 20 ms frames), computed in overlapping
     windows so an hour of audio never goes through the model at once.
  2. Transcript words romanized with uroman (MMS_FA's alphabet is Latin).
     Each original word keeps its index, so references stay in Bangla.
  3. Anchor-based CTC forced alignment: character k-grams that the
     aligner's own greedy decode and the transcript share uniquely become
     anchors; the pieces between anchors are force-aligned independently
     with a ``*`` wildcard at each end, so untranscribed audio (ads, music,
     crosstalk) is absorbed and an error cannot drift through the file.
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
        self.last_stats: Dict[str, int] = {}

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

    # ------------------------------------------------------------- anchors
    def _greedy_chars(self, em: torch.Tensor) -> Tuple[List[int], List[int]]:
        """Greedy CTC decode (excluding '*'): emitted token ids and their frames."""
        ids = em[:, : self.star_id].argmax(-1).numpy()
        chars, frames = [], []
        prev = -1
        for f, t in enumerate(ids):
            if t != prev and t != self.blank_id:
                chars.append(int(t))
                frames.append(f)
            prev = t
        return chars, frames

    @staticmethod
    def _anchor_chain(ref: List[int], hyp: List[int], k: int) -> List[Tuple[int, int]]:
        """(ref_pos, hyp_pos) of k-grams unique in both sequences, filtered to the
        longest chain that is increasing in both (so anchors never cross)."""
        import bisect
        from collections import Counter

        rg = [tuple(ref[i:i + k]) for i in range(len(ref) - k + 1)]
        hg = [tuple(hyp[j:j + k]) for j in range(len(hyp) - k + 1)]
        rc, hc = Counter(rg), Counter(hg)
        rpos = {g: i for i, g in enumerate(rg) if rc[g] == 1 and hc.get(g) == 1}
        pairs = sorted((rpos[g], j) for j, g in enumerate(hg) if g in rpos)
        # Longest increasing subsequence in hyp position (ref already sorted).
        tails, tail_idx, back = [], [], [-1] * len(pairs)
        for i, (_, hj) in enumerate(pairs):
            p = bisect.bisect_left(tails, hj)
            if p == len(tails):
                tails.append(hj)
                tail_idx.append(i)
            else:
                tails[p] = hj
                tail_idx[p] = i
            back[i] = tail_idx[p - 1] if p else -1
        chain = []
        i = tail_idx[-1] if tail_idx else -1
        while i != -1:
            chain.append(pairs[i])
            i = back[i]
        return chain[::-1]

    # ------------------------------------------------------------- alignment
    def align(self, audio: np.ndarray, words: List[str]) -> List[AlignedWord]:
        """Word-level timestamps for the whole transcript (anchor-based).

        1. Greedy-decode the aligner's own emissions to a rough romanized
           character stream with frame times.
        2. Anchors: k-grams of characters that occur exactly once in both the
           romanized transcript and the rough stream, kept only if they form a
           monotonic chain. Inside an anchor the characters match exactly, so
           their frames are known.
        3. Cut points: word starts that fall strictly inside an anchor, placed
           between the frames of the preceding and the starting character.
        4. Forced-align each piece between cut points independently with '*'
           at both ends. A misalignment therefore stays inside one piece
           instead of drifting through the rest of the recording, and
           untranscribed audio (ads, music, crosstalk) is absorbed by '*'.
        """
        cfg = self.cfg
        em = self.emissions(audio)
        n_frames = em.shape[0]
        fps = self.sample_rate / self.samples_per_frame
        tokens = self.romanize_words(words)

        ref = []
        word_first_char = {}
        for wi, toks in enumerate(tokens):
            if toks:
                word_first_char[wi] = len(ref)
            ref.extend(toks)
        if not ref:
            return []
        hyp, hyp_f = self._greedy_chars(em)
        chain = self._anchor_chain(ref, hyp, cfg.anchor_k) if hyp else []
        ref_to_frame = {}
        for r, h in chain:
            for d in range(cfg.anchor_k):
                ref_to_frame[r + d] = hyp_f[h + d]

        # Cut points: (first word index of the next piece, boundary frame).
        cuts = []
        for wi, c in word_first_char.items():
            if c in ref_to_frame and (c - 1) in ref_to_frame:
                f_prev, f_cur = ref_to_frame[c - 1], ref_to_frame[c]
                if f_cur > f_prev:
                    cuts.append((wi, (f_prev + f_cur) // 2))
        cuts.sort()
        # Thin cuts so pieces are at least min_piece_s long (fewer, sturdier pieces).
        thinned, last_f = [], 0
        for wi, f in cuts:
            if f - last_f >= cfg.min_piece_s * fps:
                thinned.append((wi, f))
                last_f = f
        bounds = [(0, 0)] + thinned + [(len(words), n_frames)]

        times: Dict[int, Tuple[float, float, float]] = {}
        alignable = set(word_first_char)
        i = 0
        while i < len(bounds) - 1:
            w0, f0 = bounds[i]
            j = i + 1
            # Merge with following pieces while this one is infeasible.
            while True:
                w1, f1 = bounds[j]
                piece_words = [w for w in range(w0, w1) if w in alignable]
                res = self._align_window(em[f0:f1], [tokens[w] for w in piece_words]) if piece_words else []
                if res is not None or j == len(bounds) - 1:
                    break
                j += 1
            if res:
                for w, (s, e, sc) in zip(piece_words, res):
                    times[w] = ((f0 + s) / fps, (f0 + e) / fps, sc)
            i = j

        result: List[AlignedWord] = []
        prev_end = 0.0
        for wi, w in enumerate(words):
            if wi in times:
                s, e, sc = times[wi]
                result.append(AlignedWord(wi, w, s, e, sc, True))
                prev_end = e
            else:
                result.append(AlignedWord(wi, w, prev_end, prev_end, float("nan"), False))
        next_start = n_frames / fps
        for wobj in reversed(result):
            if wobj.aligned:
                next_start = wobj.start
            else:
                wobj.end = max(wobj.start, next_start)
        self.last_stats = {"anchors": len(chain), "pieces": len(bounds) - 1}
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
