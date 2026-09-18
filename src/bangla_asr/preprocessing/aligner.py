"""
Sequential fuzzy alignment between chunk ASR hypotheses and reference transcripts.
Includes automated lookahead resynchronization to prevent pointer deadlock.
"""

from typing import Dict, List, Tuple
from rapidfuzz import fuzz

from bangla_asr.common.text import clean_text_for_alignment
from bangla_asr.config import AlignmentConfig


class SequentialFuzzyAligner:
    """
    Aligns sequential chunk ASR hypotheses to an untimed reference text.
    Maintains a forward-moving corpus pointer with lookahead fallback.
    """

    def __init__(self, config: AlignmentConfig = AlignmentConfig()):
        self.cfg = config

    def _find_best_match(
        self,
        hypothesis_clean: str,
        corpus_words: List[str],
        corpus_pos: int,
        lead: int,
        trail: int,
    ) -> Tuple[int, int, float]:
        """
        Searches for hypothesis_clean within [corpus_pos - lead, corpus_pos + trail].
        Returns (best_start, best_length, best_score).
        """
        hyp_words = hypothesis_clean.split()
        num_words = len(hyp_words)
        if num_words == 0 or corpus_pos >= len(corpus_words):
            return corpus_pos, 0, 0.0

        best_score = 0.0
        best_start = corpus_pos
        best_length = num_words

        search_start = max(0, corpus_pos - lead)
        search_end = min(len(corpus_words), corpus_pos + trail)

        for s in range(search_start, search_end):
            # Try variable length windows around expected length
            for len_offset in range(-3, 6):
                length = max(3, num_words + len_offset)
                e = min(s + length, len(corpus_words))
                if s >= e:
                    continue
                candidate = " ".join(corpus_words[s:e])
                score = fuzz.ratio(hypothesis_clean, candidate)
                if score > best_score:
                    best_score = score
                    best_start = s
                    best_length = e - s

        return best_start, best_length, float(best_score)

    def align(
        self,
        chunks: List[Dict],
        reference_text: str,
    ) -> List[Dict]:
        """
        Takes a list of chunk dicts (each containing 'text') and reference_text,
        and returns chunks populated with 'ground_truth' and 'align_score'.
        """
        clean_ref = clean_text_for_alignment(reference_text)
        corpus_words = clean_ref.split()

        corpus_pos = 0
        aligned_chunks = []

        for chunk in chunks:
            hyp = chunk.get("text", "")
            hyp_clean = clean_text_for_alignment(hyp)
            hyp_n_words = len(hyp_clean.split()) if hyp_clean else 0

            if not hyp_clean or corpus_pos >= len(corpus_words):
                out = dict(chunk)
                out["ground_truth"] = ""
                out["align_score"] = 0.0
                aligned_chunks.append(out)
                continue

            # Standard local search
            best_start, best_len, best_score = self._find_best_match(
                hyp_clean,
                corpus_words,
                corpus_pos,
                lead=self.cfg.search_window_lead,
                trail=self.cfg.search_window_trail,
            )

            # Deadlock Recovery: If match is low, perform a wide lookahead
            # in case speech was skipped or untranscribed
            if best_score < self.cfg.min_alignment_score:
                rec_start, rec_len, rec_score = self._find_best_match(
                    hyp_clean,
                    corpus_words,
                    corpus_pos,
                    lead=0,
                    trail=self.cfg.max_recovery_lookahead,
                )
                if rec_score >= self.cfg.recovery_score_threshold:
                    best_start, best_len, best_score = rec_start, rec_len, rec_score

            gt_end = min(best_start + best_len, len(corpus_words))
            gt_span = " ".join(corpus_words[best_start:gt_end])

            out = dict(chunk)
            out["ground_truth"] = gt_span
            out["align_score"] = round(best_score, 2)

            if best_score >= self.cfg.min_alignment_score:
                corpus_pos = max(corpus_pos, gt_end)
            elif hyp_n_words > 0 and self.cfg.failure_advance_frac > 0:
                skip = max(1, int(hyp_n_words * self.cfg.failure_advance_frac))
                corpus_pos = min(len(corpus_words), corpus_pos + skip)

            aligned_chunks.append(out)

        return aligned_chunks
