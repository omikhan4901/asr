# Why N-best Rescoring Plateaus for Bangla Speech Recognition, and How Far Fixing the Cause Goes

*CSE499 capstone, North South University (499A: Sept 2026; 499B: in progress, results frozen Nov 12, 2026).*
*This README is the working paper. Every number comes from a result file in this repository's pipeline, with
95% cluster-bootstrap confidence intervals; numbers still running are marked* **pending**.

---

## Abstract

N-best rescoring is a cheap way to improve a speech recognizer without retraining, but for Bangla media
speech (drama, narration, news) its benefit stalled at a fraction of the oracle gap. We rebuilt the
evaluation of a Bangla Whisper-medium model around forced-aligned references and a frozen scoring
harness, and found that the apparent gain of Minimum Bayes Risk (MBR) selection came from comparing
against a sampled candidate: against greedy decoding, MBR changes word error rate (WER) by +0.14 points
(95% CI −0.11 to +0.35), although a perfect selector would gain 4.12 points. Counting, for every reference
word, how many of 12 candidates contain it explains the gap: 57% of greedy errors appear in no candidate,
and MBR fixes an error only when most candidates already agree. About 41% of errors are the same word in
another spelling or dialect form, shared by every candidate; the pattern holds on FLEURS, Kathbath and a
second model, and accounts for the whole lead of a stronger open baseline. Adding learned spelling
variants to the candidate lists moves the oracle from 18.35% to 13.75% and the WER from 22.71% to 17.46%
without any training; fine-tuning the model on 60 h of in-domain data reaches **11.34%** (−11.36 points
[−12.32, −10.46]), and five predictions written before the fine-tuning runs all hold. We release the
forced-aligned segmentation of 93.1 hours of Bengali-Loop recordings and the scoring code.

**Keywords:** speech recognition, Bangla, N-best rescoring, minimum Bayes risk, error analysis, spelling variation, Whisper

---

## 1. Introduction

Rescoring is one of the cheapest ways to improve a speech recognizer. The model produces several candidate
transcripts for each utterance, and a second rule chooses among them, so accuracy can improve without
retraining. The upper bound is the *oracle*: the candidate with the fewest errors, chosen with access to the
reference. For Bangla media speech, our earlier work (499A) reported that MBR selection closes about 41% of
the gap between the first candidate and the oracle, and then stops: more candidates, other temperatures and
a learned reranker barely moved it. This work asks why it stops, and whether removing the cause helps.

Bangla makes the question interesting. Many common words have more than one accepted spelling
(ছিল/ছিলো, কি/কী), spoken forms differ from standard writing (নিয়া for নিয়ে), and public corpora
follow different conventions. The model we study is a Whisper-medium checkpoint fine-tuned by Bengali.AI on
regional Bangla. Its decoder has learned a spelling habit. If every candidate shares that habit, the
reference spelling is never in the list, and no selection rule can recover it, however good.

In practice the hard part was measurement, not modeling. The earlier evaluation clips had been selected by
the very model being evaluated; the text normalizer left punctuation that turned one word into two errors;
the "first candidate" was a random sample rather than the model's best guess; and the candidate scores used
by MBR were silently wrong under the current toolkit. Each of these changed the conclusion, so the study
rebuilds the evaluation first and then diagnoses the plateau with a count anyone can reproduce: for every
reference word, how many of the candidates contain it.

**Contributions.**
- **A corrected measurement of rescoring for Bangla:** forced-aligned references chosen without the
  evaluated model, a frozen normalizer and harness with cluster-bootstrap intervals, and the finding that
  MBR does not beat greedy decoding on media speech (it helps slightly on read speech).
- **A diagnosis of the plateau:** per-word candidate agreement separates errors no selector can reach
  (57% of greedy errors) from errors MBR out-votes; rule-based typing shows that about 41% of errors are
  spelling or dialect variants, across two models and three test sets.
- **A reference-free signal:** candidate agreement alone flags erroneous words (AUROC 0.823), and spelling
  errors are the ones on which all candidates agree.
- **Tests of the diagnosis written before running them:** hypothesis expansion with a learned convention
  rewriter and an n-gram LM, and an error-locus comparison of LoRA, decoder-only, encoder-only and full
  fine-tuning against pre-registered predictions.
- **Released resources:** the forced-aligned segmentation of the DL Sprint / Bengali-Loop recordings and the
  normalizer and scoring code (human-verified references follow when the labeling is complete).

---

## 2. Background and related work

**Bangla media-speech data and systems.** The BUET DL Sprint 4.0 competition released transcribed
recordings of public Bangla media, later published as the Bengali-Loop benchmark [1]. We use its training
recordings as our media-speech data; because the competition test transcripts are hidden, our media-speech
test set is a group of held-out training recordings. Competition systems fine-tuned the same Bengali.AI
Whisper-medium checkpoint we study and added source separation, voice-activity windowing or careful
chunking [2, 3]; the strongest, ShobdoSetu, reaches 15.55% WER on the hidden test [4]. These papers report how
low WER goes, not which errors remain. For read speech, Vistaar evaluated Indian-language models on
Kathbath, FLEURS and Common Voice [5], and Voice of India reports open and commercial systems on a Bengali
benchmark [6]; commercial systems lead there, so every claim here is scoped to *open* models on our test sets.

**N-best rescoring and its ceiling.** MBR selects the candidate with the lowest expected loss against the
others and remains competitive for Whisper [7]. The closest analysis is Novosad's decomposition of the oracle
gap of English CTC models [8]: MBR over the model's own posteriors gives nothing, while MBR weighted by an
external language model helps. PERL measured, for Chinese, how often the correct character appears among the
candidates for each 1-best error [9]; our per-word agreement count is the same coverage idea at word level,
and we add the linguistic types of the uncovered errors and their link to MBR's behavior.

**Expanding the candidate list.** Guo et al. applied a text-trained spelling corrector to every candidate
and halved the oracle WER [10]; generative error correction rewrites N-best lists with large language
models [11]. Expansion is therefore not new; we use it as a controlled test of the diagnosis.

**Spelling variation and evaluation.** Several works change the metric instead of the model: unsupervised
WER standardization [12] and Vimarsha's lattice of valid spellings for Indian languages [13]. Whisper's
default normalizer removes Indic vowel signs [14]. We do not propose a metric: we keep strict WER, publish
our normalizer, and use spelling analysis to explain rescoring and fine-tuning behavior.

**Fine-tuning Whisper.** On DL Sprint data, Bangla-WhisperDiar reports 31.32% WER with LoRA and 24.41% with
full fine-tuning [3]. A study on Russian found that Whisper fine-tuning gains vanish when the test set
accepts multiple spellings [15]. We use the LoRA / decoder-only / encoder-only / full contrast diagnostically,
to locate which part of the network fixes which error type.

---

## 3. Method

### 3.1 Forced-alignment segmentation
Each DL Sprint recording comes with one untimed transcript for 40–86 minutes of audio. The earlier pipeline
cut audio at pauses, transcribed each piece with the evaluated model and fuzzy-matched that guess against
the transcript, discarding poor matches: it lost 27% of the audio and let the model under test choose its
own test clips. We replace it with forced alignment: transcripts are romanized with uroman and aligned with
the MMS aligner (a multilingual wav2vec 2.0 CTC model, used only as an alignment tool). Anchors (character
8-grams that occur exactly once in the romanized transcript and once in the aligner's own greedy
transcription, kept as a monotonic chain) split each recording into pieces that are aligned independently,
with a wildcard token at both ends to absorb untranscribed audio (ads, music). Words are grouped into clips of
at most 25 s, cut at the longest inter-word pause; each clip's reference is exactly the transcript words
inside it. Clips are filtered only by the aligner's confidence, never by a recognizer we evaluate.
Segmentation yields **24,156 clips (93.1 h) from 112 recordings**.

### 3.2 Normalization and scoring harness
Hypotheses and references are normalized identically: Unicode NFC; one form for the nukta letters
(ড়, ঢ়, য়); removal of Bangla and Latin punctuation including hyphens, quotes, brackets and speaker marks;
whitespace collapsed. Vowel signs, virama and chandrabindu are never removed. WER is computed at corpus level,

$$\mathrm{WER} = \frac{\sum_u (S_u + D_u + I_u)}{\sum_u N_u},$$

and confidence intervals come from a cluster bootstrap over recordings (speakers or sentences for read
speech), 1,000–2,000 resamples; a comparison resamples both systems together and reports the paired
difference. Under this normalizer, the 499A 1-best / MBR / oracle WERs of 34.85 / 32.29 / 28.72% become
26.98 / 24.35 / 20.57% for the identical outputs (the earlier rule stripped only five punctuation marks).

### 3.3 Candidates, exact scores and MBR
Greedy decoding (batch 8, fp16) is the baseline for every comparison. For rescoring, the model produces 12
candidates by beam-search multinomial sampling (T = 0.7). Each candidate's score is its exact log-probability
under the model, recomputed by teacher forcing (the generation toolkit no longer returned usable beam
scores). MBR selects

$$\hat{h} = \arg\min_{h_i} \sum_{j \ne i} P(h_j \mid x)\, \mathrm{WER}(h_i, h_j),$$

with $P(h_j \mid x)$ a softmax of the candidate scores at temperature τ = 0.5.

### 3.4 Candidate-agreement anatomy
For each reference word $r_i$ we count $c_i$, the number of the $K = 12$ candidates in which it is aligned
as correct. A word with $c_i = 0$ is **selection-proof**: no candidate contains it, so no rule that chooses
among candidates can recover it. Without a reference, the same count on the greedy output ($a_j$ = number of
candidates containing greedy word $j$) is an error detector, scored by AUROC.

### 3.5 Error typing
Each error gets one type by rules on a folding key that maps common spelling variants to one form (final ো,
ি/ী, ু/ূ, chandrabindu, য়/য, ঙ/ং, hasanta): *convention* (ছিল/ছিলো, কি/কী), *vowel shift* (নিয়ে/নিয়া,
তোমার/তুমার), *segmentation* (দু পাশে/দুপাশে), near miss, substitution, number, deletion, insertion. The
first three form the **variant share**. The rules are proxies; human labels measure their precision
(labeling in progress).

### 3.6 Raising the ceiling without retraining (Phase 3)
Each clip's pool is the 12 sampled candidates plus the greedy output, scored by

$$s(h) = \log P_{\mathrm{AM}}(h \mid x) + \lambda \log P_{\mathrm{LM}}(h) + \beta |h|,$$

with a word n-gram LM (interpolated Kneser–Ney) trained only on the DL Sprint training references. Methods:
LM reranking; LM-weighted MBR; ROVER-style word voting; a convention rewriter (rules "model writes w,
reference writes w′" learned from greedy-vs-reference alignments on the training split); and **expansion**,
where each candidate's rewrite joins the pool with its parent's acoustic score and the LM chooses the
spelling. Every weight, threshold and the LM order is chosen on the development split only and applied
unchanged to every test set; human labels are never used.

### 3.7 Fine-tuning (Phase 4)
The model is fine-tuned on the cleaned DL Sprint training clips (15,325 clips, 59.6 h; bracketed tags
stripped, 67 clips with non-Bangla script dropped) in four ways: LoRA (rank 32 on every attention and MLP
projection of encoder and decoder, lr 2e-4), decoder only, encoder only, and all weights (lr 1e-5); AdamW,
batch 8×4, 2 epochs, bf16 autocast, gradient checkpointing, SpecAugment as in the checkpoint. **Before any
full run we recorded five predictions:**
- **P1** the best variant beats the strongest open zero-shot model (Hishab, 21.18%) on DL Sprint eval;
- **P2** at least half of the WER reduction comes from variant types;
- **P3** FLEURS improves less than DL Sprint eval;
- **P4** decoder-only reaches at least 80% of the best variant's gain;
- **P5** the variant share of the remaining errors falls below the zero-shot share (40.6%).

### 3.8 Failures found during development
Several conclusions changed while the evaluation was being rebuilt; each failure became a check.

| Failure | Evidence → correction → check |
|---|---|
| Normalizer | 499A WER 34.85% reproduced by stripping five marks only → full punctuation removal → frozen normalizer |
| Model-gated clips | eval clips kept only if the evaluated model matched them → forced alignment, aligner-only filter |
| Aligner drift | one drama aligned 1,408 of 4,578 words → anchor-based alignment → per-recording aligned-word counts |
| Weak baseline | "1-best" was a sampled candidate, worse than greedy → every comparison paired against greedy |
| Candidate scores | toolkit no longer returned beam scores; fallback read the wrong beams → exact teacher-forced scores |
| Spelling rule | folding merged different words (আর/আরো) → exceptions and a stricter vowel-shift rule → regression tests |
| Token limit | one transcript re-tokenized to 499 tokens crashed GPU scoring → truncation with a flag |
| Batch effect | greedy output differs between batch 1 and 8 → batch 8 for every row |
| GPU type | rows from different GPUs mixed → same-GPU baseline re-score (A100 vs RTX 3060: 96–99% identical hypotheses, WER within 0.02) |

---

## 4. Experimental setup

**Data splits** (forced-aligned DL Sprint, by recording):

| Split | Recordings | Clips | Hours |
|---|---:|---:|---:|
| Train | 83 | 15,392 | 59.92 |
| Development | 6 | 646 | 2.39 |
| Evaluation | 22 | 4,234 | 16.85 |

The split is disjoint by recording but not by speaker or series: some narrators and serialized stories appear
on both sides. **Public test sets:** FLEURS-bn test (920 utterances, read Wikipedia), Kathbath validation
(2,236; the public repository has no test split), Common Voice 27.0 test (9,408 clips). None is used for
training or tuning.

**Contamination.** Hishab's model card lists Kathbath and Common Voice among its training data, and it
transcribes 50.9% and 61.5% of those utterances exactly; BanglaASR, fine-tuned on Common Voice, does so for
59.1%. These rows are marked † and never compared.

**Hardware.** Zero-shot and Phase 1–3 inference on one RTX 3060 (12 GB); Phase 4 training and scoring on a
Modal A100-40GB, every Phase 4 row (including the zero-shot baseline) scored on that same GPU; the Hishab
baseline on a Kaggle T4. Budget: $0 out of pocket (free cloud credits).

---

## 5. Results

### 5.1 Zero-shot baselines
WER (%), greedy decoding, one normalizer. † = the model's declared training data includes this test set.

| Model | DL Sprint eval | FLEURS | Kathbath | Common Voice |
|---|---:|---:|---:|---:|
| Hishab conformer large (CTC, ~120M, CC-BY-NC) | **21.18** | **14.24** | 8.32† | 7.72† |
| tugstugi Whisper-medium (764M, Apache-2.0) | 22.71 [20.55, 25.26] | 21.54 | **14.79** | **14.81** |
| IndicWav2Vec (bn) | 42.80 | 28.18 | 22.15 | 25.80 |
| BanglaASR (Whisper-small) | 58.20 | 32.94 | 32.20 | 9.66† |
| wav2vec2-xls-r-300m-bengali | 65.20 | 47.46 | 38.57 | 43.35 |
| Whisper large-v3 | – | 79.89 | 93.24 | – |
| Whisper large-v3-turbo | – | 91.54 | 92.64 | – |

Generic Whisper models write Bangla partly in Devanagari zero-shot and are not usable without adaptation.

### 5.2 Rescoring against greedy decoding
WER points, system − greedy, 95% cluster-bootstrap CI for eval. *Sel.-proof* = share of greedy's errors that
no candidate contains; *Variant* = spelling/dialect share of greedy's errors (rule-based).

| Condition | Greedy | MBR − G | Oracle − G | Sel.-proof | Variant | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| tugstugi, DL Sprint eval | 22.71 | +0.14 [−0.11, +0.35] | −4.12 | 57.0% | 40.6% | 0.823 |
| tugstugi, DL Sprint dev | 29.33 | +0.11 | −4.97 | 54.9% | 28.0% | 0.838 |
| tugstugi, FLEURS | 21.54 | −0.51 | −5.15 | 65.3% | 40.8% | 0.771 |
| tugstugi, Kathbath | 14.79 | −0.28 | −5.46 | 54.5% | 49.8% | 0.795 |
| BanglaASR, DL Sprint eval | 58.20 | +2.82 | −4.86 | 77.5% | 14.0% | 0.784 |
| BanglaASR, FLEURS | 32.94 | +1.53 | −3.02 | 78.3% | 22.3% | 0.760 |

On media speech MBR does not differ from greedy, while the oracle is 4–5 points better; on read speech MBR
gains a little; for the weaker model it hurts. The first sampled candidate, the baseline of the 499A study, is
3.15 points *worse* than greedy on eval, which is where the old "41% recovery" came from.

### 5.3 Candidate agreement

<p align="center"><img src="figures/png/f2_mbr_fix_by_agreement.png" width="560" alt="Errors of the first sampled candidate on DL Sprint eval by the number of candidates with the reference word right, split into fixed and not fixed by MBR; the share fixed rises from 0% at c=0 to nearly 100% at c=11."></p>

**Figure 1.** DL Sprint eval, tugstugi, sampled 12-best: errors of the first candidate by the number of
candidates that have the reference word right ($c$), split into those MBR fixes and those it does not; the
line is the share MBR fixes (95% cluster-bootstrap CIs over 22 recordings).

The largest group by far has $c = 0$: 49.5% of these errors appear in no candidate. MBR fixes almost none of
the errors that only a few candidates get right and almost all of those that most candidates get right.
Without any reference, agreement flags erroneous greedy words with AUROC 0.823 [0.812, 0.834]. All 12
candidates agree on the same wrong form for 25.4% of spelling/dialect errors but only 12.8% of other errors.

### 5.4 Error types

<p align="center"><img src="figures/png/f3_variant_share_by_model.png" width="440" alt="Variant share of errors against WER for each model and test set; better models have a higher variant share."></p>

**Figure 2.** Share of each model's errors that are spelling/dialect variants (rule-based) against its WER,
per test set, with 95% CIs; contaminated rows excluded.

About 40.6% [36.1, 44.8] of tugstugi's errors on DL Sprint eval are spelling or dialect variants; without them
its WER would be 13.48%. The variant share grows as WER falls. Splitting WER into variant and other errors,
Hishab and tugstugi spend 7.8 and 9.2 points on variants on DL Sprint eval, and 13.4 and 13.5 on everything
else: **the stronger baseline's whole lead is spelling convention.**

### 5.5 Raising the ceiling without retraining (Phase 3)
All settings chosen on DL Sprint dev and applied unchanged; WER points vs greedy [95% CI].

| Set | Greedy | LM rerank | LM-MBR | ROVER | Rewriter | **LM-MBR on expanded list** | Oracle: pool → expanded |
|---|---:|---:|---:|---:|---:|---:|---:|
| DL Sprint eval | 22.71 | −1.15 | −1.15 | −0.92 | −5.13 | **−5.25 [−5.58, −4.85] → 17.46** | 18.35 → 13.75 |
| FLEURS | 21.54 | −1.29 | −1.29 | −0.90 | −1.31 | **−2.06 [−2.72, −1.47] → 19.48** | 16.34 → 14.66 |
| Kathbath | 14.79 | −0.60 | −0.64 | −0.30 | +0.43 (worse) | **−0.54 [−0.81, −0.27] → 14.25** | 9.28 → 8.72 |

Plain MBR gave nothing; an LM that knows the target spelling makes selection work (Novosad's result
reproduced for an encoder-decoder model in Bangla). Putting the reference spelling into the list is what
matters: expansion lowers the oracle by 4.6 points on eval and LM-MBR reaches 17.46%, below Hishab's
zero-shot 21.18% without any training. 75% of the rewriter's gain on eval survives with spelling/dialect rules
only. Conventions are corpus-specific: the DL Sprint rewriter *hurts* Kathbath, whose references use the
other spelling of the most frequent variant pairs. The same rewriter applied to Hishab, with rules learned from
Hishab's own training-set errors, gives 17.76% on eval: a statistical tie with ours (−0.30 [−1.05, +0.37]), so
the convention fix is model-agnostic.

### 5.6 Fine-tuning (Phase 4)

<p align="center"><img src="figures/png/f4_finetune_by_set.png" width="760" alt="WER of zero-shot tugstugi and four fine-tuning variants on four test sets with confidence intervals; full fine-tuning is best everywhere; encoder-only does not improve Kathbath or Common Voice."></p>

**Figure 3.** WER of zero-shot tugstugi and the four M1 variants (trained on DL Sprint train only), all scored
with the same harness on the same GPU, with 95% cluster-bootstrap CIs. LoRA's Kathbath and Common Voice rows
are pending.

Paired difference vs zero-shot on the same GPU (WER points [95% CI]):

| Model | DL Sprint eval | FLEURS | Kathbath | Common Voice |
|---|---:|---:|---:|---:|
| zero-shot tugstugi (WER) | 22.70 | 21.54 | 14.81 | 14.80 |
| **full** | **11.34** (−11.36 [−12.32, −10.46]) | **18.34** (−3.20 [−4.07, −2.27]) | **13.48** (−1.33 [−2.00, −0.69]) | **12.80** (−2.00 [−2.23, −1.77]) |
| LoRA | 11.76 (−10.94 [−11.86, −10.05]) | 18.40 (−3.14) | pending | pending |
| decoder-only | 12.99 (−9.71 [−10.32, −9.07]) | 18.61 (−2.93) | 13.62 (−1.19) | 13.13 (−1.68) |
| encoder-only | 13.83 (−8.87 [−9.78, −7.87]) | 19.90 (−1.64) | 14.59 (−0.22, n.s.) | 14.69 (−0.11, n.s.) |

<p align="center"><img src="figures/png/f5_gain_by_error_type.png" width="440" alt="DL Sprint eval WER split into variant and other errors for zero-shot, Phase 3 and four fine-tuning variants."></p>

**Figure 4.** DL Sprint eval WER split into spelling/dialect-variant errors and all other errors (rule-based),
for zero-shot, the Phase 3 system without training, and the four fine-tuning variants.

**The pre-registered predictions:**

| | Prediction | Outcome | Verdict |
|---|---|---|---|
| P1 | best variant < 21.18% on eval, CI below 0 | full 11.34%, −11.36 [−12.32, −10.46] | **holds** |
| P2 | ≥ 50% of the gain from variant types | 53% (full: −6.00 of −11.36 points); LoRA 54%, decoder 60%, encoder 61% | **holds** (narrowly, rule-based) |
| P3 | FLEURS improves less than eval | −3.20 vs −11.36 | **holds** |
| P4 | decoder-only ≥ 80% of the best gain | 85% (−9.71 / −11.36) | **holds** |
| P5 | variant share of remaining errors < 40.6% | 28.5% [24.9, 32.0] | **holds** |

Two results go beyond the predictions. First, **training buys more than the convention**: Phase 3 removes
variant errors (9.2 → 5.2 points) but barely touches the rest, while full fine-tuning removes 6.0 variant
points *and* 5.4 other points. Second, **encoder-only fine-tuning removes almost as many variant errors on
DL Sprint as decoder-only** (−5.4 vs −5.9 points): the spelling habit is not stored only in the decoder; the
encoder adapts to what the frozen decoder needs. But the encoder-only gains do not transfer (Kathbath and
Common Voice unchanged), while decoder-only and full fine-tuning help on every set.

### 5.7 Still running (Phase 4 continues)
- **Stacking:** 12-best of the fine-tuned model + expansion + LM-MBR, and the agreement anatomy after
  fine-tuning (does fine-tuning turn selection-proof errors into fixed ones?).
- **Train × test matrix:** the same recipe trained on Common Voice, Kathbath and Shrutilipi (34 h each) and on
  a merged set, each scored on every test set.
- **Human verification:** inter-annotator agreement, the share of DL Sprint references annotators corrected,
  every model's WER against the verified references, and the human precision of each error type.

---

## 6. Analysis

**Why rescoring plateaus.** Two mechanisms limit selection. The first is *coverage*: about half of the errors
have no correct candidate, so the best possible selector is bounded long before it reaches the reference.
The second is *consensus*: MBR is a weighted majority vote, so it can only fix a word that most candidates
already write correctly, and on those words greedy decoding is usually right already. Spelling and dialect
variants are exactly the errors the candidates share: all 12 candidates agree on the wrong form far more often
for variants than for other errors. A better voter cannot fix a convention the whole list shares; the
reference spelling has to enter the list, or the model has to learn it. Phase 3 and Phase 4 test both routes,
and both work as predicted.

**How general the diagnosis is.** MBR gives at most a small fraction of the oracle headroom for tugstugi on
every set, agreement predicts errors with AUROC 0.76–0.84, and roughly 40–50% of tugstugi's errors are
variants. The variant share rises as models improve: weak models make so many acoustic errors that spelling
hardly matters, while for the best models it becomes a large part of what is left. For the weaker BanglaASR
model MBR is harmful: when most candidates are wrong, the majority is wrong too.

**What we cannot claim.** The error types come from rules until the human labels give their precision. The
evaluation clips are cut with the help of the reference, so the numbers measure recognition on well-cut clips
from long media recordings, not end-to-end transcription of a raw recording; we never call this long-form
recognition. Commercial systems report much lower Bengali WER on other benchmarks, so every comparison is
among open models. Fine-tuned systems are compared with zero-shot ones; the Hishab model fine-tuned on the
same data would be the fully fair comparison.

**Threats to validity.** The DL Sprint split shares narrators and serialized stories between training and
evaluation, which can favor fine-tuned systems and rules learned from training text; FLEURS, Kathbath and
Common Voice share nothing with our training data and are the strict test (and there the fine-tuning gains are
smaller: −1.3 to −3.2 points). The DL Sprint references may be partly machine-generated [4]; the verified
references will bound this effect. Greedy output depends slightly on batch size and GPU, so every row of a
comparison uses one setting and one GPU type. Kathbath results use its validation split, and the DL Sprint
test is our held-out recordings rather than the hidden competition test.

---

## 7. Conclusion

N-best rescoring of a Bangla Whisper model stalls because the errors that remain are shared by the
candidates. After rebuilding the evaluation with forced-aligned references, a frozen normalizer and a greedy
baseline, MBR no longer beats greedy decoding on media speech, although a perfect selector would gain 4.12
points. Counting how many candidates contain each word shows why: 57% of errors appear in no candidate, MBR
only fixes what most candidates already agree on, and about 41% of errors are spelling and dialect variants
that the whole list shares. Fixing the cause works in both ways we predicted: adding learned spelling variants
to the candidate lists takes DL Sprint eval from 22.71% to 17.46% without training, and fine-tuning takes it to
11.34%, with all five pre-registered predictions holding. Next: stacking the two, the train × test matrix
across Bangla corpora, and the human-verified references.

---

## Reproducing

Python 3.13; the repository root doubles as a Windows virtual environment (`./Scripts/python.exe`).

| Step | Command |
|---|---|
| Segmentation (forced alignment) | `pipeline/01_segment.py`, then `pipeline/02_split.py` |
| Zero-shot scoring (the frozen harness) | `pipeline/eval_zero_shot.py --manifest <csv> --model <hf id or folder> [--adapter <LoRA>]` |
| 12-best + exact scores | `pipeline/03_generate_nbest.py`, `pipeline/rescore_nbest.py` |
| Anatomy, error types, agreement signal | `analysis/agreement.py`, `analysis/error_types.py`, `analysis/agreement_signal.py` |
| Paired comparisons | `analysis/paired_compare.py` |
| Phase 3 (no training) | `pipeline/12_raise_ceiling.py` (`runs/phase3_ceiling.sh`) |
| Fine-tuning | `pipeline/11_finetune.py` (locally) or `pipeline/modal_launch.py train` (cloud A100) |
| Phase 4 analysis | `runs/phase4_analysis.sh` |
| Cloud inference (Kaggle T4) | `notebooks/kaggle_phase4.ipynb` → `pipeline/kaggle_phase4.py` |
| Figures | `figures/scripts/*.py` (data in `figures/data/`, PDF + PNG in `figures/`) |

Tests: `PYTHONPATH=src ./Scripts/python.exe tests/test_smoke.py` and `tests/test_mbr.py`.
Step-by-step guide for the cloud runs: [`docs/PHASE4_RUNBOOK.md`](docs/PHASE4_RUNBOOK.md).

**Layout.** `src/bangla_asr/` (config, text normalization and WER, forced aligner, Whisper wrapper and N-best
generator, MBR / n-gram LM / convention rewriter / ROVER), `pipeline/` (numbered pipeline steps, fine-tuning,
cloud runners), `analysis/` (every analysis behind the numbers above), `runs/` (resumable batch runners),
`figures/`, `labeling_site/` (the team's labeling tool), `web/` (subtitle demo).

---

## References

1. Bengali-Loop: community benchmarks for long-form Bangla ASR and speaker diarization. arXiv:2602.14291.
2. DL Sprint 4.0 system papers. arXiv:2602.23070, arXiv:2602.21183, arXiv:2602.21741, arXiv:2602.22935.
3. Bhuiyan et al. Bangla-WhisperDiar. arXiv:2605.08214.
4. Sakib et al. ShobdoSetu: a data-centric framework for Bengali long-form speech recognition. arXiv:2603.19256.
5. Bhogale et al. Vistaar: diverse benchmarks and training sets for Indian language ASR. Interspeech 2023, arXiv:2305.15386.
6. Bhogale et al. Voice of India. arXiv:2604.19151.
7. Jinnai. Re-evaluating minimum Bayes risk decoding for automatic speech recognition. arXiv:2510.19471.
8. Novosad. The anatomy of the CTC oracle gap. arXiv:2606.23306.
9. PERL (word coverage of N-best lists for Chinese ASR error correction). arXiv:2412.03230.
10. Guo et al. A spelling correction model for end-to-end speech recognition. arXiv:1902.07178.
11. Chen et al. HyPoradise: an open baseline for generative speech recognition with large language models. arXiv:2309.15701.
12. Guha et al. Unsupervised language-agnostic WER standardization. arXiv:2303.05046.
13. Bhogale et al. Vimarsha. arXiv:2609.24199.
14. What is lost in normalization? Exploring pitfalls in multilingual ASR model evaluations. EMNLP 2024, arXiv:2409.02449.
15. asr_eval. arXiv:2601.20992.

Data: DL Sprint 4.0 / Bengali-Loop (CC-BY-4.0), FLEURS (CC-BY-4.0), Kathbath (CC-BY-4.0), Common Voice (CC0).
Base model: `bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium` (Apache-2.0).
