---
language:
- bn
license: apache-2.0
base_model: bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium
pipeline_tag: automatic-speech-recognition
library_name: transformers
tags:
- whisper
- bangla
- bengali
- speech-recognition
---

# Bangla ASR: Whisper-medium fine-tuned on ~300 h

A Whisper-medium model (764M parameters) for Bangla speech recognition, fine-tuned from
[tugstugi Whisper-medium](https://huggingface.co/bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium) on about
300 hours of public Bangla speech: media recordings (drama, narration, news) and read speech. It is the final model
of the CSE499 capstone at North South University; method, every result with 95% confidence intervals, and the scoring
code are in the repository: [github.com/omikhan4901/asr](https://github.com/omikhan4901/asr).

## Results

Word error rate (%), greedy decoding, one normalizer for every system (`bangla_asr.common.text.normalize_bangla_text`).
In brackets: the spelling-aware WER, which does not count differences that human annotators confirmed as spelling only
(ছিল / ছিলো).

| Test set | This model | Zero-shot base model |
|---|---:|---:|
| DL Sprint 4.0 eval (media speech, 4,234 clips) | **11.18** (8.82) | 22.70 (15.82) |
| FLEURS test | **15.60** (10.89) | 21.54 (14.56) |
| Common Voice test | **10.13** (6.61) | 14.80 (10.15) |
| Kathbath validation | **10.28** (5.87) | 14.81 (9.02) |
| Average | **11.80** (8.04) | 18.46 (12.39) |

Against the open Bangla models scored with the same harness it is best on DL Sprint, Common Voice and Kathbath among
the models not trained on those sets; on FLEURS the non-commercial Hishab conformer is ahead (14.24). Against 605
human-verified references the order of the systems is unchanged.
Whole raw recordings (22, 20.4 h, cut by voice detection): 14.2% WER. Speed: real-time factor about 0.04 on a T4.

## Use

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="OmiKhan/bangla-asr-whisper-medium")
print(asr("clip.wav", generate_kwargs={"language": "bn", "task": "transcribe"})["text"])
```

Audio is resampled to 16 kHz. The model was trained and scored on clips of 1–25 s; for long recordings, cut the audio
at pauses (for example with Silero VAD) or pass `chunk_length_s=30`.

## Training

Full fine-tuning, 2 epochs, learning rate 1e-5, from the tugstugi Whisper-medium checkpoint. The released weights are
a weight average of two runs with the same recipe, chosen by a rule fixed before any test score (lowest average WER
over three development sets): 0.25 × M1 (DL Sprint, 60 h) + 0.75 × M5-300 (all data below, ~300 h).

| Data | Hours used | License |
|---|---:|---|
| DL Sprint 4.0 long-form ASR (Bengali-Loop), training recordings, forced-aligned | ~60 | no license stated; see below |
| Common Voice 27.0, Bengali (train) | ~34 | CC0 |
| Kathbath, Bengali (train) | ~34 | CC BY 4.0 |
| Shrutilipi, Bengali | ~28 | CC BY 4.0 |
| OpenSLR 53, Large Bengali ASR training data | ~144 | CC BY-SA 4.0 |

The test sets are the corpora's official test (or validation) splits; OpenSLR 53 sentences that also occur in a test
set were removed before training.

## Limitations

- Scored on forced-aligned clips; whole recordings are harder (14.2% WER above).
- Spelling follows the training corpora's conventions, which differ between corpora (15–50% of the remaining errors
  are spelling only).
- Regional dialects are written as the transcripts of the training data write them; strongly dialectal speech is less
  well covered.

## Data provenance and takedown

The DL Sprint 4.0 audio comes from public Bangla YouTube channels and is subject to the competition's rules, which
state no data license; no audio is included in this repository. If you are a rights holder or the organizers and want
a model or derived material removed, open an issue at [github.com/omikhan4901/asr](https://github.com/omikhan4901/asr).

## Citation

Mehboob Ehsan Khan, Nabigah Bin Sayeed, Nafisa Rodela. *Why N-best Rescoring Plateaus for Bangla Speech Recognition,
and How Far Fixing the Cause Goes.* CSE499 capstone, North South University, 2026. github.com/omikhan4901/asr
