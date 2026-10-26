# Data licenses and what this repository redistributes

Checked 2026-10-27 on each dataset's own page. Not legal advice.

| Data | Used for | License / terms | Obligations |
|---|---|---|---|
| DL Sprint 4.0 long-form ASR (Bengali-Loop) | train, dev, eval (media speech) | Kaggle data tab: "Subject to Competition Rules". The rules name no data license; the audio comes from 11 public YouTube channels and the transcripts from their subtitle tracks, human-verified (Bengali-Loop paper: users should respect the original source licenses; takedown channel offered). | Do not redistribute the audio. See below for transcripts and models. |
| Common Voice 27.0, Bengali | train, test | CC0 1.0 (public domain) | none (citation customary) |
| Kathbath, Bengali (AI4Bharat) | train, test (validation split) | CC BY 4.0 (dataset metadata; the card also mentions CC0 for the packaging). Gated: accept the conditions on Hugging Face. | attribution |
| Shrutilipi, Bengali (AI4Bharat) | train | CC BY 4.0; gated | attribution, cite the paper |
| OpenSLR 53, Large Bengali ASR training data (Google) | train (final model only) | CC BY-SA 4.0 | attribution; share-alike applies to adaptations of the data |
| FLEURS, Bengali (Google) | test, model selection (validation) | CC BY 4.0 | attribution |
| tugstugi Whisper-medium (base model) | starting checkpoint | Apache-2.0 | keep the license notice |

## What the repository contains

- **No audio** from any corpus.
- **Transcripts:** `results/**/predictions.jsonl` carry each test clip's reference text next to the model output, the
  forced-aligned segmentation lists clip boundaries with their transcripts, and `results/labels/` holds the 605
  human-verified references. For Common Voice, Kathbath and FLEURS this is permitted with attribution (above). For DL
  Sprint the transcripts are derived from subtitle tracks of third-party videos and have no explicit license.

## DL Sprint: models and transcripts

- The competition rules (§10.1) require finalist teams to publish their weights "under an open-source or Creative
  Commons license of their choice", so the organizers expect models trained on this data to be public. That supports
  releasing our weights, but it is not a license for the data itself.
- The underlying media belongs to the channel owners. Releasing transcripts and segmentation derived from it is the
  main open point.
- **Before a public model or dataset release:** ask the organizers (dlsprint@buetcsefest2026.com) for written
  permission to publish (1) weights trained on the competition data and (2) the segmentation and verified references,
  and state the data provenance and the takedown route in the model and dataset cards.

## Licensing the release (decision open)

- **Weights:** trained on CC0, CC BY and CC BY-SA data plus the DL Sprint data. Whether share-alike (OpenSLR 53)
  carries over to model weights is unsettled; a CC BY-SA 4.0 or OpenRAIL-style license for the weights avoids the
  question, Apache-2.0 matches the base model.
- **Code:** the repository has no LICENSE file yet, so the code is all rights reserved until one is added.
