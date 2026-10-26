# Data licenses and what this repository redistributes

Checked 2026-10-27 on each dataset's own page. Not legal advice.

| Data | Used for | License / terms | Obligations |
|---|---|---|---|
| DL Sprint 4.0 long-form ASR (Bengali-Loop) | train, dev, eval (media speech) | Kaggle data tab: "Subject to Competition Rules". The rules name no data license; the audio comes from 11 public YouTube channels and the transcripts from their subtitle tracks, human-verified (Bengali-Loop paper: users should respect the original source licenses; takedown channel offered). | No audio is redistributed. We treat the transcripts and models derived from it as permissively usable (see below). |
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

- No license is stated for the competition data, and the competition rules (§10.1) require finalist teams to publish
  their weights "under an open-source or Creative Commons license of their choice", so the organizers expect models
  trained on this data to be public. **We therefore treat the data as permissively usable** for releasing models,
  the segmentation and the verified references, and never redistribute the audio itself.
- The underlying media belongs to the channel owners. On request by the organizers or a rights holder, we remove the
  affected transcripts, segmentation or models (contact: the repository owner); the model and dataset cards state the
  data provenance and this route.

## Licensing the release (decision open)

- **Weights:** trained on CC0, CC BY and CC BY-SA data plus the DL Sprint data (treated as permissive, above). Whether share-alike (OpenSLR 53)
  carries over to model weights is unsettled; a CC BY-SA 4.0 or OpenRAIL-style license for the weights avoids the
  question, Apache-2.0 matches the base model.
- **Code:** the repository has no LICENSE file yet, so the code is all rights reserved until one is added.
