# Notebooks

Eight notebooks walk through the paper's argument in order, one question each. They only read files that are already in
the repository (`results/`, `figures/`), so they run in seconds on any laptop: no audio, no GPU, and nothing is
recomputed. The saved outputs are in the files, so they can also be read directly on GitHub.

## Reading order

| Notebook | Question | Why it comes here |
|---|---|---|
| [01_evaluation_and_baselines](01_evaluation_and_baselines.ipynb) | How is every number measured, and where do open models start? | Every comparison later depends on one fixed scoring: the normalizer, corpus WER, confidence intervals over whole recordings, and test sets with no recording shared with training. The base model (tugstugi Whisper-medium) is chosen here: 22.7% WER on media speech. |
| [02_why_rescoring_plateaus](02_why_rescoring_plateaus.ipynb) | Does choosing among the model's own candidates help, and if not, why? | Rescoring is the cheapest improvement, so it is tried first. It does not beat greedy decoding, although a perfect selector would gain 4 points; counting which candidates contain each word shows why: most errors are in every candidate, and many are the same word in another spelling. |
| [03_list_correction](03_list_correction.ipynb) | If the missing spelling is added to the candidate list, does rescoring work? | The first test of the diagnosis, without training: adding learned spelling variants raises the ceiling and lowers WER from 22.7% to 17.5%. |
| [04_fine_tuning](04_fine_tuning.ipynb) | Which part of the model has to learn, and how much does fine-tuning gain? | The second test of the diagnosis: the model itself learns the spelling. Full fine-tuning on 60 h of media speech reaches 11.3%; training the decoder matters more than the encoder, as predicted; rescoring adds little afterwards. |
| [05_training_data_and_final_model](05_training_data_and_final_model.ipynb) | How should public corpora be mixed, and what does all the data give? | The in-domain model is weaker on read speech, so the other corpora are added: each trains its own test set best, a mix gives the best average, and all ~300 h give the final model (11.8% average; 14.2% on whole recordings). |
| [06_comparison_with_other_models](06_comparison_with_other_models.ipynb) | How does the final model compare with other open Bangla models? | Once the final model is fixed: every other open model scored with the same harness, with paired tests, plus the published numbers of larger models. |
| [07_human_verification](07_human_verification.ipynb) | Are the references right, and are the errors really the recognizer's? | The diagnosis rests on rule-based error types and on references that may be wrong; two annotators checked 605 references and the error types by hand. |
| [08_spelling_aware_wer](08_spelling_aware_wer.ipynb) | What if confirmed spelling differences are not counted as errors? | The human labels confirm which differences are spelling only, so a second metric can leave them out: the comparisons keep their direction, and about 39% of the measured fine-tuning gain lies in spelling differences. |

## Running them

```bash
python -m venv .venv && .venv/bin/pip install -e ".[notebooks]"   # once
.venv/bin/pip install jupyterlab && .venv/bin/jupyter lab notebooks/  # to open them interactively
PYTHONPATH=src .venv/bin/python notebooks/build_notebooks.py          # or: rebuild and re-execute all eight
```

The notebooks are generated from `build_notebooks.py` (cell text and code) and share the small helper `nbutil.py`;
edit those and rebuild rather than editing the `.ipynb` files. Each result shown comes with its run record (command,
git commit, runtime, library versions) where one exists.

## Cloud job notebooks

`kaggle_jobs.ipynb`, `kaggle_hishab.ipynb` and `kaggle_hishab_train.ipynb` are different: they ran the GPU jobs on
Kaggle (scoring and N-best generation) and need the private data hub; they are kept as the record of how those results
were produced. Training ran on Modal (`pipeline/modal_launch.py`).
