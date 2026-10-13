# Results summary

## Zero-shot and fine-tuned (greedy, current normalizer)

| test_set       | model                                                    |   WER% | WER 95% CI   |   CER% |   utts |   groups |      RTF |
|:---------------|:---------------------------------------------------------|-------:|:-------------|-------:|-------:|---------:|---------:|
| cv-test        | hishab/titu_stt_bn_conformer_large                       |   7.72 | [7.5, 8.0]   |   1.29 |   9408 |     7377 | nan      |
| cv-test        | bangla-speech-processing/BanglaASR                       |   9.66 | [9.4, 10.0]  |   2.67 |   9408 |     7377 |   0.055  |
| cv-test        | bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium |  14.81 | [14.5, 15.2] |   3.64 |   9408 |     7377 |   0.0434 |
| cv-test        | ai4bharat/indicwav2vec_v1_bengali                        |  25.8  | [25.4, 26.2] |   6.43 |   9408 |     7377 |   0.003  |
| cv-test        | hishab/titu_stt_bn_fastconformer                         |  41.15 | [40.1, 42.2] |  11.21 |   9408 |     7377 | nan      |
| cv-test        | arijitx/wav2vec2-xls-r-300m-bengali                      |  43.35 | [42.9, 43.9] |  12.72 |   9408 |     7377 |   0.0031 |
| fleurs-test    | hishab/titu_stt_bn_conformer_large                       |  14.24 | [12.8, 15.5] |   4.41 |    920 |      349 | nan      |
| fleurs-test    | bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium |  21.54 | [20.1, 23.0] |   6.46 |    920 |      349 |   0.0336 |
| fleurs-test    | ai4bharat/indicwav2vec_v1_bengali                        |  28.18 | [26.6, 29.8] |   7.8  |    920 |      349 |   0.0029 |
| fleurs-test    | bangla-speech-processing/BanglaASR                       |  32.94 | [31.2, 34.6] |  12.31 |    920 |      349 |   0.0451 |
| fleurs-test    | hishab/titu_stt_bn_fastconformer                         |  36.11 | [33.6, 38.6] |   8.99 |    920 |      349 | nan      |
| fleurs-test    | arijitx/wav2vec2-xls-r-300m-bengali                      |  47.46 | [45.9, 49.1] |  13.57 |    920 |      349 |   0.0029 |
| fleurs-test    | openai/whisper-large-v3                                  |  79.89 | [75.0, 85.3] |  31.72 |    920 |      349 |   0.1055 |
| fleurs-test    | openai/whisper-large-v3-turbo                            |  91.54 | [88.8, 94.7] |  46.07 |    920 |      349 |   0.0294 |
| kathbath-valid | hishab/titu_stt_bn_conformer_large                       |   8.32 | [7.9, 8.9]   |   1.48 |   2236 |       17 | nan      |
| kathbath-valid | bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium |  14.79 | [13.7, 16.1] |   2.98 |   2236 |       17 |   0.0443 |
| kathbath-valid | ai4bharat/indicwav2vec_v1_bengali                        |  22.15 | [20.8, 23.5] |   4.32 |   2236 |       17 |   0.0026 |
| kathbath-valid | hishab/titu_stt_bn_fastconformer                         |  23.96 | [19.3, 29.8] |   4.97 |   2236 |       17 | nan      |
| kathbath-valid | bangla-speech-processing/BanglaASR                       |  32.2  | [29.5, 34.9] |   8.18 |   2236 |       17 |   0.0568 |
| kathbath-valid | arijitx/wav2vec2-xls-r-300m-bengali                      |  38.57 | [35.5, 41.5] |   8.32 |   2236 |       17 |   0.0027 |
| kathbath-valid | openai/whisper-large-v3-turbo                            |  92.64 | [89.0, 96.5] |  49.83 |   2236 |       17 |   0.0546 |
| kathbath-valid | openai/whisper-large-v3                                  |  93.24 | [87.1, 99.7] |  38.7  |   2236 |       17 |   0.144  |
| splits_fa-dev  | bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium |  29.33 | [25.7, 33.7] |  12.65 |    646 |        6 |   0.0477 |
| splits_fa-eval | hishab/titu_stt_bn_conformer_large                       |  21.18 | [19.0, 23.9] |   7.26 |   4234 |       22 | nan      |
| splits_fa-eval | bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium |  22.71 | [20.6, 25.3] |   7.68 |   4234 |       22 |   0.0441 |
| splits_fa-eval | hishab/titu_stt_bn_fastconformer                         |  28.25 | [24.3, 33.0] |  12.21 |   4234 |       22 | nan      |
| splits_fa-eval | ai4bharat/indicwav2vec_v1_bengali                        |  42.8  | [39.1, 47.5] |  17.28 |   4234 |       22 |   0.003  |
| splits_fa-eval | bangla-speech-processing/BanglaASR                       |  58.2  | [54.6, 62.7] |  31.98 |   4234 |       22 |   0.0474 |
| splits_fa-eval | arijitx/wav2vec2-xls-r-300m-bengali                      |  65.2  | [62.2, 68.6] |  26.9  |   4234 |       22 |   0.003  |

## Plateau anatomy: dev_sampled_tugstugi

- nbest: data/nbest_fa/dev_sampled.jsonl
- n_utterances: 646
- n_candidates_max: 12
- ref_words: 18329
- wer_1best: 0.3286
- wer_mbr: 0.2944
- wer_utt_oracle: 0.2436
- coverage_miss_rate: 0.1398
- mbr_gap_recovery: 0.4024
- onebest_ref_errors: 5399
- onebest_ref_errors_selection_proof_frac: 0.4747
- mbr_temperature: 0.5

|   c |   onebest_errors |   mbr_fixed |
|----:|-----------------:|------------:|
|   0 |             2563 |           0 |
|   1 |              614 |          19 |
|   2 |              365 |          25 |
|   3 |              326 |          55 |
|   4 |              247 |          62 |
|   5 |              201 |          88 |
|   6 |              215 |         136 |
|   7 |              164 |         122 |
|   8 |              189 |         168 |
|   9 |              169 |         155 |
|  10 |              164 |         162 |
|  11 |              182 |         181 |
|  12 |                0 |           0 |

## Plateau anatomy: eval_banglaasr

- nbest: data/nbest_ext/eval_banglaasr.jsonl
- n_utterances: 4234
- n_candidates_max: 12
- ref_words: 120029
- wer_1best: 0.6096
- wer_mbr: 0.6102
- wer_utt_oracle: 0.5334
- coverage_miss_rate: 0.4526
- mbr_gap_recovery: -0.0074
- onebest_ref_errors: 71227
- onebest_ref_errors_selection_proof_frac: 0.7627
- mbr_temperature: 0.5

|   c |   onebest_errors |   mbr_fixed |
|----:|-----------------:|------------:|
|   0 |            54328 |           0 |
|   1 |             5234 |          72 |
|   2 |             2853 |         122 |
|   3 |             1906 |         179 |
|   4 |             1478 |         303 |
|   5 |             1136 |         364 |
|   6 |             1003 |         502 |
|   7 |              881 |         582 |
|   8 |              693 |         557 |
|   9 |              610 |         543 |
|  10 |              611 |         566 |
|  11 |              494 |         490 |
|  12 |                0 |           0 |

## Plateau anatomy: eval_sampled_tugstugi

- nbest: data/nbest_fa/eval_sampled.jsonl
- n_utterances: 4234
- n_candidates_max: 12
- ref_words: 120029
- wer_1best: 0.2585
- wer_mbr: 0.2284
- wer_utt_oracle: 0.1859
- coverage_miss_rate: 0.1144
- mbr_gap_recovery: 0.4142
- onebest_ref_errors: 27735
- onebest_ref_errors_selection_proof_frac: 0.495
- mbr_temperature: 0.5

|   c |   onebest_errors |   mbr_fixed |
|----:|-----------------:|------------:|
|   0 |            13729 |           0 |
|   1 |             3058 |          36 |
|   2 |             1849 |          95 |
|   3 |             1378 |         164 |
|   4 |             1188 |         247 |
|   5 |             1059 |         435 |
|   6 |              960 |         593 |
|   7 |              898 |         729 |
|   8 |              797 |         722 |
|   9 |              830 |         803 |
|  10 |              918 |         908 |
|  11 |             1071 |        1067 |
|  12 |                0 |           0 |

## Plateau anatomy: eval_tugstugi

- nbest: data/nbest_fa/eval_nbest.jsonl
- n_utterances: 4234
- n_candidates_max: 12
- ref_words: 120029
- wer_1best: 0.2585
- wer_mbr: 0.2297
- wer_utt_oracle: 0.1859
- coverage_miss_rate: 0.1144
- mbr_gap_recovery: 0.3969
- onebest_ref_errors: 27735
- onebest_ref_errors_selection_proof_frac: 0.495
- mbr_temperature: 0.5

|   c |   onebest_errors |   mbr_fixed |
|----:|-----------------:|------------:|
|   0 |            13729 |           0 |
|   1 |             3058 |          41 |
|   2 |             1849 |          96 |
|   3 |             1378 |         164 |
|   4 |             1188 |         267 |
|   5 |             1059 |         437 |
|   6 |              960 |         589 |
|   7 |              898 |         740 |
|   8 |              797 |         721 |
|   9 |              830 |         799 |
|  10 |              918 |         912 |
|  11 |             1071 |        1068 |
|  12 |                0 |           0 |

## Plateau anatomy: fleurs_banglaasr

- nbest: data/nbest_ext/fleurs_banglaasr.jsonl
- n_utterances: 920
- n_candidates_max: 12
- ref_words: 17766
- wer_1best: 0.356
- wer_mbr: 0.3447
- wer_utt_oracle: 0.2992
- coverage_miss_rate: 0.2556
- mbr_gap_recovery: 0.1982
- onebest_ref_errors: 5939
- onebest_ref_errors_selection_proof_frac: 0.7646
- mbr_temperature: 0.5

|   c |   onebest_errors |   mbr_fixed |
|----:|-----------------:|------------:|
|   0 |             4541 |           0 |
|   1 |              380 |           1 |
|   2 |              231 |           3 |
|   3 |              154 |           6 |
|   4 |              117 |          25 |
|   5 |              106 |          48 |
|   6 |               94 |          62 |
|   7 |               65 |          57 |
|   8 |               61 |          59 |
|   9 |               59 |          59 |
|  10 |               69 |          69 |
|  11 |               62 |          62 |
|  12 |                0 |           0 |

## Plateau anatomy: fleurs_tugstugi

- nbest: data/nbest_ext/fleurs_tugstugi.jsonl
- n_utterances: 920
- n_candidates_max: 12
- ref_words: 17766
- wer_1best: 0.2391
- wer_mbr: 0.2103
- wer_utt_oracle: 0.1639
- coverage_miss_rate: 0.1164
- mbr_gap_recovery: 0.3832
- onebest_ref_errors: 3548
- onebest_ref_errors_selection_proof_frac: 0.5829
- mbr_temperature: 0.5

|   c |   onebest_errors |   mbr_fixed |
|----:|-----------------:|------------:|
|   0 |             2068 |           0 |
|   1 |              333 |           3 |
|   2 |              170 |           5 |
|   3 |              141 |          14 |
|   4 |              140 |          20 |
|   5 |               91 |          33 |
|   6 |               84 |          60 |
|   7 |              101 |          95 |
|   8 |               81 |          78 |
|   9 |               99 |          98 |
|  10 |              122 |         121 |
|  11 |              118 |         118 |
|  12 |                0 |           0 |

## Plateau anatomy: kathbath_tugstugi

- nbest: data/nbest_ext/kathbath_tugstugi.jsonl
- n_utterances: 2236
- n_candidates_max: 12
- ref_words: 23995
- wer_1best: 0.1769
- wer_mbr: 0.1451
- wer_utt_oracle: 0.0933
- coverage_miss_rate: 0.0698
- mbr_gap_recovery: 0.38
- onebest_ref_errors: 3713
- onebest_ref_errors_selection_proof_frac: 0.4508
- mbr_temperature: 0.5

|   c |   onebest_errors |   mbr_fixed |
|----:|-----------------:|------------:|
|   0 |             1674 |           0 |
|   1 |              373 |           0 |
|   2 |              233 |           4 |
|   3 |              164 |          11 |
|   4 |              177 |          20 |
|   5 |              157 |          54 |
|   6 |              143 |         112 |
|   7 |              132 |         131 |
|   8 |              138 |         138 |
|   9 |              149 |         149 |
|  10 |              157 |         157 |
|  11 |              216 |         216 |
|  12 |                0 |           0 |
