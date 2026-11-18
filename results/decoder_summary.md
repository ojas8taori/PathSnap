# HMM/Viterbi decoder comparison

The executed decoder notebook completed with return code 0. It used the tuned HistGradientBoosting model and compared pointwise top candidate selection with Viterbi decoding. CV uses the 80 training trips in five grouped folds. The decoder grid uses student-test labels for development tuning; its selected student score is therefore not an independent test result.

## Grouped CV

| Metric | Classifier only | HMM/Viterbi | Change |
|---|---:|---:|---:|
| Top-1 accuracy | 0.785781 | 0.841191 | +0.055410 |
| Route Macro F1 | 0.628434 | 0.756252 | +0.127818 |
| Route continuity | 0.878066 | 0.956670 | +0.078604 |
| Candidate miss rate | 0.035337 | 0.035337 | unchanged |
| Exact full-sequence match | 0.000000 | 0.000000 | unchanged |

All five folds improved in Top-1, route Macro F1 and continuity. Candidate miss rate does not change because both decoders use the same candidate generator. Exact full-sequence match remained zero, so neither method reproduced every observation label for an entire trip.

## Student-test decoder settings

The notebook compared four settings. The primary selection metric was mean student-test route Macro F1; continuity was the tie breaker. The selected setting was `max_gap_seconds=120`, `max_speed_mps=45`, `sigma_m=80`, and `disconnected_log_penalty=-12`.

| Metric | Classifier only | Selected HMM/Viterbi | Change |
|---|---:|---:|---:|
| Top-1 accuracy | 0.823201 | 0.868639 | +0.045438 |
| Route Macro F1 | 0.651763 | 0.786187 | +0.134424 |
| Route continuity | 0.897427 | 0.965842 | +0.068415 |
| Candidate miss rate | 0.009193 | 0.009193 | unchanged |
| Exact full-sequence match | 0.000000 | 0.000000 | unchanged |

The tight setting reached higher continuity (0.971396) but lower route Macro F1 (0.778589), so it was not selected under the stated primary metric. The permissive and stronger-disconnection-penalty settings also had lower route Macro F1.

On student trip `00000007`, decoded Top-1 fell from 0.9018 to 0.8752 even though route Macro F1 rose from 0.8741 to 0.9307. On trip `00000056`, both Top-1 and route Macro F1 fell slightly (0.7943 to 0.7812; 0.7272 to 0.7255). The decoder improved aggregate development scores but did not help every trip or every metric.

The complete fold and per-trip outputs are in `decoder_comparison.json`. The evaluator package was not used to choose the model or decoder settings.
