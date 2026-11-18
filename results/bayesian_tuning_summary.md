# HistGradientBoosting Bayesian tuning

The executed Colab notebook completed with return code 0. It selected HistGradientBoosting and ran 12 Gaussian-process expected-improvement trials. The objective was mean route Macro F1 across five-fold trip-grouped CV on the 80 training trips. The search varied learning rate from 0.02 to 0.20, `max_iter` from 50 to 250, and `max_leaf_nodes` from 15 to 63. Reported search time was 460.2 seconds.

The best trial was trial 4: `learning_rate=0.0655946`, `max_iter=167`, `max_leaf_nodes=51`; the default `l2_regularization=1.0` was retained. Its CV route Macro F1 was 0.628434, versus 0.627129 for the untuned model's initial CV mean.

The five-fold tuned CV mean was route Macro F1 0.628434 (SD 0.038533), Top-1 0.785781 (SD 0.028284), continuity 0.878066 (SD 0.011566), and log loss 0.140454 (SD 0.013183). The untuned CV means were 0.627129, 0.785427, 0.877246 and 0.140488 respectively. Candidate miss rate was identical at 0.035337 in both runs because candidate generation did not change. Mean fit time per fold increased from 2.954 to 4.122 seconds.

| Student-test metric | Untuned | Tuned | Change |
|---|---:|---:|---:|
| Route Macro F1 | 0.646695 | 0.651763 | +0.005068 |
| Top-1 accuracy | 0.823680 | 0.823201 | −0.000479 |
| Candidate Macro F1 | 0.895011 | 0.894971 | −0.000040 |
| Route continuity | 0.897627 | 0.897427 | −0.000200 |
| Candidate log loss | 0.127333 | 0.126993 | −0.000340 (lower is better) |
| Candidate miss rate | 0.009193 | 0.009193 | 0 |
| Student-fit time (s) | 3.576 | 5.752 | +2.176 |
| Student-inference time (s) | 2.812 | 2.417 | −0.395 |

The tuned model improved student-test route Macro F1 by about 0.51 percentage points and slightly reduced log loss. Top-1 accuracy and route continuity were effectively unchanged, with tiny decreases. Fit time rose by 2.18 seconds in this run. The candidate miss rate did not change because the generator is shared by both versions.

The full `results/bayesian_tuning.json` contains all 12 parameter sets, fold records, and the tuned-versus-untuned CV details. The tuned model and its metadata are in `artifacts/`. The evaluator package was not opened or used for these development choices.
