# Initial model selection

The team selected HistGradientBoosting for Bayesian tuning after reviewing the eight initial runs. This is a development-stage choice based on the permitted 80 training trips and 10 student-test trips. The evaluator archive has not been opened or used.

HistGradientBoosting and Random Forest were nearly tied on mean trip-grouped CV route Macro F1: 0.627129 versus 0.627036 across five folds. HistGradientBoosting was slightly higher on student-test route Macro F1 (0.646695 versus 0.645974) and Top-1 accuracy (0.823680 versus 0.822737). Its candidate log loss was lower on CV (0.1405 versus 0.1513) and student test (0.1273 versus 0.1347). Mean fit time per CV fold was 2.95 seconds for HistGradientBoosting and 81.15 seconds for Random Forest in the recorded cloud runs.

The CV Macro F1 gap alone is negligible, so the selection is not based on a claim of clear accuracy separation. The student-test scores, probability quality, and much lower fit time favor HistGradientBoosting as the more practical model to tune. The tuned result will still be compared with the untuned version on matching CV folds and student-test trips.

Candidate miss rate was identical across classifiers because all models share the same generator: 3.53% averaged over CV validation trips and 0.92% over student-test trips. This error source is outside the classifier and remains a limit on the achievable route score.
