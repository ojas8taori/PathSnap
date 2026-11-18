# Eight-model experiment scope

The team's latest instruction narrows the suggested approaches to eight. The selected list is set before seeing experiment results and aims to cover materially different decision rules that can share one candidate feature pipeline:

| Model | Reason to include |
|---|---|
| Logistic Regression | Fast, interpretable linear baseline with probabilities. |
| Distance-weighted KNN | Tests whether local feature similarity ranks the right road candidate. Training is capped at 30,000 sampled rows for runtime control. |
| Random Forest | Bagged nonlinear trees; handles feature interactions and is a useful ensemble comparison. |
| Extra Trees | Randomized tree ensemble; contrasts split selection with Random Forest. |
| HistGradientBoosting | Efficient boosting option for the large candidate table. |
| RBF SVM | Tests a nonlinear margin model; training is capped at 10,000 rows because kernel fitting grows poorly with sample count. Decision scores are used for ranking; log loss is N/A. |
| Shallow MLP | One hidden layer tests a small learned nonlinear representation. |
| Deep MLP | Three hidden layers test whether added representation depth helps; early stopping is enabled. |

The reference route does not give a shared class vocabulary across trips, so all eight models use the same binary candidate-label task. The comparison reports candidate-row classification metrics and the observation-level road-arc ranking metrics separately. Training caps for KNN and RBF SVM are part of their recorded configurations and apply inside each CV fold and final student-test fit.

The following suggestions are excluded from the eight-model run:

- Ridge Classifier, SGD Classifier, Gaussian Naive Bayes and Linear SVM overlap the linear/probabilistic role covered by Logistic Regression.
- Standard KNN duplicates the instance-based comparison; the distance-weighted form keeps nearby candidate examples more influential.
- Radius Neighbours adds a radius-sensitive neighborhood rule whose empty-neighborhood behavior needs a separate search and fallback; KNN has a fixed number of neighbors.
- Decision Tree is a single-tree baseline; Random Forest and Extra Trees cover tree decisions with lower variance.
- Bagging and Extra Trees Classifier overlap the selected tree ensembles.
- Gradient Boosting and AdaBoost overlap the selected, histogram-based boosting method; XGBoost, LightGBM and CatBoost add optional library dependencies and extra environment variance.
- Linear and Polynomial SVM overlap the selected RBF kernel comparison; the RBF model tests nonlinear separation.
- Clustering approaches do not produce route labels directly. Mapping clusters to trip-local arc IDs would add an unsupervised label-mapping experiment outside the narrowed supervised comparison.

These are scope choices, not measured claims that excluded methods perform worse. The eight selected models will be evaluated empirically before a final model is chosen.
