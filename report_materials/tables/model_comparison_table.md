# PathSnap Model Comparison (Candidate Scoring Level)

| Model | Family | CV Macro F1 (Mean) | CV F1 (Std) | CV Top-1 Acc | Student-Test F1 | Student-Test Top-1 | Test Log Loss | Latency (ms/trip) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | Linear | 0.836 | 0.0305 | 0.8486 | 0.8509 | 0.8683 | 0.1774 | 4.89 |
| KNN | Instance-Based | 0.8317 | 0.1008 | 0.358 | 0.9426 | 0.4105 | 0.3063 | 369.72 |
| RandomForest | Bagging Ensemble | 0.8929 | 0.0347 | 0.8408 | 0.9139 | 0.873 | 0.109 | 21.8 |
| LightGBM | Gradient Boosting | 0.8945 | 0.0463 | 0.8643 | 0.9018 | 0.8757 | 0.1245 | 16.2 |
| MLP | Neural Network | 0.8321 | 0.0722 | 0.7376 | 0.9131 | 0.8657 | 0.1165 | 16.8 |
| Tuned_LightGBM | Gradient Boosting (Tuned) | 0.8993 | 0.0275 | 0.8538 | 0.913 | 0.8728 | 0.121 | 12.8 |

*Note: Route continuity will be populated when integrated with Person C's HMM decode module.*
