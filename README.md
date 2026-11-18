# PathSnap – Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University.

PathSnap matches noisy GPS trajectories to a road network: candidate road
segments are generated for every GPS point, a classifier scores them, and an
HMM/Viterbi decoder reconstructs a connected route. A FastAPI backend
(Laptop 1) serves predictions to an evaluator UI (Laptop 2).

## Repository layout
| Folder | Contents | Owner |
|---|---|---|
| `data_processing/` | raw-data loading, cleaning, projection, per-point labelling | A |
| `eda/` | exploratory analysis + Folium EDA maps | A |
| `features/` | candidate generation and feature engineering | A |
| `models/` | training + comparison of 5+ models | B |
| `tuning/` | Bayesian hyperparameter optimisation | B |
| `hmm/` | HMM / Viterbi route reconstruction, evaluation, error analysis | C |
| `api/` | FastAPI backend | D |
| `ui/` | evaluator interface | D |
| `viz/` | Folium route maps (`viz/route_map.py`), saved to `viz/maps/` | C |
| `results/` | CSVs, logs, plots, maps | all |
| `artifacts/` | trained model + configs | B/C |

## Setup
```bash
pip install -r requirements.txt
```
Place the provided dataset (not committed) so that the layout is:
```
data/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/student_test/<trip_id>/...
```

## Run order
```bash
# 1. EDA (train + student_test only)
python eda/eda.py
python eda/folium_eda_map.py --part train --trips 00000048 00000002

# 2. Labels, candidates and features
python features/build_dataset.py

# 3. Models            -> TODO (Person B)
# 4. Bayesian tuning   -> TODO (Person B: python tuning/tune_optuna.py)
# 5. HMM / Viterbi (needs artifacts/model.joblib from step 3)
python -m hmm.test_synthetic --edge                       # sanity test, no data needed
python -m hmm.make_scored                                 # OOF train + student_test probabilities -> artifacts/scored/
python -m hmm.evaluate --part train --estimate            # data-driven sigma / beta
python -m hmm.evaluate --part train --scored artifacts/scored/train_oof_scored.pkl --tune   # -> artifacts/hmm_config.json
python -m hmm.evaluate --part student_test --scored artifacts/scored/student_test_scored.pkl
python -m hmm.analyze                                     # error analysis
python -m hmm.plots                                       # results/hmm/plots/
python -m viz.route_map --trip data/student_test/00000091 --scored artifacts/scored/student_test_scored.pkl
# 6. API / UI          -> TODO (Person D)
```

### Viewing Folium maps
OpenStreetMap blocks tiles for pages opened as local files. Serve the repo:
```bash
python -m http.server 8080
# open http://localhost:8080/results/eda/maps/train_00000048.html
```

## HMM / Viterbi results (student_test, 10 trips)
| | Top-1 | Continuity | Route precision | Route recall |
|---|---|---|---|---|
| Classifier only | 87.3% | 95.1% | 87.0% | 98.3% |
| Classifier + HMM | 93.0% | 99.9% | 98.5% | 98.5% |

Tuned parameters (train out-of-fold only): beta = 5 m, sigma = 8 m, alpha = 0.7 (`artifacts/hmm_config.json`).
Details: `results/hmm/` and report sections 5-6.

## Deployment (final evaluation)
- **Laptop 1 (API):** TODO
- **Laptop 2 (UI):** TODO

## Data-leakage safeguards
- Evaluator-testing data is never used for training, feature selection or tuning.
- Labels (`.route`) are read only for training/evaluation, never by the prediction pipeline.
- Cross-validation is grouped by trip (`trip_id`).

See `README_personA.md` for data-processing design decisions.

## AI-assistance disclosure
TODO – describe which parts used AI tools, as required by the event policy.
