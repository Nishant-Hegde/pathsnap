# PathSnap – Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University.

### Team Members
- **Person A (Data Processing, EDA, Feature Engineering):** Saanvi Samayam
- **Person B (Model Development, Comparison & Bayesian Tuning):** Rishab Naveen
- **Person C (HMM & Viterbi Sequence Reconstruction, Route Continuity):** Nishant Hegde
- **Person D (FastAPI Serving Backend, Streamlit UI & Deployment):** Prabhu R Habagali

PathSnap matches noisy GPS trajectories to a road network: candidate road
segments are generated for every GPS point, a classifier scores them, and an
HMM/Viterbi decoder reconstructs a connected route. A FastAPI backend
(Laptop 1) serves predictions to an evaluator UI (Laptop 2).

## Repository layout
| Folder | Contents | Owner / Member |
|---|---|---|
| `data_processing/` | raw-data loading, cleaning, projection, per-point labelling | Saanvi Samayam (Person A) |
| `eda/` | exploratory analysis + Folium EDA maps | Saanvi Samayam (Person A) |
| `features/` | candidate generation and feature engineering | Saanvi Samayam (Person A) |
| `models/` | training + comparison of 5+ models | Rishab Naveen (Person B) |
| `tuning/` | Bayesian hyperparameter optimisation | Rishab Naveen (Person B) |
| `hmm/` | HMM / Viterbi route reconstruction, evaluation, error analysis | Nishant Hegde (Person C) |
| `api/` | FastAPI backend | Prabhu R Habagali (Person D) |
| `ui/` | evaluator interface | Prabhu R Habagali (Person D) |
| `viz/` | Folium route maps (`viz/route_map.py`), saved to `viz/maps/` | Nishant Hegde (Person C) |
| `report_materials/`| Complete technical report, PDF, individual section markdown files, figures, tables | Team |
| `results/` | CSVs, logs, plots, maps | Team |
| `artifacts/` | trained model (`model.joblib`) + configs | Rishab Naveen / Nishant Hegde |

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

# 2. Labels, candidates and features (Person A - Saanvi Samayam)
python features/build_dataset.py

# 3. Models (Person B - Rishab Naveen)
python -m models.m4_boosting     # Trains LightGBM on 5-fold GroupKFold
python -m models.compare         # Generates comparison tables and benchmark plots

# 4. Bayesian tuning (Person B - Rishab Naveen)
python -m tuning.tune_optuna     # 25-trial Optuna TPE tuning -> artifacts/model.joblib

# 5. HMM / Viterbi (Person C - Nishant Hegde)
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
