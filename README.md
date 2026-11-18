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
| `tuning/` | Bayesian hyperparameter optimisation | C |
| `hmm/` | HMM / Viterbi route reconstruction | C |
| `api/` | FastAPI backend | D |
| `ui/` | evaluator interface | D |
| `maps/` | Folium maps of predictions | D |
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
# 4. Bayesian tuning   -> TODO (Person C)
# 5. HMM / Viterbi     -> TODO (Person C)
# 6. API / UI          -> TODO (Person D)
```

### Viewing Folium maps
OpenStreetMap blocks tiles for pages opened as local files. Serve the repo:
```bash
python -m http.server 8080
# open http://localhost:8080/results/eda/maps/train_00000048.html
```

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
