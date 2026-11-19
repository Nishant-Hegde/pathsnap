# PathSnap – Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University.

### Team Members
- **Saanvi Samayam:** Data Processing, EDA, Candidate Feature Engineering
- **Rishab Naveen:** Model Development, 5-Model Benchmark & Bayesian Tuning
- **Nishant Hegde:** HMM & Viterbi Sequence Reconstruction, Route Continuity
- **Prabhu R Habagali:** FastAPI Serving Backend, Streamlit UI & Deployment

PathSnap matches noisy GPS trajectories to a road network: candidate road
segments are generated for every GPS point, a classifier scores them, and an
HMM/Viterbi decoder reconstructs a connected route. A FastAPI backend
(Laptop 1) serves predictions to an evaluator UI (Laptop 2).

## Repository layout
| Folder | Contents | Owner / Member |
|---|---|---|
| `data_processing/` | raw-data loading, cleaning, projection, per-point labelling | Saanvi Samayam |
| `eda/` | exploratory analysis + Folium EDA maps | Saanvi Samayam |
| `features/` | candidate generation and feature engineering | Saanvi Samayam |
| `models/` | training + comparison of 5+ models | Rishab Naveen |
| `tuning/` | Bayesian hyperparameter optimisation | Rishab Naveen |
| `hmm/` | HMM / Viterbi route reconstruction, evaluation, error analysis | Nishant Hegde |
| `api/` | FastAPI backend | Prabhu R Habagali |
| `ui/` | evaluator interface | Prabhu R Habagali |
| `viz/` | Folium route maps (`viz/route_map.py`), saved to `viz/maps/` | Nishant Hegde |
| `report/` | Complete technical report (PDF) | Team |
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

# 2. Labels, candidates and features (Saanvi Samayam)
python features/build_dataset.py

# 3. Models (Rishab Naveen)
python -m models.m4_boosting     # Trains LightGBM on 5-fold GroupKFold
python -m models.compare         # Generates comparison tables and benchmark plots

# 4. Bayesian tuning (Rishab Naveen)
python -m tuning.tune_optuna     # 25-trial Optuna TPE tuning -> artifacts/model.joblib

# 5. HMM / Viterbi (Nishant Hegde)
python -m hmm.test_synthetic --edge                       # sanity test, no data needed
python -m hmm.make_scored                                 # OOF train + student_test probabilities -> artifacts/scored/
python -m hmm.evaluate --part train --estimate            # data-driven sigma / beta
python -m hmm.evaluate --part train --scored artifacts/scored/train_oof_scored.pkl --tune   # -> artifacts/hmm_config.json
python -m hmm.evaluate --part student_test --scored artifacts/scored/student_test_scored.pkl
python -m hmm.analyze                                     # error analysis
python -m hmm.plots                                       # results/hmm/plots/
python -m viz.route_map --trip data/student_test/00000091 --scored artifacts/scored/student_test_scored.pkl

# 6. API / UI (Prabhu R Habagali)
uvicorn api.main:app --host 0.0.0.0 --port 8000
streamlit run ui/app.py
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
Details: `results/hmm/` and `report/PathSnap_Technical_Report.pdf`.

## Two-Laptop Deployment (Final Evaluation Runbook)
Both laptops must be connected to the same Wi-Fi network or mobile hotspot.

- **Laptop 1 (FastAPI Server - Prabhu R Habagali):**
  ```powershell
  $env:PATHSNAP_DATA = (Get-Location).Path + "\data"
  $env:PATHSNAP_SERVING_PARTITION = "student_test"   # or "evaluator_test"
  uvicorn api.main:app --host 0.0.0.0 --port 8000
  ```
  *(Verify by opening `http://127.0.0.1:8000/docs` or checking IP via `ipconfig`)*

- **Laptop 2 (Streamlit UI - Prabhu R Habagali):**
  ```powershell
  $env:PATHSNAP_API_URL = "http://<LAPTOP_1_IP>:8000"
  streamlit run ui/app.py
  ```

## Data-leakage safeguards
- Evaluator-testing data is never used for training, feature selection or tuning.
- Labels (`.route`) are read only for training/evaluation, never by the prediction pipeline.
- Cross-validation is grouped strictly by trip (`trip_id`).

See [report/PathSnap_Technical_Report.pdf](report/PathSnap_Technical_Report.pdf) for complete design and evaluation details.
