# PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION
## Comprehensive Technical Report
**Team Members:**
- **Saanvi Samayam:** Data Processing, EDA, Candidate Feature Engineering
- **Rishab Naveen:** Model Development, 5-Model Benchmark & Bayesian Tuning
- **Nishant Hegde:** HMM & Viterbi Sequence Reconstruction, Route Continuity
- **Prabhu R Habagali:** FastAPI Serving Backend, Streamlit UI & Deployment  
**Institution:** PES University | UE24CS352A Hackathon  

---

# PathSnap – Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University.

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
| `report/` | Complete technical report: PDF and markdown report | Team |
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
Details: `results/hmm/` and report sections 5-6.

## Deployment (final evaluation)
- **Laptop 1 (API):** TODO
- **Laptop 2 (UI):** TODO

## Data-leakage safeguards
- Evaluator-testing data is never used for training, feature selection or tuning.
- Labels (`.route`) are read only for training/evaluation, never by the prediction pipeline.
- Cross-validation is grouped by trip (`trip_id`).

See Section 2 below for data-processing design decisions.

## AI-assistance disclosure
TODO – describe which parts used AI tools, as required by the event policy.


---

# 2. Dataset Understanding and Exploratory Data Analysis
# Data-quality findings (train + student_test)

## train (80 trips)
- GPS points: total 208417, per trip min 361, median 1529, max 19313
- NaN values in raw tracks: 0; rows dropped (dup/invalid ts): 0
- Sampling: 96.50% of steps exactly 1 s; steps > 10 s: 590; > 60 s: 69; max gap 153064 s
- Trips with a gap > 60 s: ['00000002', '00000006', '00000011', '00000014', '00000021', '00000024', '00000028', '00000041', '00000042', '00000043', '00000044', '00000054', '00000055', '00000069', '00000070', '00000078', '00000081', '00000084', '00000085', '00000088', '00000089', '00000090']
- Road network: nodes 299-155003, arcs 649-307392, isolated nodes total 0, weak components median 20.0
- Arcs that have a reverse arc: 92.2% (directed network, mostly two-way)
- Route continuity (arc[i].to == arc[i+1].from): mean 0.9993; trips < 1: ['00000002', '00000019', '00000033', '00000041', '00000042', '00000069', '00000075', '00000080', '00000086', '00000089']
- GPS distance to nearest reference-route arc: median 5.6 m, p95 140.2 m, p99 203.3 m, max 569 m

## student_test (10 trips)
- GPS points: total 19160, per trip min 262, median 1420, max 3958
- NaN values in raw tracks: 0; rows dropped (dup/invalid ts): 0
- Sampling: 98.62% of steps exactly 1 s; steps > 10 s: 27; > 60 s: 1; max gap 210 s
- Trips with a gap > 60 s: ['00000065']
- Road network: nodes 1770-26635, arcs 3577-56027, isolated nodes total 0, weak components median 11.0
- Arcs that have a reverse arc: 92.7% (directed network, mostly two-way)
- Route continuity (arc[i].to == arc[i+1].from): mean 0.9997; trips < 1: ['00000076']
- GPS distance to nearest reference-route arc: median 5.2 m, p95 16.8 m, p99 27.9 m, max 106 m



---

# Candidate Road Segment Classification & Bayesian Hyperparameter Optimization

**Author:** Rishab Naveen (Model Development, Comparison & Bayesian Tuning)  
**Team:** PathSnap (UE24CS352A Hackathon, PES University)  
**Collaborators:** Saanvi Samayam, Nishant Hegde, Prabhu R Habagali  
**Input:** Candidate road segments and 21 spatial/kinematic features generated by Saanvi Samayam  
**Output:** Calibrated posterior probabilities $P(\text{arc} \mid \text{features})$, trained Pipelines, and Optuna-tuned LightGBM  

---

## 3. Model Comparison & Evaluation

### 3.1 Problem Formulation & Metric Definitions
Candidate road selection is framed as a supervised point-to-arc classification problem under extreme class imbalance. For each GPS observation $t$ with coordinates $(x_t, y_t)$, Saanvi Samayam's spatial candidate generation retrieves up to $K=10$ candidate road segments within radius $R=60\text{ m}$. Exactly one candidate (or zero in candidate misses) is the true road segment ($y=1$), while competing candidate arcs are negatives ($y=0$), creating an imbalance ratio of approximately $6:1$ to $7:1$ (positive rate $\approx 14.3\%$).

To evaluate candidate scoring models rigorously without data leakage:
1. **GroupKFold Cross-Validation (5 Folds):** Grouped strictly by `trip_id`. GPS points from the same trip exhibit strong temporal autocorrelation and sensor drift. Standard random K-Fold would leak trip trajectories across folds, giving falsely optimistic validation scores. GroupKFold guarantees entire trips are held out.
2. **Top-1 Accuracy:** The fraction of GPS observations whose highest-probability candidate arc matches the reference arc:
   $$\text{Top-1 Accuracy} = \frac{1}{N_{\text{obs}}} \sum_{i=1}^{N_{\text{obs}}} \mathbb{I}\left( \text{argmax}_{a \in \mathcal{C}_i} \hat{P}(y_{i, a}=1) = a^*_i \right)$$
3. **Macro F1:** The unweighted average of F1 scores across both negative and positive classes: $\frac{1}{2}(F_{1, \text{neg}} + F_{1, \text{pos}})$. This prevents trivial majority-class classifiers from scoring high.
4. **Log Loss (Binary Cross-Entropy):**
   $$\mathcal{L}_{\log} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$
   Probability quality is essential because Nishant Hegde's HMM directly treats $\hat{p}_i$ as observation emission probabilities $P(O_t \mid S_t = a)$. Poorly calibrated probabilities distort Viterbi trellis dynamic programming.
5. **Inference Latency:** Wall-clock scoring time per trip in milliseconds, critical for real-time serving in Prabhu R Habagali's FastAPI backend.

---

### 3.2 Consolidated Experimental Results

All experiments were executed on the real dataset (training partition: 78 valid trips, 1,259,224 candidate rows; student-test partition: 10 trips, 140,950 candidate rows) using `random_state=42`.

| Model | Family | 5-Fold CV Macro F1 (Mean $\pm$ Std) | CV Top-1 Accuracy | Student-Test Macro F1 | Student-Test Top-1 Accuracy | Student-Test Log Loss | Inference Latency (ms/trip) |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Generalized Linear Model (GLM) | $0.8360 \pm 0.0305$ | 0.8486 | 0.8509 | 0.8683 | 0.1774 | **4.89 ms** |
| **K-Nearest Neighbors (KNN)** | Instance-Based (Non-Parametric) | $0.8317 \pm 0.1008$ | 0.3580 | 0.9426 | 0.4105 | 0.3063 | 369.72 ms |
| **Random Forest** | Bagging Tree Ensemble | $0.8929 \pm 0.0347$ | 0.8408 | 0.9139 | 0.8730 | **0.1090** | 21.80 ms |
| **LightGBM (Untuned)** | Gradient Boosted Trees (GBDT) | $0.8945 \pm 0.0463$ | **0.8643** | 0.9018 | **0.8757** | 0.1245 | 16.20 ms |
| **Multi-Layer Perceptron (MLP)** | Neural Network (Deep Learning) | $0.8321 \pm 0.0722$ | 0.7376 | 0.9131 | 0.8657 | 0.1165 | 16.80 ms |
| **Tuned LightGBM (Optuna)** | Gradient Boosted Trees (Tuned) | **0.8993 $\pm$ 0.0275** | 0.8538 | **0.9130** | 0.8728 | 0.1210 | **12.80 ms** |

---

### 3.3 Detailed Model Subsections (Compulsory Questions A–C)

#### Model 1: Logistic Regression
- **A. Suitability & Why Chosen:** Serves as the canonical linear probabilistic baseline. Because the sigmoid link $\sigma(z) = \frac{1}{1 + e^{-z}}$ outputs strictly bounded probabilities in $(0, 1)$, it maps linear feature projections directly to posterior road likelihoods.
- **B. Strengths, Limitations & Loss Function:**
  - *Strengths:* Convex loss with guaranteed global minimum; near-instant inference ($4.89\text{ ms/trip}$); fully interpretable weights.
  - *Limitations:* Incapable of modeling non-linear interactions (e.g. heading difference matters only when vehicle speed exceeds $2\text{ m/s}$) without manual interaction features.
  - *Loss Function:* $L_2$-regularized Binary Cross-Entropy:
    $$\mathcal{L}(w) = -\sum_{i=1}^N [y_i \ln \sigma(w^T x_i) + (1-y_i)\ln(1-\sigma(w^T x_i))] + \frac{1}{2C} \|w\|_2^2$$
- **C. Performance & Over/Underfitting Diagnosis:**
  - *CV Macro F1:* $0.8360 \pm 0.0305$ | *Test Top-1 Acc:* $0.8683$.
  - *Diagnosis:* Underfitting. The model exhibits low variance across folds ($\pm 0.0305$), but its linear decision boundary cannot match the higher F1 scores of tree ensembles ($0.89+$), demonstrating high inductive bias.

#### Model 2: K-Nearest Neighbors (KNN)
- **A. Suitability & Why Chosen:** Non-parametric instance-based baseline. In Euclidean space, candidate road segments with similar spatial offsets and kinematics should naturally cluster near true road segments.
- **B. Strengths, Limitations & Loss Function:**
  - *Strengths:* No distributional or functional form assumptions; captures arbitrary local manifolds.
  - *Limitations:* Severe degradation due to the curse of dimensionality across 21 features; extremely slow inference ($369.72\text{ ms/trip}$) requiring distance evaluations against all training exemplars; produces coarse step probabilities.
  - *Criterion:* Implicit Euclidean distance metric on standardized features with distance-inverse weighting $w_i = 1 / d(x, x_i)$.
- **C. Performance & Over/Underfitting Diagnosis:**
  - *CV Top-1 Acc:* $0.3580$ | *Test Top-1 Acc:* $0.4105$ | *Test Latency:* $369.72\text{ ms}$.
  - *Diagnosis:* Severe underfitting in top-1 ranking despite high overall point recall. Standard Euclidean distance weights all 21 features equally, so uninformative noise dimensions dilute the critical signal of perpendicular distance and heading alignment.

#### Model 3: Random Forest
- **A. Suitability & Why Chosen:** Bagging ensemble of 100 decorrelated decision trees. Tree-based partitioning naturally captures non-linear step functions and hierarchical interactions without requiring feature normalization.
- **B. Strengths, Limitations & Loss Function:**
  - *Strengths:* Invariant to monotonic feature transforms; robust against GPS noise and outliers; sub-sampling handles variance effectively.
  - *Limitations:* Large memory footprint ($>100\text{ MB}$ uncompressed); slower inference ($21.80\text{ ms/trip}$) due to traversing 100 deep decision trees.
  - *Splitting Criterion:* Gini Impurity with balanced sub-sample class weighting:
    $$I_G(p) = 1 - \sum_{k=0}^1 p_k^2$$
- **C. Performance & Over/Underfitting Diagnosis:**
  - *CV Macro F1:* $0.8929 \pm 0.0347$ | *Test Macro F1:* $0.9139$ | *Test Log Loss:* $0.1090$.
  - *Diagnosis:* Well-calibrated fit. Test macro F1 ($0.9139$) tracks CV ($0.8929$) closely, and test log loss is the lowest among untuned models ($0.1090$), proving that bagging successfully controls variance.

#### Model 4: LightGBM (Gradient Boosted Decision Trees)
- **A. Suitability & Why Chosen:** State-of-the-art gradient boosting for tabular data. Optimizes histogram-binned split points and uses leaf-wise (best-first) tree growth. Native handling of class imbalance via `scale_pos_weight` ($6.0$).
- **B. Strengths, Limitations & Loss Function:**
  - *Strengths:* Highest single-model test accuracy ($87.57\%$ Top-1); fast tree evaluation ($16.20\text{ ms/trip}$); second-order gradient descent produces sharply calibrated probabilities.
  - *Limitations:* Vulnerable to overfitting without regularization (`reg_alpha`, `reg_lambda`, `min_child_samples`).
  - *Loss Function:* Binary Log Loss optimized via second-order Taylor expansion:
    $$\mathcal{L} = \sum_{i=1}^N \left[ g_i f(x_i) + \frac{1}{2} h_i f^2(x_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
- **C. Performance & Over/Underfitting Diagnosis:**
  - *CV Macro F1:* $0.8945 \pm 0.0463$ | *Test Top-1 Acc:* $0.8757$.
  - *Diagnosis:* Optimal baseline. Achieves the highest raw Top-1 accuracy before tuning.

#### Model 5: Multi-Layer Perceptron (MLP)
- **A. Suitability & Why Chosen:** Deep neural network baseline consisting of two dense hidden layers $(64, 32)$ with ReLU activations and Adam optimization. Tests continuous representation learning.
- **B. Strengths, Limitations & Loss Function:**
  - *Strengths:* Smooth continuous probability outputs; learns composite feature embeddings.
  - *Limitations:* Requires feature scaling; computationally expensive backpropagation; sensitive to learning rate.
  - *Loss Function:* Binary Cross-Entropy with $L_2$ weight decay ($\alpha=0.001$):
    $$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^N [y_i \ln \hat{y}_i + (1-y_i)\ln(1-\hat{y}_i)] + \frac{\alpha}{2} \|\theta\|_2^2$$
- **C. Performance & Over/Underfitting Diagnosis:**
  - *CV Macro F1:* $0.8321 \pm 0.0722$ | *Test Top-1 Acc:* $0.8657$ | *Test Log Loss:* $0.1165$.
  - *Diagnosis:* Early stopping at epoch 50 prevented severe overfitting. Training loss dropped smoothly while validation score plateaued (visualized in `results/plots/mlp_loss_curve.png`).

---

## 4. Bayesian Hyperparameter Optimization

### 4.1 Search Space & Design Rationale
We selected **LightGBM** for Bayesian optimization because it demonstrated the highest Top-1 accuracy and fastest inference latency among complex non-linear models. We utilized **Optuna** with the **Tree-structured Parzen Estimator (TPE)** sampler (`seed=42`) over 25 trials.

The search space was parameterized as follows:
- `learning_rate` $\in [0.02, 0.2]$ (log scale): Controls gradient step shrinkage. Smaller values smooth convergence.
- `num_leaves` $\in [15, 63]$: Governs model capacity in LightGBM's leaf-wise tree growth.
- `max_depth` $\in [4, 8]$: Constrains tree depth to prevent learning spurious high-order GPS noise.
- `min_child_samples` $\in [20, 100]$: Enforces minimum candidate observations per leaf, preventing splits on isolated GPS outliers.
- `subsample` $\in [0.7, 1.0]$: Bagging ratio for stochastic variance reduction.
- `colsample_bytree` $\in [0.7, 1.0]$: Feature subspace subsampling to decorrelate individual trees.
- `reg_alpha` (L1) $\in [10^{-6}, 1.0]$ (log scale): Promotes feature sparsity.
- `reg_lambda` (L2) $\in [10^{-6}, 1.0]$ (log scale): Shrinks leaf weights to stabilize emission probabilities for Viterbi decoding.
- `scale_pos_weight` $\in [4.0, 8.0]$: Addresses the $6:1$ negative-to-positive candidate class imbalance.

**Optimization Objective:** 5-fold GroupKFold Cross-Validation Macro F1 on the training partition. Macro F1 was chosen over raw accuracy to prevent majority-class collapse while penalizing false positive candidate road segments.

---

### 4.2 Optimization Progression & Best Parameters

From `results/tuning/best_params.json` (Trial #15):
```json
{
  "learning_rate": 0.08427,
  "num_leaves": 42,
  "max_depth": 7,
  "min_child_samples": 48,
  "subsample": 0.8871,
  "colsample_bytree": 0.7634,
  "reg_alpha": 0.000142,
  "reg_lambda": 0.02195,
  "scale_pos_weight": 5.482
}
```

### 4.3 Untuned vs Tuned LightGBM Comparison

| Metric | Untuned LightGBM | Tuned LightGBM (Optuna TPE) | Improvement / Delta |
|---|---|---|---|
| **CV Macro F1 (Mean)** | 0.8945 | **0.8993** | **+0.0048 (+0.54%)** |
| **CV F1 Stability ($\sigma$)**| $\pm 0.0463$ | **$\pm 0.0275$** | **40.6% variance reduction across folds** |
| **Student-Test Macro F1** | 0.9018 | **0.9130** | **+0.0112 (+1.24%)** |
| **Student-Test Log Loss** | 0.1245 | **0.1210** | **-0.0035 (Better probability calibration)** |
| **Inference Latency** | 16.20 ms/trip | **12.80 ms/trip** | **21.0% speedup** |

### 4.4 Final Model Export & HMM Integration Hand-off
The tuned pipeline (`StandardScaler` + `LGBMClassifier`) was exported to:
- `artifacts/model.joblib` (Full self-contained pipeline, 330 KB)
- `artifacts/model_meta.json` (Hyperparameters and evaluation metadata)
- `artifacts/features/scored_train_oof.parquet` (Out-of-fold scored training data for Nishant Hegde's HMM transition matrix estimation)
- `artifacts/features/scored_student_test.parquet` (Scored test set ready for Viterbi decoding)
- `models/predict.py`: Implements `score_candidates(cand_df)` matching the contract for Prabhu R Habagali's FastAPI server.


---

# 5. HMM and Viterbi Route Reconstruction

## 5.1 Route Reconstruction Formulation
Independent point-level candidate classifications predict road arcs based purely on local spatial and kinematic features. However, vehicle movement is physically constrained to a directed, topologically continuous road network. Pointwise argmax predictions can jump between disconnected parallel streets or produce invalid transitions.

We implemented a Hidden Markov Model (HMM) decoded via the Viterbi algorithm to resolve trajectory-level sequence continuity:
- **Hidden States ($S_t$):** The set of candidate road arcs $a \in \mathcal{C}_t$ generated for GPS point $t$.
- **Observations ($O_t$):** The GPS observation vector (coordinates, speed, heading, timestamp).
- **Emission Probability ($P(O_t \mid S_t = a)$):** The calibrated posterior probability $\hat{P}(a \mid x)$ output by Rishab Naveen's tuned LightGBM model:
  $$P(O_t \mid S_t = a) = (\hat{P}(a \mid x_t) + \epsilon)^\alpha$$
  where $\alpha = 0.7$ and $\epsilon = 0.0001$.
- **Transition Probability ($P(S_{t+1} = b \mid S_t = a)$):** Models the plausibility of moving from road arc $a$ to arc $b$ within time $\Delta t$:
  $$P(b \mid a) = \frac{1}{\beta} \exp\left( -\frac{|d_{\text{net}}(a, b) - d_{\text{euc}}|}{\beta} \right)$$
  subject to physical speed feasibility $v_{\text{max}} = 60.0\text{ m/s}$.

## 5.2 Experimental Comparison: Classifier-Only vs Classifier + HMM
Evaluated on the 10 student-testing trips (10 trips, 19,160 GPS points):

| Metric | Classifier-Only (Tuned LightGBM) | Classifier + HMM (Viterbi) | Absolute Gain |
|---|---|---|---|
| **Pointwise Top-1 Accuracy** | 87.29% | **92.97%** | **+5.69%** |
| **Route Continuity** | 95.14% | **99.92%** | **+4.78%** |
| **Route Precision** | 86.96% | **98.46%** | **+11.50%** |
| **Route Recall** | 98.29% | **98.47%** | **+0.18%** |
| **Route Jaccard Similarity** | 85.66% | **97.01%** | **+11.35%** |

**Conclusion:** HMM decoding boosts pointwise accuracy by 5.69% and eliminates trajectory fragmentation, achieving 99.92% route continuity.


---

# 6. Geographical Visualisation Using Folium

Folium-based interactive map generation is integrated directly into the PathSnap workflow.

## 6.1 Available Interactive Maps
The following interactive Folium maps are committed and can be opened in any web browser:
1. `viz/maps/hmm_fixed_00000091.html`: Demonstrates a case where noisy GPS observations caused classifier-only predictions to skip onto an adjacent road, and the HMM Viterbi decoder successfully restored the correct, continuous street sequence.
2. `viz/maps/hmm_struggled_00000077.html`: Diagnostic case study showing challenging dense urban intersection transitions.
3. `results/eda/maps/train_00000002.html` and `train_00000048.html`: Exploratory maps displaying GPS points, road network density, and reference route overlay.

## 6.2 Visual Styling & Legend
- **Blue Points / Line:** Raw GPS trajectory observations with timestamp popups.
- **Gray Lines:** Local directed road network graph.
- **Green Polyline:** Reference ground-truth route sequence.
- **Red Polyline:** Model-predicted reconstructed route sequence.


---

# 7. API, UI and Deployment
# PathSnap: API, UI and two-laptop runbook

## Development mode

Place the supplied dataset under `data/student_test/<trip_id>/`.  The current
student folder has no `eval_test` directory, so the server deliberately uses
`student_test` until organisers provide the evaluation partition.

Install all packages (API/UI included) from the single requirements file:

```powershell
pip install -r requirements.txt
```

On Laptop 1, from the repository root:

```powershell
$env:PATHSNAP_DATA = (Get-Location).Path + "\data"
$env:PATHSNAP_SERVING_PARTITION = "student_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` on Laptop 1 to test the API. `GET /health`
checks startup; `GET /trips` lists choices; `POST /predict` runs a route.

On Laptop 2, install `requirements.txt`, then run:

```powershell
$env:PATHSNAP_API_URL = "http://LAPTOP_1_IP:8000"
streamlit run ui/app.py
```

Use the IPv4 address reported by `ipconfig` on Laptop 1. Both devices must be
on the same Wi-Fi or phone hotspot. Allow inbound TCP port 8000 through Laptop
1's firewall if Windows prompts. The UI requests `/trips` and calls `/predict`
for every Run click; it does not use saved predictions.

## Prediction Pipeline Integration

`api/main.py` directly imports `pipeline.predict_route` for end-to-end evaluation.

The predictor receives a `Trip` copy whose `route` is `None`. Reference routes
remain available only inside `api/scoring.py`, which returns aggregate metrics
but never exposes routes or per-point labels to the UI.

Route continuity uses the same definition as `hmm.evaluate`: same or directly
connected arcs count, and a transition with a road-network gap of at most 30 m
also counts. This permits one very short skipped arc and keeps API/report values
consistent.

