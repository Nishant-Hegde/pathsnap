# PathSnap: Map Matching and Route Reconstruction
### Technical Report
**UE24CS352A Machine Learning Hackathon | PES University**  
**Team:** Saanvi Samayam (Data and EDA), Rishab Naveen (Models and Tuning), Nishant Hegde (HMM/Viterbi), Prabhu R Habagali (API/UI)  
**Repository:** [https://github.com/Nishant-Hegde/pathsnap](https://github.com/Nishant-Hegde/pathsnap)  
**PDF Report:** [PathSnap_Technical_Report.pdf](PathSnap_Technical_Report.pdf)

---

## 1. Problem Statement and Workflow
Phone GPS locations are not always exact. A GPS dot may appear beside the road, especially near tall buildings and in dense urban canyons. PathSnap takes these noisy GPS points and reconstructs the exact sequence of road segments a vehicle actually traversed, in the correct order. This is known as **map matching**.

The system pipeline consists of four stages:
1. **Candidates and features:** For each GPS point, find up to 10 road segments within 60 m and compute 21 geometric, kinematic, and topological features.
2. **Classifier:** A Bayesian-tuned LightGBM model outputs the calibrated posterior probability $P(\text{arc} \mid \text{features})$ for each candidate road segment.
3. **HMM and Viterbi:** Candidate probabilities become emission scores; shortest path road-network travel distances become transition scores; the Viterbi dynamic programming algorithm reconstructs the globally optimal, topologically continuous route.
4. **Serving & Interface:** A production FastAPI backend on Laptop 1 serves route predictions to a Streamlit UI on Laptop 2 over LAN.

---

## 2. Dataset Understanding and Exploratory Data Analysis

### 2.1 Dataset Structure
The dataset comprises 90 driving trips split into:
- **80 training trips** (used for EDA, feature engineering, and cross-validation)
- **10 student-test trips** (used for model evaluation and HMM benchmarking)
- **10 evaluator-test trips** (reserved strictly for final evaluation)

Each trip provides four whitespace-separated files:
- `.track`: GPS latitude, longitude, and timestamps
- `.nodes`: Road intersection nodes and their coordinates
- `.arcs`: Directed road connections between nodes
- `.route`: Chronological sequence of ground-truth road segments

### 2.2 Preprocessing Design
- **Local Metric Equirectangular Projection:** Coordinates span latitudes across Europe, the Americas, and Asia. A single global UTM zone induces severe distortion. We project each trip into a local equirectangular coordinate system centered on that trip's median GPS coordinate, yielding sub-0.1% metric distance accuracy.
- **Monotone Route Alignment:** The `.route` file gives an ordered arc sequence without per-point timestamps. A forward-only dynamic programming alignment pairs each GPS point with its corresponding route arc.
- **Quality Audits & Trip Exclusions:** Trips `00000033` and `00000054` had median GPS-to-route distances > 120 m (the vehicle departed from the recorded route). These two anomalous trips were strictly excluded from training.
- **Candidate Generation:** KD-Trees index arc geometries sampled every 5 m, retrieving the top 10 candidate segments within a 60 m radius. This produces 1,259,224 candidate rows with an empirical candidate miss rate of only 1.50% in train and 0.26% in student-test.

---

## 3. Machine Learning Candidate Scoring

### 3.1 5-Model Benchmark (5-Fold GroupKFold)
All models were benchmarked under 5-Fold `GroupKFold` cross-validation grouped strictly by `trip_id` (zero data leakage across folds):

| Model | Algorithm Family | CV Macro F1 | Test Macro F1 | Test Top-1 Accuracy | Test Log Loss | Inference Latency |
|---|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.8360 ± 0.0305 | 0.8509 | 86.83% | 0.1774 | 4.89 ms/trip |
| KNN | Instance-Based | 0.8317 ± 0.1008 | 0.9426 | 41.05% | 0.3063 | 369.72 ms/trip |
| Random Forest | Bagging Ensemble | 0.8929 ± 0.0347 | 0.9139 | 87.30% | 0.1090 | 21.80 ms/trip |
| **LightGBM (Baseline)** | **Gradient Boosting** | **0.8945 ± 0.0463** | **0.9018** | **87.57%** | **0.1245** | **16.20 ms/trip** |
| Neural Network (MLP) | Deep Learning | 0.8321 ± 0.0722 | 0.9131 | 86.57% | 0.1165 | 16.80 ms/trip |
| **Tuned LightGBM** | **Gradient Boosting (Tuned)** | **0.8993 ± 0.0275** | **0.9130** | **87.28%** | **0.1210** | **12.80 ms/trip** |

---

## 4. Bayesian Hyperparameter Optimization (Optuna)
Using Optuna with the Tree-structured Parzen Estimator (TPE) across 25 trials:
- **Best Hyperparameters (Trial 15):** `learning_rate=0.0843`, `num_leaves=42`, `max_depth=7`, `min_child_samples=48`, `subsample=0.887`, `colsample_bytree=0.763`, `reg_alpha=0.00014`, `reg_lambda=0.022`, `scale_pos_weight=5.482`.
- **Validation Stability:** Reduced fold-to-fold Macro F1 variance by **40.6%** ($\pm 0.0463 \rightarrow \pm 0.0275$).
- **Inference Speedup:** Reduced inference latency by **21.0%** ($16.20\text{ ms} \rightarrow 12.80\text{ ms/trip}$).
- Exported production artifact: `artifacts/model.joblib` (330 KB).

---

## 5. HMM and Viterbi Sequence Reconstruction
Independent candidate classification predicts road arcs based solely on point features, occasionally jumping between parallel streets. The Hidden Markov Model enforces physical reachability over the directed road network:
- **Emission Modeling:** $P(O_t \mid S_t = a) = (\hat{P}(a \mid x_t) + \epsilon)^\alpha$, with $\alpha=0.7, \epsilon=10^{-4}$.
- **Transition Modeling:** $P(S_{t+1}=b \mid S_t=a) = \frac{1}{\beta}\exp\left(-\frac{|d_{\text{net}}(a,b) - d_{\text{euc}}|}{\beta}\right)$, with $\beta=5\text{ m}$.

### Impact of HMM on Student-Test Trips (19,160 GPS Points):
| Evaluation Metric | Classifier-Only (Tuned LightGBM) | Classifier + HMM (Viterbi) | Net Improvement |
|---|---|---|---|
| **Pointwise Top-1 Accuracy** | 87.29% | **92.97%** | **+5.69%** |
| **Route Continuity** | 95.14% | **99.92%** | **+4.78%** |
| **Route Precision** | 86.96% | **98.46%** | **+11.50%** |
| **Route Recall** | 98.29% | **98.47%** | **+0.18%** |
| **Route Jaccard Similarity** | 85.66% | **97.01%** | **+11.35%** |

---

## 6. Two-Laptop Deployment & Visual Interface
- **Laptop 1 (FastAPI Backend):** Exposes `GET /health`, `GET /trips`, and `POST /predict`. Runs the full prediction pipeline server-side and calculates aggregate metrics without ever exposing ground-truth labels to the client.
- **Laptop 2 (Streamlit UI):** Connects to Laptop 1 via LAN (`PATHSNAP_API_URL`), displaying interactive Folium route maps with GPS fixes, candidate options, and reconstructed continuous road polylines.
