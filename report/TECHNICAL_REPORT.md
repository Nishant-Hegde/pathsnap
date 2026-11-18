# PathSnap: Map Matching and Route Reconstruction
### A simpler explanation of the complete technical report
**UE24CS352A Machine Learning Hackathon | PES University**  
**Team:** Saanvi Samayam (Data and EDA), Rishab Naveen (Models and Tuning), Nishant Hegde (HMM/Viterbi), Prabhu R Habagali (API/UI)  
**Project code:** https://github.com/Nishant-Hegde/pathsnap  
**Download PDF:** [PathSnap_Technical_Report.pdf](PathSnap_Technical_Report.pdf)

---

## 1. What problem does PathSnap solve?
Phone GPS locations are not always exact. A GPS dot may appear beside the road, especially near tall buildings. PathSnap takes these noisy GPS points and works out which roads a vehicle actually used, in the correct order. This is called **map matching**.

The system has four main parts:
1. Prepare the GPS and road data and find nearby road options;
2. Use machine learning to score each possible road;
3. Use an HMM and the Viterbi algorithm to choose a connected route; and
4. Show the result through an API and a web interface.

---

## 2. Data preparation and exploration
The data contains driving trips, GPS readings, road intersections, road segments and the actual route. The report describes 80 training trips, 10 student-test trips and 10 evaluator trips. Each trip has four files: `.nodes` (road points), `.arcs` (directed road connections), `.track` (GPS location and time) and `.route` (correct road sequence).

We convert latitude and longitude into local distances in metres. For every GPS point, we look for up to 10 nearby road segments within 60 metres and calculate 21 useful features, such as distance, direction and movement. Because the correct route file does not say exactly which GPS reading belongs to which road, a forward-only matching method connects them.

Most GPS readings are one second apart (96.5%). Some trips contain long gaps. Two unusual trips, `00000033` and `00000054`, were left out of training because their GPS traces were too far from the recorded routes. Training produced 1,259,224 candidate rows, of which 14.3% were correct matches. This means incorrect options were much more common than correct ones.

---

## 3. Testing different machine learning models
We tested five kinds of models. Each model gives a score for whether a road segment is the right match for a GPS point. We used **5-fold GroupKFold**, which keeps all data from the same trip together so that information from one trip does not leak between training and validation. Missing values were filled using training-data medians, and scaling was learned only from training folds.

### Model Comparison Table
| Model | CV Macro F1 | Test Macro F1 | Test Top-1 | Time/trip |
|---|---|---|---|---|
| Logistic Regression | 0.8360 | 0.8509 | 86.83% | 4.89 ms |
| KNN | 0.8317 | 0.9426 | 41.05% | 369.72 ms |
| Random Forest | 0.8929 | 0.9139 | 87.30% | 21.80 ms |
| LightGBM | 0.8945 | 0.9018 | 87.57% | 16.20 ms |
| Neural Network (MLP) | 0.8321 | 0.9131 | 86.57% | 16.80 ms |
| Tuned LightGBM | 0.8993 | 0.9130 | 87.28% | 12.80 ms |

**How to read the table:** Top-1 accuracy measures how often the model picks the correct road as its first choice. Macro F1 balances how well it recognizes correct and incorrect candidates. Inference time tells us how quickly the model predicts for a trip.

### What each model taught us
- **Logistic Regression:** A quick, simple starting point. It is fast but struggles when relationships between features are complex. Its test Top-1 accuracy was 86.83%. It uses a regularized binary classification loss.
- **K-Nearest Neighbors (KNN):** Looks for similar examples using distance between their features. It performed poorly at choosing the best road (41.05% Top-1) and was much slower than the other models. With 21 features, unhelpful dimensions can make nearest-neighbor comparisons less useful.
- **Random Forest:** Combines 100 decision trees. It handles unusual GPS readings well and achieved 0.9139 test Macro F1. Its test log loss was 0.1090, the lowest among the untuned models, but it was slower than LightGBM.
- **LightGBM:** Builds decision trees in stages to correct earlier errors. It handled the imbalance between correct and incorrect road candidates and gave the best untuned test Top-1 accuracy (87.57%). We chose it as the main model for the complete system.
- **Multi-Layer Perceptron (MLP):** A neural network with hidden layers of 64 and 32 units. It learned useful patterns and achieved 86.57% test Top-1 accuracy. Early stopping helped prevent overfitting.

---

## 4. Improving LightGBM with Optuna
We used **Optuna** to try 25 different LightGBM settings automatically. It searched for better learning speed, tree size, tree depth, sampling and regularization. Each trial was judged using average Macro F1 from the five grouped validation folds. The best settings were found in trial 15.

**Best settings:** `learning_rate=0.0843`; `num_leaves=42`; `max_depth=7`; `min_child_samples=48`; `subsample=0.887`; `colsample_bytree=0.763`; `reg_alpha=0.00014`; `reg_lambda=0.022`; `scale_pos_weight=5.482`.

| Measure | Before tuning | After tuning |
|---|---|---|
| CV Macro F1 | 0.8945 | **0.8993** |
| Variation across folds | 0.0463 | **0.0275** |
| Student-test Macro F1 | 0.9018 | **0.9130** |
| Student-test log loss | 0.1245 | **0.1210** |
| Prediction time per trip | 16.20 ms | **12.80 ms** |

Tuning slightly improved F1, reduced variation between validation folds, and made prediction about 21% faster. The untuned model still had a slightly higher student-test Top-1 accuracy, so the improvements depend on which metric we examine.

---

## 5. Making the route continuous with HMM and Viterbi
Even when a model chooses the correct road for most GPS points, it can jump between roads that do not connect. The **Hidden Markov Model (HMM)** helps avoid these impossible jumps. It uses two ideas:
1. **Emission scores:** how strongly LightGBM supports a road
2. **Transition scores:** whether moving from one road to the next makes sense based on the road network and travel distance.

The **Viterbi algorithm** compares possible sequences across the whole trip and selects the most likely connected route, rather than picking each road independently. The report uses emission parameters $\alpha=0.7$ and $\epsilon=0.0001$, and a distance-based transition penalty.

| Student-test measure | LightGBM only | With HMM/Viterbi |
|---|---|---|
| Top-1 accuracy | 87.29% | **92.97%** |
| Route continuity | 95.14% | **99.92%** |
| Route precision | 86.96% | **98.46%** |
| Route recall | 98.29% | **98.47%** |
| Route Jaccard similarity | 85.66% | **97.01%** |

The biggest benefit is that the final route becomes more realistic and stays connected. In one evaluator trip (`00000015`), the complete system reached **94.09% accuracy** and **99.99% route continuity** for 6,854 GPS readings. It reconstructed the route in **4.17 seconds**.

---

## 6. Showing results on a map
We use **Folium** to draw an interactive map:
- **Blue dots:** show raw GPS readings
- **Gray lines:** show the road network
- **Green lines:** show the real route
- **Red lines:** show the predicted route

The saved examples include `viz/maps/hmm_fixed_00000091.html`, where HMM fixes a wrong jump to a nearby street, and `viz/maps/hmm_struggled_00000077.html`, which shows a difficult intersection.

---

## 7. API, interface and two-laptop setup
- **Laptop 1** runs a FastAPI backend. It provides `GET /health`, `GET /trips` and `POST /predict`. It also keeps the evaluator reference routes on the server so that the client cannot access the answers directly.
- **Laptop 2** runs the Streamlit interface and connects to the backend over the local network using `PATHSNAP_API_URL`. An evaluator can choose a trip, see the matched route and view accuracy and continuity scores.

---

## 8. Final result
PathSnap turns noisy GPS readings into a connected route on a real road network. Machine learning selects promising road segments, and HMM/Viterbi checks that the complete journey makes sense. The reported evaluator result is **94.09% pointwise accuracy** and **99.99% route continuity** on the specified evaluator trip. The code, trained model and setup files are available in the team repository. Experiments used `random_state=42` for repeatability.
