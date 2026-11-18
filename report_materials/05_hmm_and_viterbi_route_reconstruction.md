# 5. HMM and Viterbi Route Reconstruction

**Author:** Nishant Hegde (HMM & Viterbi Sequence Reconstruction, Route Continuity)  
**Team:** PathSnap (UE24CS352A Hackathon, PES University)  
**Collaborators:** Saanvi Samayam, Rishab Naveen, Prabhu R Habagali  

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
