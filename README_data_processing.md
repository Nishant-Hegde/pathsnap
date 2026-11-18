# PathSnap – Saanvi Samayam: Data processing, EDA, labels, candidates

## Folder layout (drop into the team repo root)
```
data/                      <- put the provided "Student folder" contents here
   train/<trip>/<trip>.{track,nodes,arcs,route}
   student_test/<trip>/...
data_processing/io_utils.py   loader, cleaning, metre projection, geometry helpers
data_processing/labels.py     per-GPS-point labels from .route (monotone alignment)
eda/eda.py                    inventory, stats, data-quality log, plots
eda/folium_eda_map.py         Folium map: GPS + road network + reference route + gaps
features/candidates.py        candidate generation + 21 features (shared with API)
features/build_dataset.py     builds labelled candidate tables for train/student_test
results/eda/                  EDA outputs (CSV, PNG, data_quality.md)
artifacts/features/           candidate tables (*.pkl, regenerate – do not commit if large)
```

## Run order
```bash
pip install numpy pandas scipy scikit-learn matplotlib folium
python eda/eda.py --data data --out results/eda                 # ~15 s
python eda/folium_eda_map.py --part train --trips 00000048 00000002
python features/build_dataset.py --data data --out artifacts/features   # ~30 s
```

## Key design decisions (be ready to explain these in the viva)

**1. Loading/cleaning.** Tracks are sorted by timestamp; non-finite / out-of-range
coordinates and duplicate timestamps are dropped (none were found, but the code guards for it).

**2. Distances in metres.** Lon/lat → local equirectangular projection centred on each
trip's median GPS point. Trips are spread across Europe (lat ≈ 47–56°), so a single
UTM zone would not fit all trips; a per-trip local projection is accurate to well under
1 % at city scale, far below GPS noise.

**3. Labels.** `.route` is a list of arcs, not one label per GPS row. Each GPS point is
assigned to a route arc with a monotone dynamic programme (DTW-style): the index into the
route can only stay the same or move forward, minimising total point-to-arc distance
(capped at 100 m). Monotonicity resolves routes that revisit the same street. Label
quality is checked via the distance from each point to its label arc.

**4. Excluded training trips.** 00000033 and 00000054 have median GPS-to-reference
distance of 122 m and 140 m (the reference route does not follow the GPS), so they are
excluded from training (threshold: median > 30 m). Logged in `dataset_log.json`.

**5. Candidate generation.** Each trip has its own road network, so arc ids do not
transfer between trips. Problem is framed as candidate scoring: for each GPS point, arcs
within 60 m (KD-tree over arcs densified every 5 m, then exact point-to-segment
distance), keeping the 10 nearest. Radius chosen because student_test p99
GPS-to-route distance is ≈ 28 m.

**6. Features (per point–candidate pair).** distance, rank, distance minus nearest,
distance ratio; heading difference between GPS motion (±2 points) and arc direction, the
same weighted by speed, and forward-only heading difference; speed; arc length; projection
fraction and distance to the arc end; number of candidates; distance of neighbouring GPS
points (±1, ±3) to the same arc; time gaps before/after; GPS jitter; node in/out degree;
whether the arc is two-way. Target `y` = 1 if candidate is the label arc.

## Results produced (actual runs)
| | train (78 used) | student_test (10) |
|---|---|---|
| GPS points | 183,260 | 19,160 |
| candidate rows | 1,259,224 | 140,950 |
| positive rate | 14.3 % | 13.6 % |
| candidate miss rate | 1.50 % | 0.26 % |
| median label distance | 5.0 m | 5.3 m |

EDA highlights (`results/eda/data_quality.md`): no NaNs or duplicate timestamps;
96.5 % (train) / 98.6 % (test) of steps are exactly 1 s; 22 train trips have gaps > 60 s
(max 153,064 s in train, 210 s in test); networks range 649–307k arcs, no isolated nodes;
~92 % of arcs have a reverse arc (directed, mostly two-way); reference routes are
≥ 99.9 % head-to-tail connected; median GPS-to-route distance ≈ 5 m.
