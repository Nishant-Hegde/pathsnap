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

