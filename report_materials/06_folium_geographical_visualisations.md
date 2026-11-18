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
