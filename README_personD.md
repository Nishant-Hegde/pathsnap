# PathSnap: API, UI and two-laptop runbook

## Development mode

Place the supplied dataset under `data/student_test/<trip_id>/`.  The current
student folder has no `eval_test` directory, so the server deliberately uses
`student_test` until organisers provide the evaluation partition.

Install Person D's packages together with the data/feature packages:

```powershell
pip install -r requirements_personD.txt
pip install numpy pandas scipy scikit-learn
```

On Laptop 1, from the repository root:

```powershell
$env:PATHSNAP_DATA = (Get-Location).Path + "\data"
$env:PATHSNAP_SERVING_PARTITION = "student_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` on Laptop 1 to test the API. `GET /health`
checks startup; `GET /trips` lists choices; `POST /predict` runs a route.

On Laptop 2, install `requirements_personD.txt`, then run:

```powershell
$env:PATHSNAP_API_URL = "http://LAPTOP_1_IP:8000"
streamlit run ui/app.py
```

Use the IPv4 address reported by `ipconfig` on Laptop 1. Both devices must be
on the same Wi-Fi or phone hotspot. Allow inbound TCP port 8000 through Laptop
1's firewall if Windows prompts. The UI requests `/trips` and calls `/predict`
for every Run click; it does not use saved predictions.

## Switching to the final model pipeline

`api/main.py` currently imports `pipeline_stub.predict_route`, which selects the
closest road candidate and exists only to enable early UI/API integration. When
the final `pipeline.predict_route` is ready, replace that import with:

```python
from pipeline import predict_route
```

The predictor receives a `Trip` copy whose `route` is `None`. Reference routes
remain available only inside `api/scoring.py`, which returns aggregate metrics
but never exposes routes or per-point labels to the UI.
