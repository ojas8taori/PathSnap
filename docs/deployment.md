# Deployment notes

## Local development

Run the API from the repository root after training has created `artifacts/final_model.joblib` and `artifacts/final_metadata.json`:

```powershell
$env:PATHSNAP_DATA_ARCHIVE = "C:\path\to\official-evaluator-package.zip"
$env:PATHSNAP_EVALUATOR_PARTITION = "evaluator_test"
$env:PATHSNAP_ARTIFACT = "artifacts/final_model.joblib"
$env:PATHSNAP_API_HOST = "127.0.0.1"
$env:PATHSNAP_API_PORT = "8001"
uvicorn src.api:app --host $env:PATHSNAP_API_HOST --port $env:PATHSNAP_API_PORT
```

Open `ui/index.html` through a local static file server on port `8502`; `ui/config.json` points to `http://127.0.0.1:8001`. The API's `/health` endpoint reports readiness and `/trips` returns the trip IDs from the configured evaluator partition.

## Two-laptop option

Laptop 1 runs the API and holds the model artifact, road graphs, official evaluator package and reference routes. Bind the API to `0.0.0.0`; set `PATHSNAP_CORS_ORIGINS` to the exact origin serving the UI; allow API port `8001` through the local firewall. Laptop 2 serves the same `ui/` directory and changes `apiBaseUrl` in `ui/config.json` to `http://<laptop-1-address>:8001`. Start the API first, confirm `/health` returns `ready`, then open the UI and confirm it receives the trip list. The UI sends only a trip identifier. API logs are written to the receiving laptop's terminal and include the trip ID, observation count and summary metrics.

The split is a launch option only. Keep the single shared repository and the same feature/model code for local and two-laptop use. Never put evaluator reference routes in UI files or responses.

## Evaluation data handling

The official evaluator package is configured separately from the student archive. The API reports `not_ready` until both the package and final artifact exist at their configured paths. Keep route references on the API host and score only after generating a prediction from route-free trip data. Do not use student-test routes as a substitute for evaluator trips.
