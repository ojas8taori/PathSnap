# Two laptop deployment workflow

Use the same PathSnap repository and model code on both machines. Laptop 1 runs the API and holds the model, evaluator archive, road graphs and reference routes. Laptop 2 serves the UI. The UI sends a trip ID only; route references remain on Laptop 1.

## Before launch

1. Connect both laptops to the same trusted private network.
2. Use Python 3.11 and install `requirements.txt` in the PathSnap virtual environment on Laptop 1. Scikit-learn is pinned to 1.6.1 for compatibility with the saved model.
3. Put `artifacts/final_model.joblib` and `artifacts/final_metadata.json` on Laptop 1. Put the official evaluator archive on Laptop 1 as well. Do not copy the evaluator archive into the Git repository.
4. Find Laptop 1's private IPv4 address. The commands below call it `<LAPTOP1_IP>`; use the real address without angle brackets.
5. Allow inbound TCP port 8001 on Laptop 1's private network firewall. Allow port 8502 on Laptop 2 only if other devices need to open the UI from Laptop 2.

## Laptop 1 API

From the repository root in PowerShell:

```powershell
$env:PATHSNAP_DATA_ARCHIVE = "C:\path\to\evaluator_test.zip"
$env:PATHSNAP_ARTIFACT = "artifacts\final_model.joblib"
$env:PATHSNAP_EVALUATOR_PARTITION = "evaluator_test"
$env:PATHSNAP_CORS_ORIGINS = "http://<LAPTOP2_IP>:8502"
.venv\Scripts\python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8001
```

Keep this PowerShell window open. Confirm readiness locally on Laptop 1:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8001/trips
```

The first response should show `ready`; the second should list all 10 evaluator trip IDs. The API terminal records each `/predict` request and its summary metrics.

## Laptop 2 UI

Use the same repository revision. In `ui/config.json`, set the API URL to Laptop 1's private address:

```json
{
  "apiBaseUrl": "http://<LAPTOP1_IP>:8001"
}
```

From the repository root, serve the UI:

```powershell
.venv\Scripts\python.exe -m http.server 8502 --bind 0.0.0.0 --directory ui
```

Open `http://<LAPTOP2_IP>:8502` in the browser on Laptop 2. The page should show the API address and load 10 evaluator trip IDs. Select a trip and click **Predict route**. Confirm the predicted map and permitted metrics appear, and confirm the matching `prediction_complete` line appears in Laptop 1's API window.

## Troubleshooting

- If port 8001 or 8502 is already in use, stop the existing process or choose unused ports and update the API command, CORS origin, UI `apiBaseUrl`, and UI server command together.
- If `/health` is ready locally but Laptop 2 cannot load trips, check Laptop 1's private IP, firewall rule, and CORS origin. CORS must match the browser origin exactly, including the UI port.
- If the UI loads but reports an API error, call `http://<LAPTOP1_IP>:8001/health` from Laptop 2 and confirm the evaluator archive path and model artifact path in Laptop 1's environment.
- Keep both machines on the same repository version. Do not split API and UI into separate codebases.
