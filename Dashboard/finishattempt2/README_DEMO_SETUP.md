# ADACO-GNSS model-driven demo files

This package replaces the old random-value dashboard with a real FastAPI-backed model demo.

## Files to copy

Copy these into your project:

```text
ADACO-GNSS-1/
  demo/
    index.html
    dashboard_compact.css
    dashboard_model.js
  backend/
    demo_routes.py
```

Then either patch your existing `backend/main.py` using `backend/main_demo_patch.py`, or compare against `backend/main_full.py`.

## Data flow

```text
/demo/index.html
   ↓ browser fetch every 2.2 seconds
GET /demo-api/random-prediction
   ↓ backend picks random /demodata/*.json
GNSSInput validation
   ↓
predictor.predict(...)
   ↓
PredictionOutput:
  label
  label_name
  confidence
  probabilities = [P(clean), P(spoofing), P(jamming)]
   ↓
dashboard chart/status update
```

## Expected demodata JSON shape

Each `/demodata/*.json` file should already match your `GNSSInput` schema:

```json
{
  "nav_pvt": [15 numbers],
  "nav_clock": [4 numbers],
  "nav_dop": [7 numbers],
  "nav_posecef": [4 numbers],
  "spectrum_01": [256 numbers],
  "spectrum_02": [256 numbers]
}
```

The backend validates each randomly chosen file before inference. If a file does not match the schema, the dashboard shows the API error and the `/demo-api/random-prediction` endpoint returns HTTP 422.

## Run

From the project root or backend folder, run your existing server command:

```bash
cd ADACO-GNSS-1/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

```text
http://localhost:8000/demo/
```

## Notes

- The old fake series `Neither` and `Both` were removed because your current model returns exactly three classes: Clean, Spoofing, and Jamming.
- The frontend uses relative URL `/demo-api/random-prediction`, so no CORS setup is needed when served by the same FastAPI app.
- If you host the HTML somewhere else, set `window.DEMO_API_URL` before loading `dashboard_model.js`.
