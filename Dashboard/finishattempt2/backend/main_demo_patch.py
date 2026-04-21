# Patch backend/main.py with these additions.
# This keeps your existing /health and /predict endpoints and adds:
#   - /demo/                       static frontend
#   - /demo-api/random-prediction  random demodata -> model output
#   - /                            redirect to /demo/

# Add these imports near the top:
from pathlib import Path
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from demo_routes import create_demo_router

# Add this after predictor = None:
ROOT_DIR = Path(__file__).resolve().parent.parent

# Add this after app = FastAPI(...):
app.include_router(create_demo_router(lambda: predictor))
app.mount("/demo", StaticFiles(directory=str(ROOT_DIR / "demo"), html=True), name="demo")

@app.get("/")
def root():
    return RedirectResponse(url="/demo/")
