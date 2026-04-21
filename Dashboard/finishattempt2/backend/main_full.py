from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from schemas import GNSSInput, PredictionOutput
from predictor import GNSSPredictor
from demo_routes import create_demo_router

predictor = None
ROOT_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once when server starts.
    global predictor
    predictor = GNSSPredictor()
    yield
    # Cleanup on shutdown if needed.


app = FastAPI(
    title="ADACO-GNSS Inference API",
    description="Real-time GNSS spoofing and jamming detection",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: GNSSInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return predictor.predict(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Demo frontend and random demodata inference endpoint.
app.include_router(create_demo_router(lambda: predictor))
app.mount("/demo", StaticFiles(directory=str(ROOT_DIR / "demo"), html=True), name="demo")


@app.get("/")
def root():
    return RedirectResponse(url="/demo/")
