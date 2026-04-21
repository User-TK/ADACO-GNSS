from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schemas import GNSSInput, PredictionOutput
from predictor import GNSSPredictor

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model once when server starts
    global predictor
    predictor = GNSSPredictor()
    yield
    # cleanup on shutdown if needed

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