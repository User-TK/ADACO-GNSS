"""Demo routes for serving random demodata samples through the trained predictor.

Add this file to ADACO-GNSS-1/backend/demo_routes.py.
Then include the router from backend/main.py as shown in main_demo_patch.py.
"""

import json
import random
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException

from schemas import GNSSInput

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DEMODATA_DIR = ROOT_DIR / "demodata"


def _dump_pydantic(model):
    """Support both Pydantic v1 and v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _validate_gnss_input(payload: dict) -> GNSSInput:
    """Support both Pydantic v1 and v2."""
    if hasattr(GNSSInput, "model_validate"):
        return GNSSInput.model_validate(payload)
    return GNSSInput.parse_obj(payload)


def create_demo_router(
    get_predictor: Callable[[], object],
    demodata_dir: Optional[Path] = None,
) -> APIRouter:
    router = APIRouter(prefix="/demo-api", tags=["demo"])
    data_dir = Path(demodata_dir) if demodata_dir is not None else DEFAULT_DEMODATA_DIR

    @router.get("/random-prediction")
    def random_prediction(include_input: bool = False):
        """Pick one random JSON file from /demodata, run model inference, and return probabilities."""
        predictor = get_predictor()
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        json_files = sorted(data_dir.glob("*.json"))
        if not json_files:
            raise HTTPException(
                status_code=404,
                detail=f"No .json files found in demodata directory: {data_dir}",
            )

        sample_path = random.choice(json_files)

        try:
            payload = json.loads(sample_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read {sample_path.name}: {exc}",
            ) from exc

        try:
            gnss_input = _validate_gnss_input(payload)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{sample_path.name} does not match GNSSInput. Expected keys: "
                    "nav_pvt[15], nav_clock[4], nav_dop[7], nav_posecef[4], "
                    "spectrum_01[256], spectrum_02[256]. "
                    f"Validation error: {exc}"
                ),
            ) from exc

        try:
            prediction = predictor.predict(gnss_input)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed for {sample_path.name}: {exc}",
            ) from exc

        response = {
            "source_file": sample_path.name,
            "prediction": _dump_pydantic(prediction),
        }

        if include_input:
            response["input"] = payload

        return response

    return router
