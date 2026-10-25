"""FastAPI breast cancer triage service.

Label convention: target 0 = malignant (POSITIVE class), target 1 = benign.
"""

import json
import logging
import math
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, NoReturn

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings, load_settings
from app.schemas import (
    FEATURE_NAMES,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    StatsResponse,
)

MALIGNANT_CLASS = 0
MODEL_CARD_NAME = "model_card.json"
REQUIRED_CARD_KEYS = ("model_family", "feature_names", "test_metrics", "sklearn_version")


def configure_logging(level: str) -> logging.Logger:
    """Send the app logger to stdout at the configured level."""
    logger = logging.getLogger("app")
    if not logger.handlers:  # avoid duplicate handlers if the module is re-imported
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# Bad configuration fails here, at import time, before the server starts
settings: Settings = load_settings()
logger = configure_logging(settings.log_level)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------
def fail(message: str) -> NoReturn:
    """Log a startup error and abort (uvicorn then exits with a non-zero status)."""
    logger.error(message)
    raise RuntimeError(message)


def load_model(path: Path) -> Any:
    """Load the joblib-saved Pipeline."""
    if not path.is_file():
        fail(f"Model file not found: {path.resolve()}")
    try:
        return joblib.load(path)
    except Exception as exc:  # unpickling can raise many exception types
        fail(f"Could not load model file {path.resolve()}: {exc!r}")


def load_model_card(path: Path) -> dict[str, Any]:
    """Load model_card.json and check that the keys the service needs are present."""
    if not path.is_file():
        fail(f"Model card not found: {path.resolve()}")
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"Could not read model card {path.resolve()}: {exc!r}")
    missing = [key for key in REQUIRED_CARD_KEYS if key not in card]
    if missing:
        fail(f"Model card {path.resolve()} is missing keys: {missing}")
    return card


def check_feature_names(card_features: list[str]) -> None:
    """The model card's ordered features must match the request schema exactly."""
    if list(card_features) != list(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(card_features))
        extra = sorted(set(card_features) - set(FEATURE_NAMES))
        fail(
            "Model card feature_names do not match the request schema "
            f"(missing={missing}, extra={extra}; if both are empty the order differs)"
        )


def check_sklearn_version(card_version: str) -> None:
    """Warn if the model was trained with a different scikit-learn version."""
    if card_version != sklearn.__version__:
        logger.warning(
            "Model trained with scikit-learn %s but running %s",
            card_version,
            sklearn.__version__,
        )


def malignant_column(model: Any) -> int:
    """Index of the malignant class (0) in predict_proba output, looked up from classes_."""
    matches = np.flatnonzero(np.asarray(model.classes_) == MALIGNANT_CLASS)
    if matches.size != 1:
        fail(f"Model classes_ {list(model.classes_)} do not contain class {MALIGNANT_CLASS}")
    return int(matches[0])


def resolve_model_version(card: dict[str, Any]) -> str:
    """MODEL_VERSION env var wins; otherwise use the model card's model_version."""
    version = settings.model_version or card.get("model_version")
    if not version:
        fail("MODEL_VERSION is not set and the model card has no model_version")
    return str(version)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and model card once at startup."""
    logger.info(
        "Config: MODEL_PATH=%s DECISION_THRESHOLD=%s LOG_LEVEL=%s MODEL_VERSION=%s GIT_SHA=%s",
        settings.model_path,
        settings.decision_threshold,
        settings.log_level,
        settings.model_version or "(from model card)",
        settings.git_sha,
    )

    model = load_model(settings.model_path)
    card = load_model_card(settings.model_path.parent / MODEL_CARD_NAME)
    check_feature_names(card["feature_names"])
    check_sklearn_version(card["sklearn_version"])

    app.state.model = model
    app.state.model_card = card
    app.state.feature_names = list(card["feature_names"])
    app.state.malignant_index = malignant_column(model)
    app.state.model_version = resolve_model_version(card)

    logger.info(
        "Loaded %s model version %s from %s",
        card["model_family"],
        app.state.model_version,
        settings.model_path,
    )
    yield


app = FastAPI(title="Breast Cancer Triage Service", lifespan=lifespan)


def json_safe(value: Any) -> Any:
    """Replace NaN/inf (not valid JSON) with strings, recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Same 422 body as FastAPI's default, but safe when the rejected input is NaN/inf."""
    return JSONResponse(
        status_code=422,
        content={"detail": json_safe(jsonable_encoder(exc.errors()))},
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check; independent of the model."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(features: PredictRequest, request: Request) -> PredictResponse:
    """Return the malignant probability and the thresholded label for one case."""
    state = request.app.state
    row = pd.DataFrame([features.model_dump()], columns=state.feature_names)
    probability = float(state.model.predict_proba(row)[0, state.malignant_index])

    # Compare the unrounded probability with the threshold
    label = "malignant" if probability >= settings.decision_threshold else "benign"
    logger.debug("Prediction: malignant_probability=%.6f label=%s", probability, label)

    return PredictResponse(
        label=label,
        malignant_probability=round(probability, 4),
        threshold=settings.decision_threshold,
        model_version=state.model_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    """Describe the loaded model and the service configuration."""
    state = request.app.state
    card = state.model_card
    return ModelInfoResponse(
        model_version=state.model_version,
        model_family=card["model_family"],
        scikit_learn_version=sklearn.__version__,
        git_commit=settings.git_sha,
        decision_threshold=settings.decision_threshold,
        feature_count=len(state.feature_names),
        test_metrics=card["test_metrics"],
    )


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    """Placeholder until Redis-backed counters are added."""
    return StatsResponse(predictions_served=None, redis="not configured")
