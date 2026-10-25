"""Request and response models for the triage API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Strict, finite, non-negative float: rejects numeric strings, booleans, NaN/inf and negatives
Feature = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]

# First row of the scikit-learn breast cancer dataset (a malignant case), shown in /docs
_EXAMPLE = {
    "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8,
    "mean_area": 1001.0, "mean_smoothness": 0.1184, "mean_compactness": 0.2776,
    "mean_concavity": 0.3001, "mean_concave_points": 0.1471, "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871, "radius_error": 1.095, "texture_error": 0.9053,
    "perimeter_error": 8.589, "area_error": 153.4, "smoothness_error": 0.006399,
    "compactness_error": 0.04904, "concavity_error": 0.05373,
    "concave_points_error": 0.01587, "symmetry_error": 0.03003,
    "fractal_dimension_error": 0.006193, "worst_radius": 25.38, "worst_texture": 17.33,
    "worst_perimeter": 184.6, "worst_area": 2019.0, "worst_smoothness": 0.1622,
    "worst_compactness": 0.6656, "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654, "worst_symmetry": 0.4601,
    "worst_fractal_dimension": 0.1189,
}


class PredictRequest(BaseModel):
    """The 30 tumour features, all required, in training column order."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_EXAMPLE]})

    mean_radius: Feature
    mean_texture: Feature
    mean_perimeter: Feature
    mean_area: Feature
    mean_smoothness: Feature
    mean_compactness: Feature
    mean_concavity: Feature
    mean_concave_points: Feature
    mean_symmetry: Feature
    mean_fractal_dimension: Feature
    radius_error: Feature
    texture_error: Feature
    perimeter_error: Feature
    area_error: Feature
    smoothness_error: Feature
    compactness_error: Feature
    concavity_error: Feature
    concave_points_error: Feature
    symmetry_error: Feature
    fractal_dimension_error: Feature
    worst_radius: Feature
    worst_texture: Feature
    worst_perimeter: Feature
    worst_area: Feature
    worst_smoothness: Feature
    worst_compactness: Feature
    worst_concavity: Feature
    worst_concave_points: Feature
    worst_symmetry: Feature
    worst_fractal_dimension: Feature


# Ordered feature names as declared above (checked against the model card at startup)
FEATURE_NAMES: tuple[str, ...] = tuple(PredictRequest.model_fields)


class PredictResponse(BaseModel):
    # Allow field names starting with "model_" (Pydantic reserves that prefix by default)
    model_config = ConfigDict(protected_namespaces=())

    label: Literal["malignant", "benign"]
    malignant_probability: float
    threshold: float
    model_version: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    model_family: str
    scikit_learn_version: str
    git_commit: str
    decision_threshold: float
    feature_count: int
    test_metrics: dict[str, float]


class StatsResponse(BaseModel):
    predictions_served: int | None
    redis: str
