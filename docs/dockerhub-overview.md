## Overview

`rafayali990/breast-cancer-triage-api` is a FastAPI inference service wrapping a scikit-learn logistic regression pipeline (`StandardScaler` + `LogisticRegression`) trained on the scikit-learn `load_breast_cancer` dataset. It accepts 30 numeric cell-nucleus measurements and classifies a sample as likely `malignant` or likely `benign`, with `malignant` (class `0`) treated as the positive class.

**Disclaimer:** This image is an educational MLOps exercise and is **not** a medical device. It must not be used for clinical diagnosis or patient care.

## Tags

| Tag | Description |
| :--- | :--- |
| `1.0.0` | Pinned release containing the trained v1.0.0 model and serving API. |
| `latest` | Points to the same image digest as `1.0.0`. |

## Quick Start with `docker run`

Start the container on host port `8080` (without Redis, request counting is disabled):

```bash
docker run -d --name triage -p 8080:8000 rafayali990/breast-cancer-triage-api:1.0.0
```

Verify container health:

```bash
curl http://localhost:8080/health
```

Send a prediction request (all 30 numeric features from the breast cancer dataset are required, with finite values >= 0):

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8,
    "mean_area": 1001.0, "mean_smoothness": 0.1184, "mean_compactness": 0.2776,
    "mean_concavity": 0.3001, "mean_concave_points": 0.1471, "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871, "radius_error": 1.095, "texture_error": 0.9053,
    "perimeter_error": 8.589, "area_error": 153.4, "smoothness_error": 0.006399,
    "compactness_error": 0.04904, "concavity_error": 0.05373, "concave_points_error": 0.01587,
    "symmetry_error": 0.03003, "fractal_dimension_error": 0.006193, "worst_radius": 25.38,
    "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0,
    "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654, "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
  }'
```

## Run with Docker Compose (with Redis Counter)

Save the following as `docker-compose.yml` and run `docker compose up -d`:

```yaml
services:
  redis:
    image: redis:7.4.2-alpine
    command: ["redis-server", "--appendonly", "yes"]
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 5s
    restart: unless-stopped

  api:
    image: rafayali990/breast-cancer-triage-api:1.0.0
    ports:
      - "8080:8000"
    environment:
      REDIS_HOST: redis
      REDIS_PORT: 6379
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

volumes:
  redis-data:
```

## API Endpoints

| Method & Path | Description |
| :--- | :--- |
| `GET /health` | Liveness check (`{"status": "ok"}`), independent of Redis. |
| `POST /predict` | Returns `label` (`malignant` or `benign`), `malignant_probability`, `threshold`, and `model_version`. |
| `GET /model/info` | Returns model family, version, scikit-learn version, git commit, threshold, feature count, and test metrics. |
| `GET /stats` | Returns `predictions_served` count and `redis` state (`not configured`, `connected`, or `unavailable`). |

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_PATH` | `artifacts/model.joblib` | Path to the serialized scikit-learn pipeline (`model_card.json` is read from the same directory). |
| `DECISION_THRESHOLD` | `0.5` | Probability threshold in `[0, 1]` at or above which a case is classified as `malignant`. |
| `LOG_LEVEL` | `INFO` | Application logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `REDIS_HOST` | *(unset)* | Redis hostname. Request counting is enabled only when set and non-empty. |
| `REDIS_PORT` | `6379` | Redis TCP port (`1`-`65535`). Invalid values fail fast at startup. |
| `MODEL_VERSION` | *(set at build time)* | Model version string baked into the image (falls back to `model_card.json` if unset). |
| `GIT_SHA` | *(set at build time)* | Source Git commit SHA baked into the image (`unknown` if not provided at build time). |

## Runtime Details

- **Exposed port:** `8000`
- **User:** Runs as non-root user `appuser` (`uid=10001`, `gid=10001`).
- **Healthcheck:** Built-in `HEALTHCHECK` polling `http://127.0.0.1:8000/health` every 30s.

## Source Repository

GitHub: https://github.com/ProgrammingWithRafay/breast-cancer-triage-service
```