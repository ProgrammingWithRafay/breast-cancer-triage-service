# breast-cancer-triage-service

## 1. Overview

`breast-cancer-triage-service` is a FastAPI inference service wrapping a scikit-learn pipeline (`StandardScaler` + `LogisticRegression`) that classifies cell-nucleus measurements from the scikit-learn `load_breast_cancer` dataset (30 numeric features) as likely `malignant` or `benign`. Malignant is the positive class (target `0`).

All data preparation, train/test splitting, and model training steps use a fixed random seed of `42` everywhere, and direct dependencies are pinned to exact versions.

**Disclaimer:** This project is an educational MLOps exercise and is not a medical device.

## 2. Repository Layout

```text
breast-cancer-triage-service/
├── app/                      # FastAPI service (config.py, schemas.py, main.py, uvicorn_log_config.json)
├── training/                 # Data preparation, v2 dataset script, and model training
├── data/raw/                 # Raw dataset directory (breast_cancer.csv tracked by DVC)
├── dvc-remote/               # Committed local DVC remote storage
├── docs/                     # Documentation (including Docker Hub repository overview)
├── dvc.yaml                  # DVC pipeline stages (train) and tracked outputs/metrics
├── params.yaml               # Training and split hyperparameters (seed 42)
├── Dockerfile                # Multi-stage production Dockerfile
├── Dockerfile.naive          # Single-stage baseline Dockerfile for size/security comparison
├── docker-compose.yml        # Multi-container Compose stack (api + redis with build)
├── docker-compose.hub.yml    # Clean-room Compose file using the published Docker Hub image
├── .env.example              # Template for local Compose environment variables
└── requirements-*.txt        # Pinned dependencies (-dev.txt, -train.txt, -serve.txt)
```

## 3. Setup

Prerequisites: Python 3.10 or newer (tested on 3.10.11), Git, and Docker Desktop with Compose v2.

Run the following in PowerShell on Windows:

```powershell
git clone https://github.com/ProgrammingWithRafay/breast-cancer-triage-service.git
cd breast-cancer-triage-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt -r requirements-train.txt -r requirements-serve.txt
```

In bash, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## 4. Restore Data and Trained Artifacts

Pull the tracked dataset and model artifacts from the local `dvc-remote` folder:

```powershell
dvc pull
```

This restores the dataset (`data/raw/breast_cancer.csv`, version 2 with 539 rows) and the trained model files (`artifacts/model.joblib` and `artifacts/model_card.json`) from `dvc-remote`.

## 5. Dataset Versions

Two dataset versions are tracked in Git and DVC:
- **v1:** 569 rows (commit `4a079b5`)
- **v2:** 539 rows (commit `0ed7879`)

Switch to dataset v1 and then return to v2 (`HEAD`):

```powershell
git checkout 4a079b5 -- data/raw/breast_cancer.csv.dvc
dvc pull data/raw/breast_cancer.csv.dvc
(Get-Content data\raw\breast_cancer.csv).Count
git checkout HEAD -- data/raw/breast_cancer.csv.dvc
dvc pull data/raw/breast_cancer.csv.dvc
(Get-Content data\raw\breast_cancer.csv).Count
```

The two counts print `570` and `540`, which are the row counts (`569` and `539`) plus the header line.

**Warning:** All later work uses dataset v2. Do not re-run `training/prepare_data.py` after v2 because it recreates the 569-row v1 dataset.

## 6. Training Pipeline

Reproduce the DVC training pipeline and inspect the resulting evaluation metrics:

```powershell
dvc repro
dvc metrics show
```

`dvc repro` skips the training stage when nothing changed, and `dvc repro --force` retrains. Hyperparameters and split settings come from `params.yaml`. The training script enforces a quality gate and exits non-zero if test malignant recall is below `0.95`.

## 7. Run the API Locally

Start the FastAPI service locally (after running `dvc pull` or `dvc repro` so `artifacts/` exists):

```powershell
uvicorn app.main:app --port 8000 --log-config app/uvicorn_log_config.json
```

Endpoints:
- `GET /health` — Liveness probe returning `{"status": "ok"}` (independent of Redis).
- `POST /predict` — Classifies 30 cell-nucleus features and returns `label`, `malignant_probability`, `threshold`, and `model_version`.
- `GET /model/info` — Returns model metadata, version, scikit-learn version, git commit, decision threshold, feature count, and test metrics.
- `GET /stats` — Returns the Redis-backed `predictions_served` counter and `redis` connection state (`not configured`, `connected`, or `unavailable`).

## 8. Build the Images

Ensure the dataset is restored with `dvc pull` first, then build the naive baseline image and the production multi-stage image:

```powershell
docker build -f Dockerfile.naive -t naive .
$sha = git rev-parse --short HEAD
docker build --provenance=false --sbom=false -t rafayali990/breast-cancer-triage-api:1.0.0 --build-arg MODEL_VERSION=1.0.0 --build-arg GIT_SHA=$sha .
```

In bash, set the git SHA variable with `sha=$(git rev-parse --short HEAD)` before running `docker build`.

## 9. Run the Image on Its Own

Run the production container standalone (without Redis configured), check `/health`, and remove the container afterward so port `8080` is free for the Compose stack:

```powershell
docker run -d --name triage -p 8080:8000 rafayali990/breast-cancer-triage-api:1.0.0
curl.exe http://127.0.0.1:8080/health
docker rm -f triage
```

## 10. Compose Stack

Copy `.env.example` to `.env`, then edit `DOCKERHUB_USERNAME` and `GIT_SHA` in `.env` (`.env` is gitignored and never committed):

```powershell
Copy-Item .env.example .env
docker compose up -d --build
docker compose ps
curl.exe http://127.0.0.1:8080/stats
docker compose down
docker compose down -v
```

`docker compose down` stops and removes the containers while keeping the `redis-data` named volume so the counter persists. `docker compose down -v` removes the `redis-data` volume as well.

## 11. Publish to Docker Hub

Rebuild without attestations first, so the digest printed by the push matches the one on the Docker Hub tag page:

```powershell
docker login
$sha = git rev-parse --short HEAD
docker build --provenance=false --sbom=false -t rafayali990/breast-cancer-triage-api:1.0.0 --build-arg MODEL_VERSION=1.0.0 --build-arg GIT_SHA=$sha .
docker tag rafayali990/breast-cancer-triage-api:1.0.0 rafayali990/breast-cancer-triage-api:latest
docker push rafayali990/breast-cancer-triage-api:1.0.0
docker push rafayali990/breast-cancer-triage-api:latest
```

In bash, set the git SHA variable with `sha=$(git rev-parse --short HEAD)` before running `docker build`.

## 12. Clean-Room Check

In an empty folder containing only `docker-compose.hub.yml` and a `.env` file with:

```env
DOCKERHUB_USERNAME=rafayali990
IMAGE_TAG=1.0.0
HOST_PORT=8080
```

Start the stack from the published image, verify `/predict` using `sample_payload.json` from the repository, and tear it down:

```powershell
docker compose -f docker-compose.hub.yml up -d
curl.exe -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" --data-binary "@path\to\sample_payload.json"
curl.exe http://127.0.0.1:8080/stats
docker compose -f docker-compose.hub.yml down -v
```

## 13. Configuration

| Variable | Default | Scope | Description |
| :--- | :--- | :--- | :--- |
| `MODEL_PATH` | `artifacts/model.joblib` | API runtime | Path to the serialized model pipeline (`model_card.json` is loaded from the same directory). |
| `DECISION_THRESHOLD` | `0.5` | API runtime / Compose | Finite probability threshold in `[0, 1]` for classifying a sample as `malignant`. |
| `LOG_LEVEL` | `INFO` | API runtime / Compose | Python logging level name (validated via `logging.getLevelName`). |
| `REDIS_HOST` | *(unset)* | API runtime | Redis hostname; counting is enabled only when set and non-empty (`redis` in Compose). |
| `REDIS_PORT` | `6379` | API runtime | Redis port (integer in `[1, 65535]`; invalid values fail fast at startup). |
| `MODEL_VERSION` | `1.0.0` (build) / card fallback | Build / API runtime | Model version baked into the image or read from `model_card.json` when unset. |
| `GIT_SHA` | `unknown` | Build / API runtime | Git commit SHA baked into the image and reported by `/model/info`. |
| `DOCKERHUB_USERNAME` | *(required in `.env`)* | Compose | Docker Hub username for the `api` image reference. |
| `IMAGE_TAG` | `1.0.0` | Compose | Tag of `breast-cancer-triage-api` to build or pull. |
| `HOST_PORT` | `8080` | Compose | Host port mapped to container port `8000` on the `api` service. |

## 14. Notes

- `.gitattributes` forces LF for source and configuration files and disables line-ending conversion under `dvc-remote/`, so DVC hashes match after a clone on Windows.
- Docker Hub repository: https://hub.docker.com/r/rafayali990/breast-cancer-triage-api
