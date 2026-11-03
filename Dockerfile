# Stage 1: Model trainer
FROM python:3.12.11-slim-bookworm AS trainer

WORKDIR /build

# Install training dependencies first for layer caching
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Copy training inputs with preserved relative paths
COPY params.yaml ./
COPY training/ ./training/
COPY data/raw/breast_cancer.csv ./data/raw/breast_cancer.csv

# Declare version arg right before training to preserve earlier cached layers
ARG MODEL_VERSION=1.0.0
RUN python training/train.py --data-path data/raw/breast_cancer.csv --output-dir /build/artifacts --model-version "$MODEL_VERSION"

# Stage 2: Production serving image
FROM python:3.12.11-slim-bookworm

# Constant runtime defaults (nothing here changes between commits)
ENV MODEL_PATH=artifacts/model.joblib \
    DECISION_THRESHOLD=0.5 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root system user and group with fixed UID/GID 10001
RUN groupadd --system --gid 10001 appuser && \
    useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# 1. Install serving dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 2. Copy trained model artifacts from trainer stage
COPY --from=trainer /build/artifacts ./artifacts

# 3. Build metadata goes after the dependency layers: GIT_SHA changes on every commit
ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown
ENV MODEL_VERSION=$MODEL_VERSION \
    GIT_SHA=$GIT_SHA
LABEL org.opencontainers.image.version=$MODEL_VERSION \
    org.opencontainers.image.revision=$GIT_SHA

# 4. Copy application source code last
COPY app/ ./app/

# Switch to non-root user
USER 10001

EXPOSE 8000

# Container healthcheck using Python stdlib urllib.request
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); sys.exit(0 if r.status == 200 else 1)"]

# Exec-form entrypoint and default command
ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "app/uvicorn_log_config.json"]