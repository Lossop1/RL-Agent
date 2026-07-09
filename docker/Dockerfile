# Reproducible image for the Locomotion Console (FastAPI backend + built React frontend).
#
# Stage 1 builds the frontend to repo-root frontend/dist (vite build.outDir = ../frontend/dist).
# Stage 2 is a slim Python runtime with a Node runtime added only for the tiny zero-dependency
# static file server (docker/serve-frontend.mjs). The backend imports the heavy ML stack lazily,
# so the image only needs the web deps in requirements.txt.
#
# Build:  docker build -f docker/Dockerfile -t locomotion-console .
# Run:    docker run -p 8000:8000 -p 5173:5173 \
#           -e LOCOMOTION_CONSOLE_SOURCE=fake locomotion-console

# ── stage 1: frontend build ────────────────────────────────────────────────────────────
FROM node:22-bookworm-slim AS frontend
WORKDIR /src
COPY locomotion-console-ui/package.json locomotion-console-ui/
# No committed lockfile → npm install (deterministic enough for a leaf UI; pin a lockfile to use ci).
RUN cd locomotion-console-ui && npm install --no-audit --no-fund
COPY locomotion-console-ui ./locomotion-console-ui
# Emits to /src/frontend/dist (outDir ../frontend/dist) and fails the build on i18n/type errors.
RUN cd locomotion-console-ui && npm run verify

# ── stage 2: python runtime + node for the static server ────────────────────────────────
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOCOMOTION_CONSOLE_HOST=0.0.0.0 \
    LOCOMOTION_CONSOLE_PORT=8000 \
    FRONTEND_PORT=5173

# Node runtime (for serve-frontend.mjs) + a non-root user.
RUN apt-get update \
 && apt-get install -y --no-install-recommends nodejs ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Backend source + runtime scripts.
COPY autotuner ./autotuner
COPY docker ./docker
# Built frontend from stage 1.
COPY --from=frontend /src/frontend/dist ./frontend/dist

RUN chmod +x docker/entrypoint.sh && chown -R app:app /app
USER app

EXPOSE 8000 5173
# Healthcheck hits the backend liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"

ENTRYPOINT ["docker/entrypoint.sh"]
