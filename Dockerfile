# Minimal runtime image for run_match_orchestrator.py -- the previously
# missing PlayerDynamics deployment artifact (see
# PRODUCTION_READINESS_AUDIT.md §6: "PlayerDynamics itself has no
# Dockerfile/entrypoint"). Reuses the project's single requirements.txt
# as-is for correctness; trimming it down to only what
# run_match_orchestrator.py's six pipeline layers actually need (dropping
# torch/xgboost/lightgbm/scikit-learn, which belong to the separate
# analysis/orchestrator.py player-physiology pipeline, not this one) is a
# follow-up build-time optimization, not done here.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "run_match_orchestrator.py"]
CMD ["--match-id", "unset"]
