"""
Integration tests (Productionization Phase 7) -- exercise the real CLI
entrypoints (`python main.py <command>`) as actual subprocesses against the
real ingested SC Magdeburg dataset already present in data/processed/,
exactly as a coach/ops engineer would run them. No mocking of the pipeline
itself: these assert on real exit codes and real written artifacts.

Skipped (not failed) when the real dataset hasn't been ingested yet in this
environment -- a fresh checkout with no data/processed/match_inventory.json
is a normal state, not a test failure.

`train` is deliberately NOT covered here (real LSTM training against the
full dataset takes real wall-clock time unsuitable for a routine test
run) -- it's exercised instead by re-running `evaluate`/`publish` against
whatever checkpoint already exists at models/shared_backbone.pt, which is
the realistic "does the already-trained model still serve correctly"
question this layer cares about.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"

INVENTORY_PATH = PROCESSED_DIR / "match_inventory.json"
BACKBONE_PATH = MODELS_DIR / "shared_backbone.pt"

pytestmark = pytest.mark.skipif(
    not INVENTORY_PATH.exists(),
    reason="data/processed/match_inventory.json not found -- run `python main.py ingest` first",
)


def _run_main(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _real_match_ids() -> list[str]:
    inventory = json.loads(INVENTORY_PATH.read_text())
    return [m["match_id"] for m in inventory["matches"]]


class TestIngest:
    def test_ingest_exits_zero_and_updates_dataset_summary(self):
        result = _run_main("ingest", timeout=180)
        assert result.returncode == 0, result.stderr

        summary_path = PROCESSED_DIR / "dataset_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["matches_total"] >= 1
        assert summary["matches_failed_validation"] == 0

    def test_ingest_logs_start_and_finish_with_duration(self):
        result = _run_main("ingest", timeout=180)
        assert "ingest START" in result.stderr
        assert "ingest FINISH" in result.stderr
        assert "duration=" in result.stderr


@pytest.mark.skipif(not BACKBONE_PATH.exists(), reason="no trained model checkpoint at models/shared_backbone.pt")
class TestEvaluate:
    def test_evaluate_kinexon_exits_zero_and_writes_metrics(self):
        match_ids = _real_match_ids()
        assert len(match_ids) > 0
        session_id = match_ids[0]

        result = _run_main("evaluate", "--data-source", "kinexon", "--session-id", session_id, timeout=120)
        assert result.returncode == 0, result.stderr

        metrics = json.loads(result.stdout)
        assert metrics["n_windows_evaluated"] >= 0
        assert metrics["n_players_evaluated"] >= 0
        assert "anomaly_score_distribution" in metrics

    def test_evaluate_logs_start_and_finish_with_duration(self):
        match_ids = _real_match_ids()
        session_id = match_ids[0]
        result = _run_main("evaluate", "--data-source", "kinexon", "--session-id", session_id, timeout=120)
        assert "evaluate START" in result.stderr
        assert "evaluate FINISH" in result.stderr


@pytest.mark.skipif(not BACKBONE_PATH.exists(), reason="no trained model checkpoint at models/shared_backbone.pt")
class TestPublish:
    def test_publish_historical_replay_exits_zero_with_no_failures(self):
        result = _run_main("publish", timeout=120)
        assert result.returncode == 0, result.stderr
        assert "publish FINISH" in result.stderr
        assert "n_failed=0" in result.stderr

    def test_publish_is_read_only_against_the_checkpoint(self):
        """publish must never retrain -- the checkpoint's mtime must be unchanged after a publish run."""
        mtime_before = BACKBONE_PATH.stat().st_mtime
        result = _run_main("publish", timeout=120)
        assert result.returncode == 0, result.stderr
        assert BACKBONE_PATH.stat().st_mtime == mtime_before


class TestStatus:
    def test_status_exits_zero_with_structured_json(self):
        result = _run_main("status", timeout=30)
        assert result.returncode == 0, result.stderr

        report = json.loads(result.stdout)
        assert "ok" in report
        assert "model" in report
        assert "ingestion" in report
        assert "redis" in report
        assert report["ingestion"]["matches_available"] >= 1
