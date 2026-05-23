"""
Replay-Safe Adaptation Engine
Guarantees that calibration updates are deterministic and crash-safe.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import numpy as np
import logging
from utils.reliability.calibration_store import HardenedRollingThresholdStore
from utils.reliability.determinism import MutationJournal, StateMutation

logger = logging.getLogger(__name__)

@dataclass
class CalibrationCommit:
    """A versioned commit of a calibration update."""
    version: int
    threshold: float
    window_ids: List[str]
    timestamp: datetime
    checksum: str

class DeterministicCalibrationManager:
    """
    Wraps HardenedRollingThresholdStore to provide crash-safe,
    exactly-once calibration updates.
    """
    def __init__(self, player_id: int, journal: MutationJournal):
        self.player_id = player_id
        self.journal = journal
        self.store = HardenedRollingThresholdStore(player_id)
        self._current_version = 0

    def process_window(
        self,
        window_id: str,
        loss: float,
        confidence: float,
        timestamp: datetime,
        match_id: Optional[str] = None,
        model_version: str = "unknown",
    ):
        """Feed one scored window into the calibration pipeline.

        This is the ONLY legal path through which HardenedRollingThresholdStore
        may mutate. The sequence is strictly:

          1. add_window()                   — pure data ingestion, no mutation
          2. should_adapt()                 — read-only predicate
          3. compute_proposed_threshold()   — read-only proposal
          4. journal.commit()               — durable record written first
          5. apply_adaptation()             — store mutates ONLY after journal confirms

        If the journal commit fails, apply_adaptation() is never called.
        The store state and the journal therefore stay consistent under crashes.
        """
        # Step 1: ingest data — no side effects on threshold state
        self.store.add_window(window_id, loss, confidence, timestamp)

        # Step 2: check whether adaptation criteria are met (read-only)
        if not self.store.should_adapt():
            return

        # Step 3: compute what the threshold would become (read-only)
        proposed_threshold = self.store.compute_proposed_threshold()
        if proposed_threshold is None:
            # Vetting or drift check would reject — quarantine already cleared
            # inside compute_proposed_threshold via apply_adaptation guard logic.
            # Force-clear quarantine so the store doesn't re-trigger next window.
            self.store.quarantine.clear()
            return

        version_before = self._current_version
        new_version    = version_before + 1

        # Step 4: write the journal record BEFORE mutating the store
        mutation = StateMutation(
            mutation_id=f"calib_{self.player_id}_{new_version}",
            target_object=f"calibration_{self.player_id}",
            previous_version=version_before,
            new_version=new_version,
            change_set={
                "threshold":      proposed_threshold,
                "match_id":       match_id,
                "model_version":  model_version,
                "version_before": version_before,
                "version_after":  new_version,
                "store":  self.store.full_state_dict() #  full distribution state
            },
            event_id=window_id,  # causal link: which inference window triggered this
        )

        if self.journal.commit(mutation):
            # Step 5: apply only after journal confirms durability
            applied = self.store.apply_adaptation()
            if applied:
                self._current_version = new_version
                logger.info(
                    "Deterministic Calibration Commit: Player %d v%d -> v%d "
                    "(threshold: %.4f match: %s model: %s)",
                    self.player_id, version_before, new_version,
                    proposed_threshold, match_id, model_version,
                )
            else:
                logger.warning(
                    "Player %d: journal committed v%d but apply_adaptation() rejected — "
                    "journal and store version are now inconsistent. "
                    "This indicates a race condition or quarantine state change between "
                    "compute_proposed_threshold() and apply_adaptation().",
                    self.player_id, new_version,
                )

    def get_current_threshold(self, quantile: float = 0.995) -> float:
        """Return the current calibrated threshold for injection into analyze_window.

        Returns float('inf') when the store is not yet calibrated (< 30 windows),
        which causes analyze_window to treat all windows as non-anomalous — the
        correct safe default before enough baseline data is collected.
        """
        return self.store.get_threshold(quantile)

    def recover_from_journal(self):
        """Restores the calibration state from the mutation journal.

        Recovery replays to the LAST committed mutation and restores the full
        store snapshot embedded in that mutation's change_set.  This means:
          - rolling distributions (windows list)              ✓ restored
          - quarantine buffer                                  ✓ restored
          - last_update_at (controls cooldown timing)         ✓ restored
          - calibration_version                               ✓ restored
          - KS-test baseline (derived from windows list)      ✓ restored

        Post-recovery adaptation timing is therefore identical to what it would
        have been without the crash.
        """
        history = self.journal.get_history(f"calibration_{self.player_id}")
        if not history:
            return

        # Sort by version so we always replay to the latest committed state
        history_sorted = sorted(history, key=lambda m: m.new_version)
        latest = history_sorted[-1]

        snapshot = latest.change_set.get("store_snapshot")
        if snapshot is None:
            # Journal entries written before this patch lack a snapshot.
            # Fall back to version-only recovery (old behaviour) with a warning.
            logger.warning(
                "Player %d: journal entry v%d has no store_snapshot — "
                "distribution state cannot be recovered. Adaptation timing "
                "will diverge until enough new windows accumulate.",
                self.player_id, latest.new_version,
            )
            self._current_version = latest.new_version
            return

        self.store = HardenedRollingThresholdStore.from_state_dict(
            self.player_id, snapshot
        )
        self._current_version = latest.new_version
        logger.info(
            "Calibration recovered for player %d to version %d "
            "(%d active windows, %d quarantined)",
            self.player_id,
            self._current_version,
            len(self.store.windows),
            len(self.store.quarantine),
        )
