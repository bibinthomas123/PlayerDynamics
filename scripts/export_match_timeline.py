#!/usr/bin/env python
"""
Exports real per-match, per-15-minute-segment workload aggregates for
Backend's Timeline Intelligence layer -- one row per (match_id, segment)
with the segment's total distance/sprint/acceleration/deceleration/
high-intensity load (summed across SC Magdeburg's own players only) plus
each metric's segment leader (the single SCM player who contributed most
of it in that segment).

Real source: the SAME per-window computation already used for the live
coach-facing analytics.player_workload stream (analysis/player_workload.py
compute_player_workload_windows(), fed by KinexonAdapter ->
KinexonResampler -> kinexon_events_features, exactly as
scripts/publish_player_workload.py does) -- not a new metric, just bucketed
by elapsed_s into fixed 15-minute segments instead of published per-window.

Segments: 0-15, 15-30, 30-45, 45-60, 60-75, 75-90 (minutes), matching the
Timeline Intelligence spec literally. Handball matches are 60 minutes (2x30
halves), so segments 60-75 and 75-90 are structurally always empty for
every real match in this dataset -- reported as zero-window segments
rather than omitted, so a consumer can see why they're empty (no overtime
recorded) instead of silently missing data.

Goals/turnovers per segment are NOT included here -- no real coach-tracked
GameEvent exists for any of these 4 matches yet (see
PLAYER_MATCH_HISTORY_AUDIT); Backend's TimelineIntelligenceService attaches
those separately from GameEvent.gameTimeInSec when/if they exist, rather
than this script fabricating zeros for events that were never tracked.

Usage: python scripts/export_match_timeline.py [--out path]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import sys
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import CONFIG, OWNERSHIP_SCM
from ingestion.kinexon_adapter import KinexonAdapter
from ingestion.kinexon_resampler import KinexonResampler
from analysis.pilot_pipeline import discover_match_dirs
from analysis.player_workload import compute_player_workload_windows

SEGMENT_MINUTES = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
LOAD_FIELDS = ["distance_covered", "sprint_load", "acceleration_load", "deceleration_load", "high_intensity_load"]


def _segment_label(start_min: int, end_min: int) -> str:
    return f"{start_min}-{end_min}"


def _segment_for_elapsed(elapsed_s: float) -> str | None:
    minute = elapsed_s / 60.0
    for start, end in SEGMENT_MINUTES:
        if start <= minute < end:
            return _segment_label(start, end)
    return None


def build_match_timeline(match_dir: Path, use_event_features: bool = True) -> dict:
    match_id = match_dir.name[len("match_"):]
    adapter = KinexonAdapter()
    meta = adapter.load_player_meta(match_dir / "statistics.csv")
    observations = list(
        adapter.stream_positions(match_dir / "positions.csv", meta, session_id=match_id, match_id=match_id)
    )
    if not observations:
        return {"match_id": match_id, "segments": []}

    resampler = KinexonResampler()
    events_by_player, _sessions_df = resampler.resample(observations, session_id=match_id)

    if use_event_features and (match_dir / "events.csv").exists():
        from ingestion.kinexon_events_features import merge_event_features
        events_by_player = merge_event_features(
            events_by_player=events_by_player, events_csv_path=match_dir / "events.csv",
            real_player_ids=meta.keys(), bucket_seconds=resampler.bucket_seconds,
        )

    hi_threshold = CONFIG.kinexon.high_intensity_threshold_ms
    scm_pids = [pid for pid, m in meta.items() if m.ownership == OWNERSHIP_SCM]

    # segment -> field -> running total ; segment -> field -> {player_id: total}
    totals = {seg: {f: 0.0 for f in LOAD_FIELDS} for (s, e) in SEGMENT_MINUTES for seg in [_segment_label(s, e)]}
    per_player = {seg: {f: {} for f in LOAD_FIELDS} for seg in totals}
    window_count = {seg: set() for seg in totals}

    for pid in scm_pids:
        df = events_by_player.get(pid)
        if df is None or df.empty:
            continue
        rows = compute_player_workload_windows(pid, df, hi_threshold)
        for row in rows:
            seg = _segment_for_elapsed(row["elapsed_s"])
            if seg is None:
                continue
            window_count[seg].add(pid)
            for field in LOAD_FIELDS:
                totals[seg][field] += row[field]
                per_player[seg][field][pid] = per_player[seg][field].get(pid, 0.0) + row[field]

    segments_out = []
    for start, end in SEGMENT_MINUTES:
        seg = _segment_label(start, end)
        leaders = {}
        for field in LOAD_FIELDS:
            ranked = sorted(per_player[seg][field].items(), key=lambda kv: kv[1], reverse=True)
            leaders[field] = [
                {"player_id": pid, "name": meta[pid].player_name if pid in meta else f"player_{pid}", "value": round(v, 2)}
                for pid, v in ranked[:3] if v > 0
            ]
        segments_out.append({
            "segment": seg,
            "distance_m": round(totals[seg]["distance_covered"], 2),
            "sprint_load": round(totals[seg]["sprint_load"], 2),
            "acceleration_load": round(totals[seg]["acceleration_load"], 2),
            "deceleration_load": round(totals[seg]["deceleration_load"], 2),
            "high_intensity_load": round(totals[seg]["high_intensity_load"], 2),
            "n_scm_players_with_data": len(window_count[seg]),
            "workload_leaders": leaders,
            "goals": None,
            "turnovers": None,
            "coach_events_available": False,
        })

    return {"match_id": match_id, "segments": segments_out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=str(_ROOT / "data" / "processed" / "match_timeline.json"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    match_dirs = discover_match_dirs(data_dir)
    if not match_dirs:
        print("No data/match_<id>/ directories found -- run `python main.py ingest` first.")
        return

    matches_out = []
    for match_dir in match_dirs:
        print(f"Building timeline for {match_dir.name}...")
        matches_out.append(build_match_timeline(match_dir))

    Path(args.out).write_text(json.dumps({"matches": matches_out}, indent=2), encoding="utf-8")
    print(f"Wrote timeline for {len(matches_out)} matches to {args.out}")


if __name__ == "__main__":
    main()
