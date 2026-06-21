"""
stream_codec — PlayerDynamics

Dataclass <-> Redis Stream field-dict serialization for the pipeline
dataclasses that travel over Redis Streams (see REDIS_STREAM_CONTRACTS.md
and BACKEND_INTEGRATION_IMPLEMENTATION.md):

    TacticalEvent, Possession, TeamState, TeamStateTrend, CoachInsight,
    CoachSituation -- PlayerDynamics-owned, published outbound
    MatchEvent, MatchContext -- Backend-owned, decoded inbound only
    (PlayerDynamics never publishes these two; see ingestion/match_event.py)

Wire format
------------
A Redis Stream entry is a flat str -> str field map (XADD's native shape).
Each dataclass is encoded as exactly four flat fields, matching
REDIS_STREAM_CONTRACTS.md's envelope:

    schema_version  : str(int)        -- see SCHEMA_VERSION below
    type            : the dataclass's class name (used to pick the decoder)
    match_id        : str, "" if the dataclass has no match_id field
    payload         : JSON string of the dataclass's full field set

datetime fields are not natively JSON-serialisable, so they are converted
to/from ISO-8601 strings around the json.dumps/loads call. Which fields are
datetimes is tracked explicitly per dataclass in _DATETIME_FIELDS rather
than introspected from type hints, because every module in this codebase
uses `from __future__ import annotations`, which makes
`dataclasses.fields(cls)[i].type` a STRING ("datetime") rather than the
actual `datetime` class -- explicit tracking is simpler and more reliable
than parsing that string.

Versioning
-----------
SCHEMA_VERSION bumps only on a breaking change (renamed/removed/required
field, type change) to any of the five dataclasses' wire shape -- additive
optional fields do not require a bump. decode() raises ValueError on an
unrecognised schema_version rather than guessing, consistent with the
fail-closed philosophy already used by MatchState.from_dict()
(analysis/match_state.py).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Tuple, Type

from analysis.coach_insight import CoachInsight
from analysis.coach_situation import CoachSituation
from analysis.player_analytics_event import PlayerAnalyticsEvent, PilotPlayerAnalyticsEvent
from analysis.player_workload_event import PlayerWorkloadEvent
from analysis.possession import Possession
from analysis.team_state import TeamState
from analysis.team_state_trend import TeamStateTrend
from ingestion.match_event import MatchContext, MatchEvent
from ingestion.tactical_event import TacticalEvent

SCHEMA_VERSION = 1

# class name -> (class, tuple of datetime-typed field names)
_REGISTRY: Dict[str, Tuple[Type, Tuple[str, ...]]] = {
    "TacticalEvent": (TacticalEvent, ("timestamp",)),
    "Possession": (Possession, ("start_timestamp", "end_timestamp")),
    "TeamState": (TeamState, ("timestamp",)),
    "TeamStateTrend": (TeamStateTrend, ("timestamp",)),
    "CoachInsight": (CoachInsight, ("timestamp",)),
    "CoachSituation": (CoachSituation, ("timestamp",)),
    "MatchEvent": (MatchEvent, ("timestamp",)),
    "MatchContext": (MatchContext, ("timestamp",)),
    "PlayerAnalyticsEvent": (PlayerAnalyticsEvent, ("timestamp",)),
    "PilotPlayerAnalyticsEvent": (PilotPlayerAnalyticsEvent, ("timestamp",)),
    "PlayerWorkloadEvent": (PlayerWorkloadEvent, ("timestamp",)),
}


def _json_default(o: Any) -> str:
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def encode(obj: Any) -> Dict[str, str]:
    """
    Encode a TacticalEvent / Possession / TeamState / TeamStateTrend /
    CoachInsight instance into a flat field dict ready for
    RedisStreamProducer.publish().
    """
    cls_name = type(obj).__name__
    if cls_name not in _REGISTRY:
        raise ValueError(
            f"No stream codec registered for {cls_name!r}. "
            f"Registered types: {sorted(_REGISTRY.keys())}"
        )

    payload_dict = asdict(obj)
    payload_json = json.dumps(payload_dict, default=_json_default, separators=(",", ":"))
    match_id = getattr(obj, "match_id", None)

    return {
        "schema_version": str(SCHEMA_VERSION),
        "type": cls_name,
        "match_id": str(match_id) if match_id is not None else "",
        "payload": payload_json,
    }


def decode(fields: Dict[str, str]) -> Any:
    """
    Reverse of encode() — reconstructs the original dataclass instance from
    a raw Redis Stream entry's fields.

    Raises ValueError if `type` is unregistered or `schema_version` is not
    the version this codec understands (fail closed rather than guess).
    """
    schema_version = int(fields.get("schema_version", "0"))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unrecognised stream schema_version={schema_version!r} "
            f"(this codec understands {SCHEMA_VERSION!r}) -- refusing to guess at the wire shape"
        )

    cls_name = fields.get("type", "")
    if cls_name not in _REGISTRY:
        raise ValueError(
            f"No stream codec registered for {cls_name!r}. "
            f"Registered types: {sorted(_REGISTRY.keys())}"
        )
    cls, datetime_fields = _REGISTRY[cls_name]

    payload_dict = json.loads(fields["payload"])
    for fname in datetime_fields:
        raw = payload_dict.get(fname)
        if raw is not None:
            payload_dict[fname] = datetime.fromisoformat(raw)

    return cls(**payload_dict)
