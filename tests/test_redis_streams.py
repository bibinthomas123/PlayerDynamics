"""
tests/test_redis_streams.py

Integration tests for config/redis_client.py's RedisStreamProducer /
RedisStreamConsumer against a real Redis Streams implementation
(XADD/XREADGROUP/XACK/XGROUP CREATE) -- using fakeredis as the backend so
these tests run deterministically without a live Redis server.

fakeredis is monkeypatched in at exactly one seam: RedisConnectionPool.client()
(a classmethod). Every other line of production code in RedisStreamProducer/
RedisStreamConsumer runs unmodified against it -- this is a genuine exercise
of the real XADD/XREADGROUP/XACK/XGROUP CREATE call sequences, not a mock of
this module's own logic.

Skips cleanly (pytest.skip) if redis-py or fakeredis is not installed --
consistent with this project's established pattern for optional dependencies
(see tests/test_window_gap_reset.py's torch-skip pattern).

Covers:
  A. Stream / consumer group creation (idempotent)
  B. Backend -> Stream -> Consumer (publish_dataclass + read_new + decode)
  C. Producer -> Stream -> Backend Consumer (the reverse direction, same mechanism)
  D. XACK semantics (acked entries are not redelivered via '>')
  E. Replay after restart (read_pending() recovers an unacked entry for the
     SAME consumer_name, simulating a process restart)
  F. Duplicate protection (is_duplicate())
  G. Consumer groups (two independent groups each get their own full copy)

Run:
    pytest tests/test_redis_streams.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

try:
    import fakeredis
    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False

try:
    import redis  # noqa: F401
    _REDIS_PY_AVAILABLE = True
except ImportError:
    _REDIS_PY_AVAILABLE = False

if not (_FAKEREDIS_AVAILABLE and _REDIS_PY_AVAILABLE):
    pytest.skip(
        "redis-py and/or fakeredis not installed -- skipping Redis Streams "
        "integration tests (pip install redis fakeredis)",
        allow_module_level=True,
    )

from config.redis_client import (
    RedisConnectionPool,
    RedisStreamConsumer,
    RedisStreamProducer,
    StreamTopics,
)
from analysis.coach_insight import CoachInsight

BASE_TS = datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def fake_client(monkeypatch):
    """Redirect every RedisConnectionPool.client() call to a fresh fakeredis instance."""
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    monkeypatch.setattr(RedisConnectionPool, "client", classmethod(lambda cls: client))
    return client


@pytest.fixture()
def producer(fake_client) -> RedisStreamProducer:
    return RedisStreamProducer()


def _insight(category: str = "attack_activity_rising") -> CoachInsight:
    return CoachInsight(
        timestamp=BASE_TS, team_id="SC Magdeburg", severity="high",
        category=category, message=f"{category} for SC Magdeburg.", confidence=0.9,
        metadata={"source_metrics": [], "values": {}, "thresholds_crossed": {}, "window_seconds": 60},
    )


# ---------------------------------------------------------------------------
# A. Stream / consumer group creation
# ---------------------------------------------------------------------------

class TestStreamCreation:

    def test_ensure_stream_creates_an_empty_stream(self, fake_client, producer):
        producer.ensure_stream(StreamTopics.MATCH_EVENTS)
        assert fake_client.exists(StreamTopics.MATCH_EVENTS)

    def test_ensure_stream_is_idempotent(self, fake_client, producer):
        producer.ensure_stream(StreamTopics.MATCH_EVENTS)
        producer.ensure_stream(StreamTopics.MATCH_EVENTS)  # must not raise

    def test_consumer_creates_group_and_stream_together(self, fake_client):
        consumer = RedisStreamConsumer(
            StreamTopics.ANALYTICS_INSIGHTS, group="backend-ingest", consumer_name="worker-1",
            block_ms=50,
        )
        assert fake_client.exists(StreamTopics.ANALYTICS_INSIGHTS)
        groups = fake_client.xinfo_groups(StreamTopics.ANALYTICS_INSIGHTS)
        assert any(g["name"] == "backend-ingest" for g in groups)

    def test_second_consumer_same_group_does_not_recreate(self, fake_client):
        RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-1", block_ms=50)
        RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-2", block_ms=50)  # must not raise


# ---------------------------------------------------------------------------
# B. Backend -> Stream -> Consumer
# ---------------------------------------------------------------------------

class TestBackendToStreamToConsumer:

    def test_publish_dataclass_then_consume_and_decode(self, fake_client, producer):
        insight = _insight()
        entry_id = producer.publish_dataclass(StreamTopics.ANALYTICS_INSIGHTS, insight)
        assert entry_id is not None

        consumer = RedisStreamConsumer(
            StreamTopics.ANALYTICS_INSIGHTS, group="backend-ingest", consumer_name="worker-1",
            block_ms=50,
        )
        entries = consumer.read_new()
        assert len(entries) == 1
        got_id, fields = entries[0]
        assert got_id == entry_id
        restored = consumer.decode(fields)
        assert restored == insight

    def test_multiple_entries_delivered_in_order(self, fake_client, producer):
        insights = [_insight(f"category_{i}") for i in range(5)]
        for ins in insights:
            producer.publish_dataclass(StreamTopics.ANALYTICS_INSIGHTS, ins)

        consumer = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-1", block_ms=50)
        entries = consumer.read_new(count=10)
        decoded = [consumer.decode(f) for _id, f in entries]
        assert [d.category for d in decoded] == [f"category_{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# C. Producer -> Stream -> Backend Consumer (reverse direction, same mechanism)
# ---------------------------------------------------------------------------

class TestProducerToStreamToBackendConsumer:

    def test_playerdynamics_publishes_backend_consumes(self, fake_client, producer):
        """Same mechanism as Part B, exercised on an outbound analytics
        stream to demonstrate the PlayerDynamics -> Backend direction."""
        insight = _insight("workload_spike")
        producer.publish_dataclass(StreamTopics.ANALYTICS_TRENDS, insight)

        backend_consumer = RedisStreamConsumer(
            StreamTopics.ANALYTICS_TRENDS, group="backend-ingest", consumer_name="backend-worker-1",
            block_ms=50,
        )
        entries = backend_consumer.read_new()
        assert len(entries) == 1
        restored = backend_consumer.decode(entries[0][1])
        assert restored.category == "workload_spike"


# ---------------------------------------------------------------------------
# D. XACK semantics
# ---------------------------------------------------------------------------

class TestAckSemantics:

    def test_acked_entry_not_redelivered_via_new(self, fake_client, producer):
        producer.publish_dataclass(StreamTopics.ANALYTICS_INSIGHTS, _insight())
        consumer = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-1", block_ms=50)

        entries = consumer.read_new()
        assert len(entries) == 1
        consumer.ack(entries[0][0])

        again = consumer.read_new()
        assert again == []

    def test_unacked_entry_remains_pending(self, fake_client, producer):
        producer.publish_dataclass(StreamTopics.ANALYTICS_INSIGHTS, _insight())
        consumer = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-1", block_ms=50)

        consumer.read_new()  # delivered but never acked
        pending = fake_client.xpending(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest")
        assert pending["pending"] == 1


# ---------------------------------------------------------------------------
# E. Replay after restart
# ---------------------------------------------------------------------------

class TestReplayAfterRestart:

    def test_read_pending_recovers_unacked_entry_for_same_consumer_name(self, fake_client, producer):
        producer.publish_dataclass(StreamTopics.ANALYTICS_INSIGHTS, _insight())

        # "Process 1": reads but crashes before acking.
        consumer_v1 = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-1", block_ms=50)
        delivered = consumer_v1.read_new()
        assert len(delivered) == 1

        # "Process 2" after restart: SAME consumer_name -> Redis still has the
        # entry in worker-1's PEL even though this is a brand-new Python object.
        consumer_v2 = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-1", block_ms=50)
        replayed = consumer_v2.read_pending()
        assert len(replayed) == 1
        assert replayed[0][0] == delivered[0][0]  # same entry_id recovered

        consumer_v2.ack(replayed[0][0])
        assert consumer_v2.read_pending() == []

    def test_different_consumer_name_does_not_see_others_pending(self, fake_client, producer):
        producer.publish_dataclass(StreamTopics.ANALYTICS_INSIGHTS, _insight())
        consumer_a = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-A", block_ms=50)
        consumer_a.read_new()  # never acked

        consumer_b = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-B", block_ms=50)
        assert consumer_b.read_pending() == []  # worker-B has nothing pending of its own


# ---------------------------------------------------------------------------
# F. Duplicate protection
# ---------------------------------------------------------------------------

class TestDuplicateProtection:

    def test_first_check_is_not_a_duplicate(self, fake_client, producer):
        producer.publish_dataclass(StreamTopics.ANALYTICS_INSIGHTS, _insight())
        consumer = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-1", block_ms=50)
        entry_id, _ = consumer.read_new()[0]
        assert consumer.is_duplicate(entry_id) is False

    def test_second_check_of_same_id_is_a_duplicate(self, fake_client, producer):
        producer.publish_dataclass(StreamTopics.ANALYTICS_INSIGHTS, _insight())
        consumer = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-1", block_ms=50)
        entry_id, _ = consumer.read_new()[0]
        consumer.is_duplicate(entry_id)  # marks it seen
        assert consumer.is_duplicate(entry_id) is True

    def test_duplicate_namespace_is_scoped_per_stream_and_group(self, fake_client, producer):
        """The same entry_id string on a DIFFERENT (stream, group) must not
        be considered a duplicate -- the dedup key is namespaced."""
        producer.publish_dataclass(StreamTopics.ANALYTICS_INSIGHTS, _insight())
        consumer_1 = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "backend-ingest", "worker-1", block_ms=50)
        entry_id, _ = consumer_1.read_new()[0]
        consumer_1.is_duplicate(entry_id)

        consumer_2 = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "other-group", "worker-1", block_ms=50)
        assert consumer_2.is_duplicate(entry_id) is False


# ---------------------------------------------------------------------------
# G. Consumer groups (independent groups each see the full stream)
# ---------------------------------------------------------------------------

class TestConsumerGroups:

    def test_two_independent_groups_each_get_a_full_copy(self, fake_client, producer):
        producer.publish_dataclass(StreamTopics.ANALYTICS_INSIGHTS, _insight())

        group_a = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "group-a", "worker-1", block_ms=50)
        group_b = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "group-b", "worker-1", block_ms=50)

        assert len(group_a.read_new()) == 1
        assert len(group_b.read_new()) == 1  # group-b is unaffected by group-a's read

    def test_two_consumers_same_group_split_work(self, fake_client, producer):
        for i in range(4):
            producer.publish_dataclass(StreamTopics.ANALYTICS_INSIGHTS, _insight(f"cat_{i}"))

        worker_1 = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "shared-group", "worker-1", block_ms=50)
        worker_2 = RedisStreamConsumer(StreamTopics.ANALYTICS_INSIGHTS, "shared-group", "worker-2", block_ms=50)

        batch_1 = worker_1.read_new(count=2)
        batch_2 = worker_2.read_new(count=10)
        # worker-1 took the first 2; worker-2 gets only what's left (2).
        assert len(batch_1) == 2
        assert len(batch_2) == 2
        all_ids = {e[0] for e in batch_1} | {e[0] for e in batch_2}
        assert len(all_ids) == 4  # no entry delivered to both workers
