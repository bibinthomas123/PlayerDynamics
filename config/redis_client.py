from __future__ import annotations

import json
import logging
import os
import ssl
import time
from threading import RLock, Thread
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy import guard — redis-py is an optional production dependency.
# The codebase falls back to JsonFileCheckpointStore when Redis is absent.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import redis
    from redis import Redis, ConnectionPool
    from redis.exceptions import (
        ConnectionError as RedisConnectionError,
        TimeoutError   as RedisTimeoutError,
        RedisError,
    )
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    logger.warning(
        "redis-py not installed. RedisCheckpointStore and RedisPubSubClient "
        "are unavailable. Install with: pip install redis[hiredis]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in ("true", "1", "yes")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# 1. Connection pool — singleton, thread-safe
# ─────────────────────────────────────────────────────────────────────────────

class RedisConnectionPool:
    """
    Singleton connection pool shared across all Redis clients in this process.

    Call RedisConnectionPool.get() to obtain the shared pool.  The pool is
    created on first call and reused thereafter.  All configuration is read
    from environment variables so it never appears in source code.

    TLS support
    ───────────
    Set REDIS_TLS=true for TLS encryption (e.g. Redis Cloud, AWS ElastiCache).
    For mutual TLS (mTLS) set REDIS_TLS_CERTFILE, REDIS_TLS_KEYFILE, and
    REDIS_TLS_CA.

    Connection health
    ─────────────────
    socket_keepalive=True prevents NAT / load-balancer idle-timeout drops.
    health_check_interval=30 causes redis-py to send PING on idle connections
    before they are handed to callers, so stale connections are recycled
    transparently rather than surfacing as application errors.
    """

    _pool: Optional["ConnectionPool"] = None
    _lock: RLock = RLock()

    @classmethod
    def get(cls) -> "ConnectionPool":
        """Return (or create) the shared connection pool."""
        if not _REDIS_AVAILABLE:
            raise RuntimeError(
                "redis-py is not installed. Cannot create a Redis connection pool."
            )

        with cls._lock:
            if cls._pool is None:
                cls._pool = cls._build_pool()
            return cls._pool

    @classmethod
    def _build_pool(cls) -> "ConnectionPool":
        host     = os.getenv("REDIS_HOST", "localhost")
        port     = _env_int("REDIS_PORT", 6379)
        db       = _env_int("REDIS_DB", 0)
        password = os.getenv("REDIS_PASSWORD") or None
        use_tls  = _env_bool("REDIS_TLS", False)

        socket_timeout  = _env_float("REDIS_SOCKET_TIMEOUT_S", 30.0)
        connect_timeout = _env_float("REDIS_CONNECT_TIMEOUT_S", 5.0)
        max_connections = _env_int("REDIS_MAX_CONNECTIONS", 20)

        if use_tls:
            ca_path   = os.getenv("REDIS_TLS_CA")
            cert_path = os.getenv("REDIS_TLS_CERTFILE")
            key_path  = os.getenv("REDIS_TLS_KEYFILE")

            # Use from_url with rediss:// scheme — handles TLS kwargs correctly
            # across all redis-py versions without touching SSLConnection directly.
            auth_part = f":{password}@" if password else "@"
            url = f"rediss://{auth_part}{host}:{port}/{db}"

            tls_kwargs: Dict[str, Any] = dict(
                socket_timeout=socket_timeout,
                socket_connect_timeout=connect_timeout,
                socket_keepalive=True,
                health_check_interval=30,
                max_connections=max_connections,
                decode_responses=True,
                ssl_cert_reqs="required",
            )
            if ca_path:
                tls_kwargs["ssl_ca_certs"] = ca_path
            if cert_path:
                tls_kwargs["ssl_certfile"] = cert_path
            if key_path:
                tls_kwargs["ssl_keyfile"] = key_path

            pool = redis.ConnectionPool.from_url(url, **tls_kwargs)
        else:
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=connect_timeout,
                socket_keepalive=True,
                health_check_interval=30,
                max_connections=max_connections,
                decode_responses=True,
            )

        logger.info(
            "RedisConnectionPool: created | host=%s port=%d db=%d tls=%s max_conn=%d",
            host, port, db, use_tls, max_connections,
        )
        return pool

    @classmethod
    def client(cls) -> "Redis":
        """Return a Redis client using the shared pool."""
        return redis.Redis(connection_pool=cls.get())

    @classmethod
    def reset(cls) -> None:
        """Tear down the pool (test helper — not for production use)."""
        with cls._lock:
            if cls._pool is not None:
                cls._pool.disconnect()
                cls._pool = None


# ─────────────────────────────────────────────────────────────────────────────
# Retry decorator
# ─────────────────────────────────────────────────────────────────────────────

def _with_retry(fn: Callable, max_attempts: int, backoff_s: float):
    """
    Run fn() with exponential backoff on transient Redis errors.

    Transient errors (ConnectionError, TimeoutError) are retried up to
    max_attempts times.  Other RedisError subtypes propagate immediately.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except (RedisConnectionError, RedisTimeoutError) as exc:
            if attempt == max_attempts - 1:
                raise
            sleep_s = backoff_s * (2 ** attempt)
            logger.warning(
                "Redis transient error (attempt %d/%d): %s — retrying in %.2f s",
                attempt + 1, max_attempts, exc, sleep_s,
            )
            time.sleep(sleep_s)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RedisCheckpointStore — drop-in for JsonFileCheckpointStore
# ─────────────────────────────────────────────────────────────────────────────

class RedisCheckpointStore:
    """
    Redis-backed CheckpointStore.

    Protocol-compatible with JsonFileCheckpointStore so the only required
    change is in MatchStateManager's constructor — no orchestrator changes.

    Storage model
    ─────────────
    Each logical key maps to exactly ONE Redis key:
        "runtime_match_state"  →  "players_data:checkpoint:runtime_match_state"

    Payload is JSON-serialised and stored as a string value.  msgpack would
    give 3–5× better throughput (see plan item #3), but JSON is kept here to
    preserve compatibility with JsonFileCheckpointStore.load() for migration.

    TTL
    ───
    Every write sets EX = ttl_s (default 6 hours) so orphaned keys self-expire
    after an unclean shutdown.  This replaces the MAX_CHECKPOINT_AGE_HOURS
    guard that JsonFileCheckpointStore uses at read time.

    Atomicity
    ─────────
    SET key value EX ttl is atomic in Redis.  No partial write can occur.
    Cross-process writes are serialised by Redis' single-threaded command
    execution, so two orchestrator workers cannot produce split-brain state.

    Thread safety
    ─────────────
    Redis connections are thread-safe via the connection pool.  The internal
    RLock guards the in-process namespace prefix construction only.
    """

    _KEY_PREFIX = "players_data:checkpoint"

    def __init__(
        self,
        ttl_hours: Optional[float] = None,
        max_attempts: int           = 3,
        backoff_s: float            = 0.5,
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise RuntimeError(
                "redis-py is not installed. "
                "Install with: pip install redis[hiredis]"
            )

        _default_ttl = _env_float("REDIS_CHECKPOINT_TTL_HOURS", 6.0)
        self._ttl_s       = int((ttl_hours or _default_ttl) * 3600)
        self._max_attempts = max_attempts
        self._backoff_s    = backoff_s
        self._lock         = RLock()

        # Eagerly verify connectivity so misconfiguration is caught at startup.
        client = RedisConnectionPool.client()
        client.ping()
        logger.info(
            "RedisCheckpointStore: connected | ttl=%ds max_attempts=%d",
            self._ttl_s, self._max_attempts,
        )

    @classmethod
    def from_env(cls) -> "RedisCheckpointStore":
        """
        Factory method — reads all settings from environment variables.
        Preferred entry point in production.
        """
        return cls(
            ttl_hours    = _env_float("REDIS_CHECKPOINT_TTL_HOURS", 6.0),
            max_attempts = _env_int("REDIS_RETRY_MAX_ATTEMPTS", 3),
            backoff_s    = _env_float("REDIS_RETRY_BACKOFF_S", 0.5),
        )

    def _redis_key(self, key: str) -> str:
        return f"{self._KEY_PREFIX}:{key}"

    def save(self, key: str, payload: dict) -> None:
        """
        Atomically persist payload under key with TTL.

        JSON serialisation happens outside the Redis round-trip so the
        connection is held for the minimum possible duration.
        """
        rkey = self._redis_key(key)
        try:
            serialised = json.dumps(payload, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            logger.exception("RedisCheckpointStore.save: serialisation failed key=%r: %s", key, exc)
            return

        def _do_save():
            client = RedisConnectionPool.client()
            client.set(rkey, serialised, ex=self._ttl_s)

        try:
            _with_retry(_do_save, self._max_attempts, self._backoff_s)
            logger.debug(
                "RedisCheckpointStore.save: key=%r %d bytes ttl=%ds",
                rkey, len(serialised), self._ttl_s,
            )
        except RedisError:
            logger.exception("RedisCheckpointStore.save: failed for key=%r", key)

    def load(self, key: str) -> Optional[dict]:
        """
        Return the payload for key, or None if absent / unreadable.
        """
        rkey = self._redis_key(key)

        def _do_load():
            client = RedisConnectionPool.client()
            return client.get(rkey)

        try:
            raw = _with_retry(_do_load, self._max_attempts, self._backoff_s)
        except RedisError:
            logger.exception("RedisCheckpointStore.load: failed for key=%r", key)
            return None

        if raw is None:
            logger.debug("RedisCheckpointStore.load: key=%r not found", rkey)
            return None

        try:
            return json.loads(raw)
        except (ValueError, TypeError) as exc:
            logger.error(
                "RedisCheckpointStore.load: corrupt JSON for key=%r (%s) — discarding",
                rkey, exc,
            )
            self.delete(key)
            return None

    def delete(self, key: str) -> None:
        """
        Delete the stored payload.  Silent if key does not exist.
        """
        rkey = self._redis_key(key)

        def _do_delete():
            client = RedisConnectionPool.client()
            client.delete(rkey)

        try:
            _with_retry(_do_delete, self._max_attempts, self._backoff_s)
            logger.debug("RedisCheckpointStore.delete: key=%r", rkey)
        except RedisError:
            logger.exception("RedisCheckpointStore.delete: failed for key=%r", key)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Episodic memory store
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeStore:
    """
    Persistent episodic memory for cross-match retrieval.

    Key schema
    ----------
    Episode payload  :  players_data:episode:{player_id}:{episode_id}   (JSON, TTL)
    Retrieval index  :  players_data:player:{player_id}:episodes          (sorted set, score = match_minute)

    The sorted set lets us retrieve the last N episodes, or episodes by
    recency, without scanning full MatchState checkpoints.

    Only CLOSED episodes are persisted.  Ongoing episodes live in MatchState
    and are persisted when they close, when escalation changes, or when an
    intervention outcome is finalised.

    TTL
    ---
    Individual episode payloads expire after EPISODE_TTL_HOURS (default 72 h).
    The sorted set index members auto-expire via a background trim to keep it
    under MAX_INDEX_SIZE entries per player.
    """

    _PAYLOAD_PREFIX = "players_data:episode"
    _INDEX_PREFIX   = "players_data:player"
    _META_PREFIX    = "players_data:episode_meta" 
    _DEFAULT_TTL_H  = 72.0
    _MAX_INDEX_SIZE = 200  # trim older entries beyond this per player

    def __init__(
        self,
        ttl_hours: Optional[float] = None,
        max_attempts: int = 3,
        backoff_s: float = 0.05,
    ) -> None:
        _h = ttl_hours or _env_float("REDIS_EPISODE_TTL_HOURS", self._DEFAULT_TTL_H)
        self._ttl_s       = int(_h * 3600)
        self._max_attempts = max_attempts
        self._backoff_s    = backoff_s

    # ── internal key helpers ──────────────────────────────────────────────────

    def _payload_key(self, player_id: int, episode_id: str) -> str:
        return f"{self._PAYLOAD_PREFIX}:{player_id}:{episode_id}"

    def _index_key(self, player_id: int) -> str:
        return f"{self._INDEX_PREFIX}:{player_id}:episodes"

    # ── write path ───────────────────────────────────────────────────────────

    def persist_episode(
        self,
        player_id: int,
        episode_id: str,
        episode_dict: dict,
        score: float,          # match_minute or unix timestamp for ordering
    ) -> None:
        """
        Atomically store one closed episode and update the retrieval index.

        episode_dict  : PlayerEpisode.to_dict() output (already serialisable).
        score         : sort key for the sorted set (use end_minute if available,
                        else match elapsed seconds, else time.time()).
        """
        payload_key = self._payload_key(player_id, episode_id)
        index_key   = self._index_key(player_id)

        try:
            serialised = json.dumps(episode_dict, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            logger.exception("EpisodeStore.persist_episode: serialisation failed player=%d ep=%s: %s",
                             player_id, episode_id, exc)
            return

        def _do_write():
            client = RedisConnectionPool.client()
            pipe = client.pipeline()
            pipe.set(payload_key, serialised, ex=self._ttl_s)
            pipe.zadd(index_key, {episode_id: score})
            # Trim the index to the most recent MAX_INDEX_SIZE entries
            pipe.zremrangebyrank(index_key, 0, -(self._MAX_INDEX_SIZE + 1))
            pipe.execute()

        try:
            _with_retry(_do_write, self._max_attempts, self._backoff_s)
            logger.debug("EpisodeStore.persist_episode: player=%d ep=%s score=%.1f",
                         player_id, episode_id, score)
        except RedisError:
            logger.exception("EpisodeStore.persist_episode: failed player=%d ep=%s",
                             player_id, episode_id)

    # ── read path ────────────────────────────────────────────────────────────

    def get_recent_episode_ids(
        self,
        player_id: int,
        last_n: int = 20,
    ) -> list[str]:
        """
        Return up to last_n episode_ids ordered newest-first from the index.
        Fast — single ZREVRANGE call.
        """
        index_key = self._index_key(player_id)

        def _do_read():
            client = RedisConnectionPool.client()
            return client.zrevrange(index_key, 0, last_n - 1)

        try:
            raw = _with_retry(_do_read, self._max_attempts, self._backoff_s)
            return [r.decode() if isinstance(r, bytes) else r for r in (raw or [])]
        except RedisError:
            logger.exception("EpisodeStore.get_recent_episode_ids: failed player=%d", player_id)
            return []

    def load_episodes(
        self,
        player_id: int,
        episode_ids: list[str],
    ) -> list[dict]:
        """
        Batch-load episode payloads by id.  Missing / expired keys are silently
        skipped (TTL expiry is expected).  Order matches episode_ids.
        """
        if not episode_ids:
            return []

        keys = [self._payload_key(player_id, eid) for eid in episode_ids]

        def _do_mget():
            client = RedisConnectionPool.client()
            return client.mget(keys)

        try:
            raws = _with_retry(_do_mget, self._max_attempts, self._backoff_s) or []
        except RedisError:
            logger.exception("EpisodeStore.load_episodes: mget failed player=%d", player_id)
            return []

        results = []
        for raw in raws:
            if raw is None:
                continue
            try:
                results.append(json.loads(raw))
            except (ValueError, TypeError) as exc:
                logger.warning("EpisodeStore.load_episodes: corrupt JSON skipped: %s", exc)
        return results

    def retrieve_relevant(
        self,
        player_id: int,
        current_findings: list[str],
        current_trend: str = "stable",
        top_k: int = 3,
        candidate_pool: int = 20,
    ) -> list[dict]:
        """
        Retrieve the top-K historically relevant closed episodes for a player.

        Retrieval pattern (Issue 4 fix):
        ──────────────────────────────────
        Old pattern:  ZREVRANGE 0 19 → MGET all 20 → score → filter
                      → always hydrates 20 full payloads, most discarded

        New pattern:  ZREVRANGE with scores → score metadata only → MGET top-K
                      → hydrates only what will be returned

        We store a compact metadata entry alongside each payload that carries
        dominant_findings, severity, trend_direction, response, and end_minute.
        This lets us score without loading the full payload JSON.

        If metadata keys are absent (older episodes stored before this change),
        we fall back to the full MGET path transparently.

        Salience scoring (higher = more relevant):
            same_finding * 5   — dominant finding overlaps current findings
            same_trend   * 3   — trend direction matches
            unresolved   * 2   — response was 'persisted' or 'unknown'
            severity     * 2   — critical=2, high=1, medium=0.5
            recency            — newer episodes score higher (0→1 normalised)

        Parameters
        ----------
        current_findings : finding type strings from the current alert window.
        current_trend    : trend direction from the current semantic state.
        top_k            : number of episodes to return (hydrated).
        candidate_pool   : how many recent episode IDs to consider.
        """
        index_key    = self._index_key(player_id)
        target_types = set(current_findings)
        severity_map = {"critical": 2.0, "high": 1.0, "medium": 0.5, "low": 0.0}

        # ── Step 1: fetch IDs + scores from sorted set (single round-trip) ────
        def _do_ids():
            client = RedisConnectionPool.client()
            return client.zrevrange(index_key, 0, candidate_pool - 1, withscores=True)

        try:
            id_score_pairs = _with_retry(_do_ids, self._max_attempts, self._backoff_s) or []
        except RedisError:
            logger.exception("EpisodeStore.retrieve_relevant: index read failed player=%d", player_id)
            return []

        if not id_score_pairs:
            return []

        episode_ids = [
            r.decode() if isinstance(r, bytes) else r
            for r, _ in id_score_pairs
        ]
        n = len(episode_ids)

        # ── Step 2: try metadata-only scoring (avoids full payload hydration) ──
        meta_keys = [self._meta_key(player_id, eid) for eid in episode_ids]

        def _do_meta_mget():
            client = RedisConnectionPool.client()
            return client.mget(meta_keys)

        meta_available = False
        try:
            meta_raws = _with_retry(_do_meta_mget, self._max_attempts, self._backoff_s) or []
            meta_available = any(r is not None for r in meta_raws)
        except RedisError:
            meta_raws = [None] * n

        scored: list[tuple[float, int]] = []   # (score, list_index)

        if meta_available:
            # Score using lightweight metadata only
            for i, raw in enumerate(meta_raws):
                if raw is None:
                    # Metadata absent — include in hydration pass below
                    scored.append((0.0, i))
                    continue
                try:
                    meta = json.loads(raw)
                except (ValueError, TypeError):
                    scored.append((0.0, i))
                    continue

                recency       = (i + 1) / n   # newer → higher index in reversed list
                ep_findings   = set(meta.get("dominant_findings", []))
                same_finding  = 5.0 if ep_findings & target_types else 0.0
                same_trend    = 3.0 if meta.get("trend_direction") == current_trend else 0.0
                unresolved    = 2.0 if meta.get("response") in ("persisted", "unknown") else 0.0
                severity_sc   = severity_map.get(meta.get("severity", "low"), 0.0) * 2.0
                score = same_finding + same_trend + unresolved + severity_sc + recency
                scored.append((score, i))
        else:
            # No metadata — fall back to scoring by recency only before hydration
            scored = [(float(n - i) / n, i) for i in range(n)]

        # ── Step 3: select top-K indices, THEN hydrate only those payloads ────
        scored.sort(key=lambda x: -x[0])
        top_indices = [idx for _, idx in scored[:top_k]]
        top_ids     = [episode_ids[i] for i in top_indices]

        if not top_ids:
            return []

        episodes = self.load_episodes(player_id, top_ids)

        # ── Step 4: re-score with full payloads if we had to skip metadata ────
        if not meta_available and episodes:
            full_scored: list[tuple[float, dict]] = []
            for i, ep in enumerate(episodes):
                recency      = (len(episodes) - i) / max(len(episodes), 1)
                ep_findings  = set(ep.get("dominant_findings", []))
                same_finding = 5.0 if ep_findings & target_types else 0.0
                same_trend   = 3.0 if ep.get("trend_direction") == current_trend else 0.0
                unresolved   = 2.0 if ep.get("response") in ("persisted", "unknown") else 0.0
                severity_sc  = severity_map.get(ep.get("severity", "low"), 0.0) * 2.0
                score = same_finding + same_trend + unresolved + severity_sc + recency
                full_scored.append((score, ep))
            full_scored.sort(key=lambda x: -x[0])
            episodes = [ep for _, ep in full_scored[:top_k]]

        logger.debug(
            "EpisodeStore.retrieve_relevant: player=%d candidates=%d hydrated=%d returned=%d",
            player_id, n, len(top_ids), len(episodes),
        )


        hit = len(episodes)
        hit_rate = round(hit / max(candidate_pool, 1), 3)
        logger.info(
            "EpisodeStore.retrieve_relevant: player=%d candidates=%d hydrated=%d "
            "returned=%d hit_rate=%.3f meta_path=%s",
            player_id, n, len(top_ids), hit, hit_rate,
            "metadata" if meta_available else "full_payload_fallback",
        )
        return episodes

    def _meta_key(self, player_id: int, episode_id: str) -> str:
        """Lightweight metadata key — carries only scoring fields, not full payload."""
        return f"{self._META_PREFIX}:{player_id}:{episode_id}"

    def persist_episode_with_meta(
        self,
        player_id: int,
        episode_id: str,
        episode_dict: dict,
        score: float,
    ) -> None:
        """
        Store episode payload AND a compact metadata entry for efficient scoring.

        Call this instead of persist_episode() for new episodes — the metadata
        entry enables retrieve_relevant() to score without loading full payloads.

        meta carries: dominant_findings, severity, trend_direction, response
        """
        meta = {
            "dominant_findings": episode_dict.get("dominant_findings", []),
            "severity":          episode_dict.get("severity", "low"),
            "trend_direction":   episode_dict.get("trend_direction", "stable"),
            "response":          episode_dict.get("response", "unknown"),
            # "active_pattern":    episode_dict.get("active_pattern"),
            # "active_outcome":    episode_dict.get("active_outcome"),
        }

        payload_key = self._payload_key(player_id, episode_id)
        meta_key    = self._meta_key(player_id, episode_id)
        index_key   = self._index_key(player_id)

        try:
            serialised_payload = json.dumps(episode_dict, separators=(",", ":"))
            serialised_meta    = json.dumps(meta,         separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            logger.exception("EpisodeStore.persist_episode_with_meta: serialisation failed: %s", exc)
            return

        def _do_write():
            client = RedisConnectionPool.client()
            pipe = client.pipeline()
            pipe.set(payload_key, serialised_payload, ex=self._ttl_s)
            pipe.set(meta_key,    serialised_meta,    ex=self._ttl_s)
            pipe.zadd(index_key, {episode_id: score})
            pipe.zremrangebyrank(index_key, 0, -(self._MAX_INDEX_SIZE + 1))
            pipe.execute()

        try:
            _with_retry(_do_write, self._max_attempts, self._backoff_s)
            logger.debug("EpisodeStore.persist_episode_with_meta: player=%d ep=%s", player_id, episode_id)
        except RedisError:
            logger.exception("EpisodeStore.persist_episode_with_meta: failed player=%d ep=%s",
                             player_id, episode_id)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pub/Sub — channel schema + clients
# ─────────────────────────────────────────────────────────────────────────────

class PubSubConfig:
    """
    Centralised channel-name definitions for all pub/sub use cases.

    Channels are namespaced by match_id so independent matches can run in the
    same Redis instance without cross-contamination.  All names follow the
    pattern:   players_data:<match_id>:<topic>

    Topics
    ──────
    alerts          Player alert events (WARNING / CRITICAL anomaly results)
    state_updates   Match state checkpoint notifications (for cache invalidation
                    across workers — a worker receiving this should re-load state
                    from the checkpoint store rather than using its in-memory copy)
    coach_events    Coach feedback / override events fan-out
    system          Internal orchestrator lifecycle signals (match start / end)
    """

    _NAMESPACE = "players_data"

    @classmethod
    def alerts_channel(cls, match_id: str) -> str:
        """Channel for alert fan-out to dashboards and coaching tablets."""
        return f"{cls._NAMESPACE}:{match_id}:alerts"

    @classmethod
    def state_updates_channel(cls, match_id: str) -> str:
        """Channel for cache invalidation signals between orchestrator workers."""
        return f"{cls._NAMESPACE}:{match_id}:state_updates"

    @classmethod
    def coach_events_channel(cls, match_id: str) -> str:
        """Channel for coach override / feedback events."""
        return f"{cls._NAMESPACE}:{match_id}:coach_events"

    @classmethod
    def system_channel(cls, match_id: str) -> str:
        """Internal lifecycle signals: match_start, match_end, recalibrate."""
        return f"{cls._NAMESPACE}:{match_id}:system"

    @classmethod
    def all_channels_for_match(cls, match_id: str) -> list[str]:
        """Return all channels for a given match (useful for bulk subscribe)."""
        return [
            cls.alerts_channel(match_id),
            cls.state_updates_channel(match_id),
            cls.coach_events_channel(match_id),
            cls.system_channel(match_id),
        ]


class RedisPubSubClient:
    """
    Unified publisher + subscriber for the Players Data pub/sub layer.

    Each instance holds ONE dedicated Redis connection for pub/sub — the
    Redis protocol requires a separate connection for subscriptions because
    a subscribed connection can only process pub/sub commands.

    Publisher
    ─────────
    The publisher uses the shared connection pool (regular commands, no
    dedicated connection needed).  Call publish_alert(), publish_state_update(),
    etc. directly from the orchestrator or alert callback.

    Subscriber
    ──────────
    The subscriber owns a dedicated connection that is kept in subscription
    mode.  Call subscribe_*() then listen() in a worker thread (or use
    listen_in_background() which spawns the thread automatically).

    Message format
    ──────────────
    All messages are JSON objects.  Every message includes:
        {"type": "<topic>", "match_id": "<id>", "ts": <unix_epoch_float>, ...}
    Topic-specific fields are documented per publish method.

    Example (alert fan-out)
    ───────────────────────
        # Orchestrator (publishes on every alert)
        pub = RedisPubSubClient.publisher()
        pub.publish_alert(match_id="m1", payload={
            "player_id": 7,
            "alert_level": "WARNING",
            "nlg_summary": "...",
        })

        # Dashboard process (subscribes, calls handler per message)
        sub = RedisPubSubClient.subscriber()
        sub.subscribe_alerts("m1", callback=lambda msg: print(msg))
        sub.listen_in_background()   # non-blocking; runs in daemon thread
    """

    def __init__(self) -> None:
        if not _REDIS_AVAILABLE:
            raise RuntimeError(
                "redis-py is not installed. "
                "Install with: pip install redis[hiredis]"
            )
        # Dedicated connection for subscriptions (not from the shared pool)
        self._pubsub: Optional["redis.client.PubSub"] = None
        self._listener_thread: Optional[Thread] = None
        self._lock = RLock()

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def publisher(cls) -> "RedisPubSubClient":
        """Return a client configured for publishing only."""
        return cls()

    @classmethod
    def subscriber(cls) -> "RedisPubSubClient":
        """Return a client configured for subscribing."""
        instance = cls()
        # Initialise a dedicated pub/sub connection immediately
        client = RedisConnectionPool.client()
        instance._pubsub = client.pubsub(ignore_subscribe_messages=True)
        return instance

    # ── Publisher methods ─────────────────────────────────────────────────────

    def _publish(self, channel: str, payload: dict) -> int:
        """
        Publish a JSON message to channel.

        Returns the number of subscribers that received the message.
        A return value of 0 is not an error — it means no subscribers are
        currently connected (fire-and-forget semantics).
        """
        payload.setdefault("ts", time.time())
        try:
            data = json.dumps(payload, separators=(",", ":"))
            client = RedisConnectionPool.client()
            receivers = client.publish(channel, data)
            logger.debug(
                "RedisPubSubClient.publish: channel=%s receivers=%d",
                channel, receivers,
            )
            return receivers
        except RedisError as exc:
            logger.error("RedisPubSubClient.publish failed: %s", exc)
            return 0

    def publish_alert(self, match_id: str, payload: dict) -> int:
        """
        Publish a player alert event.

        payload should include:
            player_id     : int
            player_name   : str
            alert_level   : "WARNING" | "CRITICAL"
            recommendation_type : str
            confidence    : float
            anomaly_score : float
            nlg_summary   : str  (may be empty if NLG is still pending)
            elapsed_s     : int  (match clock at time of alert)

        Returns the number of subscribers that received the message.
        """
        channel = PubSubConfig.alerts_channel(match_id)
        return self._publish(channel, {"type": "alert", "match_id": match_id, **payload})

    def publish_state_update(self, match_id: str, player_id: int) -> int:
        """
        Notify other workers that a checkpoint has been written for player_id.

        Workers receiving this should invalidate their in-memory MatchState
        for this player and reload from the checkpoint store.
        """
        channel = PubSubConfig.state_updates_channel(match_id)
        return self._publish(channel, {
            "type":      "state_update",
            "match_id":  match_id,
            "player_id": player_id,
        })

    def publish_coach_event(self, match_id: str, payload: dict) -> int:
        """
        Publish a coach override or feedback event.

        payload should include:
            player_id  : int
            coach_id   : str
            decision   : str
            inference_id : int
        """
        channel = PubSubConfig.coach_events_channel(match_id)
        return self._publish(channel, {"type": "coach_event", "match_id": match_id, **payload})

    def publish_system(self, match_id: str, event: str, **extra) -> int:
        """
        Publish a lifecycle signal.

        event: "match_start" | "match_end" | "recalibrate"
        """
        channel = PubSubConfig.system_channel(match_id)
        return self._publish(channel, {
            "type":     "system",
            "match_id": match_id,
            "event":    event,
            **extra,
        })

    # ── Subscriber methods ────────────────────────────────────────────────────

    def _ensure_pubsub(self) -> "redis.client.PubSub":
        with self._lock:
            if self._pubsub is None:
                client = RedisConnectionPool.client()
                self._pubsub = client.pubsub(ignore_subscribe_messages=True)
            return self._pubsub

    def subscribe_alerts(self, match_id: str, callback: Callable[[dict], None]) -> None:
        """
        Subscribe to the alerts channel for match_id.
        callback receives the parsed JSON dict for each message.
        """
        channel = PubSubConfig.alerts_channel(match_id)
        self._subscribe(channel, callback)

    def subscribe_state_updates(self, match_id: str, callback: Callable[[dict], None]) -> None:
        """Subscribe to cross-worker state invalidation signals."""
        channel = PubSubConfig.state_updates_channel(match_id)
        self._subscribe(channel, callback)

    def subscribe_coach_events(self, match_id: str, callback: Callable[[dict], None]) -> None:
        """Subscribe to coach override events."""
        channel = PubSubConfig.coach_events_channel(match_id)
        self._subscribe(channel, callback)

    def subscribe_system(self, match_id: str, callback: Callable[[dict], None]) -> None:
        """Subscribe to lifecycle signals."""
        channel = PubSubConfig.system_channel(match_id)
        self._subscribe(channel, callback)

    def subscribe_all(self, match_id: str, callback: Callable[[dict], None]) -> None:
        """Subscribe to every channel for match_id in one call."""
        for channel in PubSubConfig.all_channels_for_match(match_id):
            self._subscribe(channel, callback)

    def _subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        ps = self._ensure_pubsub()

        def _handler(raw_message: dict) -> None:
            data_str = raw_message.get("data", "")
            if not isinstance(data_str, str):
                return
            try:
                payload = json.loads(data_str)
            except ValueError as exc:
                logger.warning(
                    "RedisPubSubClient: could not parse message on %s: %s",
                    channel, exc,
                )
                return
            try:
                callback(payload)
            except Exception:
                logger.exception(
                    "RedisPubSubClient: callback error on channel %s", channel
                )

        ps.subscribe(**{channel: _handler})
        logger.info("RedisPubSubClient: subscribed to channel=%s", channel)

    def unsubscribe_all(self) -> None:
        """Remove all subscriptions and free the dedicated connection."""
        with self._lock:
            if self._pubsub is not None:
                try:
                    self._pubsub.unsubscribe()
                    self._pubsub.close()
                except RedisError:
                    pass
                self._pubsub = None

    def listen(self) -> None:
        """
        Blocking message loop.

        Runs until the process exits or unsubscribe_all() is called from
        another thread.  Each message dispatches to the registered callback.

        Call this in a dedicated thread — it blocks indefinitely.
        """
        ps = self._ensure_pubsub()
        logger.info("RedisPubSubClient: entering listen loop")
        try:
            for _msg in ps.listen():
                pass   # dispatch is handled by per-channel handlers registered in _subscribe
        except RedisError as exc:
            logger.error("RedisPubSubClient.listen: connection lost — %s", exc)
        finally:
            logger.info("RedisPubSubClient: listen loop exited")

    def listen_in_background(self, sleep_time: float = 0.01) -> Thread:
        """
        Spawn a daemon thread that runs the message loop.

        Returns the thread so callers can join() if needed (e.g. during tests).
        The thread exits automatically when the process exits.

        sleep_time: seconds the redis-py internal loop sleeps between polls
                    (passed to redis-py's run_in_thread helper).
        """
        ps = self._ensure_pubsub()

        with self._lock:
            if self._listener_thread is not None and self._listener_thread.is_alive():
                logger.warning(
                    "RedisPubSubClient.listen_in_background: listener already running"
                )
                return self._listener_thread

            self._listener_thread = ps.run_in_thread(
                sleep_time=sleep_time,
                daemon=True,
            )
            logger.info(
                "RedisPubSubClient: background listener started (thread=%s)",
                self._listener_thread.name,
            )
            return self._listener_thread


# ─────────────────────────────────────────────────────────────────────────────
# 5. Redis Streams — production communication backbone (Backend <-> PlayerDynamics)
# ─────────────────────────────────────────────────────────────────────────────

class StreamTopics:
    """
    Fixed Redis Stream names — see REDIS_STREAM_CONTRACTS.md and
    BACKEND_INTEGRATION_IMPLEMENTATION.md for the ownership rules these
    follow.

    Ownership (strict — see BACKEND_INTEGRATION_IMPLEMENTATION.md §1):
      Backend is the source of truth for match actions and match metadata
      (shot/goal/save/turnover/timeout/substitution/card/coach annotation,
      score, clock, period state) and publishes them:
        match.events    -- discrete coach/match actions
        match.context   -- running match state (score, clock, period)
      PlayerDynamics is the source of truth for analytics and owns Kinexon
      ingestion directly (Kinexon never flows through Backend) -- it
      consumes match.events/match.context, combines them with its own
      Kinexon-derived TacticalEvent stream, and publishes:
        analytics.possessions, analytics.teamstate, analytics.trends,
        analytics.insights, analytics.situations, analytics.players

    tracking.events is NOT Backend-facing: it exists only as a possible
    PlayerDynamics-internal hand-off (a Kinexon-ingestion worker process to
    a MatchOrchestrator worker process), should those ever be split into
    separate processes. Nothing outside PlayerDynamics should ever publish
    or consume it.

    Not namespaced by match_id (unlike PubSubConfig's channels) — match_id
    travels inside each entry's fields (see ingestion/stream_codec.py),
    consistent with REDIS_STREAM_CONTRACTS.md's wire format.
    """
    MATCH_EVENTS           = "match.events"
    MATCH_CONTEXT           = "match.context"
    TRACKING_EVENTS        = "tracking.events"  # PlayerDynamics-internal only, see docstring
    ANALYTICS_POSSESSIONS  = "analytics.possessions"
    ANALYTICS_TEAMSTATE    = "analytics.teamstate"
    ANALYTICS_TRENDS       = "analytics.trends"
    ANALYTICS_INSIGHTS     = "analytics.insights"
    ANALYTICS_SITUATIONS  = "analytics.situations"
    ANALYTICS_PLAYERS      = "analytics.players"
    # Coach-facing observable workload metrics (currentLoad, loadTrend, ...).
    # Deliberately a SEPARATE topic from ANALYTICS_PLAYERS: that stream
    # carries the PlayerDynamics pilot model's output (reconstruction_loss,
    # confidence, SHAP) for the "PlayerDynamics Pilot" page. This one carries
    # zero model internals -- pure Kinexon positions.csv/events.csv
    # aggregates -- for the "Player Analytics" coach dashboard. Publishing
    # both under the same name would silently merge two incompatible
    # payload shapes onto one consumer.
    ANALYTICS_PLAYER_WORKLOAD = "analytics.player_workload"

    # Backend <-> PlayerDynamics boundary streams (excludes tracking.events,
    # which never crosses that boundary).
    FROM_BACKEND = (MATCH_EVENTS, MATCH_CONTEXT)
    TO_BACKEND   = (
        ANALYTICS_POSSESSIONS, ANALYTICS_TEAMSTATE, ANALYTICS_TRENDS,
        ANALYTICS_INSIGHTS, ANALYTICS_SITUATIONS, ANALYTICS_PLAYERS,
        ANALYTICS_PLAYER_WORKLOAD,
    )

    INBOUND  = (MATCH_EVENTS, MATCH_CONTEXT, TRACKING_EVENTS)
    OUTBOUND = TO_BACKEND
    ALL      = INBOUND + OUTBOUND


def _flatten_xreadgroup(result: Optional[list]) -> "list[tuple[str, Dict[str, str]]]":
    """
    redis-py's xreadgroup() returns [[stream_name, [(entry_id, fields), ...]]]
    (one inner list per stream requested). This client always reads one
    stream at a time, so flatten to a plain [(entry_id, fields), ...] list.
    """
    if not result:
        return []
    entries: "list[tuple[str, Dict[str, str]]]" = []
    for _stream_name, stream_entries in result:
        entries.extend(stream_entries)
    return entries


class RedisStreamProducer:
    """
    XADD wrapper — the Backend/PlayerDynamics outbound publishing side.

    Reuses RedisConnectionPool (shared pool) and the same retry-with-backoff
    helper already used by RedisCheckpointStore, so transient connection
    drops are retried transparently ("reconnect" requirement) rather than
    surfacing as a publish failure on the first hiccup.
    """

    def __init__(self, max_attempts: int = 3, backoff_s: float = 0.5) -> None:
        if not _REDIS_AVAILABLE:
            raise RuntimeError(
                "redis-py is not installed. Install with: pip install redis[hiredis]"
            )
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s

    def ensure_stream(self, stream: str) -> None:
        """
        Create `stream` if it does not exist yet ("stream creation"
        requirement), without requiring a real consumer group to exist.

        XGROUP CREATE ... MKSTREAM is the only Redis primitive that creates
        an empty stream outright (XADD also creates it, but only as a side
        effect of writing real data). A throwaway group name is used purely
        to force creation; BUSYGROUP (group/stream already exists) is
        swallowed since this call is meant to be idempotent.
        """
        def _do() -> None:
            client = RedisConnectionPool.client()
            try:
                client.xgroup_create(stream, "_bootstrap", id="0", mkstream=True)
            except RedisError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise

        try:
            _with_retry(_do, self._max_attempts, self._backoff_s)
            logger.debug("RedisStreamProducer.ensure_stream: stream=%s ready", stream)
        except RedisError:
            logger.exception("RedisStreamProducer.ensure_stream: failed for stream=%s", stream)
            raise

    def publish(self, stream: str, fields: Dict[str, str]) -> Optional[str]:
        """
        XADD fields onto stream. Returns the new entry ID, or None if the
        publish ultimately failed after retries (caller decides whether
        that is fatal — fire-and-forget callers may choose to log and continue).

        `fields` must be a flat str -> str mapping — see
        ingestion/stream_codec.encode() for the canonical envelope
        ({"schema_version", "type", "match_id", "payload"}) used by
        publish_dataclass() below.
        """
        def _do() -> str:
            client = RedisConnectionPool.client()
            return client.xadd(stream, fields)

        try:
            entry_id = _with_retry(_do, self._max_attempts, self._backoff_s)
            logger.debug("RedisStreamProducer.publish: stream=%s id=%s", stream, entry_id)
            return entry_id
        except RedisError:
            logger.exception("RedisStreamProducer.publish: failed for stream=%s", stream)
            return None

    def publish_dataclass(self, stream: str, obj: Any) -> Optional[str]:
        """Encode obj via ingestion.stream_codec.encode() and XADD it onto stream."""
        from ingestion.stream_codec import encode
        return self.publish(stream, encode(obj))


class RedisStreamConsumer:
    """
    XREADGROUP + XACK wrapper with consumer-group semantics, crash-safe
    replay, and an extra duplicate-protection layer on top of Streams' own
    delivery guarantees.

    consumer_name MUST be stable across process restarts (e.g. derived from
    a fixed worker id, not a random UUID per process) — Redis tracks each
    consumer's Pending Entries List (PEL) by this name, and read_pending()
    below depends on it to find work that was in flight when the previous
    process died.

    Reliability
    -----------
    reconnect              : every Redis call goes through the same
                              _with_retry exponential-backoff helper used by
                              RedisCheckpointStore.
    consumer group creation : _ensure_group() is called in __init__ and is
                              idempotent (BUSYGROUP swallowed), satisfying
                              "consumer group creation" + "stream creation"
                              even when this is the very first consumer ever
                              attached to a brand-new stream (mkstream=True).
    duplicate protection    : is_duplicate() does an atomic SETNX-with-TTL
                              check BEFORE a caller processes an entry. This
                              is an EXTRA layer beyond Streams' own at-most-
                              once-per-group delivery via XACK — it also
                              catches a misbehaving producer re-publishing
                              the same logical event as a new stream entry
                              (different entry_id, e.g. after a retry that
                              actually succeeded), which XACK alone cannot
                              detect since XACK only dedupes redelivery of
                              the SAME entry_id.
    replay after restart   : read_pending() (XREADGROUP ... '0') returns
                              every entry already delivered to this exact
                              consumer_name that was never XACKed — exactly
                              what was "in flight" before a crash/restart.
                              Call this once at startup, before read_new(),
                              to drain it.
    """

    _DEDUP_NAMESPACE = "players_data:stream:seen"

    def __init__(
        self,
        stream: str,
        group: str,
        consumer_name: str,
        max_attempts: int = 3,
        backoff_s: float = 0.5,
        block_ms: int = 5000,
        dedup_ttl_hours: float = 24.0,
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise RuntimeError(
                "redis-py is not installed. Install with: pip install redis[hiredis]"
            )
        self.stream = stream
        self.group = group
        self.consumer_name = consumer_name
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self._block_ms = block_ms
        self._dedup_ttl_s = int(dedup_ttl_hours * 3600)
        self._ensure_group()

    def _ensure_group(self) -> None:
        def _do() -> None:
            client = RedisConnectionPool.client()
            try:
                client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
            except RedisError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise

        _with_retry(_do, self._max_attempts, self._backoff_s)
        logger.info(
            "RedisStreamConsumer: group ready | stream=%s group=%s consumer=%s",
            self.stream, self.group, self.consumer_name,
        )

    def read_new(self, count: int = 10) -> "list[tuple[str, Dict[str, str]]]":
        """
        XREADGROUP ... '>' — only entries never delivered to this group
        before. Returns [(entry_id, fields), ...], oldest first.
        """
        def _do():
            client = RedisConnectionPool.client()
            return client.xreadgroup(
                self.group, self.consumer_name, {self.stream: ">"},
                count=count, block=self._block_ms,
            )

        try:
            result = _with_retry(_do, self._max_attempts, self._backoff_s)
            return _flatten_xreadgroup(result)
        except RedisError:
            logger.exception("RedisStreamConsumer.read_new: failed stream=%s group=%s",
                              self.stream, self.group)
            return []

    def read_pending(self, count: int = 10) -> "list[tuple[str, Dict[str, str]]]":
        """
        XREADGROUP ... '0' — entries already delivered to THIS consumer_name
        but never XACKed (this consumer's own Pending Entries List). Call
        this once at startup, before read_new(), to replay work that was in
        flight when the previous process instance died — see class
        docstring's "replay after restart" note.
        """
        def _do():
            client = RedisConnectionPool.client()
            return client.xreadgroup(
                self.group, self.consumer_name, {self.stream: "0"}, count=count,
            )

        try:
            result = _with_retry(_do, self._max_attempts, self._backoff_s)
            return _flatten_xreadgroup(result)
        except RedisError:
            logger.exception("RedisStreamConsumer.read_pending: failed stream=%s group=%s",
                              self.stream, self.group)
            return []

    def ack(self, entry_id: str) -> None:
        """XACK entry_id — call only after the entry has been fully processed."""
        def _do() -> None:
            client = RedisConnectionPool.client()
            client.xack(self.stream, self.group, entry_id)

        try:
            _with_retry(_do, self._max_attempts, self._backoff_s)
            logger.debug("RedisStreamConsumer.ack: stream=%s id=%s", self.stream, entry_id)
        except RedisError:
            logger.exception("RedisStreamConsumer.ack: failed stream=%s id=%s",
                              self.stream, entry_id)

    def is_duplicate(self, entry_id: str) -> bool:
        """
        Atomic check-and-mark: returns True if entry_id has already been
        seen by this (stream, group) within the dedup TTL window, False the
        first time (and immediately marks it seen for next time). Call this
        BEFORE processing an entry's payload; if it returns True, ack and
        skip rather than re-applying the entry's effects.
        """
        key = f"{self._DEDUP_NAMESPACE}:{self.stream}:{self.group}:{entry_id}"

        def _do() -> bool:
            client = RedisConnectionPool.client()
            was_set = client.set(key, "1", nx=True, ex=self._dedup_ttl_s)
            return not bool(was_set)

        try:
            return _with_retry(_do, self._max_attempts, self._backoff_s)
        except RedisError:
            logger.exception("RedisStreamConsumer.is_duplicate: failed for id=%s", entry_id)
            return False  # fail open -- never block processing on a dedup-check error

    def decode(self, fields: Dict[str, str]) -> Any:
        """Decode a raw stream entry's fields back into its original dataclass."""
        from ingestion.stream_codec import decode
        return decode(fields)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: verify connectivity at import time (non-fatal)
# ─────────────────────────────────────────────────────────────────────────────

def check_redis_connection() -> bool:
    """
    Ping Redis and return True if reachable.

    Call during application startup to surface misconfiguration early.
    Does NOT raise — returns False so callers can decide whether to abort
    or fall back to JsonFileCheckpointStore.
    """
    if not _REDIS_AVAILABLE:
        logger.info("check_redis_connection: redis-py not installed — skipping")
        return False
    try:
        client = RedisConnectionPool.client()
        return client.ping()
    except Exception as exc:
        logger.warning("check_redis_connection: Redis unreachable — %s", exc)
        return False