"""
Players Data — IBM CIC Germany
Data Ingestion Layer

Handles all real-time and batch data sources:
  1. GPS  — Serial NMEA / TCP NMEA stream / GPX files
  2. REST — SportRadar / Opta API polling
  3. WS   — Live match event WebSocket stream
  4. MQTT — Wearable sensor bridge (HR, accelerometer)

Each source produces normalized PlayerEvent records that feed
the pattern analysis engine via an asyncio.Queue.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable, Dict, List, Optional

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    import pynmea2
    PYNMEA2_AVAILABLE = True
except ImportError:
    pynmea2 = None
    PYNMEA2_AVAILABLE = False

from config.settings import CONFIG, GPSConfig, SportRadarConfig, LiveEventWSConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Canonical data record emitted by all sources
# ─────────────────────────────────────────────
@dataclass
class RawPlayerObservation:
    """
    Unified event record produced by every ingestion adapter.
    Downstream normalization converts this to a PlayerEvent ORM row.
    """
    source: str                             # "gps" | "api" | "ws" | "mqtt"
    player_external_id: str
    ts: datetime
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    speed_ms: Optional[float] = None
    acceleration_ms2: Optional[float] = None
    heart_rate_bpm: Optional[int] = None
    hr_recovery_time_s: Optional[float] = None
    event_type: Optional[str] = None        # sprint, walk, jog, etc.
    match_id: Optional[str] = None
    raw_payload: Optional[dict] = None      # full original record for audit

    def is_valid(self) -> bool:
        """Basic sanity checks before the record enters the pipeline."""
        if self.ts is None:
            return False
        if self.latitude is not None and not (-90 <= self.latitude <= 90):
            return False
        if self.longitude is not None and not (-180 <= self.longitude <= 180):
            return False
        if self.speed_ms is not None and self.speed_ms < 0:
            return False
        if self.heart_rate_bpm is not None and not (20 <= self.heart_rate_bpm <= 250):
            return False
        return True

    def quality_score(self) -> float:
        """Returns 0–1 data quality score based on field completeness and plausibility."""
        fields_present = sum([
            self.latitude is not None,
            self.longitude is not None,
            self.speed_ms is not None,
            self.heart_rate_bpm is not None,
        ])
        completeness = fields_present / 4.0
        validity = 1.0 if self.is_valid() else 0.0
        return round(completeness * validity, 3)


# ─────────────────────────────────────────────
# Pitch coordinate normalizer
# ─────────────────────────────────────────────
def gps_to_pitch_coords(
    lat: float, lon: float,
    pitch_origin_lat: float, pitch_origin_lon: float,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> tuple[float, float]:
    """
    Converts WGS-84 GPS coordinates to normalized pitch coords [0,100].
    Uses equirectangular approximation (valid for pitch-scale distances).
    """
    R = 6_371_000  # Earth radius in metres
    dlat = math.radians(lat - pitch_origin_lat)
    dlon = math.radians(lon - pitch_origin_lon)
    cos_lat = math.cos(math.radians(pitch_origin_lat))

    dy = R * dlat                          # North-South → pitch length axis
    dx = R * dlon * cos_lat                # East-West  → pitch width axis

    x_norm = max(0.0, min(100.0, (dx / pitch_width_m) * 100))
    y_norm = max(0.0, min(100.0, (dy / pitch_length_m) * 100))
    return round(x_norm, 2), round(y_norm, 2)


# ─────────────────────────────────────────────
# 1. GPS Ingestion Adapter
# ─────────────────────────────────────────────
class GPSIngestionAdapter:
    """
    Reads NMEA sentences from a serial port or TCP socket.
    Emits RawPlayerObservation for every valid GGA/RMC sentence.
    """

    def __init__(
        self,
        player_external_id: str,
        config: GPSConfig,
        queue: asyncio.Queue,
        pitch_origin: Optional[tuple[float, float]] = None,
    ):
        self.player_id = player_external_id
        self.cfg = config
        self.queue = queue
        self.pitch_origin = pitch_origin
        self._prev_lat: Optional[float] = None
        self._prev_lon: Optional[float] = None
        self._prev_ts: Optional[float] = None

    def _compute_speed(self, lat: float, lon: float, ts: float) -> Optional[float]:
        """Compute speed from consecutive GPS fixes (fallback if NMEA lacks speed)."""
        if self._prev_lat is None:
            return None
        R = 6_371_000
        dlat = math.radians(lat - self._prev_lat)
        dlon = math.radians(lon - self._prev_lon)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(self._prev_lat))
             * math.cos(math.radians(lat))
             * math.sin(dlon / 2) ** 2)
        dist_m = 2 * R * math.asin(math.sqrt(a))
        dt = ts - self._prev_ts
        return dist_m / dt if dt > 0 else None

    def _parse_nmea_sentence(self, raw: str, player_id: str) -> Optional[RawPlayerObservation]:
        """Parse a single NMEA sentence into a RawPlayerObservation."""
        try:
            msg = pynmea2.parse(raw.strip())
        except pynmea2.ParseError:
            return None

        if not hasattr(msg, "latitude") or msg.latitude == 0.0:
            return None

        lat = msg.latitude
        lon = msg.longitude
        now = time.time()

        speed_ms: Optional[float] = None
        if hasattr(msg, "spd_over_grnd") and msg.spd_over_grnd:
            # RMC speed is in knots
            speed_ms = float(msg.spd_over_grnd) * 0.514444
        else:
            speed_ms = self._compute_speed(lat, lon, now)

        obs = RawPlayerObservation(
            source="gps",
            player_external_id=player_id,
            ts=datetime.now(tz=timezone.utc),
            latitude=lat,
            longitude=lon,
            speed_ms=speed_ms,
            raw_payload={"sentence": raw},
        )

        self._prev_lat, self._prev_lon, self._prev_ts = lat, lon, now
        return obs

    async def stream_tcp(self) -> None:
        """Connect to a TCP NMEA server (e.g., gpsd) and stream observations."""
        host = self.cfg.tcp_host
        port = self.cfg.tcp_port
        logger.info("GPS: connecting to TCP %s:%s for player %s", host, port, self.player_id)

        while True:
            try:
                reader, _ = await asyncio.open_connection(host, port)
                logger.info("GPS: TCP connection established")
                async for line_bytes in reader:
                    line = line_bytes.decode("ascii", errors="replace")
                    obs = self._parse_nmea_sentence(line, self.player_id)
                    if obs and obs.is_valid():
                        await self.queue.put(obs)
            except (OSError, asyncio.IncompleteReadError) as exc:
                logger.warning("GPS TCP disconnected: %s — reconnecting in 2 s", exc)
                await asyncio.sleep(2)

    async def ingest_gpx_file(self, path: str) -> List[RawPlayerObservation]:
        """
        Parse a GPX file (used for post-match or training batch ingestion).
        Returns list of observations for bulk DB insert.
        """
        try:
            import gpxpy
        except ImportError:
            raise RuntimeError("gpxpy not installed — run: pip install gpxpy")

        observations: List[RawPlayerObservation] = []
        with open(path, "r") as f:
            gpx = gpxpy.parse(f)

        for track in gpx.tracks:
            for segment in track.segments:
                prev_point = None
                for point in segment.points:
                    speed_ms: Optional[float] = None
                    if point.speed is not None:
                        speed_ms = point.speed
                    elif prev_point is not None:
                        dist = point.distance_3d(prev_point) or point.distance_2d(prev_point)
                        dt = (point.time - prev_point.time).total_seconds()
                        speed_ms = dist / dt if dt > 0 else None

                    obs = RawPlayerObservation(
                        source="gps",
                        player_external_id=self.player_id,
                        ts=point.time.replace(tzinfo=timezone.utc)
                        if point.time.tzinfo is None else point.time,
                        latitude=point.latitude,
                        longitude=point.longitude,
                        speed_ms=speed_ms,
                    )
                    if obs.is_valid():
                        observations.append(obs)
                    prev_point = point

        logger.info("GPX: parsed %d observations for player %s", len(observations), self.player_id)
        return observations


# ─────────────────────────────────────────────
# 2. REST API Ingestion Adapter (SportRadar / Opta)
# ─────────────────────────────────────────────
class SportRadarAPIAdapter:
    """
    Polls the SportRadar Soccer API for match timelines and player stats.
    Converts API responses into RawPlayerObservation events.

    Endpoint used: /matches/{match_id}/timeline
    """

    def __init__(self, config: SportRadarConfig, queue: asyncio.Queue):
        self.cfg = config
        self.queue = queue
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_s)
            self._session = aiohttp.ClientSession(
                headers={"accept": "application/json"},
                timeout=timeout,
            )
        return self._session

    async def _get(self, path: str) -> Optional[dict]:
        """Authenticated GET with retry logic."""
        url = f"{self.cfg.base_url}{path}"
        params = {"api_key": self.cfg.api_key}
        session = await self._get_session()

        for attempt in range(self.cfg.retry_attempts):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", "10"))
                        logger.warning("API rate limit hit — sleeping %ds", retry_after)
                        await asyncio.sleep(retry_after)
                    else:
                        logger.error("API error %d for %s", resp.status, url)
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                wait = 2 ** attempt
                logger.warning("API request failed (attempt %d): %s — retry in %ds", attempt + 1, exc, wait)
                await asyncio.sleep(wait)
        return None

    async def fetch_match_timeline(self, match_id: str) -> None:
        """
        Fetch the timeline for a match and enqueue player-level events.
        Called once per match for post-match ingestion or pre-start enrichment.
        """
        data = await self._get(f"/matches/{match_id}/timeline")
        if not data:
            return

        for event in data.get("timeline", []):
            player_id = str(event.get("competitor", {}).get("id", ""))
            if not player_id:
                continue

            ts_str = event.get("time")
            try:
                ts = datetime.fromisoformat(ts_str)
            except (TypeError, ValueError):
                ts = datetime.now(tz=timezone.utc)

            obs = RawPlayerObservation(
                source="api",
                player_external_id=player_id,
                ts=ts,
                event_type=event.get("type"),
                match_id=match_id,
                raw_payload=event,
            )
            if obs.is_valid():
                await self.queue.put(obs)

    async def fetch_player_profile(self, player_external_id: str) -> Optional[dict]:
        """Fetch static player profile for onboarding / baseline seeding."""
        return await self._get(f"/players/{player_external_id}/profile")

    async def fetch_match_statistics(self, match_id: str) -> Optional[dict]:
        """Fetch post-match aggregate statistics for session record population."""
        return await self._get(f"/matches/{match_id}/statistics")

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


# ─────────────────────────────────────────────
# 3. WebSocket Live Event Stream Adapter
# ─────────────────────────────────────────────
class LiveEventWSAdapter:
    """
    Connects to a WebSocket live-match event stream.
    Compatible with Tracab / Stats Perform / custom providers.

    Expected message format (JSON):
    {
      "player_id": "p123",
      "ts": "2025-04-01T20:15:30Z",
      "type": "sprint_start" | "sprint_end" | "position" | "event",
      "x": 45.2,  "y": 22.1,
      "speed_ms": 7.4,
      "heart_rate": 168
    }
    """

    def __init__(self, config: LiveEventWSConfig, queue: asyncio.Queue):
        self.cfg = config
        self.queue = queue
        self._running = False

    async def stream(self) -> None:
        """Maintain a persistent WebSocket connection with auto-reconnect."""
        import websockets

        self._running = True
        attempt = 0

        while self._running and attempt < self.cfg.max_reconnect_attempts:
            try:
                logger.info("WS: connecting to %s (attempt %d)", self.cfg.url, attempt + 1)
                async with websockets.connect(
                    self.cfg.url,
                    ping_interval=self.cfg.heartbeat_interval_s,
                    ping_timeout=10,
                ) as ws:
                    logger.info("WS: connected to live event stream")
                    attempt = 0  # reset on successful connection
                    async for raw_msg in ws:
                        await self._handle_message(raw_msg)

            except Exception as exc:
                attempt += 1
                delay = min(self.cfg.reconnect_delay_s * (2 ** attempt), 60.0)
                logger.warning("WS disconnected: %s — reconnecting in %.1fs", exc, delay)
                await asyncio.sleep(delay)

        logger.error("WS: max reconnect attempts reached — stream terminated")

    async def _handle_message(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("WS: non-JSON message ignored")
            return

        player_id = str(payload.get("player_id", ""))
        if not player_id:
            return

        ts_raw = payload.get("ts")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except (TypeError, AttributeError, ValueError):
            ts = datetime.now(tz=timezone.utc)

        obs = RawPlayerObservation(
            source="ws",
            player_external_id=player_id,
            ts=ts,
            latitude=payload.get("lat"),
            longitude=payload.get("lon"),
            speed_ms=payload.get("speed_ms"),
            heart_rate_bpm=payload.get("heart_rate"),
            event_type=payload.get("type"),
            match_id=payload.get("match_id"),
            raw_payload=payload,
        )
        if obs.is_valid():
            await self.queue.put(obs)

    def stop(self) -> None:
        self._running = False


# ─────────────────────────────────────────────
# 4. MQTT Wearable Sensor Adapter
# ─────────────────────────────────────────────
class MQTTWearableAdapter:
    """
    Subscribes to MQTT topics published by a BLE/ANT+ gateway.
    Topic pattern: players_data/sensors/{player_external_id}/hr
                   players_data/sensors/{player_external_id}/accel
    """

    def __init__(self, config, queue: asyncio.Queue):
        self.cfg = config
        self.queue = queue

    async def stream(self) -> None:
        """Subscribe and forward wearable observations into the queue."""
        try:
            import asyncio_mqtt as aiomqtt
        except ImportError:
            logger.warning("asyncio_mqtt not available — MQTT adapter disabled")
            return

        async with aiomqtt.Client(self.cfg.mqtt_broker, port=self.cfg.mqtt_port) as client:
            topic_filter = f"{self.cfg.topic_prefix}/+/+"
            logger.info("MQTT: subscribing to %s", topic_filter)
            await client.subscribe(topic_filter, qos=self.cfg.qos)

            async for message in client.messages:
                await self._handle_mqtt_message(str(message.topic), message.payload)

    async def _handle_mqtt_message(self, topic: str, payload: bytes) -> None:
        """
        Topic: players_data/sensors/{player_id}/{sensor_type}
        Payload: JSON {"value": ..., "ts": "..."}
        """
        parts = topic.split("/")
        if len(parts) < 4:
            return

        player_id = parts[2]
        sensor_type = parts[3]    # "hr" | "accel"

        try:
            data = json.loads(payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        ts_raw = data.get("ts")
        try:
            ts = datetime.fromisoformat(ts_raw)
        except (TypeError, ValueError):
            ts = datetime.now(tz=timezone.utc)

        obs = RawPlayerObservation(
            source="mqtt",
            player_external_id=player_id,
            ts=ts,
            heart_rate_bpm=int(data["value"]) if sensor_type == "hr" else None,
            acceleration_ms2=float(data["value"]) if sensor_type == "accel" else None,
            raw_payload=data,
        )
        if obs.is_valid():
            await self.queue.put(obs)


# ─────────────────────────────────────────────
# 5. Ingestion Normalizer
# ─────────────────────────────────────────────
class IngestionNormalizer:
    """
    Receives RawPlayerObservation from all sources.
    Applies:
      - Unit normalization (knots → m/s, etc.)
      - GPS → pitch coordinate transform
      - Sprint / high-intensity classification
      - Sliding-window aggregate computation
      - Data quality scoring
    Produces dicts ready for bulk ORM insert.
    """

    SPRINT_THRESHOLD_MS = 7.0           # > 7 m/s = sprint (≈ 25.2 km/h)
    HIGH_INTENSITY_THRESHOLD_MS = 5.5   # > 5.5 m/s = high-intensity run

    def __init__(
        self,
        pitch_origin: Optional[tuple[float, float]] = None,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ):
        self.pitch_origin = pitch_origin
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m

        # Per-player sliding window buffers
        self._windows: Dict[str, List[RawPlayerObservation]] = {}
        self._window_seconds = CONFIG.inference.sliding_window_seconds

    def normalize(self, obs: RawPlayerObservation) -> dict:
        """Convert a RawPlayerObservation to a normalized event dict."""
        record = {
            "player_external_id": obs.player_external_id,
            "ts": obs.ts,
            "source": obs.source,
            "latitude": obs.latitude,
            "longitude": obs.longitude,
            "speed_ms": obs.speed_ms,
            "acceleration_ms2": obs.acceleration_ms2,
            "heart_rate_bpm": obs.heart_rate_bpm,
            "event_type": obs.event_type,
            "match_id": obs.match_id,
            "data_quality_score": obs.quality_score(),
            "x_pitch": None,
            "y_pitch": None,
            "is_sprint": False,
            "is_high_intensity": False,
            "window_sprint_count": None,
            "window_distance_m": None,
            "window_avg_speed_ms": None,
        }

        # Pitch coordinate transform
        if (obs.latitude and obs.longitude and self.pitch_origin):
            x, y = gps_to_pitch_coords(
                obs.latitude, obs.longitude,
                self.pitch_origin[0], self.pitch_origin[1],
                self.pitch_length_m, self.pitch_width_m,
            )
            record["x_pitch"] = x
            record["y_pitch"] = y

        # Sprint classification
        if obs.speed_ms is not None:
            record["is_sprint"] = obs.speed_ms >= self.SPRINT_THRESHOLD_MS
            record["is_high_intensity"] = obs.speed_ms >= self.HIGH_INTENSITY_THRESHOLD_MS
            if record["is_sprint"]:
                record["event_type"] = record["event_type"] or "sprint"
            elif record["is_high_intensity"]:
                record["event_type"] = record["event_type"] or "high_intensity_run"
            elif obs.speed_ms > 2.0:
                record["event_type"] = record["event_type"] or "jog"
            else:
                record["event_type"] = record["event_type"] or "walk"

        # Update sliding window
        player_id = obs.player_external_id
        window = self._windows.setdefault(player_id, [])
        window.append(obs)

        # Prune window to configured time range
        cutoff_ts = obs.ts.timestamp() - self._window_seconds
        self._windows[player_id] = [
            o for o in window if o.ts.timestamp() >= cutoff_ts
        ]
        window = self._windows[player_id]

        # Compute window aggregates
        speeds = [o.speed_ms for o in window if o.speed_ms is not None]
        sprint_obs = [o for o in window if (o.speed_ms or 0) >= self.SPRINT_THRESHOLD_MS]

        record["window_sprint_count"] = len(sprint_obs)
        record["window_avg_speed_ms"] = sum(speeds) / len(speeds) if speeds else None

        # Approximate distance from speed × sampling interval
        if len(speeds) >= 2:
            dt = self._window_seconds / len(window)
            record["window_distance_m"] = sum(speeds) * dt
        else:
            record["window_distance_m"] = None

        return record


# ─────────────────────────────────────────────
# 6. Ingestion Pipeline Orchestrator
# ─────────────────────────────────────────────
class IngestionPipeline:
    """
    Top-level orchestrator that:
      - Starts all adapters concurrently
      - Drains the shared queue
      - Runs normalization
      - Calls a user-supplied callback (e.g., write to DB, push to analysis engine)
    """

    def __init__(
        self,
        on_event: Callable[[dict], None],
        pitch_origin: Optional[tuple[float, float]] = None,
    ):
        self.on_event = on_event
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        self.normalizer = IngestionNormalizer(pitch_origin=pitch_origin)

    async def run(
        self,
        enable_gps: bool = True,
        enable_api: bool = True,
        enable_ws: bool = True,
        enable_mqtt: bool = True,
        gps_player_id: Optional[str] = None,
    ) -> None:
        """Start all enabled adapters and the consumer loop."""
        tasks = []

        if enable_gps and gps_player_id:
            gps = GPSIngestionAdapter(
                player_external_id=gps_player_id,
                config=CONFIG.gps,
                queue=self.queue,
            )
            tasks.append(asyncio.create_task(gps.stream_tcp(), name="gps_stream"))

        if enable_api:
            # API adapter is event-driven, not a persistent loop — excluded from task list
            # Call pipeline.ingest_match(match_id) explicitly to trigger
            pass

        if enable_ws:
            ws = LiveEventWSAdapter(config=CONFIG.live_ws, queue=self.queue)
            tasks.append(asyncio.create_task(ws.stream(), name="ws_stream"))

        if enable_mqtt:
            mqtt = MQTTWearableAdapter(config=CONFIG.wearable, queue=self.queue)
            tasks.append(asyncio.create_task(mqtt.stream(), name="mqtt_stream"))

        # Consumer
        tasks.append(asyncio.create_task(self._consume(), name="queue_consumer"))

        logger.info("Ingestion pipeline started with %d adapters", len(tasks) - 1)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _consume(self) -> None:
        """Drain the queue, normalize, and call on_event callback."""
        while True:
            try:
                obs: RawPlayerObservation = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )
                normalized = self.normalizer.normalize(obs)
                self.on_event(normalized)
                self.queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.exception("Consumer error: %s", exc)
