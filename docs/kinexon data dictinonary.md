# Kinexon CSV Export — Data Dictionary

**Session**: HSG Wetzlar vs. SC Magdeburg
**Session ID**: 3387
**Date**: 06/07/2026
**Source**: Kinexon UWB (Ultra-Wideband) tracking system

---

## Player Identity Resolution

Kinexon assigns each tracked entity a numeric identifier. This integer appears as `mapped id`
in positions/Inertial files and as `Player ID` in events/statistics files. These are
**Kinexon's own identifiers** — they are **NOT** the backend database `Player.id`
(PostgreSQL autoincrement starting at 1).

| File | Identity Column | Example Values |
|------|----------------|----------------|
| positions.csv | `mapped id` | 1796, 2260, 2331 |
| Inertial.csv | `mapped id` | 1796, 2260, 2331 |
| events.csv | `Player ID` | 1796, 2260, 2331 |
| statistics.csv | `Player ID` | 1796, 2260, 2331 |

**Cross-system mapping required** if Kinexon data must be linked to the backend `Player.id`.
Phase 1 (`KinexonAdapter`) uses Kinexon `mapped id` directly as `player_id` in PlayerDynamics.

Ball entities must be excluded: `group name == "Ball"` (mapped_ids: 369, 371).

---

## 1. positions.csv — PRIMARY SOURCE

**Role**: Real-time 20 Hz position and kinematics for all tracked entities.
**Rows**: 1,081,793 | **Delimiter**: `;` | **Encoding**: UTF-8 | **Sample rate**: 50 ms (20 Hz)

| Column | Meaning | Unit | Level | Category | Adapter Use |
|--------|---------|------|-------|----------|-------------|
| `ts in ms` | Epoch timestamp (Kinexon clock) | ms | Player | Required | → `ts`, `timestamp_ms` |
| `formatted local time` | Human-readable timestamp | text | Player | Info | Ignore |
| `sensor id` | Hardware sensor serial number | int | Player | Metadata | Ignore |
| `mapped id` | **Kinexon player identity** | int | Player | Required | → `player_id`, `player_external_id` |
| `number` | Jersey number | int | Player | Metadata | → `jersey_number` |
| `full name` | Full player name | text | Player | Metadata | → `player_name` |
| `league id` | League database identifier | int | Team | Metadata | Ignore |
| `group id` | Team database identifier | int | Team | Metadata | Ignore |
| `group name` | Team name or `"Ball"` | text | Team | Required | → ball entity filter |
| `x in m` | Long-axis position, pitch-centred | m | Player | Required | → `x_m`, normalised → `x_pitch` |
| `y in m` | Short-axis position, pitch-centred | m | Player | Required | → `y_m`, normalised → `y_pitch` |
| `z in m` | Height (tracker chest height ≈ 1.6 m) | m | Player | Info | Ignore |
| `speed in m/s` | Instantaneous speed (Kinexon-computed) | m/s | Player | Required | → `speed_ms` |
| `direction of movement in deg` | Movement heading angle | deg | Player | Optional | Ignore |
| `acceleration in m/s2` | Signed acceleration (Kinexon-computed) | m/s² | Player | Required | → `acceleration_ms2` |
| `total distance in m` | Cumulative session distance | m | Player | **⚠ Always 0** | Ignore — compute from Δ(x,y) |
| `heart rate in bpm` | HR from wearable sensor | bpm | Player | Optional | → `heart_rate_bpm` (absent in export) |
| `core temperature in celsius` | Core body temperature | °C | Player | Optional | Ignore |
| `metabolic power in W/kg` | Instantaneous metabolic power | W/kg | Player | Optional | Ignore |
| `player orientation in deg` | Body facing direction | deg | Player | Optional | Ignore |
| `player orientation category` | Facing label (Forward/Back/Left/Right) | text | Player | Optional | Ignore |
| `ball possession (id of possessed ball)` | ID of possessed ball entity | int | Player | Optional | Ignore |
| `acceleration load` | Unit-less cumulative acceleration load | — | Player | Optional | Ignore |

**Observed value ranges (sample)**:

| Field | Min | Max | Notes |
|-------|-----|-----|-------|
| x in m | -22.23 | +22.68 | Centred origin; ±22 m > court boundary (40 m long axis) |
| y in m | -12.85 | +11.01 | Centred origin; ±12 m > court boundary (20 m short axis) |
| speed in m/s | 0.0 | 33.70 | **33.70 is a sensor artefact** — cap at 12.0 m/s |
| acceleration in m/s2 | negative | positive | Signed; typically ±8 m/s² in handball |

**Known issues in this export**:
- `total distance in m` is always 0. Compute `distance_delta_m` as Euclidean distance
  between consecutive `(x in m, y in m)` positions per player.
- `heart rate in bpm` is entirely absent (wearable HR sensor not worn/synced).
- Speed outlier at 33.70 m/s (≈ 121 km/h) — adapter caps at `KinexonConfig.max_speed_ms`.

---

## 2. Inertial.csv — NOT USED IN ADAPTER

**Role**: Raw IMU (Inertial Measurement Unit) data from the sensor's orientation module.
**Rows**: 8,420,005 | **Delimiter**: `;` | **Encoding**: UTF-8 | **Sample rate**: ~11 ms (~91 Hz, variable)

| Column | Meaning | Unit | Notes |
|--------|---------|------|-------|
| `ts in ms` | Epoch timestamp | ms | |
| `formatted local time` | Human-readable timestamp | text | |
| `sensor id` | Hardware sensor identifier | int | |
| `mapped id` | Kinexon player identity | int | Same namespace as positions.csv |
| `number` | Jersey number | int | |
| `full name` | Player name | text | |
| `league id` | League identifier | int | |
| `group id` | Team identifier | int | |
| `group name` | Team name or "Ball" | text | |
| `x in m` | Position X | m | **Sparse: ~15% of rows populated** |
| `y in m` | Position Y | m | **Sparse: ~15% of rows populated** |
| `z in m` | Position Z | m | **Sparse: ~15% of rows populated** |
| `heart rate in bpm` | Heart rate | bpm | **NEVER populated (0 rows with data)** |
| `rr interval in ms as n1,n2,...` | RR interval list | ms | Absent in export |
| `accelerometer x` | Raw accelerometer X axis | g | Populated ~85% |
| `accelerometer y` | Raw accelerometer Y axis | g | Populated ~85% |
| `accelerometer z` | Raw accelerometer Z axis | g | Populated ~85% |
| `gyroscope x` | Angular velocity X | rad/s | Populated ~85% |
| `gyroscope y` | Angular velocity Y | rad/s | Populated ~85% |
| `gyroscope z` | Angular velocity Z | rad/s | Populated ~85% |
| `quaternion 1` | Orientation quaternion W component | — | Populated ~85% |
| `quaternion i` | Orientation quaternion X component | — | Populated ~85% |
| `quaternion j` | Orientation quaternion Y component | — | Populated ~85% |
| `quaternion k` | Orientation quaternion Z component | — | Populated ~85% |

**Why not used**: HR never populated. x/y populated in only 15% of rows and less
reliable than positions.csv. Not suited as primary kinematic source.

---

## 3. events.csv — NOT USED IN ADAPTER

**Role**: Discrete, event-level activity records from Kinexon's automated event detector.
**Rows**: 7,528 (data rows; rows 0–12 are a multi-row compound header)
**Delimiter**: `;` | **Encoding**: UTF-8

| Column | Meaning | Unit | Notes |
|--------|---------|------|-------|
| `Timestamp (ms)` | Event start epoch | ms | |
| `Timestamp in local format` | Human-readable event start | text | |
| `Player ID` | Kinexon player identity | int | Same namespace as `mapped id` |
| `Name` | Player name | text | |
| `Event type` | Activity classification | text | See event types below |
| Sub-columns (vary by event type) | Duration, distance, peak speed, count, etc. | varies | |

**Event types in this export**: Acceleration, Ball Possession, Ball Possession (lost),
Ball Possession (recovered), Change of Direction, Deceleration, Exertion, Impact,
Jump, Pass, Shot, Sprint.

**Why not used**: Only 7,528 aggregate event rows across the full match — insufficient
temporal resolution for 20 Hz anomaly detection. Suitable for post-match analysis.

---

## 4. statistics.csv — SESSION METADATA (NOT USED IN REAL-TIME STREAM)

**Role**: Per-player session aggregate statistics. Used for player registry population
and baseline seeding (`sessions_df` input to `BaselineBuilder.compute()`).
**Rows**: 31 players | **Delimiter**: `;` | **Encoding**: Latin-1 (contains ≥, Ø, etc.)

Selected columns:

| Column | Meaning | Unit | Adapter Use |
|--------|---------|------|-------------|
| `Player ID` | Kinexon player identity | int | → player registry key |
| `Name` | Player name | text | **EMPTY** — use `full name` from positions.csv |
| `Position` | IHF position code | text | → position label |
| `Session ID` | Kinexon session identifier | int | → `match_id` context |
| `Group name` | Team name | text | → team grouping |
| `Description` | Match label | text | → match context |
| `Distance (m)` | Total session distance | m | → baseline seeding |
| `Distance (high speed) (m)` | High-speed distance | m | → baseline seeding |
| `Speed (avg.) (km/h)` | Average session speed | km/h | → baseline seeding |
| `Speed (max.) (km/h)` | Maximum session speed | km/h | → baseline seeding |
| `Heart rate (avg.) (bpm)` | Average HR | bpm | → baseline seeding |
| `Heart rate (min.) (bpm)` | Minimum HR | bpm | → baseline seeding |
| `Heart rate (max.) (bpm)` | Maximum HR | bpm | → baseline seeding |
| `Sprints` | Sprint count | int | → baseline seeding |
| `Accelerations` | Acceleration event count | int | → baseline seeding |
| HR zone columns | Time in HR zones (%) | % | → baseline seeding |
| Speed zone columns | Distance in speed zones (m) | m | → baseline seeding |

**IHF Position Codes (observed in this export)**:

| Code | German | English |
|------|--------|---------|
| TW | Torwart | Goalkeeper |
| KR | Kreisläufer | Pivot |
| RM | Rückraum Mitte | Centre Back |
| RR | Rückraum Rechts | Right Back |
| RL | Rückraum Links | Left Back |
| RA | Rechtsaußen | Right Wing |
| LA | Linksaußen | Left Wing |

**Players in session (confirmed from statistics.csv)**:

| Kinexon ID | Position | Team |
|-----------|----------|------|
| 2260 | LA | HSG Wetzlar |
| 1796 | RR | SC Magdeburg |
| 39 | TW | HSG Wetzlar |
| 2407 | TW | HSG Wetzlar |
| 296 | RM | SC Magdeburg |
| 2057 | RA | SC Magdeburg |
| 2262 | RA | HSG Wetzlar |
| 1974 | RM | HSG Wetzlar |
| 2279 | RL | SC Magdeburg |
| 1824 | RL | SC Magdeburg |
| 2266 | RM | HSG Wetzlar |
| 1977 | KR | HSG Wetzlar |
| 732 | RM | SC Magdeburg |
| 1973 | RL | HSG Wetzlar |
| 2261 | KR | HSG Wetzlar |
| 1975 | RM | HSG Wetzlar |
| 2058 | LA | SC Magdeburg |
| 1164 | KR | SC Magdeburg |
| 2059 | RM | SC Magdeburg |
| 2331 | TW | SC Magdeburg |
| 290 | LA | SC Magdeburg |
| 2096 | RR | HSG Wetzlar |
| 771 | RR | SC Magdeburg |
| 1797 | KR | SC Magdeburg |
| 2413 | LA | HSG Wetzlar |
| 1978 | RL | HSG Wetzlar |
| 1161 | RL | SC Magdeburg |
| 1823 | TW | SC Magdeburg |
| 25 | RR | HSG Wetzlar |
| 2056 | RA | SC Magdeburg |
| 2404 | RA | HSG Wetzlar |

---

## Coordinate Normalisation

The pipeline expects `x_pitch`, `y_pitch` ∈ [0, 100].
Kinexon delivers `x in m`, `y in m` in a pitch-centred coordinate system where (0, 0) is the
court centre.

```
x_pitch = clamp((x_m + pitch_length_m / 2) / pitch_length_m × 100,  0, 100)
y_pitch = clamp((y_m + pitch_width_m  / 2) / pitch_width_m  × 100,  0, 100)
```

For handball (40 m × 20 m):

```
x_pitch = clamp((x_m + 20.0) / 40.0 × 100,  0, 100)
y_pitch = clamp((y_m + 10.0) / 20.0 × 100,  0, 100)
```

Positions slightly outside court boundaries are clamped to [0, 100].

### Axis convention

| Axis | Physical dimension | Kinexon range | Pitch range |
|------|--------------------|---------------|-------------|
| X | Long axis (court length) | −22 to +22 m | [0, 100] left → right |
| Y | Short axis (court width) | −13 to +13 m | [0, 100] bottom → top |

---

## distance_delta_m Derivation

`total distance in m` is always 0 in the provided export. Compute per-player distance delta
from consecutive position ticks:

```python
distance_delta_m = sqrt((x2 - x1)**2 + (y2 - y1)**2)
```

At 20 Hz (50 ms ticks), top handball speed (~8 m/s) → max delta ≈ 0.4 m per tick.
First tick per player uses delta = 0.0.

---

## Pipeline Integration Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| player_id from Kinexon mapped_id | ✓ | Direct integer, no transform |
| GPS lat/lon not used | ✓ | UWB metre coords only |
| distance_delta_m derivation | ✓ | Euclidean √(Δx²+Δy²) per player |
| Coordinate normalisation | ✓ | Centred-origin → [0, 100] |
| Ball entity exclusion | ✓ | group_name == "Ball" filter |
| Speed outlier handling | ✓ | Cap at KinexonConfig.max_speed_ms |
| heart_rate_bpm | ⚠ | Absent in export → None |
| is_real flag in window builder | ⚠ | Requires both speed_ms AND heart_rate_bpm non-None |
| Backend Player.id mapping | ❌ | Out of scope (Phase 1) |
| Inertial.csv integration | ❌ | Out of scope (Phase 1); no usable HR or position |
| events.csv integration | ❌ | Out of scope (Phase 1); event aggregates only |
