# Handball Calibration Audit
## PlayerDynamics — SC Magdeburg (Kinexon UWB)

**Audit date**: 2026-06-19  
**Scope**: `analysis/anomaly_detection.py`, `analysis/regime.py`, `explainability/semantics_layer.py`, `analysis/telemetry_validity.py`, `config/settings.py`, `analysis/baseline.py`  
**Constraint**: No code changes permitted. Document only.

---

## Reference: Sport Dimensions

| Dimension | Football (current) | Handball (actual) | Source |
|---|---|---|---|
| Pitch length | 105 m | 40 m | IHF Rule 1 |
| Pitch width | 68 m | 20 m | IHF Rule 1 |
| Sprint threshold | 7.0 m/s (25.2 km/h) | 5.5 m/s (19.8 km/h) | IHF / KinexonConfig |
| High-intensity threshold | — | 4.17 m/s (15.0 km/h) | KinexonConfig |
| Max speed (sensor cap) | — | 12.0 m/s | KinexonConfig |
| Max acceleration (sensor cap) | — | 25.0 m/s² | KinexonConfig |
| Match duration | 90 min | 60 min (2 × 30 min) | IHF Rule 7 |
| Half duration | 45 min (2700 s) | 30 min (1800 s) | IHF Rule 7 |

The analytics engine was built against football GPS data. Kinexon provides UWB positioning in metres, centred at court midpoint: x ∈ [−22, +22], y ∈ [−13, +13]. The adapter normalises both axes to [0, 100] pitch units before feeding the engine.

---

## D. MUST REPLACE  
*Actively wrong: output category is incorrect, not just miscalibrated. Deploy handball values before any production inference.*

---

### D1. `SPRINT_THRESHOLD_MS = 7.0` — highest impact
**File**: `analysis/anomaly_detection.py` line 163  
**Used in**: `_extract()` → `sprint = 1.0 if speed >= SPRINT_THRESHOLD_MS else 0.0`

Football value (25.2 km/h). The maximum recorded speed for a handball player under match conditions is roughly 6.5 m/s (23.4 km/h). At threshold 7.0 m/s, **no handball player ever registers as sprinting**.

**Cascade effects** (all downstream of this single constant):

| Downstream | Effect |
|---|---|
| `sprint_flag` feature (index 3) | Permanently 0 for all handball events |
| `SessionRegimeClassifier` intensity | `sprint_frac = 0` always → all windows are `__low` intensity |
| `RegimeAwareThresholdStore` | Only `defensive__low`, `midfield__low`, `attacking__low` ever populated; 6 of 9 regime buckets permanently empty |
| `sprint_low = fv.get("sprint_flag") == 0` in `analyze_window()` | Always `True` (line 4667) |
| `fatigue_flag` condition | `sprint_low or speed_low` → `sprint_low=True` satisfies this unconditionally |
| `locomotor_suppression` rule (semantics_layer) | `sprint_count=0` always → suppression appears chronic |
| `fatigue_accumulation` rule (semantics_layer) | `window_sprint_count=0` contributes to spurious fatigue signals |

**Handball value**: 5.5 m/s. Already in `KinexonConfig.sprint_threshold_ms`. The analytics engine must read this value.

---

### D2. `PITCH_LENGTH_M = 105.0` / `PITCH_WIDTH_M = 68.0` — second-highest impact
**File**: `analysis/anomaly_detection.py` lines 169–170  
**Used in**: `_extract()` lines 1232–1234

```python
dx_m = (x - prev_x) / 100.0 * PITCH_WIDTH_M   # * 68.0  instead of * 20.0
dy_m = (y - prev_y) / 100.0 * PITCH_LENGTH_M   # * 105.0 instead of * 40.0
distance_delta = math.sqrt(dx_m**2 + dy_m**2)
```

The KinexonAdapter normalises Kinexon UWB metres to [0, 100] pitch units relative to the handball court (40 m × 20 m). When `_extract()` converts back to metres it uses FIFA dimensions, inflating each displacement by **2.6–3.4×** depending on movement direction.

| Axis | Pitch unit → real metres | Pitch unit → computed metres | Inflation |
|---|---|---|---|
| x (short axis) | 1 unit = 0.20 m | 1 unit = 0.68 m | 3.4× |
| y (long axis) | 1 unit = 0.40 m | 1 unit = 1.05 m | 2.6× |

**Cascade effects**:

| Downstream | Effect |
|---|---|
| `distance_delta_m` feature (index 6) | ~3× inflated. LSTM trains on inflated values — only safe if ALL training data is Kinexon, never mixed with football GPS. |
| `baseline.distance_mean` | Accumulates inflated `distance_delta_m` → inflated by ~3×. |
| `baseline_speed` in `analyze_window()` | Input (`distance_mean`) is inflated → baseline speed inflated (see D4). |
| `PositionalDriftAnalyzer` | Drift operates in pitch units, not metres — **not** affected by this constant (see C3). |
| `fatigue_decay_residual` (if computed from distance) | Inflated → triggers `fatigue_residual_high=80m` with only ~27 m actual movement. |

**Handball values**: `PITCH_LENGTH_M = 40.0`, `PITCH_WIDTH_M = 20.0`.  
Already in `KinexonConfig.pitch_length_m` and `KinexonConfig.pitch_width_m`. The analytics engine must read these.

---

### D3. `late_in_game = elapsed > 2700`
**File**: `analysis/anomaly_detection.py` line 4668

2700 s = 45 minutes = football first-half duration. Used in `analyze_window()` to escalate fatigue signals to "high" severity and to gate `fatigue_flag` (anomaly AND late).

A handball match is 60 minutes (3600 s). Half-time is at 1800 s. Using 2700 s:
- During the first 30-minute half: `elapsed` reaches 1800 s maximum → `late_in_game = False` always
- During the second 30-minute half: `elapsed` is 1800–3600 s → `late_in_game` is True only after 2700 s, i.e., only in the **last 15 minutes**

Effect: fatigue severity escalation is available for only the final 15 minutes of a match rather than the entire second half. The qualitative label on fatigue signals ("medium" vs "high") is systematically incorrect for the first 15 minutes of any handball second half.

**Handball value**: 1800 s (30 min). The signal should escalate whenever `elapsed > match_half_duration_s`.

---

### D4. `baseline_speed = distance_mean / (90 * 60)`
**File**: `analysis/anomaly_detection.py` line 4663

```python
baseline_speed = (baseline.distance_mean / (90 * 60)
                  if baseline.distance_mean > 0 else 3.5)
```

Two compounding errors:

1. **Denominator**: 90 × 60 = 5 400 s. Handball match is 60 × 60 = 3 600 s. This alone underestimates baseline speed by 1.5×.

2. **Numerator**: `distance_mean` is the cumulative average of `distance_delta_m` values, which are inflated ~3× by D2. So `baseline_speed` is inflated by approximately **4.5×** relative to actual movement speed.

Effect on `speed_ratio = speed_ms / baseline_speed`:
- Real handball cruising speed: ~3.5 m/s
- `baseline_speed` from inflated data: ~3.5 × 4.5 ≈ 15.75 m/s (for a player covering ~9 450 m of inflated distance in 3 600 s... more precisely it depends on actual data, but inflation is of this order)
- `speed_ratio ≈ 3.5 / 15.75 ≈ 0.22`, which is always below the `speed_low` threshold of 0.55

Effect: `speed_low = True` **permanently** → `fatigue_flag = anomaly and (speed_low or sprint_low) and late_in_game`. Since both `speed_low` and `sprint_low` are always True, any detected anomaly after 1800 s is automatically a fatigue event regardless of actual player output.

**Handball fix**: denominator should be `60 * 60 = 3 600`. Also requires D2 fix to remove distance inflation from the numerator.

---

## B. NEEDS CALIBRATION  
*The correct handball value is known. No analytical failure occurs under normal conditions, but output magnitude or label is systematically biased.*

---

### B1. TVL acceleration threshold: hard-coded `12.0 m/s²`
**File**: `analysis/telemetry_validity.py` line 78

```python
if accel > 12.0:
    issues.append(f"implausible_accel_{accel:.2f}")
    confidence = 0.0
```

Note: the class constant `self.MAX_ACCEL_MS2 = 15.0` (line 42) is **not used** in the actual check — the check uses the bare literal `12.0`. This is a pre-existing inconsistency.

For handball, peak acceleration during change-of-direction is documented at 20–25 m/s² (`KinexonConfig.max_accel_ms2 = 25.0`). However, the Kinexon adapter emits the key `"acceleration_ms2"` while TVL checks `event.get("accel", 0.0)` — a **key mismatch**. This means TVL currently never checks Kinexon acceleration (always reads 0.0 → check passes silently).

The key mismatch is doubly important:
- Fixing the key without raising the threshold would cause genuine handball accelerations (>12 m/s²) to be marked INVALID.
- Both must be fixed together: key alignment **and** threshold raised to at least 25.0 m/s².

**Action**: Align TVL key to `"acceleration_ms2"` AND raise threshold to `KinexonConfig.max_accel_ms2 = 25.0`.

---

### B2. TVL maximum speed: `MAX_SPEED_MS = 13.5 m/s`
**File**: `analysis/telemetry_validity.py` line 40

13.5 m/s = 48.6 km/h. The hard-reject fires at >20% over ceiling = 16.2 m/s.

`KinexonConfig.max_speed_ms = 12.0 m/s` (43 km/h). Since TVL's soft ceiling (13.5) is above the KinexonConfig cap (12.0), no valid handball reading will be rejected or penalised for speed. The check is **permissive** — safe to leave as-is for Phase 1, but the ceiling is misleading for handball documentation.

**Action**: Update `MAX_SPEED_MS` to 12.0 or introduce a Kinexon-specific override for clarity. Not urgent.

---

### B3. `SequenceWindowConfig.event_interval_s = 15` (GPS cadence)
**File**: `config/settings.py` line 104

The window config was designed for GPS at 1 Hz downsampled to 1 event per 15 seconds (DT_OUT=15 in the original data generator). Kinexon streams at 20 Hz. The KinexonAdapter must decimate to 1 event per 15 seconds before feeding the window builder.

Currently unverified whether the adapter enforces this downsampling in live mode. If it does not, the 8-tick window fills in 0.4 s (8 × 50 ms at 20 Hz) instead of 120 s.

Additionally, the 15 s/tick granularity was chosen for detecting fatigue over 2-minute GPS windows. Handball's high-intensity bursts last 5–15 s. Finer granularity (e.g., 5 s/tick = 600 ms intervals at 20 Hz decimation → `window_steps = 24`) would capture intensity transitions that the current 15 s tick misses.

**Action**: Confirm adapter downsampling in live mode. Evaluate whether window resolution should change for handball (requires model retraining).

---

### B4. `AnomalyScoringConfig.accel` clamp: `±10.0 m/s²` in `_extract()`
**File**: `analysis/anomaly_detection.py` line 1224

```python
accel = float(np.clip(accel, -10.0, 10.0))
```

This is a feature-space clamp on the value stored in the LSTM sequence. It does not reject the event (unlike TVL). KinexonConfig.max_accel_ms2=25.0 means genuine handball peak accelerations are truncated in the feature vector.

Effect: the LSTM sees maximum ±10 m/s² regardless of actual acceleration. Peak handball change-of-direction efforts (which are the most diagnostically meaningful events) are compressed to the same feature value as moderate accelerations. This reduces the model's ability to distinguish elite bursts from moderate efforts.

**Action**: Raise the feature clamp to ±25.0 m/s² once D2 (pitch geometry) is resolved and model is retrained.

---

## C. NEEDS REAL-WORLD VALIDATION  
*Direction is plausible but the correct handball value is not derivable from first principles alone. Requires match data analysis or handball sports science input.*

---

### C1. Semantic layer: window quality gate suppresses all findings for Kinexon
**File**: `explainability/semantics_layer.py` line 379

```python
if hr == 0.0 and speed > 0.5:
    reasons.append("HR=0 bpm while speed=... — HR sensor dropout")
    # → degraded=True → all SemanticFindings suppressed
```

The feature vector has `heart_rate_bpm = 0.0` for all Kinexon events (TVL fix: None → safe_float → 0.0). The window quality gate fires for every moving Kinexon player and returns `degraded=True`, which suppresses **all** semantic findings — including the purely movement-based ones (`locomotor_overload`, `tactical_instability`, `fatigue_accumulation`, `locomotor_suppression`).

This is an unintended cascade of the TVL fix: the TVL correctly distinguishes `None` (absent sensor) from `0` (malfunction), but the feature vector collapses both to `0.0`. The SemanticInterpreter cannot recover the distinction.

**Impact**: None of the six semantic finding types will be generated for any SC Magdeburg player until HR sensors are integrated, regardless of what the LSTM anomaly model detects.

**Resolution path**: Add `hr_sensor_absent` boolean to `fv` dict at line 4598 of `analyze_window()` (adjacent to `mask_completeness`). The gate in `_assess_window_quality()` would then check this flag before concluding dropout. This is a targeted, minimal change.

---

### C2. `PositionalDriftConfig.zone_radius_meters = 5.0` (pitch-unit floor)
**File**: `config/settings.py` line 205

The field name is misleading: `PositionalDriftAnalyzer` receives positions in pitch units [0, 100] and computes Euclidean distance in those same units. The `zone_radius_meters` value is therefore a pitch-unit quantity, not a metre quantity.

On a 40 m × 20 m handball court, 5 pitch units corresponds to:
- 1.0 m on the short (20 m) axis
- 2.0 m on the long (40 m) axis

This is the **minimum** zone floor (`thr = max(std_r * 2.0, 5.0)`). In practice, the personal `std_r` from the baseline dominates unless the player's historical positions are very tightly clustered (e.g., a stationary goalkeeper).

For a handball goalkeeper who rarely leaves the 6-metre crease, a personal `std_r` of < 2.5 pitch units is plausible, meaning the 5.0 floor binds — and corresponds to roughly 1 m of actual court space. This is probably too tight.

**Action**: Analyse actual Kinexon `std_r` values per position. If goalkeeper std_r < 2.5 pitch units in real sessions, raise the floor for handball (suggested: 8–10 pitch units = 1.6–2.0 m on short axis). Also rename the field to `zone_radius_pitch_units` to remove the misleading "meters" label.

---

### C3. Regime territory thresholds: `33 / 67` pitch-unit thirds
**File**: `analysis/regime.py` lines 69–70

```python
_TERRITORY_DEFENSIVE_MAX  = 33.0
_TERRITORY_ATTACKING_MIN  = 67.0
```

These divide the [0, 100] pitch axis into three equal thirds: defensive (<33), midfield (33–67), attacking (>67). In football this maps to own third / centre / opponent third.

For handball the analogous zones are:
- Own half (0–50): defensive
- Opponent half (50–100): offensive
- No football-style "midfield" third in handball — the game transitions rapidly between halves

The 33/67 split will misclassify handball midfield positions (near the centre line, x≈50) as "midfield" rather than the attack or defence intent.

Additionally, because D1 (sprint threshold) makes all windows `__low` intensity, the territory classification is the only dimension that currently varies. Regime keys produced under handball with current constants will all be `{defensive,midfield,attacking}__low`.

**Action**: Evaluate whether 50-based split (own half / opponent half) better represents handball tactical context. This would collapse 9 regimes to 6 (2 territories × 3 intensities), requiring retraining of `RegimeAwareThresholdStore` buckets.

---

### C4. ACWR injury-risk band: `1.5 / 0.8`
**File**: `analysis/baseline.py` line 447, `analysis/anomaly_detection.py` line 4588

ACWR bands [0.8, 1.5] are from Hulin et al. (2016), validated primarily in cricket, rugby, and Australian football. Application to handball has limited published evidence.

The ACWR computation itself uses `total_distance_m` from `sessions_df`. If session distances come from the Kinexon system (as they will for handball), they carry the ~3× D2 inflation. Crucially, this inflation is **consistent across all sessions** — the ratio cancels out mathematically: `(inflated_acute) / (inflated_chronic) = actual_ratio`. The ACWR value is therefore correct as long as all sessions in `sessions_df` use the same coordinate system.

However, if any sessions in `sessions_df` are from football GPS (with true distances), the ratio is poisoned.

**Action**: Validate that `sessions_df` for handball players contains only Kinexon-derived distances before deploying. Once D2 is fixed, all stored distances must be retroactively recalculated.

---

### C5. `fatigue_residual_high = 80.0` in semantics layer
**File**: `explainability/semantics_layer.py` line 125

Used in `_rule_recovery_degradation()` and `_rule_fatigue_accumulation()`. This threshold governs when `fatigue_decay_residual` (cumulative distance above the exponential decay curve) is considered elevated.

The 80 m value was calibrated against football distances. With D2 inflating `distance_delta_m` ~3×, the effective handball equivalent is approximately 27 m of actual court distance. This makes the threshold extremely sensitive — minor movement bursts would push residual above 80 m before the fix, and the threshold would need revisiting after D2 is resolved.

**Action**: Defer validation until D2 is fixed. After geometry correction, re-evaluate against 3–5 real match sessions to establish a handball-appropriate fatigue residual threshold.

---

## A. SAFE TO KEEP  
*Football-specific origin but value is either universal or directionally correct for handball.*

---

| # | Item | File | Why safe |
|---|---|---|---|
| 1 | `MAX_HR_BPM = 220`, `MIN_HR_BPM = 30` | `telemetry_validity.py` | Universal physiological bounds. Not sport-specific. |
| 2 | EMA alpha `score_ema_alpha = 0.25` | `config/settings.py` | Statistical smoothing parameter. Sport-agnostic. |
| 3 | ACWR windows: 7-day acute / 28-day chronic | `config/settings.py` `BaselineConfig` | Sports science consensus across team sports, including handball (provided session data is single-source — see C4). |
| 4 | `min_calibration_windows = 30` | `config/settings.py` | Statistical floor for adaptive threshold. Sport-agnostic. |
| 5 | `mad_multiplier = 5.0` | `config/settings.py` | Robust statistics parameter. Sport-agnostic. |
| 6 | Semantic `hr_high = 175`, `hr_critical = 185` | `semantics_layer.py` | Elite handball matches sustain HR above 175 bpm. Physiologically appropriate. Currently unreachable (HR=None), but correct for future HR integration. |
| 7 | Semantic `z_score_high = 1.5`, `z_score_very_high = 2.5` | `semantics_layer.py` | Standard deviation thresholds. Sport-agnostic. |
| 8 | Semantic `drift_elevated = 1.2×`, `drift_high = 1.8×` | `semantics_layer.py` | Relative multipliers of personal zone radius. Sport-agnostic. |
| 9 | Semantic `shap_relevant = 0.05`, `shap_strong = 0.15` | `semantics_layer.py` | Attribution cutoffs. Sport-agnostic. |
| 10 | Semantic `persistence_confirmed = 3`, `persistence_severe = 6` (windows) | `semantics_layer.py` | Count of consecutive windows. Once window duration is validated (B3), this count is sport-agnostic. |
| 11 | Semantic `hr_recovery_negative = -0.05`, `hr_recovery_flat = 0.02` | `semantics_layer.py` | Fractional HR change rates. Physiologically calibrated, not sport-specific. |
| 12 | Semantic `speed_ms_low = 2.5 m/s` | `semantics_layer.py` | Walking pace threshold. Applies to handball without change. |
| 13 | Semantic `acwr_high_risk = 1.30`, `acwr_low_readiness = 0.80` | `semantics_layer.py` | Literature-derived. Directionally valid for handball (see C4 for caveat on data provenance). |
| 14 | Regime intensity fractions: `0.15 / 0.04` sprint fraction | `regime.py` | Once D1 (sprint threshold) is fixed, these fraction cutoffs are reasonable for handball intensity bucketing. Currently inert. |

---

## Cascade Impact Summary

```
D1 SPRINT_THRESHOLD_MS = 7.0
  └─► sprint_flag = 0 always
       ├─► regime intensity = __low always       (all 6 non-low buckets empty)
       ├─► sprint_low = True always              ─┐
       └─► fatigue_accumulation / suppression      │
            semantics always show 0 sprints        │
                                                   │
D2 PITCH_LENGTH_M = 105, PITCH_WIDTH_M = 68        │
  └─► distance_delta inflated ~3x                  │
       ├─► baseline.distance_mean inflated          │
       │   └─► D4 baseline_speed inflated ─────────┤
       └─► fatigue_decay_residual inflated          │
            └─► C5 fatigue_residual_high too easy  │
                                                   │
D4 baseline_speed = distance_mean / (90*60)  ◄─────┘
  └─► speed_low = True always (ratio << 0.55)
       └─► fatigue_flag = anomaly AND True AND late_in_game

D3 late_in_game = elapsed > 2700
  └─► never True in 30-min handball half
       └─► fatigue_flag permanently False in first half
           fatigue severity = "medium" for first 75% of second half

C1 hr = 0.0 in feature vector (None→safe_float→0.0)
  └─► semantic window quality gate fires for all moving Kinexon players
       └─► ALL 6 SemanticFinding types suppressed
            incl. locomotor_overload, tactical_instability, fatigue_accumulation
```

---

## Recommended Fix Order

Fix dependencies must be respected:

```
Phase A — prerequisite geometry (unblocks everything downstream)
  [1] D2: PITCH_LENGTH_M = 40.0, PITCH_WIDTH_M = 20.0
  [2] D1: SPRINT_THRESHOLD_MS = 5.5
  [3] D3: late_in_game threshold → 1800 s
  [4] D4: baseline_speed denominator → 3600 s
  Retrain LSTM on corrected-geometry handball sessions.

Phase B — refine once geometry is correct
  [5] B1: TVL acceleration: align key to "acceleration_ms2", raise threshold to 25.0
  [6] B4: Feature-space accel clamp ±10 → ±25
  [7] C1: Add hr_sensor_absent to fv dict in analyze_window() to unblock semantics

Phase C — validate with real match data (post-Phase A)
  [8] C5: Re-derive fatigue_residual_high from corrected sessions
  [9] C2: Measure goalkeeper std_r in pitch units; adjust zone_radius floor if needed
  [10] C3: Evaluate regime territory 33/67 vs 50 split against handball match tapes
  [11] B3: Confirm Kinexon adapter downsampling rate; evaluate window_seconds for handball
```

Items [1]–[4] are hard prerequisites. Without them, any LSTM calibration or semantic threshold tuning is calibrated against wrong geometry and wrong sprint classification.

---

*No code was modified during this audit.*
