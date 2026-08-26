"""
Pump-01 real-time condition-monitoring service.

Startup self-calibration + online inference:
MQTT -> healthy calibration -> feature windows -> Isolation Forest
-> persistent anomaly decision -> fault hypothesis -> recommendation
-> MQTT diagnostics.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import paho.mqtt.client as mqtt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------- Configuration ----------------

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

INPUT_TOPIC = os.getenv("INPUT_TOPIC", "factory/pump01/state")
OUTPUT_TOPIC = os.getenv(
    "OUTPUT_TOPIC", "factory/pump01/ml_diagnostics"
)
STATUS_TOPIC = os.getenv(
    "STATUS_TOPIC", "factory/pump01/ml_status"
)

ASSET_ID = os.getenv("ASSET_ID", "Pump-01")

CALIBRATION_SAMPLES = int(
    os.getenv("CALIBRATION_SAMPLES", "100")
)
WINDOW_SIZE = int(
    os.getenv("WINDOW_SIZE", "30")
)
WARNING_CONFIRM_COUNT = int(
    os.getenv("WARNING_CONFIRM_COUNT", os.getenv("ANOMALY_CONFIRM_COUNT", "3"))
)
CRITICAL_CONFIRM_COUNT = int(
    os.getenv("CRITICAL_CONFIRM_COUNT", "2")
)

WARNING_Z = float(os.getenv("WARNING_Z", "2.0"))
CRITICAL_Z = float(os.getenv("CRITICAL_Z", "3.0"))
EXTREME_Z = float(os.getenv("EXTREME_Z", "6.0"))

WARNING_ANOMALY_INDEX = float(
    os.getenv("WARNING_ANOMALY_INDEX", "0.70")
)
CRITICAL_ANOMALY_INDEX = float(
    os.getenv("CRITICAL_ANOMALY_INDEX", "0.95")
)

IFOREST_ESTIMATORS = int(
    os.getenv("IFOREST_ESTIMATORS", "300")
)
IFOREST_CONTAMINATION = float(
    os.getenv("IFOREST_CONTAMINATION", "0.05")
)
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

SENSORS = ("temperature", "vibration", "current")


# ---------------- Logging ----------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("pump-ml")


# ---------------- Helpers ----------------

def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def finite_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_iso_timestamp(value: object) -> Optional[float]:
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


# ---------------- Data types ----------------

@dataclass
class Sample:
    source_ts: float
    temperature: float
    vibration: float
    current: float


@dataclass
class Baseline:
    mean: dict[str, float]
    std: dict[str, float]
    median: dict[str, float]
    mad: dict[str, float]


# ---------------- Feature engineering ----------------

class FeatureEngine:
    FEATURE_NAMES = [
        "temperature_mean",
        "temperature_std",
        "temperature_slope",
        "temperature_range",
        "vibration_mean",
        "vibration_std",
        "vibration_rms",
        "vibration_slope",
        "vibration_range",
        "vibration_spike_rate",
        "current_mean",
        "current_std",
        "current_slope",
        "current_range",
    ]

    @classmethod
    def build(cls, window: list[Sample]) -> dict[str, float]:
        if len(window) < 3:
            raise ValueError("At least 3 samples are required.")

        t = np.arange(len(window), dtype=float)

        arrays = {
            sensor: np.array(
                [getattr(sample, sensor) for sample in window],
                dtype=float,
            )
            for sensor in SENSORS
        }

        features: dict[str, float] = {}

        for sensor in SENSORS:
            values = arrays[sensor]
            features[f"{sensor}_mean"] = float(np.mean(values))
            features[f"{sensor}_std"] = float(np.std(values))
            features[f"{sensor}_slope"] = float(
                np.polyfit(t, values, 1)[0]
            )
            features[f"{sensor}_range"] = float(
                np.max(values) - np.min(values)
            )

        vibration = arrays["vibration"]
        features["vibration_rms"] = float(
            np.sqrt(np.mean(vibration ** 2))
        )

        vibration_std = float(np.std(vibration))
        if vibration_std > 1e-9:
            threshold = float(np.mean(vibration)) + 2.5 * vibration_std
            features["vibration_spike_rate"] = float(
                np.mean(vibration > threshold)
            )
        else:
            features["vibration_spike_rate"] = 0.0

        return features

    @classmethod
    def vector(cls, features: dict[str, float]) -> np.ndarray:
        return np.array(
            [features[name] for name in cls.FEATURE_NAMES],
            dtype=float,
        )


# ---------------- Healthy baseline ----------------

def build_baseline(samples: list[Sample]) -> Baseline:
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    median: dict[str, float] = {}
    mad: dict[str, float] = {}

    for sensor in SENSORS:
        values = np.array(
            [getattr(sample, sensor) for sample in samples],
            dtype=float,
        )
        mean[sensor] = float(np.mean(values))
        std[sensor] = max(float(np.std(values, ddof=1)), 1e-6)
        median[sensor] = float(np.median(values))
        mad[sensor] = max(
            float(np.median(np.abs(values - median[sensor]))),
            1e-6,
        )

    return Baseline(mean, std, median, mad)


# ---------------- Isolation Forest ----------------

class AnomalyDetector:
    """Startup-calibrated Isolation Forest with empirical anomaly index."""

    def __init__(self, calibration_windows: list[dict[str, float]]) -> None:
        if len(calibration_windows) < 10:
            raise ValueError(
                "Not enough calibration windows for Isolation Forest."
            )

        X = np.vstack(
            [FeatureEngine.vector(f) for f in calibration_windows]
        )

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=IFOREST_ESTIMATORS,
            contamination=IFOREST_CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        # Preserve the healthy score distribution instead of mapping every
        # score below one fixed quantile directly to anomaly=1.0.
        self.healthy_scores = np.sort(
            self.model.decision_function(X_scaled)
        )

    def score(
        self,
        features: dict[str, float],
    ) -> tuple[bool, float, float]:
        vector = FeatureEngine.vector(features).reshape(1, -1)
        scaled = self.scaler.transform(vector)

        decision_score = float(
            self.model.decision_function(scaled)[0]
        )

        percentile = float(
            np.searchsorted(
                self.healthy_scores,
                decision_score,
                side="right",
            ) / len(self.healthy_scores)
        )

        # 0 ~= very normal, 1 ~= extremely unusual relative to healthy data.
        anomaly_index = clamp(1.0 - percentile)

        # sklearn's outlier flag is retained as supporting evidence only.
        model_outlier = decision_score < 0.0

        return (
            model_outlier,
            anomaly_index,
            decision_score,
        )


# ---------------- Sensor evidence ----------------

def calculate_evidence(
    window: list[Sample],
    baseline: Baseline,
) -> dict:
    arrays = {
        sensor: np.array(
            [getattr(sample, sensor) for sample in window],
            dtype=float,
        )
        for sensor in SENSORS
    }

    z_scores: dict[str, float] = {}
    robust_z_scores: dict[str, float] = {}
    trend_z_scores: dict[str, float] = {}

    t = np.arange(len(window), dtype=float)
    seconds = max(len(window) - 1, 1)

    for sensor in SENSORS:
        mean_value = float(np.mean(arrays[sensor]))

        z = (
            mean_value - baseline.mean[sensor]
        ) / baseline.std[sensor]

        robust_z = (
            0.6745
            * (
                mean_value - baseline.median[sensor]
            )
            / baseline.mad[sensor]
        )

        slope = float(np.polyfit(t, arrays[sensor], 1)[0])

        trend_z = abs(
            slope * seconds / baseline.std[sensor]
        )

        z_scores[sensor] = float(z)
        robust_z_scores[sensor] = float(robust_z)
        trend_z_scores[sensor] = float(trend_z)

    dominant_sensor = max(
        SENSORS,
        key=lambda sensor: abs(z_scores[sensor]),
    )

    return {
        "z_scores": z_scores,
        "robust_z_scores": robust_z_scores,
        "trend_z_scores": trend_z_scores,
        "dominant_sensor": dominant_sensor,
        "dominant_z": abs(z_scores[dominant_sensor]),
    }


# ---------------- Evidence-based diagnosis ----------------

class FaultHypothesisEngine:
    """Evidence-based diagnosis aligned with the simulator's fault signatures.

    These are engineering heuristics, not calibrated probabilities and not a
    supervised classifier. Diagnosis is only called after a persistent anomaly.
    """

    @staticmethod
    def z_evidence(z: float, scale: float = 3.0) -> float:
        # Smooth response; unlike clamp(abs(z)/5), large z-scores do not all
        # collapse to the same value immediately.
        return float(np.tanh(abs(float(z)) / scale))

    def diagnose(
        self,
        window: list[Sample],
        evidence: dict,
        baseline: Baseline,
    ) -> dict:
        z = evidence["z_scores"]
        trend = evidence["trend_z_scores"]

        arrays = {
            sensor: np.array(
                [getattr(sample, sensor) for sample in window],
                dtype=float,
            )
            for sensor in SENSORS
        }

        vibration_variability = (
            float(np.std(arrays["vibration"]))
            / baseline.std["vibration"]
        )
        current_variability = (
            float(np.std(arrays["current"]))
            / baseline.std["current"]
        )

        # Detect transient vibration events, not simply a high vibration
        # level. A smooth 7 mm/s vibration trend should not be classified as
        # cavitation merely because every point is above the healthy level.
        vib_diff = np.diff(arrays["vibration"])
        if len(vib_diff) > 0:
            spike_threshold = 3.0 * baseline.std["vibration"]
            vibration_spike_rate = float(
                np.mean(
                    np.abs(vib_diff) > spike_threshold
                )
            )
        else:
            vibration_spike_rate = 0.0

        # Relative deviation retains information when a very tight baseline
        # makes the z-score enormous.
        mean_vibration = float(np.mean(arrays["vibration"]))
        mean_temperature = float(np.mean(arrays["temperature"]))
        mean_current = float(np.mean(arrays["current"]))

        vib_rel = (
            mean_vibration - baseline.mean["vibration"]
        ) / max(baseline.mean["vibration"], 1e-6)

        temp_rel = (
            mean_temperature - baseline.mean["temperature"]
        ) / max(baseline.mean["temperature"], 1e-6)

        curr_rel = (
            mean_current - baseline.mean["current"]
        ) / max(baseline.mean["current"], 1e-6)

        # Relative response scales chosen from the current simulator profiles:
        # vibration changes are large for mechanical faults, current changes
        # are especially important for motor/electrical degradation, and
        # temperature is supporting evidence.
        V_rel = clamp(vib_rel / 2.0)
        T_rel = clamp(temp_rel / 0.10)
        C_rel = clamp(curr_rel / 0.80)

        # Smooth z/trend evidence.
        V = self.z_evidence(z["vibration"], 3.0)
        T = self.z_evidence(z["temperature"], 3.0)
        C = self.z_evidence(z["current"], 3.0)
        VT = self.z_evidence(trend["vibration"], 3.0)
        TT = self.z_evidence(trend["temperature"], 3.0)
        CT = self.z_evidence(trend["current"], 3.0)

        VV = clamp(
            max(0.0, vibration_variability - 1.0) / 3.0
        )
        CV = clamp(
            max(0.0, current_variability - 1.0) / 3.0
        )
        SPIKES = clamp(vibration_spike_rate * 6.0)

        # ---------------------------------------------------------------
        # Fault scores aligned to FAULT_PROFILES
        # ---------------------------------------------------------------
        scores = {
            # Bearing: strong vibration + meaningful temperature rise,
            # comparatively weak current response.
            "bearing_wear": (
                0.35 * V_rel
                + 0.25 * T_rel
                + 0.18 * VT
                + 0.12 * TT
                + 0.10 * VV
            ),

            # Misalignment: strong vibration + moderate current response
            # + temperature support + progressive trend.
            "misalignment": (
                0.42 * V_rel
                + 0.24 * C_rel
                + 0.14 * T_rel
                + 0.12 * VT
                + 0.08 * CV
            ),

            # Cavitation: unstable vibration / spikes are more important than
            # a smooth vibration drift.
            "cavitation": (
                0.34 * VV
                + 0.34 * SPIKES
                + 0.18 * CV
                + 0.10 * V
                + 0.04 * (1.0 - T)
            ),

            # Motor/electrical: current dominates; temperature supports it.
            "motor_electrical": (
                0.50 * C_rel
                + 0.20 * T_rel
                + 0.16 * CV
                + 0.10 * CT
                + 0.04 * V
            ),

            # Impeller imbalance: vibration-dominant and comparatively low
            # current response; high variability/spikes favour cavitation.
            "impeller_imbalance": (
                0.58 * V_rel
                + 0.20 * VT
                + 0.14 * VV
                + 0.08 * (1.0 - C_rel)
            ),
        }

        # Contradiction penalties make the hypotheses more physically coherent.
        if C_rel > 0.65 and C_rel > V_rel + 0.15:
            scores["bearing_wear"] *= 0.65
            scores["misalignment"] *= 0.70
            scores["impeller_imbalance"] *= 0.55

        if V_rel > 0.55 and C_rel < 0.35:
            scores["motor_electrical"] *= 0.55

        if VV > 0.65 or SPIKES > 0.45:
            scores["impeller_imbalance"] *= 0.65

        if VV < 0.20 and SPIKES < 0.15 and V_rel > 0.45:
            scores["cavitation"] *= 0.65

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_fault, top_score = ranked[0]
        total_score = sum(max(score, 0.0) for _, score in ranked)
        share = (
            top_score / total_score
            if total_score > 1e-9
            else 0.0
        )

        confidence = (
            "HIGH"
            if share >= 0.55
            else "MEDIUM"
            if share >= 0.40
            else "LOW"
        )

        return {
            "status": "HYPOTHESIS_AVAILABLE",
            "most_likely": top_fault,
            "confidence": confidence,
            "support_score": round(float(top_score), 3),
            "alternatives": [
                {
                    "fault": fault,
                    "support": round(float(score), 3),
                }
                for fault, score in ranked[1:3]
            ],
            "note": (
                "Evidence-based hypothesis, not a validated supervised "
                "fault classifier."
            ),
        }


# ---------------- Decision + recommendation ----------------

class DecisionEngine:
    """Gradual severity engine: NORMAL -> WARNING -> CRITICAL."""

    def __init__(self) -> None:
        self.warning_streak = 0
        self.critical_streak = 0

    def reset(self) -> None:
        self.warning_streak = 0
        self.critical_streak = 0

    def evaluate(
        self,
        model_outlier: bool,
        anomaly_index: float,
        dominant_z: float,
    ) -> dict:
        # A single sklearn outlier flag must NOT immediately trigger WARNING or
        # CRITICAL. We require corroborating anomaly percentile or sensor evidence.
        warning_evidence = (
            anomaly_index >= WARNING_ANOMALY_INDEX
            or dominant_z >= WARNING_Z
        )

        if warning_evidence:
            self.warning_streak += 1
        else:
            self.warning_streak = max(
                0,
                self.warning_streak - 1,
            )

        warning_confirmed = (
            self.warning_streak >= WARNING_CONFIRM_COUNT
        )

        # Critical requires BOTH strong anomaly evidence and strong sensor
        # evidence, except an extreme sensor deviation. Even an extreme deviation
        # must persist across two windows before we escalate.
        critical_evidence = (
            anomaly_index >= CRITICAL_ANOMALY_INDEX
            and dominant_z >= CRITICAL_Z
        ) or dominant_z >= EXTREME_Z

        if critical_evidence:
            self.critical_streak += 1
        else:
            self.critical_streak = max(
                0,
                self.critical_streak - 1,
            )

        critical_confirmed = (
            self.critical_streak >= CRITICAL_CONFIRM_COUNT
        )

        if critical_confirmed:
            severity = "CRITICAL"
        elif warning_confirmed:
            severity = "WARNING"
        else:
            severity = "NORMAL"

        return {
            "severity": severity,
            "is_anomaly": severity != "NORMAL",
            "persistent": warning_confirmed,
            "critical": critical_confirmed,
            "diagnosis_allowed": warning_confirmed,
            "warning_streak": self.warning_streak,
            "critical_streak": self.critical_streak,
            "model_outlier": bool(model_outlier),
        }


# ---------------- Recommendation ----------------

# Maintenance guidance keyed by fault hypothesis. Kept separate from
# FaultHypothesisEngine so the diagnosis (evidence) and the recommended
# action (response) can evolve independently.
FAULT_ACTIONS: dict[str, dict[str, str]] = {
    "bearing_wear": {
        "action": "INSPECT_BEARINGS",
        "message": (
            "Elevated vibration with a supporting temperature rise "
            "suggests bearing wear. Schedule a bearing inspection and "
            "check lubrication."
        ),
    },
    "misalignment": {
        "action": "CHECK_ALIGNMENT",
        "message": (
            "Vibration rising together with current draw suggests "
            "shaft/coupling misalignment. Verify shaft alignment at the "
            "next planned stop."
        ),
    },
    "cavitation": {
        "action": "CHECK_SUCTION_CONDITIONS",
        "message": (
            "Unstable, spiky vibration points to cavitation. Check "
            "suction pressure, strainer condition, and NPSH margin."
        ),
    },
    "motor_electrical": {
        "action": "INSPECT_MOTOR_ELECTRICAL",
        "message": (
            "Current draw is the dominant signal, indicating a possible "
            "motor or electrical issue. Inspect motor windings, supply "
            "voltage balance, and drive/VFD settings."
        ),
    },
    "impeller_imbalance": {
        "action": "CHECK_IMPELLER_BALANCE",
        "message": (
            "Steady, vibration-dominant deviation with low current "
            "response suggests impeller imbalance or fouling. Inspect "
            "the impeller for wear, damage, or debris."
        ),
    },
}

# Fallback guidance when no fault hypothesis is available yet, keyed by the
# sensor currently showing the largest deviation.
SENSOR_FALLBACK_ACTIONS: dict[str, dict[str, str]] = {
    "temperature": {
        "action": "MONITOR_TEMPERATURE",
        "message": (
            "Temperature is the dominant deviation. Monitor cooling and "
            "lubrication; a specific fault hypothesis is not yet available."
        ),
    },
    "vibration": {
        "action": "MONITOR_VIBRATION",
        "message": (
            "Vibration is the dominant deviation. Monitor closely; a "
            "specific fault hypothesis is not yet available."
        ),
    },
    "current": {
        "action": "MONITOR_CURRENT",
        "message": (
            "Motor current is the dominant deviation. Monitor closely; a "
            "specific fault hypothesis is not yet available."
        ),
    },
}

SEVERITY_PRIORITY: dict[str, str] = {
    "NORMAL": "NONE",
    "WARNING": "MEDIUM",
    "CRITICAL": "HIGH",
}


def build_recommendation(
    severity: str,
    dominant_sensor: str,
    dominant_z: float,
    diagnosis: Optional[dict],
) -> dict:
    """Turn severity + (optional) fault diagnosis into an actionable
    recommendation for the diagnostics output.

    This intentionally stays a simple rules layer on top of
    FaultHypothesisEngine's output rather than re-deriving evidence itself.
    """

    if severity == "NORMAL":
        return {
            "action": "CONTINUE_MONITORING",
            "priority": "NONE",
            "message": "No persistent anomaly detected. Continue normal monitoring.",
        }

    priority = SEVERITY_PRIORITY.get(severity, "MEDIUM")

    has_hypothesis = (
        diagnosis is not None
        and diagnosis.get("status") == "HYPOTHESIS_AVAILABLE"
    )

    if has_hypothesis:
        fault = diagnosis.get("most_likely")
        confidence = diagnosis.get("confidence", "LOW")
        fault_info = FAULT_ACTIONS.get(fault)

        if fault_info is not None:
            action = fault_info["action"]
            message = fault_info["message"]
        else:
            # Unknown fault key (e.g. FaultHypothesisEngine extended without
            # updating FAULT_ACTIONS) — fall back to the dominant sensor.
            fallback = SENSOR_FALLBACK_ACTIONS.get(
                dominant_sensor,
                SENSOR_FALLBACK_ACTIONS["vibration"],
            )
            action = fallback["action"]
            message = fallback["message"]

        # Escalate priority for CRITICAL + HIGH-confidence hypotheses.
        if severity == "CRITICAL" and confidence == "HIGH":
            priority = "URGENT"

        return {
            "action": action,
            "priority": priority,
            "message": message,
            "based_on": {
                "fault_hypothesis": fault,
                "confidence": confidence,
            },
        }

    # No confirmed hypothesis yet (or diagnosis wasn't allowed): fall back
    # to guidance based purely on which sensor is deviating most.
    fallback = SENSOR_FALLBACK_ACTIONS.get(
        dominant_sensor,
        SENSOR_FALLBACK_ACTIONS["vibration"],
    )

    message = fallback["message"]
    if severity == "CRITICAL":
        message += " Deviation is critical — investigate as soon as possible."

    return {
        "action": fallback["action"],
        "priority": priority,
        "message": message,
        "based_on": {
            "dominant_sensor": dominant_sensor,
            "dominant_z": round(float(dominant_z), 2),
        },
    }


# ---------------- MQTT service ----------------

class PumpMLService:
    def __init__(self) -> None:
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"pump-ml-{ASSET_ID.lower()}",
            protocol=mqtt.MQTTv5,
        )

        self.client.reconnect_delay_set(
            min_delay=1,
            max_delay=60,
        )

        self.client.will_set(
            STATUS_TOPIC,
            json.dumps({
                "service": "pump-ml",
                "asset_id": ASSET_ID,
                "status": "OFFLINE",
                "timestamp": utc_now_iso(),
            }),
            qos=1,
            retain=True,
        )

        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

        self.calibration: list[Sample] = []
        self.window: deque[Sample] = deque(maxlen=WINDOW_SIZE)

        self.baseline: Optional[Baseline] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None

        self.diagnosis_engine = FaultHypothesisEngine()
        self.decision_engine = DecisionEngine()

        self.calibrated = False
        self.last_source_ts: Optional[float] = None
        self.running = True

    def publish_status(self, status: str) -> None:
        payload = {
            "service": "pump-ml",
            "asset_id": ASSET_ID,
            "status": status,
            "timestamp": utc_now_iso(),
        }

        try:
            self.client.publish(
                STATUS_TOPIC,
                json.dumps(payload),
                qos=1,
                retain=True,
            )
        except Exception:
            log.exception("Failed to publish ML status.")

    def on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties,
    ) -> None:
        if reason_code.is_failure:
            log.error("MQTT connection failed: %s", reason_code)
            return

        log.info(
            "Connected to %s:%s",
            MQTT_HOST,
            MQTT_PORT,
        )

        result, _ = client.subscribe(
            INPUT_TOPIC,
            qos=1,
        )

        if result != mqtt.MQTT_ERR_SUCCESS:
            log.error(
                "Subscription failed: %s",
                mqtt.error_string(result),
            )
            return

        log.info("Subscribed to %s", INPUT_TOPIC)

        self.publish_status(
            "CALIBRATING" if not self.calibrated else "ONLINE"
        )

    def on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties,
    ) -> None:
        log.warning("MQTT disconnected: %s", reason_code)

    def parse_payload(self, payload: dict) -> Optional[Sample]:
        if payload.get("asset") != ASSET_ID:
            return None

        measurements = payload.get("measurements")
        if not isinstance(measurements, dict):
            return None

        temperature = finite_float(
            measurements.get("temperature")
        )
        vibration = finite_float(
            measurements.get("vibration")
        )
        current = finite_float(
            measurements.get("current")
        )

        if any(
            value is None
            for value in (
                temperature,
                vibration,
                current,
            )
        ):
            return None

        source_ts = parse_iso_timestamp(
            payload.get("timestamp")
        )

        if source_ts is None:
            log.warning(
                "Ignoring state without a valid source timestamp."
            )
            return None

        return Sample(
            source_ts=source_ts,
            temperature=float(temperature),
            vibration=float(vibration),
            current=float(current),
        )

    def on_message(
        self,
        client,
        userdata,
        message,
    ) -> None:
        if message.topic != INPUT_TOPIC:
            return

        # Ignore retained historical state while creating a new baseline.
        if message.retain and not self.calibrated:
            log.info(
                "Ignoring retained state during calibration."
            )
            return

        try:
            payload = json.loads(
                message.payload.decode("utf-8")
            )

            if not isinstance(payload, dict):
                return

            sample = self.parse_payload(payload)
            if sample is None:
                return

            # One physical source timestamp = one ML sample.
            if (
                self.last_source_ts is not None
                and sample.source_ts <= self.last_source_ts
            ):
                log.debug(
                    "Ignoring duplicate/out-of-order source timestamp."
                )
                return

            self.last_source_ts = sample.source_ts
            self.process_sample(sample)

        except json.JSONDecodeError:
            log.warning("Invalid JSON received on %s", INPUT_TOPIC)
        except Exception:
            log.exception("Unexpected message-processing error.")

    def process_sample(self, sample: Sample) -> None:
        # ---------------- Calibration ----------------
        if not self.calibrated:
            self.calibration.append(sample)

            count = len(self.calibration)
            log.info(
                "Calibration %d/%d",
                count,
                CALIBRATION_SAMPLES,
            )

            if count >= CALIBRATION_SAMPLES:
                self.finish_calibration()

            return

        # ---------------- Inference ----------------
        self.window.append(sample)

        if len(self.window) < WINDOW_SIZE:
            return

        if (
            self.baseline is None
            or self.anomaly_detector is None
        ):
            return

        window = list(self.window)

        features = FeatureEngine.build(window)

        model_outlier, anomaly_index, model_score = (
            self.anomaly_detector.score(features)
        )

        evidence = calculate_evidence(
            window,
            self.baseline,
        )

        decision = self.decision_engine.evaluate(
            model_outlier=model_outlier,
            anomaly_index=anomaly_index,
            dominant_z=float(
                evidence["dominant_z"]
            ),
        )

        # Do NOT diagnose normal operation.
        diagnosis: Optional[dict] = None

        if decision["diagnosis_allowed"]:
            diagnosis = self.diagnosis_engine.diagnose(
                window,
                evidence,
                self.baseline,
            )

        recommendation = build_recommendation(
            severity=decision["severity"],
            dominant_sensor=str(
                evidence["dominant_sensor"]
            ),
            dominant_z=float(
                evidence["dominant_z"]
            ),
            diagnosis=diagnosis,
        )

        output = {
            "asset_id": ASSET_ID,
            "timestamp": utc_now_iso(),
            "is_anomaly": bool(
                decision["is_anomaly"]
            ),
            "anomaly_score": round(
                float(anomaly_index),
                3,
            ),
            "severity": decision["severity"],
            "evidence": {
                "dominant_feature": str(
                    evidence["dominant_sensor"]
                ),
                "z_score_deviation": round(
                    float(evidence["dominant_z"]),
                    2,
                ),
                "z_scores": {
                    key: round(float(value), 2)
                    for key, value in evidence["z_scores"].items()
                },
                "robust_z_scores": {
                    key: round(float(value), 2)
                    for key, value
                    in evidence["robust_z_scores"].items()
                },
                "model_decision_score": round(
                    float(model_score),
                    4,
                ),
            },
            "diagnosis": (
                diagnosis
                if diagnosis is not None
                else {
                    "status": "NO_DIAGNOSIS",
                    "reason": (
                        "No persistent anomaly confirmed."
                    ),
                }
            ),
            "recommendation": recommendation,
        }

        result = self.client.publish(
            OUTPUT_TOPIC,
            json.dumps(
                output,
                separators=(",", ":"),
            ),
            qos=1,
            retain=False,
        )

        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            log.error(
                "Diagnostic publish failed: %s",
                mqtt.error_string(result.rc),
            )

        log.info(
            "severity=%s anomaly=%.3f dominant=%s diagnosis=%s",
            output["severity"],
            output["anomaly_score"],
            output["evidence"]["dominant_feature"],
            output["diagnosis"].get(
                "most_likely",
                "none",
            ),
        )

    def finish_calibration(self) -> None:
        if len(self.calibration) < CALIBRATION_SAMPLES:
            return

        self.baseline = build_baseline(
            self.calibration
        )

        calibration_features: list[
            dict[str, float]
        ] = []

        for end in range(
            WINDOW_SIZE,
            len(self.calibration) + 1,
        ):
            window = self.calibration[
                end - WINDOW_SIZE:end
            ]
            calibration_features.append(
                FeatureEngine.build(window)
            )

        self.anomaly_detector = AnomalyDetector(
            calibration_windows=calibration_features
        )

        self.calibrated = True

        self.window.clear()
        self.window.extend(
            self.calibration[-WINDOW_SIZE:]
        )

        self.decision_engine.reset()

        log.info(
            "======================================================"
        )
        log.info(
            "CALIBRATED: T %.2f±%.2f | V %.2f±%.2f | I %.2f±%.2f",
            self.baseline.mean["temperature"],
            self.baseline.std["temperature"],
            self.baseline.mean["vibration"],
            self.baseline.std["vibration"],
            self.baseline.mean["current"],
            self.baseline.std["current"],
        )
        log.info(
            "Isolation Forest fitted from %d healthy windows.",
            len(calibration_features),
        )
        log.info(
            "======================================================"
        )

        self.publish_status("ONLINE")

    def run(self) -> None:
        log.info("Starting Pump-01 ML service")
        log.info("Input : %s", INPUT_TOPIC)
        log.info("Output: %s", OUTPUT_TOPIC)

        while self.running:
            try:
                self.client.connect(
                    MQTT_HOST,
                    MQTT_PORT,
                    keepalive=60,
                )
                self.client.loop_forever(
                    retry_first_connection=True
                )

            except KeyboardInterrupt:
                break

            except Exception as exc:
                log.error(
                    "Service loop error: %s",
                    exc,
                )
                if self.running:
                    time.sleep(3)

        try:
            self.client.disconnect()
        except Exception:
            pass

        log.info("ML service stopped.")


# ---------------- Signals / entry point ----------------

SERVICE = PumpMLService()


def shutdown(signum, frame) -> None:
    log.info(
        "Shutdown signal received: %s",
        signum,
    )
    SERVICE.running = False


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


if __name__ == "__main__":
    try:
        SERVICE.run()
    except Exception:
        log.exception("Fatal ML service error.")
        sys.exit(1)
