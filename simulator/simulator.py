# simulator/simulator.py
# ─────────────────────────────────────────────────────────────────────────────
# IIoT Predictive Maintenance Platform
# Pump Simulator + OPC UA Server
#
# This script:
#   1. Generates stochastic sensor data for Pump-01
#   2. Exposes the sensor data over OPC UA for Node-RED
#   3. Runs one 4-minute health-degradation scenario and then stops
#   4. Randomly selects a hidden fault mechanism for each run
#   5. Stores ground-truth labels locally for ML training/evaluation
#
# IMPORTANT:
# The fault type is deliberately NOT exposed through OPC UA.
# The ML pipeline must diagnose it from sensor behavior, not from a label.
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import csv
import random
from datetime import datetime, timezone
from pathlib import Path

from asyncua import Server, ua


# ─── MACHINE IDENTITY ────────────────────────────────────────────────────────

MACHINE_ID = "Pump-01"
MACHINE_ZONE = "Compressor Station A"


# ─── SIMULATION TIMING ────────────────────────────────────────────────────────
#
# One complete run lasts 4 minutes:
#   0–120 s   : normal operation
#   120–200 s : degradation develops
#   200–240 s : fault becomes imminent
#
# The simulator DOES NOT restart automatically.
# Restarting the script represents maintenance + returning the pump to service.

SAMPLE_INTERVAL_S = 1.0

TOTAL_RUN_TIME_S = 240.0
NORMAL_PHASE_S = 120.0
DEGRADING_PHASE_S = 80.0
FAULT_IMMINENT_PHASE_S = 40.0


# ─── NORMAL OPERATING BEHAVIOR ────────────────────────────────────────────────
#
# These are nominal operating points, not diagnostic labels.
# The noise is intentionally smaller than in the old simulator because we
# separately model correlated process variation below.

NORMAL = {
    "temperature": {
        "base": 65.0,
        "noise": 0.8,
        "min": 45.0,
        "max": 100.0,
    },
    "vibration": {
        "base": 2.5,
        "noise": 0.12,
        "min": 0.2,
        "max": 15.0,
    },
    "current": {
        "base": 14.0,
        "noise": 0.25,
        "min": 5.0,
        "max": 25.0,
    },
}


# ─── FAULT PROFILES ──────────────────────────────────────────────────────────
#
# These profiles describe HOW a hidden fault tends to affect the measured
# signals. They are generator parameters, NOT the ML diagnosis.
#
# sensitivity:
#     How strongly the hidden fault affects the sensor as damage increases.
#
# noise_mult:
#     How much additional variability the fault introduces.
#
# spike_prob:
#     Probability of an abnormal transient/spike.
#
# spike_scale:
#     Maximum size of that transient.
#
# The exact numbers are simulation parameters and should be calibrated against
# real plant data if this simulator is later used for serious benchmarking.

FAULT_PROFILES = {
    "bearing": {
        "description": "Bearing degradation",
        "temperature": {
            "sensitivity": 0.80,
            "noise_mult": 1.25,
            "spike_prob": 0.015,
            "spike_scale": 1.5,
        },
        "vibration": {
            "sensitivity": 2.40,
            "noise_mult": 1.80,
            "spike_prob": 0.10,
            "spike_scale": 1.2,
        },
        "current": {
            "sensitivity": 0.65,
            "noise_mult": 1.15,
            "spike_prob": 0.01,
            "spike_scale": 0.25,
        },
    },

    "misalignment": {
        "description": "Shaft/coupling misalignment",
        "temperature": {
            "sensitivity": 0.45,
            "noise_mult": 1.20,
            "spike_prob": 0.01,
            "spike_scale": 0.8,
        },
        "vibration": {
            "sensitivity": 3.20,
            "noise_mult": 1.90,
            "spike_prob": 0.08,
            "spike_scale": 1.0,
        },
        "current": {
            "sensitivity": 0.95,
            "noise_mult": 1.30,
            "spike_prob": 0.02,
            "spike_scale": 0.35,
        },
    },

    "cavitation": {
        "description": "Hydraulic cavitation",
        "temperature": {
            "sensitivity": 0.15,
            "noise_mult": 1.10,
            "spike_prob": 0.10,
            "spike_scale": 0.6,
        },
        "vibration": {
            "sensitivity": 2.00,
            "noise_mult": 2.50,
            "spike_prob": 0.25,
            "spike_scale": 2.0,
        },
        "current": {
            "sensitivity": 0.35,
            "noise_mult": 1.80,
            "spike_prob": 0.18,
            "spike_scale": 0.8,
        },
    },

    "motor_electrical": {
        "description": "Motor/electrical degradation",
        "temperature": {
            "sensitivity": 1.00,
            "noise_mult": 1.35,
            "spike_prob": 0.03,
            "spike_scale": 1.2,
        },
        "vibration": {
            "sensitivity": 0.55,
            "noise_mult": 1.25,
            "spike_prob": 0.03,
            "spike_scale": 0.5,
        },
        "current": {
            "sensitivity": 2.80,
            "noise_mult": 1.80,
            "spike_prob": 0.08,
            "spike_scale": 0.9,
        },
    },

    "impeller_imbalance": {
        "description": "Impeller imbalance",
        "temperature": {
            "sensitivity": 0.30,
            "noise_mult": 1.15,
            "spike_prob": 0.02,
            "spike_scale": 0.7,
        },
        "vibration": {
            "sensitivity": 3.00,
            "noise_mult": 1.70,
            "spike_prob": 0.07,
            "spike_scale": 1.2,
        },
        "current": {
            "sensitivity": 0.80,
            "noise_mult": 1.25,
            "spike_prob": 0.02,
            "spike_scale": 0.35,
        },
    },
}


# ─── PUMP SIMULATOR ──────────────────────────────────────────────────────────

class PumpSimulator:
    """
    Generates a realistic-looking multivariate sensor stream.

    The sensors are NOT generated independently.

    A slowly varying hidden operating-load component affects all sensors,
    while each sensor also has its own correlated noise and fault response.

    The actual fault type is hidden from OPC UA/Node-RED.
    """

    def __init__(self):
        self.rng = random.Random()

        self.elapsed_s = 0.0
        self.mode = "normal"

        # A different fault is selected for every new simulator run.
        self.fault_type = self.rng.choice(list(FAULT_PROFILES))
        self.fault = FAULT_PROFILES[self.fault_type]

        # Hidden degradation state.
        # This is NOT sent to the ML system.
        self.damage_level = 0.0

        # Slowly changing common operating/load condition.
        self.load_state = 0.0

        # Previous sensor values are used to create temporal correlation.
        self.last_values = {
            name: spec["base"]
            for name, spec in NORMAL.items()
        }

    def phase(self):
        """Return generic machine lifecycle state and phase progress."""

        if self.elapsed_s < NORMAL_PHASE_S:
            return "normal", 0.0

        if self.elapsed_s < NORMAL_PHASE_S + DEGRADING_PHASE_S:
            progress = (
                (self.elapsed_s - NORMAL_PHASE_S)
                / DEGRADING_PHASE_S
            )
            return "degrading", progress

        progress = (
            self.elapsed_s
            - NORMAL_PHASE_S
            - DEGRADING_PHASE_S
        ) / FAULT_IMMINENT_PHASE_S

        return "fault_imminent", min(1.0, progress)

    def update_hidden_damage(self, mode):
        """
        Evolve a hidden degradation state stochastically.

        There is no single fixed linear degradation formula.
        The damage progression itself contains randomness.
        """

        if mode == "normal":
            return

        if mode == "degrading":
            mean_increment = 0.020
        else:
            mean_increment = 0.045

        increment = max(
            0.002,
            self.rng.gauss(
                mean_increment,
                mean_increment * 0.35,
            ),
        )

        self.damage_level = min(
            1.0,
            self.damage_level + increment,
        )

    def update_load(self):
        """
        Slowly vary the operating load.

        Real machines do not produce perfectly constant sensor values.
        The same load variation influences multiple sensors.
        """

        self.load_state = (
            0.90 * self.load_state
            + 0.10 * self.rng.gauss(0.0, 1.0)
        )

    def correlated_noise(self, sensor, sigma):
        """
        Generate temporally correlated sensor noise.

        The current reading retains some relationship to the previous
        reading instead of being an independent random number.
        """

        residual = (
            self.last_values[sensor]
            - NORMAL[sensor]["base"]
        )

        return (
            0.82 * residual
            + 0.57 * self.rng.gauss(0.0, sigma)
        )

    def get_sensor_values(self):
        """Generate one new multivariate sensor sample."""

        self.mode, _ = self.phase()

        # Update hidden physical state first.
        self.update_hidden_damage(self.mode)
        self.update_load()

        values = {}

        # Common operating-load influence.
        load_effect = {
            "temperature": 0.8,
            "vibration": 0.12,
            "current": 0.9,
        }

        for sensor, spec in NORMAL.items():

            if self.mode == "normal":
                profile = None
                severity = 0.0
            else:
                profile = self.fault[sensor]
                severity = self.damage_level

            # Fault changes the statistical behavior progressively.
            if profile is None:
                noise_mult = 1.0
                fault_effect = 0.0
                spike_prob = 0.0
                spike_scale = 0.0
            else:
                noise_mult = (
                    1.0
                    + (profile["noise_mult"] - 1.0)
                    * severity
                )

                fault_effect = (
                    profile["sensitivity"]
                    * severity
                )

                spike_prob = (
                    profile["spike_prob"]
                    * severity
                )

                spike_scale = (
                    profile["spike_scale"]
                    * severity
                )

            # Normal operating variability + temporal correlation.
            noise = self.correlated_noise(
                sensor,
                spec["noise"] * noise_mult,
            )

            # Occasional transient abnormal behavior.
            spike = (
                self.rng.uniform(0.0, spike_scale)
                if self.rng.random() < spike_prob
                else 0.0
            )

            value = (
                spec["base"]
                + load_effect[sensor] * self.load_state
                + fault_effect
                + noise
                + spike
            )

            # Physical sanity limits for this simplified model.
            value = max(
                spec["min"],
                min(spec["max"], value),
            )

            values[sensor] = round(value, 2)

        self.last_values = values
        return values

    def step(self):
        self.elapsed_s += SAMPLE_INTERVAL_S

    def finished(self):
        return self.elapsed_s >= TOTAL_RUN_TIME_S


# ─── GROUND-TRUTH LOG ────────────────────────────────────────────────────────
#
# This file is deliberately separate from OPC UA/MQTT.
#
# The ML model must NOT receive:
#     bearing / cavitation / misalignment / ...
#
# Otherwise the simulator would be giving the answer to the model.
#
# The CSV is useful later for:
#   - supervised model training
#   - evaluating diagnosis accuracy
#   - comparing ML prediction vs actual simulated fault

def append_ground_truth(timestamp, elapsed_s, mode, fault_type, values):
    path = Path("simulation_ground_truth.csv")
    new_file = not path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        if new_file:
            writer.writerow([
                "timestamp",
                "elapsed_s",
                "mode",
                "fault_type",
                "temperature",
                "vibration",
                "current",
            ])

        writer.writerow([
            timestamp,
            round(elapsed_s, 1),
            mode,
            fault_type,
            values["temperature"],
            values["vibration"],
            values["current"],
        ])


# ─── OPC UA SERVER ───────────────────────────────────────────────────────────

async def run_opcua_server():
    """
    Creates the OPC UA server and publishes the simulated sensor stream.

    Node-RED continues to consume these variables exactly as before.
    """

    simulator = PumpSimulator()

    server = Server()
    await server.init()

    server.set_endpoint(
        "opc.tcp://0.0.0.0:4840/freeopcua/server/"
    )

    server.set_server_name(
        f"IIoT Simulator - {MACHINE_ID}"
    )

    # Namespace
    uri = "http://iiot.factory.com"
    idx = await server.register_namespace(uri)

    # Machine object
    pump = await server.nodes.objects.add_object(
        idx,
        MACHINE_ID,
    )

    # Sensor nodes
    temp_node = await pump.add_variable(
        ua.NodeId("Pump-01.Temperature", idx),
        "Temperature",
        0.0,
    )

    vibration_node = await pump.add_variable(
        ua.NodeId("Pump-01.Vibration", idx),
        "Vibration",
        0.0,
    )

    current_node = await pump.add_variable(
        ua.NodeId("Pump-01.Current", idx),
        "Current",
        0.0,
    )

    # Generic lifecycle state.
    # This is NOT the fault diagnosis.
    mode_node = await pump.add_variable(
        ua.NodeId("Pump-01.Mode", idx),
        "Mode",
        "normal",
    )

    # Allow writes
    await temp_node.set_writable()
    await vibration_node.set_writable()
    await current_node.set_writable()
    await mode_node.set_writable()

    print()
    print("=" * 70)
    print("  IIoT PUMP SIMULATOR + OPC UA")
    print("=" * 70)
    print(f"  Machine        : {MACHINE_ID}")
    print(f"  Zone           : {MACHINE_ZONE}")
    print("  Endpoint       : opc.tcp://localhost:4840")
    print("  Sensors        : Temperature | Vibration | Current")
    print("  Run duration   : 4 minutes")
    print("  Restart        : MANUAL")
    print()
    print(
        f"  Hidden fault selected for this run: "
        f"{simulator.fault_type}"
    )
    print(
        "  NOTE: fault type is NOT exposed through OPC UA/MQTT."
    )
    print(
        "  Ground truth is saved locally in "
        "simulation_ground_truth.csv"
    )
    print("=" * 70)
    print()

    print(
        f"{'CYCLE':<8}"
        f"{'MODE':<17}"
        f"{'TEMP':>10}"
        f"{'VIB':>12}"
        f"{'CURRENT':>12}"
    )
    print("-" * 65)

    async with server:

        cycle = 0

        while not simulator.finished():

            cycle += 1

            values = simulator.get_sensor_values()

            timestamp = datetime.now(
                timezone.utc
            ).isoformat()

            # Publish the actual sensor stream.
            await temp_node.write_value(
                values["temperature"]
            )

            await vibration_node.write_value(
                values["vibration"]
            )

            await current_node.write_value(
                values["current"]
            )

            await mode_node.write_value(
                simulator.mode
            )

            # Save ground truth ONLY for offline ML training/evaluation.
            append_ground_truth(
                timestamp,
                simulator.elapsed_s,
                simulator.mode,
                simulator.fault_type,
                values,
            )

            print(
                f"{cycle:<8}"
                f"{simulator.mode:<17}"
                f"{values['temperature']:>7.2f} C"
                f"{values['vibration']:>9.2f} mm/s"
                f"{values['current']:>9.2f} A"
            )

            simulator.step()

            await asyncio.sleep(
                SAMPLE_INTERVAL_S
            )

    print()
    print("=" * 70)
    print("  SIMULATION COMPLETE")
    print("  Fault-imminent stage reached.")
    print("  Pump stopped.")
    print()
    print(
        "  Perform the virtual maintenance intervention, "
        "then restart simulator.py manually."
    )
    print("=" * 70)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_opcua_server())
