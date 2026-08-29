"""
Module 2: Data Collection & Preprocessing - Telemetry Processor.
Performs data cleaning, sliding window feature extraction, and statistical normalization.
"""

import collections
import numpy as np
from typing import Dict, Any, List
from config import GRID_CONFIG, CYBER_CONFIG

class TelemetryProcessor:
    """
    Ingests synchronized telemetry from SCADA, PMUs, and Smart Meters.
    Computes rolling metrics, Z-score deviations, and data quality indicators.
    """
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        # Rolling histories for bus voltage and active power: bus_id -> deque
        self.voltage_history: Dict[int, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=window_size))
        self.power_history: Dict[int, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=window_size))
        self.frequency_history = collections.deque(maxlen=window_size)

    def process_telemetry_stream(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cleans telemetry, extracts statistical deviations, and computes baseline Z-scores.
        """
        processed_buses = {}
        timestamp = telemetry.get("timestamp", 0.0)
        sys_freq = telemetry.get("frequency_hz", 50.0)
        self.frequency_history.append(sys_freq)

        for b_id, b_raw in telemetry.get("buses", {}).items():
            b_id = int(b_id)
            v_val = b_raw.get("voltage_pu", 1.0)
            p_val = b_raw.get("active_power_mw", 0.0)

            # Update rolling buffers
            self.voltage_history[b_id].append(v_val)
            self.power_history[b_id].append(p_val)

            v_arr = np.array(self.voltage_history[b_id])
            p_arr = np.array(self.power_history[b_id])

            v_mean = float(np.mean(v_arr))
            v_std = float(np.std(v_arr)) if len(v_arr) > 3 and np.std(v_arr) > 1e-4 else 0.01
            v_z_score = abs(v_val - v_mean) / v_std

            p_mean = float(np.mean(p_arr))
            p_std = float(np.std(p_arr)) if len(p_arr) > 3 and np.std(p_arr) > 1e-4 else 0.1
            p_z_score = abs(p_val - p_mean) / p_std

            # Packet drop rate check (SCADA RTU health)
            loss_rate = b_raw.get("rtu_packet_loss_rate", 0.0)

            processed_buses[b_id] = {
                "bus_id": b_id,
                "bus_name": b_raw.get("bus_name", f"Bus-{b_id:02d}"),
                "status": b_raw.get("status", "ENERGIZED"),
                "scada_rtu_id": b_raw.get("scada_rtu_id", f"RTU-SUB{b_id:02d}"),
                "voltage_pu": v_val,
                "voltage_mean": round(v_mean, 4),
                "voltage_std": round(v_std, 4),
                "voltage_z_score": round(float(v_z_score), 3),
                "voltage_angle_deg": b_raw.get("voltage_angle_deg", 0.0),
                "active_power_mw": p_val,
                "power_mean": round(p_mean, 3),
                "power_z_score": round(float(p_z_score), 3),
                "reactive_power_mvar": b_raw.get("reactive_power_mvar", 0.0),
                "rtu_packet_loss_rate": loss_rate,
                "is_outlier": bool(v_z_score > CYBER_CONFIG["ANOMALY_Z_SCORE_THRESHOLD"] or p_z_score > CYBER_CONFIG["ANOMALY_Z_SCORE_THRESHOLD"]),
            }

        # Calculate system-wide frequency statistics
        freq_arr = np.array(self.frequency_history)
        freq_mean = float(np.mean(freq_arr))
        freq_dev = abs(sys_freq - GRID_CONFIG["NOMINAL_FREQUENCY_HZ"])

        return {
            "timestamp": timestamp,
            "frequency_hz": sys_freq,
            "frequency_deviation_hz": round(freq_dev, 4),
            "processed_buses": processed_buses,
            "branches": telemetry.get("branches", {}),
            "pmus": telemetry.get("pmus", {}),
            "smart_meters": telemetry.get("smart_meters", {}),
            "restoration_ratio_pct": telemetry.get("restoration_ratio_pct", 100.0),
        }
