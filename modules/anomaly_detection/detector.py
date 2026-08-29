"""
Module 4: Anomaly Detection Module.
Monitors operational grid bounds, computes multi-variate statistical anomalies,
and evaluates Asset Health Index (AHI).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from config import GRID_CONFIG, CYBER_CONFIG

@dataclass
class AnomalyItem:
    entity_type: str  # "BUS", "BRANCH", "SYSTEM_FREQUENCY"
    entity_id: int
    anomaly_type: str  # "VOLTAGE_VIOLATION", "FREQUENCY_VIOLATION", "THERMAL_OVERLOAD", "GNN_STRUCTURAL_ANOMALY", "DATA_OUTLIER", "PACKET_LOSS"
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    value: float
    threshold: float
    message: str

@dataclass
class AnomalyReport:
    total_anomalies: int
    max_severity: str
    anomalies: List[AnomalyItem]
    bus_health_indices: Dict[int, float]  # bus_id -> 0-100% health
    system_health_score: float  # 0-100%
    timestamp: float

class AnomalyDetector:
    """
    Hybrid physical & machine-learning anomaly detection engine.
    """
    def __init__(self):
        pass

    def detect_anomalies(self, processed_telemetry: Dict[str, Any], gnn_results: Dict[str, Any]) -> AnomalyReport:
        """
        Executes physical boundary checks, GNN anomaly score checks, and computes Asset Health Index.
        """
        anomalies: List[AnomalyItem] = []
        buses = processed_telemetry.get("processed_buses", {})
        branches = processed_telemetry.get("branches", {})
        sys_freq = processed_telemetry.get("frequency_hz", 50.0)
        timestamp = processed_telemetry.get("timestamp", 0.0)
        gnn_scores = gnn_results.get("node_anomaly_scores", {})

        # 1. System Frequency Violations
        freq_dev = abs(sys_freq - GRID_CONFIG["NOMINAL_FREQUENCY_HZ"])
        if sys_freq < GRID_CONFIG["FREQUENCY_CRITICAL_LOW_HZ"] or sys_freq > GRID_CONFIG["FREQUENCY_CRITICAL_HIGH_HZ"]:
            anomalies.append(AnomalyItem(
                entity_type="SYSTEM_FREQUENCY",
                entity_id=0,
                anomaly_type="FREQUENCY_VIOLATION",
                severity="CRITICAL",
                value=sys_freq,
                threshold=GRID_CONFIG["FREQUENCY_CRITICAL_LOW_HZ"],
                message=f"Critical system frequency deviation: {sys_freq:.2f} Hz (Standard: 50.0 Hz)",
            ))
        elif freq_dev > GRID_CONFIG["FREQUENCY_TOLERANCE_HZ"]:
            anomalies.append(AnomalyItem(
                entity_type="SYSTEM_FREQUENCY",
                entity_id=0,
                anomaly_type="FREQUENCY_VIOLATION",
                severity="HIGH",
                value=sys_freq,
                threshold=GRID_CONFIG["FREQUENCY_TOLERANCE_HZ"],
                message=f"Frequency out of tolerance: {sys_freq:.2f} Hz",
            ))

        # 2. Bus Voltage & GNN Anomalies
        bus_health: Dict[int, float] = {}
        for b_id, b_info in buses.items():
            health = 100.0
            v_val = b_info.get("voltage_pu", 1.0)
            v_z = b_info.get("voltage_z_score", 0.0)
            p_z = b_info.get("power_z_score", 0.0)
            pkt_loss = b_info.get("rtu_packet_loss_rate", 0.0)
            gnn_err = gnn_scores.get(b_id, 0.0)

            # Check Critical Voltage
            if v_val < GRID_CONFIG["VOLTAGE_CRITICAL_MIN_PU"] or v_val > GRID_CONFIG["VOLTAGE_CRITICAL_MAX_PU"]:
                health -= 50.0
                anomalies.append(AnomalyItem(
                    entity_type="BUS",
                    entity_id=b_id,
                    anomaly_type="VOLTAGE_VIOLATION",
                    severity="CRITICAL",
                    value=v_val,
                    threshold=GRID_CONFIG["VOLTAGE_CRITICAL_MIN_PU"],
                    message=f"Bus-{b_id:02d} Critical Voltage excursion: {v_val:.3f} pu",
                ))
            elif v_val < GRID_CONFIG["VOLTAGE_MIN_PU"] or v_val > GRID_CONFIG["VOLTAGE_MAX_PU"]:
                health -= 25.0
                anomalies.append(AnomalyItem(
                    entity_type="BUS",
                    entity_id=b_id,
                    anomaly_type="VOLTAGE_VIOLATION",
                    severity="MEDIUM",
                    value=v_val,
                    threshold=GRID_CONFIG["VOLTAGE_MIN_PU"],
                    message=f"Bus-{b_id:02d} Voltage out of standard range: {v_val:.3f} pu",
                ))

            # Check GNN Spatial Structural Anomaly
            if gnn_err > CYBER_CONFIG["GNN_RECONSTRUCTION_ERROR_THRESHOLD"]:
                health -= 35.0
                anomalies.append(AnomalyItem(
                    entity_type="BUS",
                    entity_id=b_id,
                    anomaly_type="GNN_STRUCTURAL_ANOMALY",
                    severity="HIGH",
                    value=gnn_err,
                    threshold=CYBER_CONFIG["GNN_RECONSTRUCTION_ERROR_THRESHOLD"],
                    message=f"Bus-{b_id:02d} GNN Structural Anomaly detected (score: {gnn_err:.3f})",
                ))

            # Check Statistical Outlier
            if v_z > CYBER_CONFIG["ANOMALY_Z_SCORE_THRESHOLD"] or p_z > CYBER_CONFIG["ANOMALY_Z_SCORE_THRESHOLD"]:
                health -= 15.0
                anomalies.append(AnomalyItem(
                    entity_type="BUS",
                    entity_id=b_id,
                    anomaly_type="DATA_OUTLIER",
                    severity="MEDIUM",
                    value=max(v_z, p_z),
                    threshold=CYBER_CONFIG["ANOMALY_Z_SCORE_THRESHOLD"],
                    message=f"Bus-{b_id:02d} Telemetry statistical outlier (Z-score: {max(v_z, p_z):.2f})",
                ))

            # Check SCADA RTU Packet Loss
            if pkt_loss > CYBER_CONFIG["DDOS_PACKET_DROP_RATE_THRESHOLD"]:
                health -= 40.0
                anomalies.append(AnomalyItem(
                    entity_type="BUS",
                    entity_id=b_id,
                    anomaly_type="PACKET_LOSS",
                    severity="HIGH",
                    value=pkt_loss * 100.0,
                    threshold=CYBER_CONFIG["DDOS_PACKET_DROP_RATE_THRESHOLD"] * 100.0,
                    message=f"Bus-{b_id:02d} SCADA RTU experiencing {pkt_loss*100.0:.1f}% packet drop",
                ))

            bus_health[b_id] = max(0.0, min(100.0, health))

        # 3. Branch Thermal Overloads
        for br_id, br_info in branches.items():
            loading = br_info.get("loading_percent", 0.0)
            if loading > GRID_CONFIG["CRITICAL_LINE_LOADING_PERCENT"]:
                anomalies.append(AnomalyItem(
                    entity_type="BRANCH",
                    entity_id=br_id,
                    anomaly_type="THERMAL_OVERLOAD",
                    severity="CRITICAL",
                    value=loading,
                    threshold=GRID_CONFIG["CRITICAL_LINE_LOADING_PERCENT"],
                    message=f"Branch-{br_id} (Bus {br_info['from_bus']} -> Bus {br_info['to_bus']}) Critical Thermal Overload: {loading:.1f}%",
                ))
            elif loading > GRID_CONFIG["MAX_LINE_LOADING_PERCENT"]:
                anomalies.append(AnomalyItem(
                    entity_type="BRANCH",
                    entity_id=br_id,
                    anomaly_type="THERMAL_OVERLOAD",
                    severity="HIGH",
                    value=loading,
                    threshold=GRID_CONFIG["MAX_LINE_LOADING_PERCENT"],
                    message=f"Branch-{br_id} Thermal Overload: {loading:.1f}%",
                ))

        # Calculate overall system health
        if not anomalies:
            system_health = 100.0
        else:
            avg_bus_health = sum(bus_health.values()) / max(len(bus_health), 1)
            system_health = avg_bus_health if not any(a.severity == "CRITICAL" for a in anomalies) else min(avg_bus_health, 45.0)

        # Determine highest severity
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        max_sev = "LOW"
        max_val = 0
        for a in anomalies:
            val = severity_order.get(a.severity, 0)
            if val > max_val:
                max_val = val
                max_sev = a.severity

        return AnomalyReport(
            total_anomalies=len(anomalies),
            max_severity=max_sev if anomalies else "NONE",
            anomalies=anomalies,
            bus_health_indices=bus_health,
            system_health_score=round(system_health, 1),
            timestamp=timestamp,
        )
