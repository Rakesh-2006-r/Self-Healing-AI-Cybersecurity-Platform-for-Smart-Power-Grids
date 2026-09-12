"""
Module 5: Cybersecurity Agent - Threat Classifier & MITRE ICS Engine.
Distinguishes cyber intrusions from physical faults and generates tactical cyber threat alerts.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from config import MITRE_ICS_TECHNIQUES, CYBER_CONFIG

@dataclass
class CyberThreatAlert:
    alert_id: str
    threat_category: str  # "CYBER_ATTACK", "PHYSICAL_FAULT", "EQUIPMENT_DEGRADATION"
    attack_type: str  # "FDIA", "DDOS", "BREAKER_HIJACK", "REPLAY_ATTACK", "PHYSICAL_FAULT", "NORMAL"
    target_bus_id: int
    target_branch_id: Optional[int]
    threat_severity_index: float  # 0 to 100
    severity_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    mitre_technique_id: str
    mitre_technique_name: str
    mitre_description: str
    detected_at: float
    confidence_score: float  # 0.0 to 1.0
    evidence: List[str]
    suggested_cyber_response: str

class CybersecurityAgent:
    """
    Intelligent Cybersecurity Agent that analyzes anomalies, GNN features, and telemetry
    to classify threats and map them to MITRE ATT&CK for Industrial Control Systems (ICS).
    """
    def __init__(self):
        pass

    def analyze_threats(
        self,
        anomaly_report,
        processed_telemetry: Dict[str, Any],
        gnn_results: Dict[str, Any],
        raw_branches: Dict[int, Any] = None
    ) -> List[CyberThreatAlert]:
        """
        Synthesizes multi-sensor signals to classify cyber vs physical threats.
        """
        alerts: List[CyberThreatAlert] = []
        buses = processed_telemetry.get("processed_buses", {})
        pmus = processed_telemetry.get("pmus", {})
        gnn_scores = gnn_results.get("node_anomaly_scores", {})
        timestamp = processed_telemetry.get("timestamp", time.time())

        # 1. First check for Breaker Hijacking / Unauthorized Open Switch
        if raw_branches:
            for br_id, br in raw_branches.items():
                if br.status == 0 and not getattr(br, "is_tie_switch", False):
                    mitre = MITRE_ICS_TECHNIQUES["BREAKER_HIJACK"]
                    alerts.append(CyberThreatAlert(
                        alert_id=f"CYBER-HIJACK-BR-{br_id}",
                        threat_category="CYBER_ATTACK",
                        attack_type="BREAKER_HIJACK",
                        target_bus_id=br.from_bus,
                        target_branch_id=br_id,
                        threat_severity_index=92.0,
                        severity_level="CRITICAL",
                        mitre_technique_id=mitre["technique_id"],
                        mitre_technique_name=mitre["name"],
                        mitre_description=mitre["description"],
                        detected_at=timestamp,
                        confidence_score=0.95,
                        evidence=[
                            f"Breaker on Branch-{br_id} (Bus {br.from_bus} -> Bus {br.to_bus}) opened without operator dispatch ticket",
                            "Sudden line power disruption",
                        ],
                        suggested_cyber_response="Revoke unauthorized SCADA command credentials; execute automated feeder tie-switch power restoration."
                    ))

        # 2. Scan each bus for intrusion patterns
        for b_id, b_info in buses.items():
            v_val = b_info.get("voltage_pu", 1.0)
            v_z = b_info.get("voltage_z_score", 0.0)
            p_z = b_info.get("power_z_score", 0.0)
            loss_rate = b_info.get("rtu_packet_loss_rate", 0.0)
            gnn_err = gnn_scores.get(b_id, 0.0)
            has_pmu = b_id in pmus
            pmu_sync = pmus[b_id].get("sync_lock", True) if has_pmu else True

            # 2a. Check for DDoS Flooding Attack on SCADA RTU
            if loss_rate > CYBER_CONFIG["DDOS_PACKET_DROP_RATE_THRESHOLD"] or not pmu_sync:
                mitre = MITRE_ICS_TECHNIQUES["DDOS"]
                alerts.append(CyberThreatAlert(
                    alert_id=f"CYBER-DDOS-BUS-{b_id:02d}",
                    threat_category="CYBER_ATTACK",
                    attack_type="DDOS",
                    target_bus_id=b_id,
                    target_branch_id=None,
                    threat_severity_index=88.0,
                    severity_level="HIGH",
                    mitre_technique_id=mitre["technique_id"],
                    mitre_technique_name=mitre["name"],
                    mitre_description=mitre["description"],
                    detected_at=timestamp,
                    confidence_score=0.94,
                    evidence=[
                        f"RTU telemetry packet drop rate: {loss_rate*100.0:.1f}%",
                        f"PMU Synchrophasor sync lock lost: {not pmu_sync}",
                    ],
                    suggested_cyber_response="Engage SCADA Firewall rate limiting, switch to secondary fiber communication channel."
                ))

            # 2b. Check for Replay attacks (Telemetry steady/nominal while flagged or discordance with neighboring dynamics)
            elif b_info.get("status") == "COMPROMISED" and (0.98 <= v_val <= 1.04):
                mitre = MITRE_ICS_TECHNIQUES["REPLAY_ATTACK"]
                alerts.append(CyberThreatAlert(
                    alert_id=f"CYBER-REPLAY-BUS-{b_id:02d}",
                    threat_category="CYBER_ATTACK",
                    attack_type="REPLAY_ATTACK",
                    target_bus_id=b_id,
                    target_branch_id=None,
                    threat_severity_index=82.0,
                    severity_level="HIGH",
                    mitre_technique_id=mitre["technique_id"],
                    mitre_technique_name=mitre["name"],
                    mitre_description=mitre["description"],
                    detected_at=timestamp,
                    confidence_score=0.88,
                    evidence=[
                        "Telemetry channel marked compromised while measurements remain implausibly steady",
                        "Live state must be cross-validated against trusted PMU data",
                    ],
                    suggested_cyber_response="Quarantine the replayed telemetry stream, re-authenticate the RTU, and re-estimate state from trusted measurements."
                ))

            # 2c. Check for Physical Deep Voltage Sag / Equipment Short Circuit
            elif v_val < 0.75:
                mitre = MITRE_ICS_TECHNIQUES["PHYSICAL_FAULT"]
                alerts.append(CyberThreatAlert(
                    alert_id=f"FAULT-PHYS-BUS-{b_id:02d}",
                    threat_category="PHYSICAL_FAULT",
                    attack_type="PHYSICAL_FAULT",
                    target_bus_id=b_id,
                    target_branch_id=None,
                    threat_severity_index=90.0,
                    severity_level="CRITICAL",
                    mitre_technique_id=mitre["technique_id"],
                    mitre_technique_name=mitre["name"],
                    mitre_description=mitre["description"],
                    detected_at=timestamp,
                    confidence_score=0.92,
                    evidence=[
                        f"Severe voltage sag: {v_val:.3f} pu",
                        "High current fault signature detected across bus feeder",
                    ],
                    suggested_cyber_response="Isolate faulted bus section; reconfigure downstream tie-switches."
                ))

            # 2d. Check for False Data Injection Attack (FDIA)
            elif b_info.get("status") == "COMPROMISED" or ((gnn_err > CYBER_CONFIG["GNN_RECONSTRUCTION_ERROR_THRESHOLD"] or v_z > CYBER_CONFIG["ANOMALY_Z_SCORE_THRESHOLD"]) and (v_val > 1.10 or (0.75 <= v_val < 0.90) or p_z > 2.0)):
                mitre = MITRE_ICS_TECHNIQUES["FDIA"]
                alerts.append(CyberThreatAlert(
                    alert_id=f"CYBER-FDIA-BUS-{b_id:02d}",
                    threat_category="CYBER_ATTACK",
                    attack_type="FDIA",
                    target_bus_id=b_id,
                    target_branch_id=None,
                    threat_severity_index=94.0,
                    severity_level="CRITICAL",
                    mitre_technique_id=mitre["technique_id"],
                    mitre_technique_name=mitre["name"],
                    mitre_description=mitre["description"],
                    detected_at=timestamp,
                    confidence_score=0.96,
                    evidence=[
                        f"GNN Graph reconstruction error: {gnn_err:.3f} (Threshold: {CYBER_CONFIG['GNN_RECONSTRUCTION_ERROR_THRESHOLD']})",
                        f"Artificial voltage injection: {v_val:.3f} pu (Z-score: {v_z:.2f})",
                        "Physical grid frequency invariant with reported voltage spike",
                    ],
                    suggested_cyber_response="Quarantine corrupted SCADA/PMU sensor stream; trigger Weighted Least Squares bad data state re-estimation."
                ))

        return alerts
