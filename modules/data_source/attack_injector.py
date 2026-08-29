"""
Module 1 (Sub-module): Cyber-Physical Attack Simulator and Injector.
Supports False Data Injection (FDIA), DDoS, Breaker Hijacking, Replay Attacks, and Physical Faults.
"""

import time
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from config import MITRE_ICS_TECHNIQUES

@dataclass
class AttackEvent:
    id: str
    attack_type: str  # "FDIA", "DDOS", "BREAKER_HIJACK", "REPLAY_ATTACK", "PHYSICAL_FAULT"
    target_bus: int
    target_branch: Optional[int] = None
    start_time: float = field(default_factory=time.time)
    duration_seconds: float = 60.0
    intensity: float = 1.0
    mitre_info: Dict[str, str] = field(default_factory=dict)
    is_active: bool = True
    description: str = ""

class AttackInjector:
    """
    Injects realistic cyber-physical attacks into SCADA, PMU, and network streams.
    """
    def __init__(self):
        self.active_attacks: Dict[str, AttackEvent] = {}
        self.replay_buffer: List[Dict[str, Any]] = []

    def inject_attack(self, attack_type: str, target_bus: int, target_branch: Optional[int] = None, intensity: float = 1.0) -> AttackEvent:
        """Launches a cyber or physical attack on the grid simulation."""
        attack_id = f"ATK-{attack_type}-{target_bus}-{int(time.time()*1000)%10000}"
        mitre_info = MITRE_ICS_TECHNIQUES.get(attack_type, {
            "technique_id": "T0800",
            "name": "Custom Cyber-Physical Attack",
            "description": "Adversary action targeting smart grid infrastructure."
        })

        descriptions = {
            "FDIA": f"Stealthy False Data Injection on Bus-{target_bus:02d} perturbing voltage and power measurements.",
            "DDOS": f"DDoS traffic flooding Substation RTU-{target_bus:02d}, causing 85% packet loss and synchrophasor dropout.",
            "BREAKER_HIJACK": f"Unauthorized trip signal issued to Breaker on Branch-{target_branch or target_bus} to disconnect key feeder.",
            "REPLAY_ATTACK": f"Replaying recorded normal telemetry from Bus-{target_bus:02d} to mask active grid overloading.",
            "PHYSICAL_FAULT": f"Phase-to-ground physical short circuit fault on Bus-{target_bus:02d} inducing voltage collapse.",
        }

        event = AttackEvent(
            id=attack_id,
            attack_type=attack_type,
            target_bus=target_bus,
            target_branch=target_branch,
            intensity=intensity,
            mitre_info=mitre_info,
            is_active=True,
            description=descriptions.get(attack_type, "Active Cyber Threat"),
        )
        self.active_attacks[attack_id] = event
        return event

    def clear_all_attacks(self):
        """Clears all active cyber-physical attacks."""
        self.active_attacks.clear()

    def clear_attack(self, attack_id: str):
        """Stops a specific attack."""
        if attack_id in self.active_attacks:
            del self.active_attacks[attack_id]

    def apply_attacks_to_telemetry(self, raw_telemetry: Dict[str, Any], grid_sim) -> Dict[str, Any]:
        """
        Modifies raw grid telemetry stream and grid state in accordance with active attacks.
        """
        telemetry = raw_telemetry.copy()
        if not self.active_attacks:
            return telemetry

        for attack in self.active_attacks.values():
            if not attack.is_active:
                continue

            target_b = attack.target_bus

            # 1. False Data Injection Attack (FDIA)
            if attack.attack_type == "FDIA":
                if target_b in telemetry["buses"]:
                    bus_data = telemetry["buses"][target_b]
                    # Stealthy bias addition
                    bus_data["voltage_pu"] = round(bus_data["voltage_pu"] + 0.18 * attack.intensity, 4)
                    bus_data["active_power_mw"] = round(bus_data["active_power_mw"] * (1.0 + 0.65 * attack.intensity), 3)
                    bus_data["voltage_angle_deg"] = round(bus_data["voltage_angle_deg"] + 15.0 * attack.intensity, 3)
                    telemetry["buses"][target_b]["status"] = "COMPROMISED"
                if target_b in telemetry["pmus"]:
                    pmu = telemetry["pmus"][target_b]
                    pmu["voltage_magnitude_pu"] = round(pmu["voltage_magnitude_pu"] + 0.18 * attack.intensity, 4)
                    pmu["frequency_hz"] = round(pmu["frequency_hz"] + 0.45 * attack.intensity, 4)

            # 2. Denial of Service (DDoS)
            elif attack.attack_type == "DDOS":
                if target_b in telemetry["buses"]:
                    telemetry["buses"][target_b]["rtu_packet_loss_rate"] = 0.85 * attack.intensity
                if target_b in telemetry["pmus"]:
                    telemetry["pmus"][target_b]["sync_lock"] = False
                    telemetry["pmus"][target_b]["rocof_hz_s"] = 0.0

            # 3. Breaker Hijacking / Unauthorized Switching
            elif attack.attack_type == "BREAKER_HIJACK":
                br_id = attack.target_branch or target_b
                if br_id in grid_sim.branches:
                    grid_sim.set_breaker_status(br_id, 0)  # Forcibly open breaker
                    if target_b in telemetry["buses"]:
                        telemetry["buses"][target_b]["status"] = "FAULTED"

            # 4. Replay Attack
            elif attack.attack_type == "REPLAY_ATTACK":
                if target_b in telemetry["buses"]:
                    # Freezes reading to a steady state while real physical load increases
                    telemetry["buses"][target_b]["voltage_pu"] = 1.0100
                    telemetry["buses"][target_b]["active_power_mw"] = 15.000
                    telemetry["buses"][target_b]["status"] = "COMPROMISED"

            # 5. Physical Equipment Fault
            elif attack.attack_type == "PHYSICAL_FAULT":
                if target_b in telemetry["buses"]:
                    telemetry["buses"][target_b]["voltage_pu"] = 0.6500  # Deep sag
                    telemetry["buses"][target_b]["status"] = "FAULTED"
                if target_b in telemetry["pmus"]:
                    telemetry["pmus"][target_b]["frequency_hz"] = 48.95
                    telemetry["pmus"][target_b]["rocof_hz_s"] = -1.25

        return telemetry
