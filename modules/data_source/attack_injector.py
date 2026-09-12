"""
Module 1 (Sub-module): Cyber-Physical Attack Simulator and Injector.
Supports False Data Injection (FDIA), DDoS, Breaker Hijacking, Replay Attacks, and Physical Faults.
"""

import time
import math
import copy
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
        Dynamically modulates frequency, voltages, packet loss, and power restoration across
        all concurrent active attacks.
        """
        telemetry = copy.deepcopy(raw_telemetry)
        if not self.active_attacks:
            return telemetry

        cur_t = time.time()
        base_freq = 50.0 + random.gauss(0, 0.003)
        total_f_delta = 0.0

        for attack in self.active_attacks.values():
            if not attack.is_active:
                continue

            target_b = attack.target_bus
            intensity = max(0.2, attack.intensity)

            # 1. False Data Injection Attack (FDIA)
            if attack.attack_type == "FDIA":
                # Dynamic sinusoidal oscillation on spoofed measurements with unique bus phase
                v_drift = (0.16 + 0.04 * math.sin(cur_t * 2.5 + target_b)) * intensity
                p_factor = 1.0 + (0.50 + 0.15 * math.cos(cur_t * 2.0 + target_b)) * intensity
                
                if target_b in telemetry["buses"]:
                    bus_data = telemetry["buses"][target_b]
                    bus_data["voltage_pu"] = round(bus_data["voltage_pu"] + v_drift, 4)
                    bus_data["active_power_mw"] = round(bus_data["active_power_mw"] * p_factor, 3)
                    bus_data["voltage_angle_deg"] = round(bus_data["voltage_angle_deg"] + 14.0 * intensity, 3)
                    bus_data["status"] = "COMPROMISED"
                
                if target_b in telemetry["pmus"]:
                    pmu = telemetry["pmus"][target_b]
                    pmu["voltage_magnitude_pu"] = round(pmu["voltage_magnitude_pu"] + v_drift, 4)
                    pmu["frequency_hz"] = round(50.0 + (0.35 + 0.10 * math.sin(cur_t * 3.0 + target_b)) * intensity, 4)
                
                # Compound frequency deviation
                total_f_delta -= (0.045 + 0.025 * math.sin(cur_t * 2.0 + target_b)) * intensity + random.gauss(0, 0.004)

            # 2. Denial of Service (DDoS)
            elif attack.attack_type == "DDOS":
                loss_rate = min(1.0, (0.80 + 0.10 * math.sin(cur_t * 4.0 + target_b)) * intensity)
                if target_b in telemetry["buses"]:
                    telemetry["buses"][target_b]["rtu_packet_loss_rate"] = round(loss_rate, 3)
                    telemetry["buses"][target_b]["status"] = "DEGRADED"
                
                if target_b in telemetry["pmus"]:
                    telemetry["pmus"][target_b]["sync_lock"] = False
                    telemetry["pmus"][target_b]["rocof_hz_s"] = round(random.gauss(0, 0.15) * intensity, 4)
                
                # Frequency estimation noise surges during DDoS flood
                total_f_delta += (0.08 * math.sin(cur_t * 4.5 + target_b) * intensity) + random.gauss(0, 0.015)

            # 3. Breaker Hijacking / Unauthorized Switching
            elif attack.attack_type == "BREAKER_HIJACK":
                br_id = attack.target_branch or target_b
                if br_id in grid_sim.branches:
                    grid_sim.set_breaker_status(br_id, 0)  # Forcibly open breaker
                    if target_b in telemetry["buses"]:
                        telemetry["buses"][target_b]["status"] = "FAULTED"
                        telemetry["buses"][target_b]["voltage_pu"] = round(max(0.0, 0.35 - 0.15 * intensity), 4)
                        telemetry["buses"][target_b]["active_power_mw"] = 0.0
                
                # Disconnection sags system frequency
                total_f_delta -= (0.35 * intensity + 0.04 * math.sin(cur_t * 1.5 + target_b)) + random.gauss(0, 0.01)

            # 4. Replay Attack
            elif attack.attack_type == "REPLAY_ATTACK":
                if target_b in telemetry["buses"]:
                    telemetry["buses"][target_b]["voltage_pu"] = 1.0100
                    telemetry["buses"][target_b]["active_power_mw"] = 15.000
                    telemetry["buses"][target_b]["status"] = "COMPROMISED"
                
                total_f_delta -= (0.15 * intensity + 0.03 * math.cos(cur_t * 1.8 + target_b))

            # 5. Physical Equipment Fault
            elif attack.attack_type == "PHYSICAL_FAULT":
                if target_b in telemetry["buses"]:
                    telemetry["buses"][target_b]["voltage_pu"] = round(max(0.05, 0.50 - 0.15 * intensity), 4)
                    telemetry["buses"][target_b]["status"] = "FAULTED"
                    telemetry["buses"][target_b]["active_power_mw"] = 0.0
                
                if target_b in telemetry["pmus"]:
                    telemetry["pmus"][target_b]["frequency_hz"] = round(48.80 - 0.35 * intensity, 4)
                    telemetry["pmus"][target_b]["rocof_hz_s"] = round(-1.50 * intensity, 4)
                
                # Deep system frequency sag under short circuit
                total_f_delta -= (0.95 * intensity + 0.06 * math.sin(cur_t * 2.5 + target_b)) + random.gauss(0, 0.02)

        # Set compound system frequency
        telemetry["frequency_hz"] = round(max(45.0, min(55.0, base_freq + total_f_delta)), 3)

        # Dynamically calculate actual served load and power restoration ratio
        total_demand = sum(b.get("load_p_mw", 0.0) for b in raw_telemetry.get("buses", {}).values())
        served_load = 0.0
        for b_id, b in telemetry.get("buses", {}).items():
            b_load = b.get("load_p_mw", 0.0)
            v_pu = b.get("voltage_pu", 1.0)
            st = b.get("status", "ENERGIZED")
            if st in ["FAULTED", "ISOLATED"]:
                # Zero power served on isolated or faulted nodes
                served_load += 0.0
            elif st == "DEGRADED":
                # Partial power under brownout / communication degradation
                served_load += b_load * max(0.5, min(1.0, v_pu))
            elif st == "COMPROMISED":
                # Compromised bus delivery
                served_load += b_load * max(0.65, min(1.0, v_pu))
            else:
                # Normal or Restored bus
                served_load += b_load * (1.0 if v_pu >= 0.90 else max(0.0, v_pu / 0.90))

        restoration_ratio = (served_load / max(total_demand, 1e-3)) * 100.0

        telemetry["total_demand_mw"] = round(total_demand, 2)
        telemetry["total_load_mw"] = round(served_load, 2)
        telemetry["restoration_ratio_pct"] = round(max(0.0, min(100.0, restoration_ratio)), 1)
        telemetry["system_status"] = "NORMAL" if restoration_ratio > 98.0 and abs(telemetry["frequency_hz"] - 50.0) < 0.2 else "DEGRADED"

        return telemetry
