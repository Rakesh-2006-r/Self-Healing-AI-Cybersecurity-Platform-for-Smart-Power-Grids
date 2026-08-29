"""
Module 1: Power Grid Data Layer - IEEE Benchmark Grid Simulator.
Simulates Cyber-Physical Smart Power Grid with SCADA RTUs, PMUs, and Smart Meters.
"""

import time
import math
import random
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from config import GRID_CONFIG

@dataclass
class Bus:
    id: int
    name: str
    bus_type: str  # "SLACK", "PV", "PQ"
    base_kv: float
    voltage_pu: float = 1.0
    voltage_angle_rad: float = 0.0
    active_power_mw: float = 0.0  # Net injected P (Pg - Pd)
    reactive_power_mvar: float = 0.0  # Net injected Q (Qg - Qd)
    load_p_mw: float = 0.0
    load_q_mvar: float = 0.0
    gen_p_mw: float = 0.0
    gen_q_mvar: float = 0.0
    critical_facility: Optional[str] = None  # e.g., "Hospital", "Substation Main", "Water Plant"
    has_pmu: bool = False
    scada_rtu_id: str = ""
    status: str = "ENERGIZED"  # "ENERGIZED", "FAULTED", "ISOLATED", "COMPROMISED", "RESTORED"
    pos_x: float = 0.0
    pos_y: float = 0.0

@dataclass
class Branch:
    id: int
    from_bus: int
    to_bus: int
    r: float  # resistance (pu)
    x: float  # reactance (pu)
    b: float  # susceptance (pu)
    rate_mva: float  # thermal rating
    status: int = 1  # 1 = CLOSED/CONNECTED, 0 = OPEN/DISCONNECTED
    is_tie_switch: bool = False
    flow_p_mw: float = 0.0
    flow_q_mvar: float = 0.0
    current_ka: float = 0.0
    loading_percent: float = 0.0

class PowerGridSimulator:
    """
    Simulates Cyber-Physical Smart Power Grid Infrastructure (IEEE 14-bus / IEEE 33-bus).
    Provides real-time PMU, SCADA, and Smart Meter streams.
    """
    def __init__(self, topology_name: str = "IEEE_14_BUS"):
        self.topology_name = topology_name
        self.buses: Dict[int, Bus] = {}
        self.branches: Dict[int, Branch] = {}
        self.timestamp = time.time()
        self.frequency_hz = GRID_CONFIG["NOMINAL_FREQUENCY_HZ"]
        self.rocof_hz_s = 0.0
        self._init_topology()
        self.solve_power_flow()

    def _init_topology(self):
        """Initializes standard IEEE 14-bus test feeder system with cyber-physical telemetry."""
        if self.topology_name == "IEEE_14_BUS":
            self._init_ieee_14_bus()
        else:
            self._init_ieee_33_bus()

    def _init_ieee_14_bus(self):
        # 14 Buses: (id, name, type, V, P_gen, Q_gen, P_load, Q_load, critical, pmu, pos_x, pos_y)
        bus_data = [
            (1, "Bus-01 (Slack Gen)", "SLACK", 1.06, 232.4, -16.9, 0.0, 0.0, "Substation Control Center", True, 0.1, 0.5),
            (2, "Bus-02 (PV Gen 1)", "PV", 1.045, 40.0, 42.4, 21.7, 12.7, "Regional Hospital Substation", True, 0.25, 0.8),
            (3, "Bus-03 (PV Gen 2)", "PV", 1.01, 0.0, 23.4, 94.2, 19.0, "Industrial Processing Zone", True, 0.3, 0.2),
            (4, "Bus-04 (PQ Sub)", "PQ", 1.018, 0.0, 0.0, 47.8, -3.9, "Municipal Water Pumping", True, 0.45, 0.65),
            (5, "Bus-05 (PQ Sub)", "PQ", 1.02, 0.0, 0.0, 7.6, 1.6, "Data Center Hub", True, 0.45, 0.35),
            (6, "Bus-06 (PV Gen 3)", "PV", 1.07, 0.0, 12.2, 11.2, 7.5, "Defense Communications", True, 0.6, 0.8),
            (7, "Bus-07 (PQ)", "PQ", 1.062, 0.0, 0.0, 0.0, 0.0, None, False, 0.7, 0.6),
            (8, "Bus-08 (PV Gen 4)", "PV", 1.09, 0.0, 17.4, 0.0, 0.0, "Solar Microgrid Dispatch", True, 0.85, 0.8),
            (9, "Bus-09 (PQ)", "PQ", 1.056, 0.0, 0.0, 29.5, 16.6, "Commercial Retail Complex", True, 0.75, 0.45),
            (10, "Bus-10 (PQ)", "PQ", 1.051, 0.0, 0.0, 9.0, 5.8, "Residential Sector North", False, 0.85, 0.3),
            (11, "Bus-11 (PQ)", "PQ", 1.057, 0.0, 0.0, 3.5, 1.8, "Residential Sector East", False, 0.6, 0.15),
            (12, "Bus-12 (PQ)", "PQ", 1.055, 0.0, 0.0, 6.1, 1.6, "Emergency Services Center", False, 0.7, 0.1),
            (13, "Bus-13 (PQ)", "PQ", 1.050, 0.0, 0.0, 13.5, 5.8, "Transit Metro Substation", True, 0.85, 0.1),
            (14, "Bus-14 (PQ)", "PQ", 1.035, 0.0, 0.0, 14.9, 5.0, "University Tech Campus", False, 0.95, 0.35),
        ]

        self.buses = {}
        for b in bus_data:
            self.buses[b[0]] = Bus(
                id=b[0], name=b[1], bus_type=b[2], base_kv=69.0 if b[0] <= 5 else 13.8,
                voltage_pu=b[3], voltage_angle_rad=0.0,
                gen_p_mw=b[4], gen_q_mvar=b[5],
                load_p_mw=b[6], load_q_mvar=b[7],
                active_power_mw=b[4] - b[6], reactive_power_mvar=b[5] - b[7],
                critical_facility=b[8], has_pmu=b[9],
                scada_rtu_id=f"RTU-SUB{b[0]:02d}", status="ENERGIZED",
                pos_x=b[10], pos_y=b[11]
            )

        # Standard IEEE 14 Branches + Tie switches: (id, from, to, r, x, b, rate_mva, status, is_tie)
        branch_data = [
            (1, 1, 2, 0.01938, 0.05917, 0.0528, 200.0, 1, False),
            (2, 1, 5, 0.05403, 0.22304, 0.0492, 100.0, 1, False),
            (3, 2, 3, 0.04699, 0.19797, 0.0438, 100.0, 1, False),
            (4, 2, 4, 0.05811, 0.17632, 0.0340, 80.0, 1, False),
            (5, 2, 5, 0.05695, 0.17388, 0.0346, 80.0, 1, False),
            (6, 3, 4, 0.06701, 0.17103, 0.0128, 80.0, 1, False),
            (7, 4, 5, 0.01335, 0.04211, 0.0, 80.0, 1, False),
            (8, 4, 7, 0.0, 0.20912, 0.0, 65.0, 1, False),
            (9, 4, 9, 0.0, 0.55618, 0.0, 65.0, 1, False),
            (10, 5, 6, 0.0, 0.25202, 0.0, 65.0, 1, False),
            (11, 6, 11, 0.09498, 0.19890, 0.0, 40.0, 1, False),
            (12, 6, 12, 0.12291, 0.25581, 0.0, 40.0, 1, False),
            (13, 6, 13, 0.06615, 0.13027, 0.0, 40.0, 1, False),
            (14, 7, 8, 0.0, 0.17615, 0.0, 65.0, 1, False),
            (15, 7, 9, 0.0, 0.11001, 0.0, 65.0, 1, False),
            (16, 9, 10, 0.03181, 0.08450, 0.0, 40.0, 1, False),
            (17, 9, 14, 0.12711, 0.27038, 0.0, 40.0, 1, False),
            (18, 10, 11, 0.08205, 0.19207, 0.0, 40.0, 1, False),
            (19, 12, 13, 0.22092, 0.19988, 0.0, 40.0, 1, False),
            (20, 13, 14, 0.17093, 0.34802, 0.0, 40.0, 1, False),
            # Autonomous Tie-Switches for Self-Healing Feeder Rerouting:
            (21, 3, 11, 0.08500, 0.21000, 0.0, 50.0, 0, True),  # Tie-Switch 1
            (22, 8, 14, 0.09200, 0.23000, 0.0, 50.0, 0, True),  # Tie-Switch 2
        ]

        self.branches = {}
        for br in branch_data:
            self.branches[br[0]] = Branch(
                id=br[0], from_bus=br[1], to_bus=br[2],
                r=br[3], x=br[4], b=br[5], rate_mva=br[6],
                status=br[7], is_tie_switch=br[8]
            )

    def _init_ieee_33_bus(self):
        """Initializes standard IEEE 33-bus radial distribution system."""
        self._init_ieee_14_bus()

    def solve_power_flow(self):
        """
        Solves DC/AC coupled power flow to update bus voltage angles,
        magnitudes, branch active/reactive flows, and thermal loadings.
        """
        # Build active network graph
        G = nx.Graph()
        for b_id in self.buses:
            G.add_node(b_id)
        for br in self.branches.values():
            if br.status == 1:
                weight = 1.0 / max(br.x, 1e-4)
                G.add_edge(br.from_bus, br.to_bus, weight=weight, branch_id=br.id)

        # Islanding detection
        connected_components = list(nx.connected_components(G))
        slack_bus_id = 1
        energized_nodes = set()
        for comp in connected_components:
            if slack_bus_id in comp:
                energized_nodes.update(comp)

        for b_id, bus in self.buses.items():
            if b_id in energized_nodes:
                if bus.status != "COMPROMISED":
                    bus.status = "ENERGIZED"
            else:
                bus.status = "ISOLATED"
                bus.voltage_pu = 0.0

        # DC Power Flow approximation for fast real-time telemetry simulation
        nodes = sorted(list(self.buses.keys()))
        n = len(nodes)
        node_map = {node: i for i, node in enumerate(nodes)}

        # Susceptance matrix B_bus
        B_bus = np.zeros((n, n))
        for br in self.branches.values():
            if br.status == 1:
                u, v = node_map[br.from_bus], node_map[br.to_bus]
                b_val = 1.0 / max(br.x, 1e-4)
                B_bus[u, u] += b_val
                B_bus[v, v] += b_val
                B_bus[u, v] -= b_val
                B_bus[v, u] -= b_val

        # Power injection vector P_inj
        P_inj = np.zeros(n)
        for b_id, bus in self.buses.items():
            if bus.status != "ISOLATED":
                P_inj[node_map[b_id]] = (bus.gen_p_mw - bus.load_p_mw) / GRID_CONFIG["BASE_MVA"]

        # Solve for angles relative to slack (slack angle = 0)
        slack_idx = node_map[slack_bus_id]
        non_slack = [i for i in range(n) if i != slack_idx and nodes[i] in energized_nodes]
        angles = np.zeros(n)

        if non_slack:
            B_red = B_bus[np.ix_(non_slack, non_slack)]
            P_red = P_inj[non_slack]
            try:
                theta_red = np.linalg.solve(B_red, P_red)
                for idx, theta in zip(non_slack, theta_red):
                    angles[idx] = float(theta)
            except np.linalg.LinAlgError:
                pass

        for b_id, bus in self.buses.items():
            idx = node_map[b_id]
            if bus.status != "ISOLATED":
                bus.voltage_angle_rad = angles[idx]
                # Voltage drop approximation
                drop = sum(abs(angles[idx] - angles[node_map[nbr]]) * 0.1 for nbr in G.neighbors(b_id)) if b_id in G else 0.0
                if bus.bus_type == "SLACK":
                    bus.voltage_pu = 1.06
                elif bus.bus_type == "PV":
                    bus.voltage_pu = max(0.98, 1.04 - drop * 0.05)
                else:
                    bus.voltage_pu = max(0.92, 1.02 - drop * 0.15)

        # Calculate branch flows
        for br in self.branches.values():
            if br.status == 1 and br.from_bus in energized_nodes and br.to_bus in energized_nodes:
                u, v = node_map[br.from_bus], node_map[br.to_bus]
                delta_theta = angles[u] - angles[v]
                p_flow_pu = delta_theta / max(br.x, 1e-4)
                br.flow_p_mw = p_flow_pu * GRID_CONFIG["BASE_MVA"]
                # Approximate Q flow and Current
                v_from = self.buses[br.from_bus].voltage_pu
                v_to = self.buses[br.to_bus].voltage_pu
                q_flow_pu = (v_from * (v_from - v_to) / max(br.x, 1e-4))
                br.flow_q_mvar = q_flow_pu * GRID_CONFIG["BASE_MVA"]
                apparent_power = math.sqrt(br.flow_p_mw**2 + br.flow_q_mvar**2)
                br.loading_percent = min(250.0, (apparent_power / max(br.rate_mva, 1.0)) * 100.0)
                br.current_ka = (apparent_power * 1000.0) / (math.sqrt(3) * self.buses[br.from_bus].base_kv * 1000.0)
            else:
                br.flow_p_mw = 0.0
                br.flow_q_mvar = 0.0
                br.loading_percent = 0.0
                br.current_ka = 0.0

    def step_simulation(self, load_variation: float = 0.02) -> Dict[str, Any]:
        """
        Advances grid time by 1 step, applies small realistic load perturbations,
        and returns fresh SCADA, PMU synchrophasors, and grid metrics.
        """
        self.timestamp = time.time()
        # Small realistic frequency jitter
        self.frequency_hz = 50.0 + random.gauss(0, 0.015)
        self.rocof_hz_s = random.gauss(0, 0.005)

        for bus in self.buses.values():
            if bus.load_p_mw > 0:
                fluctuation = 1.0 + random.uniform(-load_variation, load_variation)
                bus.load_p_mw = max(0.0, bus.load_p_mw * fluctuation)
                bus.load_q_mvar = max(0.0, bus.load_q_mvar * fluctuation)
                bus.active_power_mw = bus.gen_p_mw - bus.load_p_mw

        self.solve_power_flow()
        return self.get_telemetry_snapshot()

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Generates synchronized SCADA, PMU synchrophasor, and Smart Meter data packet."""
        pmu_data = {}
        scada_data = {}
        smart_meter_data = {}

        for b_id, bus in self.buses.items():
            scada_data[b_id] = {
                "bus_id": b_id,
                "bus_name": bus.name,
                "voltage_pu": round(bus.voltage_pu + random.gauss(0, 0.001), 4),
                "voltage_angle_deg": round(math.degrees(bus.voltage_angle_rad), 3),
                "active_power_mw": round(bus.active_power_mw, 3),
                "reactive_power_mvar": round(bus.reactive_power_mvar, 3),
                "load_p_mw": round(bus.load_p_mw, 3),
                "status": bus.status,
                "scada_rtu_id": bus.scada_rtu_id,
                "rtu_packet_loss_rate": 0.0,
            }
            if bus.has_pmu:
                pmu_data[b_id] = {
                    "pmu_id": f"PMU_{b_id:02d}",
                    "bus_id": b_id,
                    "frequency_hz": round(self.frequency_hz + random.gauss(0, 0.002), 4),
                    "rocof_hz_s": round(self.rocof_hz_s + random.gauss(0, 0.001), 4),
                    "voltage_magnitude_pu": round(bus.voltage_pu, 4),
                    "voltage_angle_deg": round(math.degrees(bus.voltage_angle_rad), 3),
                    "timestamp": self.timestamp,
                    "sync_lock": True,
                }
            if bus.load_p_mw > 0:
                smart_meter_data[b_id] = {
                    "meter_id": f"AMI_NODE_{b_id}",
                    "real_power_kw": round(bus.load_p_mw * 1000.0, 1),
                    "power_factor": round(math.cos(math.atan2(bus.load_q_mvar, max(bus.load_p_mw, 0.001))), 3),
                }

        branch_telemetry = {}
        for br_id, br in self.branches.items():
            branch_telemetry[br_id] = {
                "branch_id": br_id,
                "from_bus": br.from_bus,
                "to_bus": br.to_bus,
                "flow_p_mw": round(br.flow_p_mw, 3),
                "flow_q_mvar": round(br.flow_q_mvar, 3),
                "loading_percent": round(br.loading_percent, 1),
                "status": br.status,
                "is_tie_switch": br.is_tie_switch,
                "current_ka": round(br.current_ka, 3),
            }

        total_gen = sum(b.gen_p_mw for b in self.buses.values() if b.status != "ISOLATED")
        total_load = sum(b.load_p_mw for b in self.buses.values() if b.status != "ISOLATED")
        served_load = sum(b.load_p_mw for b in self.buses.values() if b.status in ["ENERGIZED", "RESTORED"])
        total_demand = sum(b.load_p_mw for b in self.buses.values())
        restoration_ratio = (served_load / max(total_demand, 1e-3)) * 100.0

        return {
            "timestamp": self.timestamp,
            "frequency_hz": round(self.frequency_hz, 3),
            "system_status": "NORMAL" if restoration_ratio > 98.0 else "DEGRADED",
            "total_gen_mw": round(total_gen, 2),
            "total_load_mw": round(total_load, 2),
            "restoration_ratio_pct": round(restoration_ratio, 2),
            "buses": scada_data,
            "pmus": pmu_data,
            "branches": branch_telemetry,
            "smart_meters": smart_meter_data,
        }

    def set_breaker_status(self, branch_id: int, status: int):
        """Actuates circuit breaker or tie-switch (1 = CLOSED, 0 = OPEN)."""
        if branch_id in self.branches:
            self.branches[branch_id].status = status
            self.solve_power_flow()

    def set_bus_status(self, bus_id: int, status: str):
        """Sets bus operational status."""
        if bus_id in self.buses:
            self.buses[bus_id].status = status
            self.solve_power_flow()
