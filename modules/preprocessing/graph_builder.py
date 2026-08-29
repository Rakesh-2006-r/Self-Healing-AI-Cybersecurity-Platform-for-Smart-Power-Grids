"""
Module 2 (Sub-module): Graph Data Builder.
Constructs PyTorch graph tensors and NetworkX topologies from grid telemetry.
"""

import math
import torch
import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple
from config import GRID_CONFIG

class GraphDataBuilder:
    """
    Transforms smart grid telemetry into PyTorch Graph representation (Node features X, Edge Index, Edge Features).
    """
    def __init__(self, num_nodes: int = 14):
        self.num_nodes = num_nodes

    def build_graph_tensors(self, processed_telemetry: Dict[str, Any], branches_dict: Dict[int, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, nx.Graph]:
        """
        Builds:
          - x: [num_nodes, node_features_dim]
          - edge_index: [2, num_edges]
          - edge_attr: [num_edges, edge_features_dim]
          - nx_graph: NetworkX graph
        """
        buses = processed_telemetry.get("processed_buses", {})
        sorted_bus_ids = sorted(list(buses.keys()))
        bus_to_idx = {b_id: i for i, b_id in enumerate(sorted_bus_ids)}
        n = len(sorted_bus_ids)

        # Node Feature Vector for each bus:
        # [voltage_pu, voltage_angle_rad, P_pu, Q_pu, v_z_score, p_z_score, packet_loss, has_pmu]
        node_features = []
        pmus = processed_telemetry.get("pmus", {})

        for b_id in sorted_bus_ids:
            b_info = buses[b_id]
            v_pu = b_info.get("voltage_pu", 1.0)
            angle_rad = math.radians(b_info.get("voltage_angle_deg", 0.0))
            p_pu = b_info.get("active_power_mw", 0.0) / GRID_CONFIG["BASE_MVA"]
            q_pu = b_info.get("reactive_power_mvar", 0.0) / GRID_CONFIG["BASE_MVA"]
            v_z = b_info.get("voltage_z_score", 0.0)
            p_z = b_info.get("power_z_score", 0.0)
            loss_rate = b_info.get("rtu_packet_loss_rate", 0.0)
            has_pmu = 1.0 if b_id in pmus else 0.0

            node_features.append([v_pu, angle_rad, p_pu, q_pu, v_z, p_z, loss_rate, has_pmu])

        x_tensor = torch.tensor(node_features, dtype=torch.float32)

        # Edge construction
        src_nodes = []
        dst_nodes = []
        edge_attributes = []
        nx_graph = nx.Graph()

        for b_id in sorted_bus_ids:
            nx_graph.add_node(b_id, **buses[b_id])

        for br_id, br in branches_dict.items():
            u, v = br.from_bus, br.to_bus
            status = br.status
            if u in bus_to_idx and v in bus_to_idx:
                u_idx, v_idx = bus_to_idx[u], bus_to_idx[v]
                r_val = getattr(br, "r", 0.05)
                x_val = getattr(br, "x", 0.1)
                loading = getattr(br, "loading_percent", 0.0) / 100.0

                # Bidirectional edges
                src_nodes.extend([u_idx, v_idx])
                dst_nodes.extend([v_idx, u_idx])
                attr = [r_val, x_val, loading, float(status)]
                edge_attributes.extend([attr, attr])

                if status == 1:
                    nx_graph.add_edge(u, v, branch_id=br.id, loading_pct=getattr(br, "loading_percent", 0.0), is_tie=getattr(br, "is_tie_switch", False))

        if len(src_nodes) == 0:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 4), dtype=torch.float32)
        else:
            edge_index_tensor = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            edge_attr_tensor = torch.tensor(edge_attributes, dtype=torch.float32)

        return x_tensor, edge_index_tensor, edge_attr_tensor, nx_graph
