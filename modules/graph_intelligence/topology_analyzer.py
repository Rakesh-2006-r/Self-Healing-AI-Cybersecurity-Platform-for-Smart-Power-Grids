"""
Module 3 (Sub-module): Topology Analyzer.
Graph-theoretic topology analysis, electrical islanding detection, and tie-switch pathfinding.
"""

import networkx as nx
from typing import Dict, List, Any, Set, Tuple

class TopologyAnalyzer:
    """
    Performs dynamic topological reasoning on the power network graph.
    """
    def __init__(self):
        pass

    def analyze_topology(self, nx_graph: nx.Graph, slack_bus_id: int = 1, tie_branches: Dict[int, Any] = None) -> Dict[str, Any]:
        """
        Analyzes network graph structure, detects isolated islands, and computes graph centrality.
        """
        if nx_graph.number_of_nodes() == 0:
            return {
                "is_connected": True,
                "islands": [],
                "energized_buses": [],
                "isolated_buses": [],
                "critical_bridges": [],
                "betweenness_centrality": {},
            }

        is_connected = nx.is_connected(nx_graph)
        components = list(nx.connected_components(nx_graph))

        energized_buses = set()
        isolated_buses = set()

        for comp in components:
            if slack_bus_id in comp:
                energized_buses.update(comp)
            else:
                isolated_buses.update(comp)

        # Graph Centrality
        try:
            betweenness = nx.betweenness_centrality(nx_graph)
        except Exception:
            betweenness = {node: 0.0 for node in nx_graph.nodes()}

        # Critical Bridges (single lines whose tripping causes blackout)
        try:
            bridges = list(nx.bridges(nx_graph))
        except Exception:
            bridges = []

        return {
            "is_connected": is_connected,
            "num_islands": len(components),
            "islands": [list(c) for c in components],
            "energized_buses": sorted(list(energized_buses)),
            "isolated_buses": sorted(list(isolated_buses)),
            "critical_bridges": bridges,
            "betweenness_centrality": {node: round(score, 4) for node, score in betweenness.items()},
        }

    def find_restoration_paths(self, nx_graph: nx.Graph, isolated_buses: List[int], all_branches: Dict[int, Any], slack_bus_id: int = 1) -> List[Dict[str, Any]]:
        """
        Identifies candidate tie-switches to close that will restore power to isolated sub-grids.
        """
        if not isolated_buses:
            return []

        isolated_set = set(isolated_buses)
        candidate_actions = []

        # Check all available open tie-switches or branches
        for br_id, br in all_branches.items():
            if br.status == 0:  # Open switch
                u, v = br.from_bus, br.to_bus
                # Check if this switch bridges an energized bus and an isolated bus
                u_isolated = u in isolated_set
                v_isolated = v in isolated_set

                if (u_isolated and not v_isolated) or (v_isolated and not u_isolated):
                    candidate_actions.append({
                        "branch_id": br_id,
                        "from_bus": u,
                        "to_bus": v,
                        "action_type": "CLOSE_TIE_SWITCH",
                        "restores_buses": list(isolated_set),
                        "priority": "HIGH" if br.is_tie_switch else "MEDIUM",
                        "description": f"Close Tie-Switch {br_id} between Bus-{u} and Bus-{v} to restore {len(isolated_set)} de-energized buses."
                    })

        return candidate_actions
