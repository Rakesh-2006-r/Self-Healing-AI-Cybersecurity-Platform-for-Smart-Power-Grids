"""
Module 9 (Sub-module): Graph Knowledge Base.
Models graph-structured relationships between critical facilities, substations, and cybersecurity assets.
"""

from typing import Dict, List, Any, Optional

class GraphKnowledgeBase:
    """
    Graph-structured repository of asset criticality and cyber-physical dependencies.
    """
    def __init__(self):
        self.critical_assets = {
            1: {"name": "Substation Control Center (HQ)", "tier": 1, "priority": "CRITICAL", "backup_gen": True},
            2: {"name": "Regional Hospital Substation", "tier": 1, "priority": "CRITICAL", "backup_gen": True},
            3: {"name": "Industrial Processing Zone", "tier": 3, "priority": "MEDIUM", "backup_gen": False},
            4: {"name": "Municipal Water Pumping", "tier": 2, "priority": "HIGH", "backup_gen": True},
            5: {"name": "Data Center Hub", "tier": 2, "priority": "HIGH", "backup_gen": True},
            6: {"name": "Defense Communications Substation", "tier": 1, "priority": "CRITICAL", "backup_gen": True},
            8: {"name": "Solar Microgrid Dispatch", "tier": 2, "priority": "HIGH", "backup_gen": False},
            9: {"name": "Commercial Retail Complex", "tier": 3, "priority": "MEDIUM", "backup_gen": False},
            12: {"name": "Emergency Services Center", "tier": 1, "priority": "CRITICAL", "backup_gen": True},
            13: {"name": "Transit Metro Substation", "tier": 2, "priority": "HIGH", "backup_gen": False},
        }

    def get_asset_context(self, bus_id: int) -> Dict[str, Any]:
        """Returns criticality metadata for a specified power grid bus."""
        return self.critical_assets.get(bus_id, {
            "name": f"Substation Feeder Bus-{bus_id:02d}",
            "tier": 4,
            "priority": "LOW",
            "backup_gen": False
        })
