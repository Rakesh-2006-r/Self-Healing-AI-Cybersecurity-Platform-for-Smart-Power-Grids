"""
Configuration settings for the Self-Healing Multi-Agent AI Cybersecurity Platform
for Smart Power Grid Infrastructure.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_store"
LOGS_DIR = BASE_DIR / "logs"

for p in [DATA_DIR, KNOWLEDGE_BASE_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Grid Electrical Standards & Limits (IEEE 1547 / IEEE Standard 1159)
GRID_CONFIG = {
    "NOMINAL_VOLTAGE_PU": 1.0,
    "VOLTAGE_MIN_PU": 0.94,
    "VOLTAGE_MAX_PU": 1.10,
    "VOLTAGE_CRITICAL_MIN_PU": 0.85,
    "VOLTAGE_CRITICAL_MAX_PU": 1.15,
    "NOMINAL_FREQUENCY_HZ": 50.0,  # 50 Hz standard
    "FREQUENCY_TOLERANCE_HZ": 0.5,
    "FREQUENCY_CRITICAL_LOW_HZ": 48.5,
    "FREQUENCY_CRITICAL_HIGH_HZ": 51.5,
    "MAX_LINE_LOADING_PERCENT": 100.0,
    "CRITICAL_LINE_LOADING_PERCENT": 120.0,
    "BASE_MVA": 100.0,
}

# Supported Grid Topologies
SUPPORTED_GRIDS = ["IEEE_14_BUS", "IEEE_33_BUS"]
DEFAULT_GRID = "IEEE_14_BUS"

# Cyber Threat & Anomaly Scoring Parameters
CYBER_CONFIG = {
    "ANOMALY_Z_SCORE_THRESHOLD": 3.0,
    "GNN_RECONSTRUCTION_ERROR_THRESHOLD": 0.15,
    "FDIA_RESIDUAL_EPSILON": 0.08,
    "DDOS_PACKET_DROP_RATE_THRESHOLD": 0.35,
    "REPLAY_CORRELATION_THRESHOLD": 0.99,
    "THREAT_SEVERITY_LEVELS": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
}

# MITRE ATT&CK for ICS Mapping
MITRE_ICS_TECHNIQUES = {
    "FDIA": {
        "technique_id": "T0831",
        "name": "Manipulation of Control",
        "description": "Adversaries manipulate state estimation telemetry to fool automatic generation control.",
    },
    "DDOS": {
        "technique_id": "T0814",
        "name": "Denial of Service (SCADA Telemetry)",
        "description": "Adversaries overwhelm substation RTU communication channels with traffic.",
    },
    "BREAKER_HIJACK": {
        "technique_id": "T0855",
        "name": "Unauthorized Command Message",
        "description": "Adversaries send malicious open/trip control commands to circuit breakers.",
    },
    "REPLAY_ATTACK": {
        "technique_id": "T0830",
        "name": "Adversary-in-the-Middle / Replay",
        "description": "Adversaries record normal operating telemetry and replay it during a physical contingency.",
    },
    "PHYSICAL_FAULT": {
        "technique_id": "N/A",
        "name": "Non-Malicious Grid Contingency",
        "description": "Equipment insulation failure, lightning strike, or vegetation contact.",
    },
}

# ChromaDB & Vector RAG Settings
VECTOR_DB_CONFIG = {
    "PERSIST_DIRECTORY": str(KNOWLEDGE_BASE_DIR / "chroma_db"),
    "COLLECTION_NAME": "grid_cybersecurity_knowledge",
    "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
    "TOP_K_RESULTS": 3,
}

# LLM & Multi-Agent LangGraph Settings
LLM_CONFIG = {
    "PROVIDER": os.getenv("LLM_PROVIDER", "local_expert"),  # "local_expert", "ollama", "openai", "gemini", "anthropic", "deepseek", "mistral", "huggingface"
    "MODEL_NAME": os.getenv("LLM_MODEL", "llama3.2"),
    "TEMPERATURE": 0.1,
    "TIMEOUT_SECONDS": 30,
    "SUPPORTED_PROVIDERS": [
        "local_expert",
        "ollama",
        "openai",
        "gemini",
        "anthropic",
        "deepseek",
        "mistral",
        "huggingface",
    ],
    "DEFAULT_MODELS": {
        "local_expert": "PowerGrid-Agent-v1.0",
        "ollama": "llama3.2",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-5-sonnet-20241022",
        "deepseek": "deepseek-chat",
        "mistral": "mistral-small-latest",
        "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
    }
}

# Self-Healing System Policies
SELF_HEALING_CONFIG = {
    "AUTONOMOUS_MODE_DEFAULT": True,
    "MAX_RECOVERY_STEPS": 5,
    "POWER_RESTORE_PRIORITY": ["CRITICAL_INFRASTRUCTURE", "INDUSTRIAL", "COMMERCIAL", "RESIDENTIAL"],
    "ALLOW_AUTOMATIC_TIE_SWITCHING": True,
    "ALLOW_AUTOMATIC_ISOLATION": True,
}
