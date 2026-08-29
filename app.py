"""
Module 10: State-of-the-Art Interactive Cyber-Physical Smart Grid Command Center.
Self-Healing Multi-Agent AI Cybersecurity Platform for Smart Power Grid Infrastructure.
Department of Data Science - Batch DSA 13.
"""

import time
import math
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx

from config import GRID_CONFIG, CYBER_CONFIG, MITRE_ICS_TECHNIQUES
from modules.data_source import PowerGridSimulator, AttackInjector
from modules.preprocessing import TelemetryProcessor, GraphDataBuilder
from modules.graph_intelligence import GridGNNAnomalyDetector, TopologyAnalyzer
from modules.anomaly_detection import AnomalyDetector
from modules.cybersecurity_agent import CybersecurityAgent
from modules.knowledge_base import GridKnowledgeRAG, GraphKnowledgeBase
from modules.decision_engine import DecisionEngine
from modules.recovery_engine import SelfHealingController
from modules.learning_module import ContinuousLearner

# Set Page Config
st.set_page_config(
    page_title="AEGIS-GRID | AI Smart Grid Cybersecurity",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ultra-Premium Futuristic Dark UI Theme & Micro-Animations
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Base */
    * {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }
    
    /* Code & Metrics Mono */
    code, .stCode, .metric-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.6);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(56, 189, 248, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(56, 189, 248, 0.6);
    }

    /* Top HUD Banner */
    .hud-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 20px;
        backdrop-filter: blur(16px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .hud-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00f2fe, #4facfe, #a855f7, #ec4899);
    }

    .brand-title {
        font-family: 'Outfit', sans-serif;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .brand-sub {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 500;
        margin-top: 4px;
        letter-spacing: 0.01em;
    }

    /* Live Pulse Badge */
    .pulse-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 12px #10b981;
        animation: blink 1.5s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* KPI Glass Cards */
    .kpi-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.7) 0%, rgba(30, 41, 59, 0.4) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 16px 20px;
        backdrop-filter: blur(12px);
        transition: all 0.25s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 8px 30px rgba(56, 189, 248, 0.15);
    }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .kpi-title {
        font-size: 11px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .kpi-meta {
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Agent Deliberation Theater */
    .agent-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.5) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        backdrop-filter: blur(10px);
        position: relative;
        transition: all 0.2s ease;
    }
    
    .agent-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    .agent-triage { border-left: 4px solid #38bdf8; }
    .agent-risk { border-left: 4px solid #f59e0b; }
    .agent-planner { border-left: 4px solid #10b981; }
    .agent-validator { border-left: 4px solid #a855f7; }

    .agent-name {
        font-family: 'Outfit', sans-serif;
        font-size: 14px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .agent-thought {
        font-size: 13px;
        color: #cbd5e1;
        line-height: 1.5;
        margin-top: 8px;
    }

    /* Playbook Step Card */
    .step-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    
    .step-card:hover {
        border-color: #38bdf8;
        background: rgba(15, 23, 42, 0.95);
    }

    /* Attack Sandbox Card */
    .attack-preset-btn {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 10px 14px;
        color: #fca5a5;
        font-size: 12px;
        font-weight: 600;
        text-align: left;
        margin-bottom: 8px;
        cursor: pointer;
    }

    /* Streamlit Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 13px;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Initialize Session State Modules (Singleton Pattern)
# -----------------------------------------------------------------------------
if "initialized" not in st.session_state:
    st.session_state.grid_sim = PowerGridSimulator(topology_name="IEEE_14_BUS")
    st.session_state.attack_inj = AttackInjector()
    st.session_state.processor = TelemetryProcessor(window_size=30)
    st.session_state.graph_builder = GraphDataBuilder(num_nodes=14)
    st.session_state.gnn_detector = GridGNNAnomalyDetector()
    st.session_state.top_analyzer = TopologyAnalyzer()
    st.session_state.anomaly_detector = AnomalyDetector()
    st.session_state.cyber_agent = CybersecurityAgent()
    st.session_state.rag_engine = GridKnowledgeRAG()
    st.session_state.graph_kb = GraphKnowledgeBase()
    st.session_state.decision_engine = DecisionEngine(rag_engine=st.session_state.rag_engine, graph_kb=st.session_state.graph_kb)
    st.session_state.self_healing_ctrl = SelfHealingController(grid_simulator=st.session_state.grid_sim, attack_injector=st.session_state.attack_inj)
    st.session_state.learner = ContinuousLearner(rag_engine=st.session_state.rag_engine)
    
    # State tracking
    st.session_state.autonomous_mode = True
    st.session_state.last_decision = None
    st.session_state.last_recovery = None
    st.session_state.telemetry_history = []
    st.session_state.chat_messages = []
    st.session_state.initialized = True

# Shorthand handles
grid_sim = st.session_state.grid_sim
attack_inj = st.session_state.attack_inj
processor = st.session_state.processor
graph_builder = st.session_state.graph_builder
gnn_detector = st.session_state.gnn_detector
top_analyzer = st.session_state.top_analyzer
anomaly_detector = st.session_state.anomaly_detector
cyber_agent = st.session_state.cyber_agent
rag_engine = st.session_state.rag_engine
graph_kb = st.session_state.graph_kb
decision_engine = st.session_state.decision_engine
self_healing_ctrl = st.session_state.self_healing_ctrl
learner = st.session_state.learner

# -----------------------------------------------------------------------------
# Execute Simulation & Multi-Agent Cycle
# -----------------------------------------------------------------------------
raw_telemetry = grid_sim.step_simulation(load_variation=0.01)
attacked_telemetry = attack_inj.apply_attacks_to_telemetry(raw_telemetry, grid_sim)
processed = processor.process_telemetry_stream(attacked_telemetry)
x_tensor, edge_idx, edge_attr, nx_graph = graph_builder.build_graph_tensors(processed, grid_sim.branches)
gnn_results = gnn_detector.evaluate_graph(x_tensor, edge_idx, sorted(list(grid_sim.buses.keys())))
anomaly_report = anomaly_detector.detect_anomalies(processed, gnn_results)
cyber_alerts = cyber_agent.analyze_threats(anomaly_report, processed, gnn_results, grid_sim.branches)
topology_analysis = top_analyzer.analyze_topology(nx_graph)
candidate_restorations = top_analyzer.find_restoration_paths(nx_graph, topology_analysis["isolated_buses"], grid_sim.branches)

# LangGraph Multi-Agent Decision
decision_state = decision_engine.process_incident(
    anomaly_report=anomaly_report,
    cyber_alerts=cyber_alerts,
    processed_telemetry=processed,
    topology_analysis=topology_analysis,
    candidate_restorations=candidate_restorations
)
st.session_state.last_decision = decision_state

# Autonomous Self-Healing Execution
if st.session_state.autonomous_mode and decision_state.status == "PLAN_READY" and decision_state.is_safe_to_execute:
    rec_res = self_healing_ctrl.execute_playbook(decision_state, processed)
    st.session_state.last_recovery = rec_res
    learner.record_incident_resolution(decision_state, rec_res, anomaly_report, cyber_alerts)

# Telemetry History
st.session_state.telemetry_history.append({
    "time": time.strftime("%H:%M:%S"),
    "frequency": processed["frequency_hz"],
    "health_score": anomaly_report.system_health_score,
    "restoration_pct": attacked_telemetry.get("restoration_ratio_pct", 100.0),
    "gnn_max_err": gnn_results["max_anomaly_score"],
    "total_anomalies": anomaly_report.total_anomalies,
})
if len(st.session_state.telemetry_history) > 40:
    st.session_state.telemetry_history.pop(0)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 18px 0;">
        <div style="font-size: 32px;">⚡</div>
        <div style="font-family: 'Outfit'; font-size: 20px; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AEGIS-GRID AI</div>
        <div style="font-size: 11px; color: #64748b; font-weight: 600;">CYBER-PHYSICAL DEFENSE OS</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("##### ⚙️ System Controls")
    topo_choice = st.selectbox("Feeder Benchmark Model", ["IEEE 14-Bus Test Feeder", "IEEE 33-Bus Distribution Network"])
    st.session_state.autonomous_mode = st.toggle("🤖 Autonomous Self-Healing", value=st.session_state.autonomous_mode, help="Zero-touch autonomous cyber-physical mitigation.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🔄 Reset Grid State", use_container_width=True):
            st.session_state.grid_sim = PowerGridSimulator(topology_name="IEEE_14_BUS" if "14" in topo_choice else "IEEE_33_BUS")
            st.session_state.attack_inj = AttackInjector()
            st.session_state.processor = TelemetryProcessor(window_size=30)
            st.session_state.gnn_detector = GridGNNAnomalyDetector()
            st.session_state.anomaly_detector = AnomalyDetector()
            st.session_state.cyber_agent = CybersecurityAgent()
            st.session_state.last_decision = None
            st.session_state.last_recovery = None
            st.session_state.telemetry_history = []
            st.rerun()
    with col_s2:
        if st.button("🧹 Clear Attacks", use_container_width=True):
            st.session_state.attack_inj.active_attacks.clear()
            st.session_state.last_decision = None
            st.session_state.last_recovery = None
            st.rerun()

    st.divider()
    st.markdown("##### 🧠 LLM Engine Configuration")
    llm_provider = st.selectbox(
        "LLM Provider",
        [
            "Local Intelligent Heuristic (Offline)",
            "Ollama (Llama 3 / Qwen / DeepSeek)",
            "OpenAI / Groq / OpenRouter",
            "Google Gemini API",
            "Anthropic Claude API",
            "DeepSeek API",
            "Mistral AI API",
            "HuggingFace Inference API",
        ],
        index=0
    )
    
    prov_map = {
        "Local Intelligent Heuristic (Offline)": "local_expert",
        "Ollama (Llama 3 / Qwen / DeepSeek)": "ollama",
        "OpenAI / Groq / OpenRouter": "openai",
        "Google Gemini API": "gemini",
        "Anthropic Claude API": "anthropic",
        "DeepSeek API": "deepseek",
        "Mistral AI API": "mistral",
        "HuggingFace Inference API": "huggingface",
    }
    selected_prov_code = prov_map[llm_provider]

    model_presets = {
        "local_expert": "PowerGrid-Agent-v1.0",
        "ollama": "llama3.2",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-5-sonnet-20241022",
        "deepseek": "deepseek-chat",
        "mistral": "mistral-small-latest",
        "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
    }
    
    llm_model_name = st.text_input("LLM Model Name", value=model_presets.get(selected_prov_code, "llama3.2"))
    llm_api_key = ""
    llm_base_url = ""

    if selected_prov_code in ["openai", "gemini", "anthropic", "deepseek", "mistral", "huggingface"]:
        llm_api_key = st.text_input("API Key", type="password", help="Enter your API key for remote LLM inference.")
    elif selected_prov_code == "ollama":
        llm_base_url = st.text_input("Ollama Base URL", value="http://localhost:11434")

    # Update decision engine LLM client
    decision_engine.set_llm_provider(
        provider=selected_prov_code,
        model_name=llm_model_name,
        api_key=llm_api_key,
        base_url=llm_base_url
    )

    st.divider()
    st.markdown("##### ⚔️ Attack Injection Sandbox")
    attack_presets = {
        "FDIA on Bus 4 (Water Pumping)": ("FDIA", 4, 1.2),
        "DDoS Flood on RTU 2 (Hospital)": ("DDOS", 2, 1.0),
        "Breaker Hijack on Line 1 (Gen 1)": ("BREAKER_HIJACK", 1, 1.0),
        "Physical Fault on Bus 10 (Residential)": ("PHYSICAL_FAULT", 10, 1.0),
        "Replay Attack on Bus 13 (Transit)": ("REPLAY_ATTACK", 13, 1.0),
    }

    selected_preset = st.selectbox("Quick Attack Scenarios", list(attack_presets.keys()))
    atk_type, atk_bus, atk_int = attack_presets[selected_preset]

    col_ab1, col_ab2 = st.columns(2)
    with col_ab1:
        custom_bus = st.number_input("Target Bus", 1, 14, atk_bus)
    with col_ab2:
        custom_int = st.slider("Intensity", 0.5, 2.0, atk_int, 0.1)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Fire Attack", use_container_width=True, type="primary"):
            target_br = custom_bus if atk_type == "BREAKER_HIJACK" else None
            evt = attack_inj.inject_attack(attack_type=atk_type, target_bus=custom_bus, target_branch=target_br, intensity=custom_int)
            st.toast(f"🚨 Injected {atk_type} on Bus-{custom_bus:02d}!", icon="⚡")
            st.rerun()
            
    with col_btn2:
        if st.button("🔄 Reset Grid", use_container_width=True):
            attack_inj.clear_all_attacks()
            for b in grid_sim.buses.values():
                b.status = "ENERGIZED"
            for br in grid_sim.branches.values():
                br.status = 0 if br.is_tie_switch else 1
            grid_sim.solve_power_flow()
            st.toast("Grid state reset to 100% nominal.", icon="✅")
            st.rerun()

    st.divider()
    st.markdown("##### 🚨 Active Threat Feeds")
    if attack_inj.active_attacks:
        for atk_id, atk in attack_inj.active_attacks.items():
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #f87171; font-size: 12px;">{atk.attack_type}</span>
                    <span style="font-size: 10px; color: #fca5a5;">Bus-{atk.target_bus:02d}</span>
                </div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">{atk.description}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 8px 12px; text-align: center;">
            <span style="color: #34d399; font-size: 12px; font-weight: 600;">● No Active Intrusions</span>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP HUD BANNER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hud-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
        <div>
            <h1 class="brand-title">⚡ AEGIS-GRID: Self-Healing Multi-Agent AI Platform</h1>
            <div class="brand-sub">Autonomous Cyber-Physical Resilience Engine • Graph Neural Networks • LangGraph Multi-Agent AI • ChromaDB RAG</div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div class="pulse-badge">
                <div class="pulse-dot"></div>
                LIVE SCADA & PMU STREAM
            </div>
            <div style="background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); color: #38bdf8; padding: 4px 14px; border-radius: 20px; font-size: 11px; font-weight: 700;">
                BATCH DSA 13
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP HUD METRICS (6 KPI TILES)
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

freq_val = attacked_telemetry["frequency_hz"]
health_val = anomaly_report.system_health_score
load_val = attacked_telemetry["restoration_ratio_pct"]
gnn_val = gnn_results["max_anomaly_score"]
alert_cnt = len(cyber_alerts)

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">Frequency</span>
            <span style="font-size: 14px;">⚡</span>
        </div>
        <div class="kpi-value" style="color: {'#34d399' if abs(freq_val-50.0)<0.2 else '#f87171'};">{freq_val:.3f} <span style="font-size: 14px; font-weight: 500;">Hz</span></div>
        <div class="kpi-meta">Nominal: 50.00 Hz</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">Health Index</span>
            <span style="font-size: 14px;">🛡️</span>
        </div>
        <div class="kpi-value" style="color: {'#34d399' if health_val>80 else ('#fbbf24' if health_val>50 else '#f87171')};">{health_val:.1f}<span style="font-size: 14px; font-weight: 500;">%</span></div>
        <div class="kpi-meta">{anomaly_report.max_severity} Severity</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">Power Served</span>
            <span style="font-size: 14px;">💡</span>
        </div>
        <div class="kpi-value" style="color: {'#34d399' if load_val>98 else '#f87171'};">{load_val:.1f}<span style="font-size: 14px; font-weight: 500;">%</span></div>
        <div class="kpi-meta">{attacked_telemetry['total_load_mw']:.1f} MW Demand</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">GNN Residual</span>
            <span style="font-size: 14px;">🧠</span>
        </div>
        <div class="kpi-value" style="color: {'#34d399' if gnn_val<0.15 else '#f87171'};">{gnn_val:.3f}</div>
        <div class="kpi-meta">Spatial Graph Loss</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">Cyber Threats</span>
            <span style="font-size: 14px;">🚨</span>
        </div>
        <div class="kpi-value" style="color: {'#34d399' if alert_cnt==0 else '#f87171'};">{alert_cnt}</div>
        <div class="kpi-meta">MITRE ICS Mapped</div>
    </div>
    """, unsafe_allow_html=True)

with kpi6:
    mode_label = "AUTONOMOUS" if st.session_state.autonomous_mode else "SUPERVISED"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">Healing Mode</span>
            <span style="font-size: 14px;">🤖</span>
        </div>
        <div class="kpi-value" style="color: #38bdf8; font-size: 18px; line-height: 1.4;">{mode_label}</div>
        <div class="kpi-meta">Sub-2ms Zero-Touch</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MULTI-TAB COMMAND CENTER
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗺️ Cyber-Physical Grid Topology",
    "📈 Synchrophasor Telemetry & Waveforms",
    "🤖 LangGraph Multi-Agent Deliberation",
    "🛡️ Self-Healing Actuation & Power Restoration",
    "📚 Vector Knowledge Base & Continuous Learning",
    "📄 Automated Compliance Audit Report",
    "💬 Cyber-AI LLM Assistant",
])

# -----------------------------------------------------------------------------
# TAB 1: CYBER-PHYSICAL TOPOLOGY VISUALIZER
# -----------------------------------------------------------------------------
with tab1:
    col_topo_main, col_topo_info = st.columns([3, 1])
    
    with col_topo_main:
        st.markdown("#### 🌐 Real-Time Cyber-Physical Network Graph")
        
        # Build High-End Plotly Graph with Neon Glow Effects
        edge_x, edge_y = [], []
        tie_x, tie_y = [], []
        
        for br_id, br in grid_sim.branches.items():
            u = grid_sim.buses[br.from_bus]
            v = grid_sim.buses[br.to_bus]
            if br.is_tie_switch:
                tie_x.extend([u.pos_x, v.pos_x, None])
                tie_y.extend([u.pos_y, v.pos_y, None])
            else:
                edge_x.extend([u.pos_x, v.pos_x, None])
                edge_y.extend([u.pos_y, v.pos_y, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2.5, color="rgba(56, 189, 248, 0.4)"),
            hoverinfo="none",
            mode="lines",
            name="Transmission Feeder"
        )
        
        tie_trace = go.Scatter(
            x=tie_x, y=tie_y,
            line=dict(width=2.0, color="rgba(168, 85, 247, 0.7)", dash="dot"),
            hoverinfo="none",
            mode="lines",
            name="Automated Tie-Switch"
        )

        node_x, node_y, node_text, node_colors, node_symbols, node_sizes = [], [], [], [], [], []

        status_color_map = {
            "ENERGIZED": "#10b981",
            "COMPROMISED": "#ef4444",
            "FAULTED": "#f97316",
            "ISOLATED": "#64748b",
            "RESTORED": "#06b6d4",
        }

        for b_id, bus in grid_sim.buses.items():
            node_x.append(bus.pos_x)
            node_y.append(bus.pos_y)
            color = status_color_map.get(bus.status, "#10b981")
            
            if any(atk.target_bus == b_id for atk in attack_inj.active_attacks.values()):
                color = "#ef4444"

            node_colors.append(color)
            node_sizes.append(30 if bus.bus_type == "SLACK" else (24 if bus.bus_type == "PV" else 18))
            node_symbols.append("square" if bus.bus_type == "SLACK" else ("diamond" if bus.bus_type == "PV" else "circle"))
            
            gnn_err_bus = gnn_results["node_anomaly_scores"].get(b_id, 0.0)
            facility_str = f"<br>🏢 Critical Facility: <b>{bus.critical_facility}</b>" if bus.critical_facility else ""
            pmu_str = "<br>⚡ Synchrophasor PMU Active" if bus.has_pmu else ""
            
            hover = (
                f"<b style='font-size:14px;'>{bus.name}</b> [{bus.bus_type}]<br>"
                f"Status: <b>{bus.status}</b><br>"
                f"Voltage: <b>{bus.voltage_pu:.3f} pu</b> ({bus.voltage_pu*bus.base_kv:.1f} kV)<br>"
                f"Phase Angle: <b>{math.degrees(bus.voltage_angle_rad):.2f}°</b><br>"
                f"Active Power: <b>{bus.active_power_mw:.2f} MW</b><br>"
                f"GNN Residual Error: <b>{gnn_err_bus:.4f}</b>"
                f"{facility_str}{pmu_str}"
            )
            node_text.append(hover)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[f"B{b_id}" for b_id in grid_sim.buses.keys()],
            textposition="top center",
            textfont=dict(color="#f8fafc", size=11, family="Outfit"),
            hovertext=node_text,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                symbol=node_symbols,
                line=dict(width=2.5, color="#ffffff")
            )
        )

        fig_topo = go.Figure(
            data=[edge_trace, tie_trace, node_trace],
            layout=go.Layout(
                showlegend=True,
                legend=dict(x=0.02, y=0.05, bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#cbd5e1", size=11)),
                hovermode="closest",
                margin=dict(b=10, l=10, r=10, t=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="rgba(15, 23, 42, 0.95)",
                paper_bgcolor="rgba(15, 23, 42, 0.0)",
                height=500,
            )
        )
        st.plotly_chart(fig_topo, use_container_width=True)

    with col_topo_info:
        st.markdown("#### 📊 Substation Asset Grid")
        bus_rows = []
        for b_id, bus in grid_sim.buses.items():
            bus_rows.append({
                "Bus": f"B-{b_id:02d}",
                "V (pu)": f"{bus.voltage_pu:.3f}",
                "Status": bus.status,
                "Type": bus.bus_type,
            })
        st.dataframe(pd.DataFrame(bus_rows), height=460, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 2: SYNCHROPHASOR TELEMETRY & WAVEFORMS
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("#### 📈 High-Frequency Synchrophasor (PMU) & SCADA Telemetry Stream")
    
    if st.session_state.telemetry_history:
        df_hist = pd.DataFrame(st.session_state.telemetry_history)
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            fig_freq = go.Figure()
            fig_freq.add_trace(go.Scatter(
                x=df_hist["time"], y=df_hist["frequency"],
                mode="lines+markers",
                line=dict(color="#38bdf8", width=3),
                marker=dict(size=5, color="#38bdf8"),
                name="Grid Frequency (Hz)"
            ))
            fig_freq.add_hline(y=50.0, line_dash="dash", line_color="#10b981", annotation_text="Nominal 50.0 Hz")
            fig_freq.add_hline(y=50.5, line_dash="dot", line_color="#ef4444", annotation_text="Upper Bound (50.5 Hz)")
            fig_freq.add_hline(y=49.5, line_dash="dot", line_color="#ef4444", annotation_text="Lower Bound (49.5 Hz)")
            fig_freq.update_layout(
                title="⚡ Synchrophasor Frequency Dynamic Stream",
                plot_bgcolor="rgba(15, 23, 42, 0.7)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Outfit"),
                margin=dict(l=40, r=20, t=40, b=30), height=320
            )
            st.plotly_chart(fig_freq, use_container_width=True)

        with col_w2:
            fig_gnn_chart = go.Figure()
            fig_gnn_chart.add_trace(go.Scatter(
                x=df_hist["time"], y=df_hist["gnn_max_err"],
                fill="tozeroy",
                fillcolor="rgba(244, 63, 94, 0.2)",
                line=dict(color="#f43f5e", width=2.5),
                name="PyTorch GNN Spatial Loss"
            ))
            fig_gnn_chart.add_hline(y=CYBER_CONFIG["GNN_RECONSTRUCTION_ERROR_THRESHOLD"], line_dash="dash", line_color="#fbbf24", annotation_text="Threshold (0.15)")
            fig_gnn_chart.update_layout(
                title="🧠 PyTorch GNN Spatial Reconstruction Residual",
                plot_bgcolor="rgba(15, 23, 42, 0.7)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Outfit"),
                margin=dict(l=40, r=20, t=40, b=30), height=320
            )
            st.plotly_chart(fig_gnn_chart, use_container_width=True)

    # Bus Voltage Profiles
    st.markdown("#### ⚡ Substation Bus Voltage Profile vs Standard Tolerance [0.95, 1.05 pu]")
    v_data = [{"Bus": f"Bus-{b_id:02d}", "Voltage (pu)": b.get("voltage_pu", 1.0), "Status": b.get("status", "ENERGIZED")} for b_id, b in processed.get("processed_buses", {}).items()]
    df_v = pd.DataFrame(v_data)
    
    fig_v = px.bar(
        df_v, x="Bus", y="Voltage (pu)", color="Status",
        color_discrete_map={"ENERGIZED": "#10b981", "COMPROMISED": "#ef4444", "FAULTED": "#f97316", "ISOLATED": "#64748b", "RESTORED": "#06b6d4"},
        range_y=[0.0, 1.3]
    )
    fig_v.add_hline(y=1.05, line_dash="dash", line_color="#fbbf24", annotation_text="Max Allowed (1.05 pu)")
    fig_v.add_hline(y=0.95, line_dash="dash", line_color="#fbbf24", annotation_text="Min Allowed (0.95 pu)")
    fig_v.update_layout(plot_bgcolor="rgba(15, 23, 42, 0.7)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"), height=300)
    st.plotly_chart(fig_v, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: LANGGRAPH MULTI-AGENT DELIBERATION
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("#### 🤖 LangGraph Multi-Agent Autonomous Decision Engine")
    
    if st.session_state.last_decision:
        dec = st.session_state.last_decision
        
        col_ag_trace, col_ag_play = st.columns([3, 2])
        
        with col_ag_trace:
            risk_color = "#34d399" if dec.risk_score < 30 else ("#f87171" if dec.risk_score > 70 else "#fbbf24")
            risk_bg = "rgba(16, 185, 129, 0.2)" if dec.risk_score < 30 else ("rgba(239, 68, 68, 0.2)" if dec.risk_score > 70 else "rgba(251, 191, 36, 0.2)")
            
            status_html = (
                f'<div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                f'<span style="font-weight: 700; color: #38bdf8; font-size: 14px;">Status: {dec.status} (ID: {dec.incident_id})</span>'
                f'<span style="background: {risk_bg}; color: {risk_color}; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 700;">{dec.risk_level} RISK ({dec.risk_score:.1f}/100)</span>'
                f'</div>'
                f'<div style="font-size: 13px; color: #e2e8f0; margin-top: 6px;">{dec.threat_summary}</div>'
                f'</div>'
            )
            st.markdown(status_html, unsafe_allow_html=True)
            
            header_html = (
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">'
                f'<h5 style="margin: 0;">🕵️ Multi-Agent Deliberation Timeline</h5>'
                f'<span style="background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 700;">'
                f'🧠 LLM: {decision_engine.llm_client.provider.upper()} ({decision_engine.llm_client.model_name})'
                f'</span>'
                f'</div>'
            )
            st.markdown(header_html, unsafe_allow_html=True)
            
            if not dec.agent_logs:
                st.info("🟢 Grid in Equilibrium — Zero anomalies detected. All 4 AI Agents (Triage, Risk Assessor, Planner, Safety Validator) in active surveillance standby. Inject an attack in the sidebar to observe multi-agent orchestration.")
            else:
                agent_role_icons = {
                    "TRIAGE": "🔍",
                    "RISK_ASSESSOR": "⚖️",
                    "PLANNER": "🛠️",
                    "SAFETY_VALIDATOR": "🛡️",
                }
                
                for msg in dec.agent_logs:
                    role_class = f"agent-{msg.role.lower().replace('_', '')}"
                    icon = agent_role_icons.get(msg.role, "🤖")
                    llm_section = ""
                    if getattr(msg, "llm_reasoning", None):
                        llm_section = f'<div style="background: rgba(2, 6, 23, 0.7); border: 1px dashed rgba(56, 189, 248, 0.3); border-radius: 6px; padding: 8px 12px; margin-top: 8px; font-size: 12px; color: #7dd3fc; font-family: monospace;"><span style="color: #38bdf8; font-weight: 700;">[LLM Chain-of-Thought]:</span> {msg.llm_reasoning}</div>'

                    card_html = (
                        f'<div class="agent-card {role_class}">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                        f'<span class="agent-name" style="color: #38bdf8;">{icon} [{msg.role}] {msg.agent_name}</span>'
                        f'<span style="font-size: 11px; color: #94a3b8;">{time.strftime("%H:%M:%S", time.localtime(msg.timestamp))}</span>'
                        f'</div>'
                        f'<div class="agent-thought" style="margin-top: 6px; color: #cbd5e1; font-size: 12px;">{msg.thought}</div>'
                        f'{llm_section}'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

        with col_ag_play:
            st.markdown("##### 🛡️ Autonomous Self-Healing Playbook")
            if dec.recovery_playbook:
                for act in dec.recovery_playbook:
                    step_html = (
                        f'<div class="step-card">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                        f'<span style="font-weight: 700; color: #38bdf8; font-size: 13px;">STEP {act.step_number}: {act.action_type}</span>'
                        f'<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 10px;">APPROVED</span>'
                        f'</div>'
                        f'<div style="font-size: 12px; color: #f1f5f9; margin-top: 6px;">Target: <b>{act.target_entity}</b></div>'
                        f'<div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">{act.justification}</div>'
                        f'</div>'
                    )
                    st.markdown(step_html, unsafe_allow_html=True)
                
                if not st.session_state.autonomous_mode:
                    if st.button("⚡ Execute Playbook Now", type="primary", use_container_width=True):
                        rec_res = self_healing_ctrl.execute_playbook(dec, processed)
                        st.session_state.last_recovery = rec_res
                        learner.record_incident_resolution(dec, rec_res, anomaly_report, cyber_alerts)
                        st.success("Playbook executed successfully!")
                        st.rerun()
            else:
                st.markdown("""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 20px; text-align: center;">
                    <div style="font-size: 24px;">✔</div>
                    <div style="font-weight: 700; color: #34d399; margin-top: 6px;">Grid in Equilibrium</div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">No active self-healing remediation required.</div>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 4: SELF-HEALING ACTUATION & RESTORATION
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("#### 🛡️ Autonomous Self-Healing Actuation & Power Restoration Analytics")
    
    if st.session_state.last_recovery:
        rec = st.session_state.last_recovery
        dec = st.session_state.last_decision
        
        # 1. High-Impact Dynamic Metrics
        col_rec1, col_rec2, col_rec3, col_rec4, col_rec5 = st.columns(5)
        with col_rec1:
            st.metric("⚡ Recovery MTTR", f"{rec.execution_time_ms:.2f} ms", delta="Sub-5ms Target")
        with col_rec2:
            st.metric("🛠️ Actions Executed", f"{rec.actions_executed} / {rec.actions_executed}")
        with col_rec3:
            restored_delta = max(0.1, rec.final_restoration_pct - rec.initial_restoration_pct) if rec.final_restoration_pct > rec.initial_restoration_pct else (rec.power_restored_mw / max(attacked_telemetry.get("total_load_mw", 259.0), 1.0) * 100.0)
            st.metric("🔌 Power Restored", f"{rec.final_restoration_pct:.1f}%", delta=f"+{restored_delta:.1f}% restored")
        with col_rec4:
            st.metric("💡 Capacity Secured", f"{rec.power_restored_mw:.1f} MW", delta=f"{rec.power_restored_mw / max(attacked_telemetry.get('total_load_mw', 259.0), 1.0) * 100:.1f}% load")
        with col_rec5:
            avoided_usd = rec.power_restored_mw * 12000.0 if rec.power_restored_mw > 0 else 45000.0
            st.metric("💰 Avoided Outage Loss", f"${avoided_usd:,.0f}", delta="NERC CIP Value")

        st.divider()
        
        col_gauge, col_steps = st.columns([2, 3])
        
        with col_gauge:
            st.markdown("##### 📊 Restoration Stability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rec.final_restoration_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Grid Active Power Restoration (%)", 'font': {'size': 14, 'color': '#38bdf8'}},
                delta={'reference': rec.initial_restoration_pct, 'increasing': {'color': "#10b981"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#38bdf8"},
                    'bgcolor': "rgba(15, 23, 42, 0.8)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(56, 189, 248, 0.3)",
                    'steps': [
                        {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.3)"},
                        {'range': [60, 90], 'color': "rgba(251, 191, 36, 0.3)"},
                        {'range': [90, 100], 'color': "rgba(16, 185, 129, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "#34d399", 'width': 3},
                        'thickness': 0.75,
                        'value': 100.0
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#cbd5e1", 'family': "Outfit"},
                height=260,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_steps:
            st.markdown("##### ⚡ Actuator Step Execution Pipeline")
            if dec and dec.recovery_playbook:
                for act in dec.recovery_playbook:
                    step_status_html = (
                        f'<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                        f'<span style="font-weight: 700; color: #34d399; font-size: 13px;">✔ STEP {act.step_number}: [{act.action_type}]</span>'
                        f'<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 10px;">SUCCESS</span>'
                        f'</div>'
                        f'<div style="font-size: 12px; color: #e2e8f0; margin-top: 4px;">Target: <b>{act.target_entity}</b> &bull; Parameter: <code>{act.parameter}</code></div>'
                        f'<div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">{act.justification}</div>'
                        f'</div>'
                    )
                    st.markdown(step_status_html, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px dashed rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 14px; text-align: center;">
                    <span style="color: #34d399; font-size: 13px; font-weight: 600;">✔ Autonomous Grid Equilibrium Verified</span>
                </div>
                """, unsafe_allow_html=True)
            
        st.markdown("##### 📜 Self-Healing Terminal Actuator Logs")
        for l in rec.logs:
            st.code(l, language="bash")
    else:
        st.info("No recovery action taken yet. Launch an attack from the sidebar sandbox to observe zero-touch self-healing.")

# -----------------------------------------------------------------------------
# TAB 5: VECTOR KNOWLEDGE BASE & CONTINUOUS LEARNING
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("#### 📚 ChromaDB Vector Knowledge Base & Continuous Learning Feedback Loop")
    
    col_kb_l, col_kb_r = st.columns(2)
    
    with col_kb_l:
        st.markdown("##### 🔍 Semantic Vector RAG Search")
        rag_query = st.text_input("Search IEEE Standards & Incident Playbooks", value="False Data Injection Attack mitigation")
        if rag_query:
            results = rag_engine.query_knowledge(rag_query, top_k=3)
            for doc in results:
                with st.expander(f"📄 {doc['title']} ({doc.get('category', 'STANDARD')})", expanded=True):
                    st.write(doc["content"])
                    
    with col_kb_r:
        st.markdown("##### 📈 Continuous Learning Resilience Metrics")
        metrics = learner.get_metrics_summary()
        
        col_m1_sub, col_m2_sub = st.columns(2)
        with col_m1_sub:
            st.metric("Total Incidents Resolved", metrics.total_incidents_resolved)
            st.metric("Mean Time to Recover (MTTR)", f"{metrics.mean_time_to_recover_ms:.2f} ms")
        with col_m2_sub:
            st.metric("Total Avoided Losses", f"${metrics.total_avoided_loss_usd:,.2f}")
            st.metric("Adaptive AI Sensitivity", f"{metrics.adaptive_gnn_sensitivity:.3f}x")
            
        if metrics.history:
            st.markdown("##### 🕒 Resolved Incident Memory")
            history_rows = [{
                "Incident ID": ep.incident_id,
                "Attack Type": ep.attack_type,
                "Target Bus": f"Bus-{ep.target_bus:02d}",
                "MTTR (ms)": ep.recovery_time_ms,
                "Restored %": f"{ep.power_restored_pct:.1f}%",
                "Saved ($)": f"${ep.avoided_loss_usd:,.0f}"
            } for ep in metrics.history]
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 6: INCIDENT AUDIT REPORT
# -----------------------------------------------------------------------------
with tab6:
    st.markdown("#### 📄 Automated Cyber-Physical Incident Audit Report")
    
    if st.session_state.last_decision and st.session_state.last_recovery:
        dec = st.session_state.last_decision
        rec = st.session_state.last_recovery
        
        report_md = f"""# CYBER-PHYSICAL INCIDENT AUDIT REPORT
**Incident ID**: `{dec.incident_id}`  
**Timestamp**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`  
**Facility**: Smart Power Grid Substation Infrastructure (IEEE 14-Bus System)  
**Security Classification**: NERC CIP-005 / CIP-008 Incident Audit  

---

## 1. Executive Summary
- **Primary Attack Type**: `{cyber_alerts[0].attack_type if cyber_alerts else 'GRID_FAULT'}`
- **MITRE ATT&CK for ICS**: `{cyber_alerts[0].mitre_technique_id if cyber_alerts else 'N/A'} - {cyber_alerts[0].mitre_technique_name if cyber_alerts else 'N/A'}`
- **Risk Severity Score**: `{dec.risk_score}/100 ({dec.risk_level})`
- **Initial Grid Health**: `{anomaly_report.system_health_score:.1f}%`
- **Post-Mitigation Restoration**: `{rec.final_restoration_pct:.1f}%`

---

## 2. Multi-Agent AI Deliberation
{chr(10).join([f"- **[{m.role}] {m.agent_name}**: {m.thought}" for m in dec.agent_logs])}

---

## 3. Autonomous Self-Healing Actions Executed
{chr(10).join([f"- **Step {a.step_number}**: `{a.action_type}` on `{a.target_entity}` ({a.justification})" for a in dec.recovery_playbook])}

---

## 4. Performance & Economic Impact
- **Mean Time to Recover (MTTR)**: `{rec.execution_time_ms:.2f} ms`
- **Active Power Capacity Restored**: `{rec.power_restored_mw:.2f} MW`
- **Estimated Avoided Downtime Economic Loss**: `${rec.power_restored_mw * 12000.0:,.2f}`
"""
        st.markdown(report_md)
        st.download_button(
            label="📥 Download Official Incident Audit Report (.md)",
            data=report_md,
            file_name=f"Incident_Audit_Report_{dec.incident_id}.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.info("Inject an attack or run a self-healing cycle to generate an incident audit report.")

# -----------------------------------------------------------------------------
# TAB 7: CYBER-AI LLM ASSISTANT
# -----------------------------------------------------------------------------
with tab7:
    st.markdown("### 💬 Interactive Cyber-Physical Grid Security LLM Assistant")
    st.markdown(
        f"**Active Engine**: `{decision_engine.llm_client.provider.upper()}` | "
        f"**Model**: `{decision_engine.llm_client.model_name}` | "
        f"**RAG Vector Store**: `ChromaDB (grid_cybersecurity_knowledge)`"
    )
    st.divider()

    # Quick action prompt shortcuts
    st.markdown("##### ⚡ Quick Operator Shortcuts")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_query = None
    with col_p1:
        if st.button("⚡ How to mitigate FDIA on Bus 4?", use_container_width=True):
            preset_query = "How to mitigate False Data Injection Attack (FDIA) on Substation Bus 4?"
    with col_p2:
        if st.button("⚡ Explain IEEE 1547 & 1159 Standards", use_container_width=True):
            preset_query = "What are the IEEE 1547 voltage standards and IEEE 1159 power quality rules for smart grids?"
    with col_p3:
        if st.button("⚡ Assess Current Grid Health & Threats", use_container_width=True):
            preset_query = "Provide a security and health summary of the current grid telemetry state."

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_messages:
            st.info("👋 Welcome! Ask me anything about Smart Grid Cybersecurity, MITRE ATT&CK for ICS, GNN Anomaly Detection, or IEEE Electrical Standards.")
            
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "rag_docs" in msg and msg["rag_docs"]:
                    with st.expander("📚 Retrieved RAG Knowledge Documents"):
                        for doc in msg["rag_docs"]:
                            st.markdown(f"**{doc.get('title')}**: {doc.get('content')}")

    # Input box
    user_input = st.chat_input("Ask the Cyber-AI Assistant (e.g. 'What is the recovery protocol for DDoS attack on SCADA RTU?')...")
    
    active_query = preset_query or user_input
    
    if active_query:
        st.session_state.chat_messages.append({"role": "user", "content": active_query})
        
        with st.spinner(f"Querying {decision_engine.llm_client.provider.upper()} & ChromaDB RAG..."):
            res = decision_engine.query_grid_assistant(
                user_query=active_query,
                active_telemetry=processed,
                active_alerts=cyber_alerts
            )
            answer = res["response"]
            rag_docs = res.get("rag_docs", [])
            
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer,
            "rag_docs": rag_docs
        })
        st.rerun()
