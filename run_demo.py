"""
End-to-End Demonstration Script for Self-Healing Multi-Agent AI Cybersecurity Platform.
Demonstrates all 10 modules working together in real time.
"""

import time
import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from modules.data_source import PowerGridSimulator, AttackInjector
from modules.preprocessing import TelemetryProcessor, GraphDataBuilder
from modules.graph_intelligence import GridGNNAnomalyDetector, TopologyAnalyzer
from modules.anomaly_detection import AnomalyDetector
from modules.cybersecurity_agent import CybersecurityAgent
from modules.knowledge_base import GridKnowledgeRAG, GraphKnowledgeBase
from modules.decision_engine import DecisionEngine
from modules.recovery_engine import SelfHealingController
from modules.learning_module import ContinuousLearner

def run_simulation_pipeline():
    print("=" * 80)
    print(">>> SELF-HEALING MULTI-AGENT AI CYBERSECURITY PLATFORM FOR SMART POWER GRIDS <<<")
    print("    Department of Data Science - Batch DSA 13")
    print("=" * 80)

    # 1. Initialize all 10 Modules
    print("\n[INIT] Initializing 10 Architecture Modules...")
    grid_sim = PowerGridSimulator(topology_name="IEEE_14_BUS")
    attack_inj = AttackInjector()
    processor = TelemetryProcessor(window_size=20)
    graph_builder = GraphDataBuilder(num_nodes=14)
    gnn_detector = GridGNNAnomalyDetector()
    top_analyzer = TopologyAnalyzer()
    anomaly_detector = AnomalyDetector()
    cyber_agent = CybersecurityAgent()
    rag_engine = GridKnowledgeRAG()
    graph_kb = GraphKnowledgeBase()
    decision_engine = DecisionEngine(rag_engine=rag_engine, graph_kb=graph_kb)
    self_healing_ctrl = SelfHealingController(grid_simulator=grid_sim, attack_injector=attack_inj)
    learner = ContinuousLearner(rag_engine=rag_engine)
    print("[OK] All 10 Modules successfully loaded and connected.\n")

    # -------------------------------------------------------------
    # PHASE 1: Baseline Normal Grid Operation
    # -------------------------------------------------------------
    print("-" * 80)
    print("[PHASE 1] Normal Grid Operation Baseline (SCADA + PMU Synchrophasor)")
    print("-" * 80)
    raw_telemetry = grid_sim.step_simulation()
    processed = processor.process_telemetry_stream(raw_telemetry)
    x_tensor, edge_idx, edge_attr, nx_graph = graph_builder.build_graph_tensors(processed, grid_sim.branches)
    gnn_res = gnn_detector.evaluate_graph(x_tensor, edge_idx, sorted(list(grid_sim.buses.keys())))
    anomalies = anomaly_detector.detect_anomalies(processed, gnn_res)
    alerts = cyber_agent.analyze_threats(anomalies, processed, gnn_res, grid_sim.branches)
    top_res = top_analyzer.analyze_topology(nx_graph)

    print(f"System Status: {raw_telemetry['system_status']} | Frequency: {raw_telemetry['frequency_hz']} Hz | Restored: {raw_telemetry['restoration_ratio_pct']}%")
    print(f"Health Score: {anomalies.system_health_score}/100 | Anomalies Detected: {anomalies.total_anomalies} | Cyber Alerts: {len(alerts)}")

    # -------------------------------------------------------------
    # PHASE 2: Attack Injection - False Data Injection Attack (FDIA)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PHASE 2] Cyber Attack Injection (MITRE T0831: False Data Injection Attack on Bus 4)")
    print("-" * 80)
    attack_event = attack_inj.inject_attack(attack_type="FDIA", target_bus=4, intensity=1.2)
    print(f"[ATTACK LAUNCHED] {attack_event.id} ({attack_event.description})")

    # Step simulation with attack applied
    raw_telemetry = grid_sim.step_simulation()
    attacked_telemetry = attack_inj.apply_attacks_to_telemetry(raw_telemetry, grid_sim)
    processed = processor.process_telemetry_stream(attacked_telemetry)

    # -------------------------------------------------------------
    # PHASE 3: AI Intelligence Layer (GNN + Anomaly + Cybersecurity Agent)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PHASE 3] AI Intelligence Layer Processing (PyTorch GNN + Threat Classifier)")
    print("-" * 80)
    x_tensor, edge_idx, edge_attr, nx_graph = graph_builder.build_graph_tensors(processed, grid_sim.branches)
    gnn_res = gnn_detector.evaluate_graph(x_tensor, edge_idx, sorted(list(grid_sim.buses.keys())))
    anomalies = anomaly_detector.detect_anomalies(processed, gnn_res)
    alerts = cyber_agent.analyze_threats(anomalies, processed, gnn_res, grid_sim.branches)
    top_res = top_analyzer.analyze_topology(nx_graph)
    candidate_restorations = top_analyzer.find_restoration_paths(nx_graph, top_res["isolated_buses"], grid_sim.branches)

    print(f"* PyTorch GNN Max Anomaly Score: {gnn_res['max_anomaly_score']:.4f} (Bus-04 Error: {gnn_res['node_anomaly_scores'].get(4, 0.0):.4f})")
    print(f"* Grid Health Score: {anomalies.system_health_score}/100 ({anomalies.max_severity} severity)")
    for alert in alerts:
        print(f"  [CYBER ALERT] {alert.alert_id} | Type: {alert.attack_type} | MITRE: {alert.mitre_technique_id} ({alert.mitre_technique_name}) | TSI: {alert.threat_severity_index}")

    # -------------------------------------------------------------
    # PHASE 4: LangGraph Multi-Agent Decision Making & RAG Context
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PHASE 4] LangGraph Multi-Agent Decision Making (LLM + RAG + Playbook Generation)")
    print("-" * 80)
    decision = decision_engine.process_incident(
        anomaly_report=anomalies,
        cyber_alerts=alerts,
        processed_telemetry=processed,
        topology_analysis=top_res,
        candidate_restorations=candidate_restorations
    )

    print(f"Incident ID: {decision.incident_id} | Status: {decision.status} | Risk: {decision.risk_score}/100 ({decision.risk_level})")
    print(f"RAG Retrieved Documents: {[d['title'] for d in decision.rag_context]}")
    for log in decision.agent_logs:
        print(f"  [{log.role}] {log.agent_name}: {log.thought}")

    print(f"\nGenerated Self-Healing Playbook ({len(decision.recovery_playbook)} Steps):")
    for act in decision.recovery_playbook:
        print(f"  Step {act.step_number}: [{act.action_type}] on {act.target_entity} -> {act.justification}")

    # -------------------------------------------------------------
    # PHASE 5: Autonomous Self-Healing Execution
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PHASE 5] Autonomous Recovery Module Execution (Self-Healing in Action)")
    print("-" * 80)
    recovery_result = self_healing_ctrl.execute_playbook(decision, processed)
    for log_line in recovery_result.logs:
        print(f"  {log_line}")

    print(f"\n[OK] Recovery Completed in {recovery_result.execution_time_ms} ms | Power Restored: {recovery_result.final_restoration_pct}%")

    # -------------------------------------------------------------
    # PHASE 6: Continuous Learning & Knowledge Base Feedback Loop
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PHASE 6] Continuous Learning Agent & Knowledge Base Update")
    print("-" * 80)
    episode = learner.record_incident_resolution(decision, recovery_result, anomalies, alerts)
    metrics = learner.get_metrics_summary()

    print(f"[OK] Logged Incident Episode: {episode.incident_id}")
    print(f"[OK] Avoided Economic Damage: ${episode.avoided_loss_usd:,.2f}")
    print(f"[OK] Total Incidents Resolved: {metrics.total_incidents_resolved} | Mean Time to Recover: {metrics.mean_time_to_recover_ms} ms")
    print(f"[OK] ChromaDB Knowledge Base updated with new resolution pattern.")

    print("\n" + "=" * 80)
    print(">>> END-TO-END SMART GRID CYBERSECURITY PLATFORM SIMULATION SUCCESSFUL <<<")
    print("=" * 80)

if __name__ == "__main__":
    run_simulation_pipeline()
