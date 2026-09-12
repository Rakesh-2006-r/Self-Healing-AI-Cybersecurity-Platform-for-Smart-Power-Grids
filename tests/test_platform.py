"""
Comprehensive Test Suite for Self-Healing Multi-Agent AI Cybersecurity Platform.
Verifies all 10 modules and end-to-end integration workflows.
"""

import unittest
import torch
import networkx as nx
from modules.data_source import PowerGridSimulator, AttackInjector
from modules.preprocessing import TelemetryProcessor, GraphDataBuilder
from modules.graph_intelligence import GridGNNAnomalyDetector, TopologyAnalyzer
from modules.anomaly_detection import AnomalyDetector
from modules.cybersecurity_agent import CybersecurityAgent
from modules.knowledge_base import GridKnowledgeRAG, GraphKnowledgeBase
from modules.decision_engine import DecisionEngine
from modules.recovery_engine import SelfHealingController
from modules.learning_module import ContinuousLearner

class TestSmartGridPlatform(unittest.TestCase):

    def setUp(self):
        self.grid_sim = PowerGridSimulator(topology_name="IEEE_14_BUS")
        self.attack_inj = AttackInjector()
        self.processor = TelemetryProcessor(window_size=15)
        self.graph_builder = GraphDataBuilder(num_nodes=14)
        self.gnn_detector = GridGNNAnomalyDetector()
        self.top_analyzer = TopologyAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.cyber_agent = CybersecurityAgent()
        self.rag_engine = GridKnowledgeRAG()
        self.graph_kb = GraphKnowledgeBase()
        self.decision_engine = DecisionEngine(rag_engine=self.rag_engine, graph_kb=self.graph_kb)
        self.self_healing_ctrl = SelfHealingController(grid_simulator=self.grid_sim, attack_injector=self.attack_inj)
        self.learner = ContinuousLearner(rag_engine=self.rag_engine)

    def test_01_grid_simulator_initialization(self):
        """Verify IEEE 14-bus test feeder configuration."""
        self.assertEqual(len(self.grid_sim.buses), 14)
        self.assertGreaterEqual(len(self.grid_sim.branches), 20)
        snapshot = self.grid_sim.get_telemetry_snapshot()
        self.assertEqual(snapshot["system_status"], "NORMAL")
        self.assertAlmostEqual(snapshot["restoration_ratio_pct"], 100.0, places=1)
        self.assertIn(1, snapshot["pmus"])

    def test_02_attack_injection_fdia_and_ddos(self):
        """Verify cyber attack simulation and telemetry distortion."""
        event = self.attack_inj.inject_attack(attack_type="FDIA", target_bus=4, intensity=1.0)
        self.assertEqual(event.attack_type, "FDIA")
        self.assertEqual(event.target_bus, 4)
        self.assertEqual(event.mitre_info["technique_id"], "T0831")

        raw_telemetry = self.grid_sim.get_telemetry_snapshot()
        attacked = self.attack_inj.apply_attacks_to_telemetry(raw_telemetry, self.grid_sim)
        self.assertGreater(attacked["buses"][4]["voltage_pu"], 1.10)
        self.assertEqual(attacked["buses"][4]["status"], "COMPROMISED")

    def test_03_preprocessing_and_graph_builder(self):
        """Verify data cleaning, Z-score computation, and PyTorch tensor creation."""
        raw = self.grid_sim.step_simulation()
        processed = self.processor.process_telemetry_stream(raw)
        self.assertIn(1, processed["processed_buses"])
        
        x, edge_idx, edge_attr, g = self.graph_builder.build_graph_tensors(processed, self.grid_sim.branches)
        self.assertEqual(x.shape[0], 14)
        self.assertEqual(x.shape[1], 8)
        self.assertEqual(edge_idx.shape[0], 2)
        self.assertIsInstance(g, nx.Graph)

    def test_04_gnn_model_inference(self):
        """Verify PyTorch Spatial GNN AutoEncoder forward pass and reconstruction loss."""
        raw = self.grid_sim.step_simulation()
        processed = self.processor.process_telemetry_stream(raw)
        x, edge_idx, edge_attr, _ = self.graph_builder.build_graph_tensors(processed, self.grid_sim.branches)
        res = self.gnn_detector.evaluate_graph(x, edge_idx, list(self.grid_sim.buses.keys()))
        self.assertIn("node_anomaly_scores", res)
        self.assertEqual(len(res["node_anomaly_scores"]), 14)

    def test_05_topology_analyzer_and_tie_switch_pathfinding(self):
        """Verify electrical islanding detection and self-healing tie-switch discovery."""
        # Trip branch 1 (Bus 1 to Bus 2)
        self.grid_sim.set_breaker_status(1, 0)
        _, _, _, g = self.graph_builder.build_graph_tensors(
            self.processor.process_telemetry_stream(self.grid_sim.get_telemetry_snapshot()),
            self.grid_sim.branches
        )
        analysis = self.top_analyzer.analyze_topology(g)
        self.assertIn("betweenness_centrality", analysis)
        self.assertIn("energized_buses", analysis)

    def test_06_cybersecurity_agent_mitre_classification(self):
        """Verify classification of FDIA vs DDoS vs Physical Faults."""
        self.attack_inj.inject_attack(attack_type="FDIA", target_bus=5)
        raw = self.attack_inj.apply_attacks_to_telemetry(self.grid_sim.get_telemetry_snapshot(), self.grid_sim)
        processed = self.processor.process_telemetry_stream(raw)
        x, edge_idx, _, g = self.graph_builder.build_graph_tensors(processed, self.grid_sim.branches)
        gnn_res = self.gnn_detector.evaluate_graph(x, edge_idx, list(self.grid_sim.buses.keys()))
        anomalies = self.anomaly_detector.detect_anomalies(processed, gnn_res)
        alerts = self.cyber_agent.analyze_threats(anomalies, processed, gnn_res, self.grid_sim.branches)

        self.assertGreater(len(alerts), 0)
        alert = alerts[0]
        self.assertEqual(alert.attack_type, "FDIA")
        self.assertEqual(alert.mitre_technique_id, "T0831")

    def test_07_rag_knowledge_base_retrieval(self):
        """Verify vector semantic retrieval of IEEE standards and playbooks."""
        results = self.rag_engine.query_knowledge("False Data Injection mitigation playbook", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("FDIA", results[0]["title"] + results[0]["content"])

    def test_08_langgraph_decision_engine_and_self_healing(self):
        """Verify end-to-end multi-agent plan generation and automated remediation."""
        self.attack_inj.inject_attack(attack_type="FDIA", target_bus=3)
        raw = self.attack_inj.apply_attacks_to_telemetry(self.grid_sim.get_telemetry_snapshot(), self.grid_sim)
        processed = self.processor.process_telemetry_stream(raw)
        x, edge_idx, _, g = self.graph_builder.build_graph_tensors(processed, self.grid_sim.branches)
        gnn_res = self.gnn_detector.evaluate_graph(x, edge_idx, list(self.grid_sim.buses.keys()))
        anomalies = self.anomaly_detector.detect_anomalies(processed, gnn_res)
        alerts = self.cyber_agent.analyze_threats(anomalies, processed, gnn_res, self.grid_sim.branches)
        top_res = self.top_analyzer.analyze_topology(g)

        # Decision
        decision = self.decision_engine.process_incident(anomalies, alerts, processed, top_res, [])
        self.assertEqual(decision.status, "PLAN_READY")
        self.assertTrue(decision.is_safe_to_execute)
        self.assertGreater(len(decision.recovery_playbook), 0)

        # Execution
        result = self.self_healing_ctrl.execute_playbook(decision, processed)
        self.assertTrue(result.success)
        self.assertGreater(result.actions_executed, 0)
        self.assertEqual(result.final_restoration_pct, 100.0)

    def test_09_continuous_learning_feedback_loop(self):
        """Verify logging incident resolution and continuous learning metrics."""
        self.attack_inj.inject_attack(attack_type="DDOS", target_bus=2)
        raw = self.attack_inj.apply_attacks_to_telemetry(self.grid_sim.get_telemetry_snapshot(), self.grid_sim)
        processed = self.processor.process_telemetry_stream(raw)
        x, edge_idx, _, g = self.graph_builder.build_graph_tensors(processed, self.grid_sim.branches)
        gnn_res = self.gnn_detector.evaluate_graph(x, edge_idx, list(self.grid_sim.buses.keys()))
        anomalies = self.anomaly_detector.detect_anomalies(processed, gnn_res)
        alerts = self.cyber_agent.analyze_threats(anomalies, processed, gnn_res, self.grid_sim.branches)
        top_res = self.top_analyzer.analyze_topology(g)

        decision = self.decision_engine.process_incident(anomalies, alerts, processed, top_res, [])
        recovery = self.self_healing_ctrl.execute_playbook(decision, processed)
        
        episode = self.learner.record_incident_resolution(decision, recovery, anomalies, alerts)
        self.assertEqual(episode.incident_id, decision.incident_id)
        self.assertGreater(episode.avoided_loss_usd, 0.0)

        metrics = self.learner.get_metrics_summary()
        self.assertEqual(metrics.total_incidents_resolved, 1)

    def test_10_llm_client_and_agent_reasoning(self):
        """Verify LLM Client multi-provider inference, JSON parsing, and Agent chat assistant reasoning."""
        from modules.decision_engine import LLMClient
        
        # 1. Local Expert Heuristic Provider
        llm = LLMClient(provider="local_expert", model_name="PowerGrid-Agent-v1.0")
        response = llm.query("You are a Triage Agent", "FDIA detected on Bus 4")
        self.assertIn("FDIA", response)
        self.assertGreater(len(response), 20)

        # 2. Multi-Provider Provider Initialization & Offline Fallbacks
        for prov in ["anthropic", "deepseek", "mistral", "huggingface", "ollama", "gemini", "openai"]:
            prov_llm = LLMClient(provider=prov)
            res = prov_llm.query("You are a Risk Assessor", "Assess risk of DDoS attack")
            self.assertIsInstance(res, str)
            self.assertGreater(len(res), 10)

        # 3. Structured JSON Query Extraction
        json_res = llm.query_json("System prompt", "Return mitigation steps")
        self.assertIsInstance(json_res, dict)

        # 4. Interactive Grid LLM Assistant Query
        chat_res = self.decision_engine.query_grid_assistant("How to handle FDIA on Bus 4?")
        self.assertIn("response", chat_res)
        self.assertIn("rag_docs", chat_res)
        self.assertGreater(len(chat_res["response"]), 20)

    def test_11_langgraph_workflow_runs_all_specialist_agents(self):
        """Verify the decision engine uses all four LangGraph nodes in order."""
        self.attack_inj.inject_attack(attack_type="FDIA", target_bus=4)
        raw = self.attack_inj.apply_attacks_to_telemetry(self.grid_sim.get_telemetry_snapshot(), self.grid_sim)
        processed = self.processor.process_telemetry_stream(raw)
        x, edge_idx, _, graph = self.graph_builder.build_graph_tensors(processed, self.grid_sim.branches)
        gnn_res = self.gnn_detector.evaluate_graph(x, edge_idx, list(self.grid_sim.buses.keys()))
        anomalies = self.anomaly_detector.detect_anomalies(processed, gnn_res)
        alerts = self.cyber_agent.analyze_threats(anomalies, processed, gnn_res, self.grid_sim.branches)
        topology = self.top_analyzer.analyze_topology(graph)

        decision = self.decision_engine.process_incident(anomalies, alerts, processed, topology, [])
        workflow_nodes = self.decision_engine.workflow.get_graph().nodes

        self.assertTrue({"triage", "risk_assessment", "recovery_planning", "safety_validation"}.issubset(workflow_nodes))
        self.assertEqual(
            [message.role for message in decision.agent_logs],
            ["TRIAGE", "RISK_ASSESSOR", "PLANNER", "SAFETY_VALIDATOR"],
        )

    def test_12_replay_attack_is_classified_and_remediated(self):
        """Replay telemetry must not fall through without an agent action."""
        self.attack_inj.inject_attack(attack_type="REPLAY_ATTACK", target_bus=13)
        raw = self.attack_inj.apply_attacks_to_telemetry(self.grid_sim.get_telemetry_snapshot(), self.grid_sim)
        processed = self.processor.process_telemetry_stream(raw)
        x, edge_idx, _, graph = self.graph_builder.build_graph_tensors(processed, self.grid_sim.branches)
        gnn_res = self.gnn_detector.evaluate_graph(x, edge_idx, list(self.grid_sim.buses.keys()))
        anomalies = self.anomaly_detector.detect_anomalies(processed, gnn_res)
        alerts = self.cyber_agent.analyze_threats(anomalies, processed, gnn_res, self.grid_sim.branches)
        topology = self.top_analyzer.analyze_topology(graph)

        self.assertEqual(alerts[0].attack_type, "REPLAY_ATTACK")
        decision = self.decision_engine.process_incident(anomalies, alerts, processed, topology, [])
        self.assertTrue(decision.is_safe_to_execute)
        self.assertIn("REAUTHENTICATE_SENSOR_STREAM", [action.parameter for action in decision.recovery_playbook])

        recovery = self.self_healing_ctrl.execute_playbook(decision, processed)
        self.assertTrue(recovery.success)
        self.assertFalse(self.attack_inj.active_attacks)

    def test_13_breaker_hijack_restores_authorized_breaker(self):
        """A hijacked breaker is quarantined and closed by the recovery controller."""
        self.attack_inj.inject_attack(attack_type="BREAKER_HIJACK", target_bus=1, target_branch=1)
        raw = self.attack_inj.apply_attacks_to_telemetry(self.grid_sim.get_telemetry_snapshot(), self.grid_sim)
        processed = self.processor.process_telemetry_stream(raw)
        x, edge_idx, _, graph = self.graph_builder.build_graph_tensors(processed, self.grid_sim.branches)
        gnn_res = self.gnn_detector.evaluate_graph(x, edge_idx, list(self.grid_sim.buses.keys()))
        anomalies = self.anomaly_detector.detect_anomalies(processed, gnn_res)
        alerts = self.cyber_agent.analyze_threats(anomalies, processed, gnn_res, self.grid_sim.branches)
        topology = self.top_analyzer.analyze_topology(graph)
        candidates = self.top_analyzer.find_restoration_paths(graph, topology["isolated_buses"], self.grid_sim.branches)

        decision = self.decision_engine.process_incident(anomalies, alerts, processed, topology, candidates)
        self.assertIn("RESTORE_BREAKER", [action.action_type for action in decision.recovery_playbook])

        recovery = self.self_healing_ctrl.execute_playbook(decision, processed)
        self.assertTrue(recovery.success)
        self.assertEqual(self.grid_sim.branches[1].status, 1)
        self.assertFalse(self.attack_inj.active_attacks)

if __name__ == "__main__":
    unittest.main()
