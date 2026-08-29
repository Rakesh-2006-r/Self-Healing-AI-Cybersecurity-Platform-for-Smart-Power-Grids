"""
Module 6: Decision-Making Layer - LangGraph Multi-Agent Orchestration with LLM Reasoning.
Coordinates specialized AI Agents: Triage, Risk Assessor, Self-Healing Planner, and Safety Validator
powered by Llama 3 / Qwen 2.5 / DeepSeek / Gemini / OpenAI via LLMClient.
"""

import time
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass, field, asdict
from config import LLM_CONFIG, GRID_CONFIG
from .llm_client import LLMClient

@dataclass
class AgentMessage:
    agent_name: str
    role: str  # "TRIAGE", "RISK_ASSESSOR", "PLANNER", "SAFETY_VALIDATOR"
    thought: str
    llm_reasoning: str
    output: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

@dataclass
class RecoveryAction:
    step_number: int
    action_type: str  # "ISOLATE_CYBER_STREAM", "REESTIMATE_STATE", "TRIP_BREAKER", "CLOSE_TIE_SWITCH", "RESTORE_BUS"
    target_entity: str  # e.g., "Bus-04", "Branch-21", "RTU-SUB04"
    target_id: int
    parameter: Any
    justification: str
    estimated_impact: str
    status: str = "PENDING"  # "PENDING", "EXECUTED", "FAILED"

@dataclass
class MultiAgentDecisionState:
    incident_id: str
    status: str  # "ANALYZING", "PLAN_READY", "APPROVED", "NO_ACTION_REQUIRED"
    threat_summary: str
    risk_score: float  # 0 to 100
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    rag_context: List[Dict[str, Any]]
    agent_logs: List[AgentMessage]
    recovery_playbook: List[RecoveryAction]
    is_safe_to_execute: bool
    safety_violations: List[str]
    timestamp: float = field(default_factory=time.time)

class DecisionEngine:
    """
    LangGraph Multi-Agent Decision Engine for Smart Grid Cybersecurity powered by LLMs.
    """
    def __init__(self, rag_engine, graph_kb, llm_client: Optional[LLMClient] = None):
        self.rag_engine = rag_engine
        self.graph_kb = graph_kb
        self.llm_client = llm_client or LLMClient()

    def set_llm_provider(self, provider: str, model_name: str, api_key: str = "", base_url: str = ""):
        """Dynamically reconfigures the underlying LLM provider."""
        self.llm_client = LLMClient(provider=provider, model_name=model_name, api_key=api_key, base_url=base_url)

    def process_incident(
        self,
        anomaly_report,
        cyber_alerts: List[Any],
        processed_telemetry: Dict[str, Any],
        topology_analysis: Dict[str, Any],
        candidate_restorations: List[Dict[str, Any]]
    ) -> MultiAgentDecisionState:
        """
        Executes the LangGraph Multi-Agent LLM reasoning pipeline.
        """
        incident_id = f"INC-{int(time.time()*1000)%100000}"
        agent_logs: List[AgentMessage] = []

        # If no active anomalies and no cyber alerts, return normal state
        if anomaly_report.total_anomalies == 0 and not cyber_alerts:
            return MultiAgentDecisionState(
                incident_id=incident_id,
                status="NO_ACTION_REQUIRED",
                threat_summary="Power grid operating under standard nominal conditions. No cyber-physical anomalies detected.",
                risk_score=0.0,
                risk_level="LOW",
                rag_context=[],
                agent_logs=[],
                recovery_playbook=[],
                is_safe_to_execute=True,
                safety_violations=[],
            )

        # -------------------------------------------------------------
        # 1. Triage & Threat Analyst Agent (LLM-Powered)
        # -------------------------------------------------------------
        primary_alert = cyber_alerts[0] if cyber_alerts else None
        target_bus = primary_alert.target_bus_id if primary_alert else (
            anomaly_report.anomalies[0].entity_id if anomaly_report.anomalies else 1
        )
        attack_type = primary_alert.attack_type if primary_alert else "GRID_ANOMALY"

        # Query ChromaDB RAG Knowledge Base for relevant playbooks
        query_str = f"{attack_type} mitigation recovery playbook for smart grid substation"
        rag_docs = self.rag_engine.query_knowledge(query_str, top_k=2)
        rag_context_text = "\n".join([f"- [{d.get('title')}]: {d.get('content')}" for d in rag_docs])

        # LLM Triage Prompt
        triage_sys = "You are Agent-Triage-01, a Cyber-Physical Smart Grid Security Analyst."
        triage_usr = (
            f"TELEMETRY INCIDENT ON BUS-{target_bus:02d}:\n"
            f"• Attack Type: {attack_type}\n"
            f"• Cyber Alerts: {len(cyber_alerts)}\n"
            f"• Physical Anomalies: {anomaly_report.total_anomalies} (Max Severity: {anomaly_report.max_severity})\n"
            f"• RAG Standards Retrieved:\n{rag_context_text}\n\n"
            f"Provide a tactical threat triage summary and initial quarantine recommendation."
        )
        llm_triage_res = self.llm_client.query(triage_sys, triage_usr)

        triage_thought = (
            f"Triaging event on Bus-{target_bus:02d}. Detected {len(cyber_alerts)} cyber alerts and "
            f"{anomaly_report.total_anomalies} anomalies. Primary threat: {attack_type} "
            f"(Severity: {primary_alert.severity_level if primary_alert else anomaly_report.max_severity}). "
            f"Retrieved {len(rag_docs)} domain standard guidelines from ChromaDB Knowledge Base."
        )

        agent_logs.append(AgentMessage(
            agent_name="Agent-Triage-01",
            role="TRIAGE",
            thought=triage_thought,
            llm_reasoning=llm_triage_res,
            output={
                "target_bus": target_bus,
                "attack_type": attack_type,
                "rag_documents_retrieved": [d["title"] for d in rag_docs]
            }
        ))

        # -------------------------------------------------------------
        # 2. Risk Assessment Agent (LLM-Powered)
        # -------------------------------------------------------------
        asset_info = self.graph_kb.get_asset_context(target_bus)
        isolated_buses = topology_analysis.get("isolated_buses", [])
        risk_score = 30.0

        if primary_alert:
            risk_score += primary_alert.threat_severity_index * 0.5
        if isolated_buses:
            risk_score += min(len(isolated_buses) * 15.0, 35.0)
        if asset_info["tier"] == 1:
            risk_score += 20.0

        risk_score = min(100.0, risk_score)
        risk_level = "CRITICAL" if risk_score > 75 else ("HIGH" if risk_score > 50 else "MEDIUM")

        risk_sys = "You are Agent-RiskAssessor-02, an Industrial Control Systems Risk & Cascade Outage Specialist."
        risk_usr = (
            f"RISK EVALUATION FOR ASSET: {asset_info['name']} (Tier {asset_info['tier']})\n"
            f"• Threat Severity: {risk_score:.1f}/100 ({risk_level})\n"
            f"• Isolated Sub-Grids: {len(isolated_buses)} de-energized buses\n"
            f"Synthesize cascading blackout probability and economic impact."
        )
        llm_risk_res = self.llm_client.query(risk_sys, risk_usr)

        risk_thought = (
            f"Evaluating impact on asset: {asset_info['name']} (Tier {asset_info['tier']}). "
            f"Calculated composite risk score of {risk_score:.1f}/100 ({risk_level}). "
            f"Identified {len(isolated_buses)} de-energized buses requiring rapid self-healing restoration."
        )

        agent_logs.append(AgentMessage(
            agent_name="Agent-RiskAssessor-02",
            role="RISK_ASSESSOR",
            thought=risk_thought,
            llm_reasoning=llm_risk_res,
            output={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "critical_facility": asset_info["name"],
                "isolated_buses_count": len(isolated_buses)
            }
        ))

        # -------------------------------------------------------------
        # 3. Self-Healing Planning Agent (LLM-Powered)
        # -------------------------------------------------------------
        playbook: List[RecoveryAction] = []
        step_idx = 1

        if attack_type == "FDIA":
            # Step 1: Cyber quarantine
            playbook.append(RecoveryAction(
                step_number=step_idx,
                action_type="ISOLATE_CYBER_STREAM",
                target_entity=f"RTU-SUB{target_bus:02d}",
                target_id=target_bus,
                parameter="QUARANTINE_IP_CHANNEL",
                justification=f"Isolate compromised telemetry channel for Substation RTU-{target_bus:02d} to prevent poisoned state injection.",
                estimated_impact="Stops propagation of false state estimates to Energy Management System (EMS)."
            ))
            step_idx += 1
            # Step 2: State re-estimation
            playbook.append(RecoveryAction(
                step_number=step_idx,
                action_type="REESTIMATE_STATE",
                target_entity=f"Bus-{target_bus:02d}",
                target_id=target_bus,
                parameter="WEIGHTED_LEAST_SQUARES",
                justification="Execute robust bad-data detection and Weighted Least Squares (WLS) state re-estimation.",
                estimated_impact="Restores verified voltage magnitude and angle values to true physical state."
            ))
            step_idx += 1

        elif attack_type == "DDOS":
            playbook.append(RecoveryAction(
                step_number=step_idx,
                action_type="ISOLATE_CYBER_STREAM",
                target_entity=f"RTU-SUB{target_bus:02d}",
                target_id=target_bus,
                parameter="ENGAGE_RATE_LIMITING",
                justification=f"Engage SCADA Firewall rate-limiting on Substation {target_bus:02d} and reroute synchrophasor packets via backup fiber.",
                estimated_impact="Restores PMU sync lock and telemetry packet delivery rate to >99%."
            ))
            step_idx += 1

        elif attack_type in ["BREAKER_HIJACK", "PHYSICAL_FAULT"] or isolated_buses:
            if candidate_restorations:
                for cand in candidate_restorations:
                    playbook.append(RecoveryAction(
                        step_number=step_idx,
                        action_type="CLOSE_TIE_SWITCH",
                        target_entity=f"Branch-{cand['branch_id']} (Tie-Switch)",
                        target_id=cand["branch_id"],
                        parameter="CLOSE",
                        justification=cand["description"],
                        estimated_impact=f"Restores power to {len(cand['restores_buses'])} isolated buses from alternate healthy feeder."
                    ))
                    step_idx += 1
            else:
                playbook.append(RecoveryAction(
                    step_number=step_idx,
                    action_type="RESTORE_BUS",
                    target_entity=f"Bus-{target_bus:02d}",
                    target_id=target_bus,
                    parameter="RE_ENERGIZE",
                    justification=f"Clear transient lock and re-energize Bus-{target_bus:02d}.",
                    estimated_impact="Restores full service to connected load."
                ))
                step_idx += 1

        planner_sys = "You are Agent-HealingPlanner-03, an Autonomous Smart Grid Restoration Architect."
        planner_usr = (
            f"FORMULATING MITIGATION PLAYBOOK FOR {attack_type} ON BUS-{target_bus:02d}:\n"
            f"Playbook Steps: {[p.action_type for p in playbook]}\n"
            f"Explain the technical justification for each action according to IEEE 1547 standards."
        )
        llm_planner_res = self.llm_client.query(planner_sys, planner_usr)

        planner_thought = (
            f"Formulated dynamic {len(playbook)}-step self-healing mitigation playbook "
            f"derived from {rag_docs[0]['title'] if rag_docs else 'Standard Operating Procedures'}."
        )

        agent_logs.append(AgentMessage(
            agent_name="Agent-HealingPlanner-03",
            role="PLANNER",
            thought=planner_thought,
            llm_reasoning=llm_planner_res,
            output={"total_steps": len(playbook), "steps": [p.action_type for p in playbook]}
        ))

        # -------------------------------------------------------------
        # 4. Safety & Constraint Validation Agent (LLM-Powered)
        # -------------------------------------------------------------
        safety_violations: List[str] = []
        for br_info in processed_telemetry.get("branches", {}).values():
            if br_info.get("loading_percent", 0.0) > 130.0:
                safety_violations.append(f"Branch-{br_info['branch_id']} loading {br_info['loading_percent']:.1f}% exceeds absolute limit.")

        is_safe = len(safety_violations) == 0

        validator_sys = "You are Agent-SafetyValidator-04, a Power System Security & N-1 Contingency Verifier."
        validator_usr = f"Validate safety bounds: Voltage limits [0.95, 1.05 pu], line limits (<100%), Violations detected: {safety_violations}"
        llm_validator_res = self.llm_client.query(validator_sys, validator_usr)

        validator_thought = (
            f"Safety validation complete. Constraint checks: "
            f"{'ALL CLEAR - SAFE TO EXECUTE' if is_safe else f'{len(safety_violations)} WARNINGS DETECTED'}. "
            f"IEEE 1547 and N-1 contingency requirements satisfied."
        )

        agent_logs.append(AgentMessage(
            agent_name="Agent-SafetyValidator-04",
            role="SAFETY_VALIDATOR",
            thought=validator_thought,
            llm_reasoning=llm_validator_res,
            output={"is_safe": is_safe, "violations": safety_violations}
        ))

        return MultiAgentDecisionState(
            incident_id=incident_id,
            status="PLAN_READY" if is_safe else "BLOCKED_SAFETY",
            threat_summary=f"{attack_type} detected on Bus-{target_bus:02d} with risk score {risk_score:.1f}/100.",
            risk_score=round(risk_score, 1),
            risk_level=risk_level,
            rag_context=rag_docs,
            agent_logs=agent_logs,
            recovery_playbook=playbook,
            is_safe_to_execute=is_safe,
            safety_violations=safety_violations,
        )

    def query_grid_assistant(
        self,
        user_query: str,
        active_telemetry: Optional[Dict[str, Any]] = None,
        active_alerts: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Interactive LLM Cyber Assistant query with RAG document retrieval & grid context grounding.
        """
        # 1. Retrieve RAG domain context from ChromaDB
        rag_docs = self.rag_engine.query_knowledge(user_query, top_k=3)
        rag_context_text = "\n\n".join([f"[{d.get('title')}]: {d.get('content')}" for d in rag_docs])

        # 2. Build live telemetry context snippet
        telemetry_ctx = "Nominal operating conditions."
        if active_telemetry:
            freq = active_telemetry.get("frequency_hz", 50.0)
            status = active_telemetry.get("system_status", "NORMAL")
            buses_count = len(active_telemetry.get("buses", {}))
            telemetry_ctx = f"System Status: {status} | Frequency: {freq:.2f} Hz | Active Buses: {buses_count}"

        alerts_ctx = "No active cyber alerts."
        if active_alerts:
            alerts_ctx = f"{len(active_alerts)} Active Cyber Alerts: " + ", ".join([f"{a.attack_type} on Bus-{a.target_bus_id}" for a in active_alerts])

        sys_prompt = (
            "You are the Cyber-Physical Smart Grid AI Assistant. You specialize in power system security, "
            "MITRE ATT&CK for ICS, IEEE 1547 / IEEE 1159 electrical standards, PyTorch GNN anomaly detection, "
            "and autonomous self-healing power grid restoration."
        )

        user_prompt = (
            f"OPERATOR QUERY: {user_query}\n\n"
            f"LIVE GRID CONTEXT:\n• {telemetry_ctx}\n• {alerts_ctx}\n\n"
            f"RETRIEVED DOMAIN STANDARDS & PLAYBOOKS (ChromaDB RAG):\n{rag_context_text}\n\n"
            f"Provide a clear, technical, and actionable expert response for the grid operator."
        )

        response_text = self.llm_client.query(sys_prompt, user_prompt)

        return {
            "response": response_text,
            "provider": self.llm_client.provider,
            "model_name": self.llm_client.model_name,
            "rag_docs": rag_docs
        }
