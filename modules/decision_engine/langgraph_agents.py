"""
Module 6: LangGraph multi-agent decision layer for smart-grid cybersecurity.

The workflow follows the project architecture: Triage -> Risk Assessment ->
Recovery Planning -> Safety Validation. Each node has a narrowly scoped
responsibility and passes structured state to the next node.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from config import GRID_CONFIG
from .llm_client import LLMClient


@dataclass
class AgentMessage:
    agent_name: str
    role: str
    thought: str
    llm_reasoning: str
    output: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class RecoveryAction:
    step_number: int
    action_type: str
    target_entity: str
    target_id: int
    parameter: Any
    justification: str
    estimated_impact: str
    status: str = "PENDING"


@dataclass
class MultiAgentDecisionState:
    incident_id: str
    status: str
    threat_summary: str
    risk_score: float
    risk_level: str
    rag_context: List[Dict[str, Any]]
    agent_logs: List[AgentMessage]
    recovery_playbook: List[RecoveryAction]
    is_safe_to_execute: bool
    safety_violations: List[str]
    timestamp: float = field(default_factory=time.time)


class IncidentWorkflowState(TypedDict, total=False):
    """The data contract shared by the LangGraph agent nodes."""

    incident_id: str
    anomaly_report: Any
    cyber_alerts: List[Any]
    processed_telemetry: Dict[str, Any]
    topology_analysis: Dict[str, Any]
    candidate_restorations: List[Dict[str, Any]]
    primary_alert: Any
    target_bus: int
    attack_type: str
    rag_docs: List[Dict[str, Any]]
    rag_context_text: str
    asset_info: Dict[str, Any]
    risk_score: float
    risk_level: str
    recovery_playbook: List[RecoveryAction]
    safety_violations: List[str]
    is_safe_to_execute: bool
    agent_logs: List[AgentMessage]


class DecisionEngine:
    """Coordinates the four specialized agents through a LangGraph workflow."""

    def __init__(self, rag_engine, graph_kb, llm_client: Optional[LLMClient] = None):
        self.rag_engine = rag_engine
        self.graph_kb = graph_kb
        self.llm_client = llm_client or LLMClient()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Compile the ordered, inspectable LangGraph agent workflow once."""
        workflow = StateGraph(IncidentWorkflowState)
        workflow.add_node("triage", self._triage_agent)
        workflow.add_node("risk_assessment", self._risk_assessment_agent)
        workflow.add_node("recovery_planning", self._recovery_planning_agent)
        workflow.add_node("safety_validation", self._safety_validation_agent)
        workflow.add_edge(START, "triage")
        workflow.add_edge("triage", "risk_assessment")
        workflow.add_edge("risk_assessment", "recovery_planning")
        workflow.add_edge("recovery_planning", "safety_validation")
        workflow.add_edge("safety_validation", END)
        return workflow.compile()

    def set_llm_provider(self, provider: str, model_name: str, api_key: str = "", base_url: str = ""):
        """Dynamically reconfigure the LLM used by every agent node."""
        self.llm_client = LLMClient(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )

    def _ask_agent(self, system_prompt: str, user_prompt: str) -> str:
        """Return usable agent output even when a remote provider is unavailable."""
        response = self.llm_client.query(system_prompt, user_prompt)
        if isinstance(response, str) and response.strip():
            return response.strip()
        return "The configured LLM returned no response; local safety rules remain in effect."

    def _triage_agent(self, state: IncidentWorkflowState) -> Dict[str, Any]:
        anomaly_report = state["anomaly_report"]
        cyber_alerts = state["cyber_alerts"]
        primary_alert = cyber_alerts[0] if cyber_alerts else None
        target_bus = int(
            primary_alert.target_bus_id
            if primary_alert
            else (anomaly_report.anomalies[0].entity_id if anomaly_report.anomalies else 1)
        )
        attack_type = primary_alert.attack_type if primary_alert else "GRID_ANOMALY"

        rag_docs = self.rag_engine.query_knowledge(
            f"{attack_type} mitigation recovery playbook for smart grid substation",
            top_k=2,
        )
        rag_context_text = "\n".join(
            f"- [{document.get('title', 'Untitled')}]: {document.get('content', '')}"
            for document in rag_docs
        ) or "- No matching knowledge-base documents were retrieved."

        llm_reasoning = self._ask_agent(
            "You are Agent-Triage-01, a Cyber-Physical Smart Grid Security Analyst.",
            (
                f"TELEMETRY INCIDENT ON BUS-{target_bus:02d}:\n"
                f"- Attack Type: {attack_type}\n"
                f"- Cyber Alerts: {len(cyber_alerts)}\n"
                f"- Physical Anomalies: {anomaly_report.total_anomalies} "
                f"(Max Severity: {anomaly_report.max_severity})\n"
                f"- RAG Standards Retrieved:\n{rag_context_text}\n\n"
                "Provide a tactical threat triage summary and initial quarantine recommendation."
            ),
        )
        severity = primary_alert.severity_level if primary_alert else anomaly_report.max_severity
        thought = (
            f"Triaging event on Bus-{target_bus:02d}. Detected {len(cyber_alerts)} cyber alerts and "
            f"{anomaly_report.total_anomalies} anomalies. Primary threat: {attack_type} "
            f"(Severity: {severity}). Retrieved {len(rag_docs)} relevant knowledge-base documents."
        )
        log = AgentMessage(
            agent_name="Agent-Triage-01",
            role="TRIAGE",
            thought=thought,
            llm_reasoning=llm_reasoning,
            output={
                "target_bus": target_bus,
                "attack_type": attack_type,
                "rag_documents_retrieved": [document.get("title", "Untitled") for document in rag_docs],
            },
        )
        return {
            "primary_alert": primary_alert,
            "target_bus": target_bus,
            "attack_type": attack_type,
            "rag_docs": rag_docs,
            "rag_context_text": rag_context_text,
            "agent_logs": [*state.get("agent_logs", []), log],
        }

    def _risk_assessment_agent(self, state: IncidentWorkflowState) -> Dict[str, Any]:
        primary_alert = state.get("primary_alert")
        target_bus = state["target_bus"]
        isolated_buses = state["topology_analysis"].get("isolated_buses", [])
        asset_info = self.graph_kb.get_asset_context(target_bus)

        risk_score = 30.0
        if primary_alert:
            risk_score += float(primary_alert.threat_severity_index) * 0.5
        if isolated_buses:
            risk_score += min(len(isolated_buses) * 15.0, 35.0)
        if asset_info.get("tier") == 1:
            risk_score += 20.0
        risk_score = min(100.0, risk_score)
        risk_level = "CRITICAL" if risk_score > 75 else ("HIGH" if risk_score > 50 else "MEDIUM")

        llm_reasoning = self._ask_agent(
            "You are Agent-RiskAssessor-02, an Industrial Control Systems Risk and Cascade Outage Specialist.",
            (
                f"RISK EVALUATION FOR ASSET: {asset_info['name']} (Tier {asset_info['tier']})\n"
                f"- Threat Severity: {risk_score:.1f}/100 ({risk_level})\n"
                f"- Isolated Sub-Grids: {len(isolated_buses)} de-energized buses\n"
                "Synthesize cascading blackout probability and economic impact."
            ),
        )
        thought = (
            f"Evaluating impact on asset: {asset_info['name']} (Tier {asset_info['tier']}). "
            f"Calculated composite risk score of {risk_score:.1f}/100 ({risk_level}). "
            f"Identified {len(isolated_buses)} de-energized buses requiring restoration."
        )
        log = AgentMessage(
            agent_name="Agent-RiskAssessor-02",
            role="RISK_ASSESSOR",
            thought=thought,
            llm_reasoning=llm_reasoning,
            output={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "critical_facility": asset_info["name"],
                "isolated_buses_count": len(isolated_buses),
            },
        )
        return {
            "asset_info": asset_info,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "agent_logs": [*state.get("agent_logs", []), log],
        }

    @staticmethod
    def _action(
        step_number: int,
        action_type: str,
        target_entity: str,
        target_id: int,
        parameter: str,
        justification: str,
        estimated_impact: str,
    ) -> RecoveryAction:
        return RecoveryAction(
            step_number=step_number,
            action_type=action_type,
            target_entity=target_entity,
            target_id=target_id,
            parameter=parameter,
            justification=justification,
            estimated_impact=estimated_impact,
        )

    def _make_playbook(self, state: IncidentWorkflowState) -> List[RecoveryAction]:
        """Create executable actions for every attack type exposed by the dashboard."""
        target_bus = state["target_bus"]
        attack_type = state["attack_type"]
        candidates = state["candidate_restorations"]
        playbook: List[RecoveryAction] = []

        def add(action_type, target_entity, target_id, parameter, justification, impact):
            playbook.append(
                self._action(
                    len(playbook) + 1,
                    action_type,
                    target_entity,
                    target_id,
                    parameter,
                    justification,
                    impact,
                )
            )

        if attack_type == "FDIA":
            add(
                "ISOLATE_CYBER_STREAM",
                f"RTU-SUB{target_bus:02d}",
                target_bus,
                "QUARANTINE_IP_CHANNEL",
                f"Quarantine the compromised telemetry channel for RTU-{target_bus:02d}.",
                "Stops poisoned measurements from reaching state estimation.",
            )
            add(
                "REESTIMATE_STATE",
                f"Bus-{target_bus:02d}",
                target_bus,
                "WEIGHTED_LEAST_SQUARES",
                "Run bad-data detection and weighted least-squares state re-estimation.",
                "Restores a verified voltage magnitude and angle estimate.",
            )
        elif attack_type == "DDOS":
            add(
                "ISOLATE_CYBER_STREAM",
                f"RTU-SUB{target_bus:02d}",
                target_bus,
                "ENGAGE_RATE_LIMITING",
                f"Rate-limit RTU-{target_bus:02d} traffic and reroute synchrophasor packets.",
                "Restores the authenticated telemetry path.",
            )
        elif attack_type == "REPLAY_ATTACK":
            add(
                "ISOLATE_CYBER_STREAM",
                f"RTU-SUB{target_bus:02d}",
                target_bus,
                "REAUTHENTICATE_SENSOR_STREAM",
                f"Quarantine replayed measurements and re-authenticate RTU-{target_bus:02d}.",
                "Prevents stale telemetry from masking live grid conditions.",
            )
            add(
                "REESTIMATE_STATE",
                f"Bus-{target_bus:02d}",
                target_bus,
                "PMU_CROSS_VALIDATION",
                "Re-estimate state using trusted PMU and SCADA measurements.",
                "Restores a live, cross-validated operational state.",
            )
        elif attack_type == "BREAKER_HIJACK":
            add(
                "ISOLATE_CYBER_STREAM",
                f"RTU-SUB{target_bus:02d}",
                target_bus,
                "REVOKE_UNAUTHORIZED_COMMAND",
                "Revoke the unauthorized breaker-control command channel before reconfiguration.",
                "Prevents the malicious command from reopening restored feeders.",
            )
            branch_id = int(getattr(state.get("primary_alert"), "target_branch_id", 0) or target_bus)
            add(
                "RESTORE_BREAKER",
                f"Branch-{branch_id}",
                branch_id,
                "CLOSE_AUTHORIZED_BREAKER",
                f"Close Branch-{branch_id} only after the control channel is quarantined.",
                "Restores the authorized feeder path without replaying the malicious command.",
            )

        if attack_type in {"BREAKER_HIJACK", "PHYSICAL_FAULT", "GRID_ANOMALY"} or state["topology_analysis"].get("isolated_buses"):
            if candidates:
                for candidate in candidates:
                    branch_id = int(candidate["branch_id"])
                    add(
                        "CLOSE_TIE_SWITCH",
                        f"Branch-{branch_id} (Tie-Switch)",
                        branch_id,
                        "CLOSE",
                        candidate["description"],
                        f"Restores power to {len(candidate['restores_buses'])} isolated buses from a healthy feeder.",
                    )
            elif attack_type != "BREAKER_HIJACK":
                add(
                    "RESTORE_BUS",
                    f"Bus-{target_bus:02d}",
                    target_bus,
                    "RE_ENERGIZE",
                    f"Isolate the affected section and re-energize Bus-{target_bus:02d} after verification.",
                    "Restores service to the connected load when electrical bounds permit.",
                )
        return playbook

    def _recovery_planning_agent(self, state: IncidentWorkflowState) -> Dict[str, Any]:
        playbook = self._make_playbook(state)
        target_bus = state["target_bus"]
        attack_type = state["attack_type"]
        llm_reasoning = self._ask_agent(
            "You are Agent-HealingPlanner-03, an Autonomous Smart Grid Restoration Architect.",
            (
                f"FORMULATING MITIGATION PLAYBOOK FOR {attack_type} ON BUS-{target_bus:02d}:\n"
                f"Playbook Steps: {[action.action_type for action in playbook]}\n"
                "Explain the technical justification for each action according to IEEE 1547 standards."
            ),
        )
        source = state["rag_docs"][0].get("title", "Standard Operating Procedures") if state["rag_docs"] else "Standard Operating Procedures"
        thought = f"Formulated an executable {len(playbook)}-step recovery playbook derived from {source}."
        log = AgentMessage(
            agent_name="Agent-HealingPlanner-03",
            role="PLANNER",
            thought=thought,
            llm_reasoning=llm_reasoning,
            output={"total_steps": len(playbook), "steps": [action.action_type for action in playbook]},
        )
        return {
            "recovery_playbook": playbook,
            "agent_logs": [*state.get("agent_logs", []), log],
        }

    def _safety_validation_agent(self, state: IncidentWorkflowState) -> Dict[str, Any]:
        violations: List[str] = []
        for branch_info in state["processed_telemetry"].get("branches", {}).values():
            loading = float(branch_info.get("loading_percent", 0.0))
            if loading > GRID_CONFIG["CRITICAL_LINE_LOADING_PERCENT"]:
                violations.append(
                    f"Branch-{branch_info['branch_id']} loading {loading:.1f}% exceeds the critical thermal limit."
                )

        if not state.get("recovery_playbook"):
            violations.append("No executable recovery action was produced for the detected incident.")

        is_safe = not violations
        llm_reasoning = self._ask_agent(
            "You are Agent-SafetyValidator-04, a Power System Security and N-1 Contingency Verifier.",
            (
                "Validate the proposed response against voltage limits [0.95, 1.05 pu], "
                "the critical line loading limit, and N-1 constraints. "
                f"Detected violations: {violations}"
            ),
        )
        thought = (
            "Safety validation complete. "
            + ("ALL CLEAR - SAFE TO EXECUTE." if is_safe else f"{len(violations)} SAFETY BLOCKER(S) DETECTED.")
        )
        log = AgentMessage(
            agent_name="Agent-SafetyValidator-04",
            role="SAFETY_VALIDATOR",
            thought=thought,
            llm_reasoning=llm_reasoning,
            output={"is_safe": is_safe, "violations": violations},
        )
        return {
            "safety_violations": violations,
            "is_safe_to_execute": is_safe,
            "agent_logs": [*state.get("agent_logs", []), log],
        }

    def process_incident(
        self,
        anomaly_report,
        cyber_alerts: List[Any],
        processed_telemetry: Dict[str, Any],
        topology_analysis: Dict[str, Any],
        candidate_restorations: List[Dict[str, Any]],
    ) -> MultiAgentDecisionState:
        """Run an incident through the four-node LangGraph workflow."""
        incident_id = f"INC-{int(time.time() * 1000) % 100000}"
        if anomaly_report.total_anomalies == 0 and not cyber_alerts:
            return MultiAgentDecisionState(
                incident_id=incident_id,
                status="NO_ACTION_REQUIRED",
                threat_summary="Power grid operating under nominal conditions. No cyber-physical anomalies detected.",
                risk_score=0.0,
                risk_level="LOW",
                rag_context=[],
                agent_logs=[],
                recovery_playbook=[],
                is_safe_to_execute=True,
                safety_violations=[],
            )

        result = self.workflow.invoke(
            {
                "incident_id": incident_id,
                "anomaly_report": anomaly_report,
                "cyber_alerts": cyber_alerts,
                "processed_telemetry": processed_telemetry,
                "topology_analysis": topology_analysis,
                "candidate_restorations": candidate_restorations,
                "agent_logs": [],
            }
        )
        is_safe = bool(result["is_safe_to_execute"])
        attack_type = result["attack_type"]
        target_bus = result["target_bus"]
        return MultiAgentDecisionState(
            incident_id=incident_id,
            status="PLAN_READY" if is_safe else "BLOCKED_SAFETY",
            threat_summary=(
                f"{attack_type} detected on Bus-{target_bus:02d} with risk score "
                f"{result['risk_score']:.1f}/100."
            ),
            risk_score=round(float(result["risk_score"]), 1),
            risk_level=result["risk_level"],
            rag_context=result["rag_docs"],
            agent_logs=result["agent_logs"],
            recovery_playbook=result["recovery_playbook"],
            is_safe_to_execute=is_safe,
            safety_violations=result["safety_violations"],
        )

    def query_grid_assistant(
        self,
        user_query: str,
        active_telemetry: Optional[Dict[str, Any]] = None,
        active_alerts: Optional[List[Any]] = None,
        anomaly_report: Optional[Any] = None,
        gnn_results: Optional[Dict[str, Any]] = None,
        active_attacks: Optional[List[Any]] = None,
        last_decision: Optional[Any] = None,
        last_recovery: Optional[Any] = None,
        grid_sim: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Answer an operator question with RAG context and current grid telemetry."""
        rag_docs = self.rag_engine.query_knowledge(user_query, top_k=3)
        rag_context_text = "\n\n".join(
            f"[{document.get('title')}]: {document.get('content')}" for document in rag_docs
        )

        # 1. System-wide telemetry metrics
        sys_freq = 50.0
        freq_dev = 0.0
        restoration_pct = 100.0
        if active_telemetry:
            sys_freq = active_telemetry.get("frequency_hz", 50.0)
            freq_dev = active_telemetry.get("frequency_deviation_hz", 0.0)
            restoration_pct = active_telemetry.get("restoration_ratio_pct", 100.0)

        health_score = getattr(anomaly_report, "system_health_score", 100.0) if anomaly_report else 100.0

        # 2. Extract Bus-level Measurements
        buses_dict: Dict[int, Any] = {}
        if active_telemetry and "processed_buses" in active_telemetry:
            buses_dict = active_telemetry["processed_buses"]
        elif active_telemetry and "buses" in active_telemetry:
            buses_dict = active_telemetry["buses"]
        elif grid_sim and hasattr(grid_sim, "buses"):
            buses_dict = {
                b_id: {
                    "bus_id": b_id,
                    "bus_name": f"Bus-{b_id:02d}",
                    "voltage_pu": b.voltage_pu,
                    "voltage_angle_deg": getattr(b, "voltage_angle_deg", 0.0),
                    "active_power_mw": b.active_power_mw,
                    "reactive_power_mvar": getattr(b, "reactive_power_mvar", 0.0),
                    "status": b.status,
                    "rtu_packet_loss_rate": getattr(b, "rtu_packet_loss_rate", 0.0),
                }
                for b_id, b in grid_sim.buses.items()
            }

        bus_summary_lines = []
        for b_id, b_data in sorted(buses_dict.items(), key=lambda x: int(x[0])):
            b_id_int = int(b_id)
            v = b_data.get("voltage_pu", 1.0)
            p = b_data.get("active_power_mw", 0.0)
            q = b_data.get("reactive_power_mvar", 0.0)
            st_val = b_data.get("status", "ENERGIZED")
            loss = b_data.get("rtu_packet_loss_rate", 0.0)
            gnn_err = 0.0
            if gnn_results and "node_anomaly_scores" in gnn_results:
                gnn_err = gnn_results["node_anomaly_scores"].get(b_id_int, 0.0)
            bus_summary_lines.append(
                f"Bus-{b_id_int:02d}: V={v:.3f} pu, P={p:.2f} MW, Q={q:.2f} MVar, RTU_Loss={loss:.1f}%, Status={st_val}, GNN_Loss={gnn_err:.4f}"
            )
        bus_summary_str = "\n".join(bus_summary_lines) if bus_summary_lines else "Nominal bus state across feeder network."

        # 3. Active Cyber Attacks
        attacks_lines = []
        if active_attacks:
            for atk in active_attacks:
                if isinstance(atk, dict):
                    attacks_lines.append(
                        f"Attack ID={atk.get('id', 'N/A')}, Type={atk.get('attack_type', 'N/A')}, Target Bus={atk.get('target_bus', 'N/A')}, Description={atk.get('description', 'N/A')}"
                    )
                else:
                    attacks_lines.append(
                        f"Attack ID={getattr(atk, 'id', 'N/A')}, Type={getattr(atk, 'attack_type', 'N/A')}, Target Bus={getattr(atk, 'target_bus', 'N/A')}, Description={getattr(atk, 'description', 'N/A')}"
                    )
        attacks_ctx = "\n".join(attacks_lines) if attacks_lines else "No active cyber-physical attacks currently injected."

        # 4. Cyber Threat Alerts
        alerts_lines = []
        if active_alerts:
            for al in active_alerts:
                evidence_str = ", ".join(al.evidence[:2]) if hasattr(al, "evidence") and al.evidence else "Telemetry anomaly"
                alerts_lines.append(
                    f"[{getattr(al, 'severity_level', 'HIGH')}] Alert {getattr(al, 'alert_id', 'N/A')}: {getattr(al, 'attack_type', 'THREAT')} on Bus-{getattr(al, 'target_bus_id', 'N/A')} "
                    f"(MITRE {getattr(al, 'mitre_technique_id', 'T0800')} {getattr(al, 'mitre_technique_name', '')}, TSI={getattr(al, 'threat_severity_index', 0.0):.1f}). Evidence: {evidence_str}"
                )
        alerts_ctx = "\n".join(alerts_lines) if alerts_lines else "0 active cyber threat alerts."

        # 5. Anomalies
        anomalies_lines = []
        if anomaly_report and hasattr(anomaly_report, "anomalies") and anomaly_report.anomalies:
            for an in anomaly_report.anomalies:
                anomalies_lines.append(
                    f"[{an.severity}] {an.entity_type} {an.entity_id}: {an.anomaly_type} - {an.message} (Value={an.value:.3f}, Threshold={an.threshold:.3f})"
                )
        anomalies_ctx = "\n".join(anomalies_lines) if anomalies_lines else "0 anomalies detected across grid buses."

        full_user_prompt = (
            f"OPERATOR QUERY: {user_query}\n\n"
            f"LIVE GRID CONTEXT:\n"
            f"- System Frequency: {sys_freq:.3f} Hz (Deviation: {freq_dev:.4f} Hz)\n"
            f"- Restoration Ratio: {restoration_pct:.1f}%\n"
            f"- Grid Health Score: {health_score:.1f}/100\n"
            f"- Total Active Feeder Buses: {len(buses_dict)}\n\n"
            f"ACTIVE CYBER ATTACKS:\n{attacks_ctx}\n\n"
            f"ACTIVE CYBER THREAT ALERTS:\n{alerts_ctx}\n\n"
            f"ACTIVE PHYSICAL & ML ANOMALIES:\n{anomalies_ctx}\n\n"
            f"LIVE BUS TELEMETRY & STATUS:\n{bus_summary_str}\n\n"
            f"RETRIEVED DOMAIN STANDARDS AND PLAYBOOKS:\n{rag_context_text}\n\n"
            "Instructions: Provide a clear, technically rigorous, and actionable response directly addressing the operator inquiry. "
            "If the question inquires about a specific bus (e.g. Bus-02 or Bus 4), extract its real-time telemetry, analyze any active issues/attacks/anomalies, and provide step-by-step engineering mitigation and self-healing solutions conforming to IEEE 1547 and NERC CIP."
        )

        response_text = self._ask_agent(
            (
                "You are the Cyber-Physical Smart Grid AI Assistant. You specialize in power system security, "
                "MITRE ATT&CK for ICS, IEEE 1547 / IEEE 1159 electrical standards, PyTorch GNN anomaly detection, "
                "and autonomous self-healing power grid restoration."
            ),
            full_user_prompt,
        )
        return {
            "response": response_text,
            "provider": self.llm_client.provider,
            "model_name": self.llm_client.model_name,
            "rag_docs": rag_docs,
        }
